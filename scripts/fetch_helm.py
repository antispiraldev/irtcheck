"""Fetch HELM Lite per-instance predictions and emit irtcheck's input format.

    python scripts/fetch_helm.py --runs helm_runs.json -o helm_responses.jsonl

One `display_predictions.json` per (scenario, model), deduplicated to the latest
published version of each. Records come out as irtcheck's three required fields
plus `subject`, which is what lets docs/validation.md group per-item output by
benchmark:

    {"model_id": ..., "item_id": "<scenario>_<instance_id>", "correct": 0|1,
     "subject": "<scenario>"}

## The eighteen scenarios, named rather than implied

docs/validation.md section 2 described the matrix as "5 MMLU subjects,
OpenBookQA, 7 MATH level-1 subjects, 5 LegalBench subsets" and left the exact
subsets to be inferred. That was enough to lose: after the interval change the
real-data numbers could not be re-measured, because nobody could rebuild the
same 3,551 items. Every subset is therefore listed in SCENARIOS below, and
`--runs` is checked against it so a silent drift in what the bucket publishes
fails loudly instead of producing a differently-shaped matrix with the same name.

## Metrics

`stats` carries several numbers per instance and only some are the response. The
order below is the same precedence `irtcheck fit --metric` uses, and a scenario
whose metric is absent is an error rather than a skipped row: a missing metric
silently dropping a whole scenario is how a 3,551-item matrix quietly becomes a
2,500-item one.

Values must be 0 or 1. HELM's `exact_match` is already binary; anything else
raises, because CLAUDE.md is explicit that a continuous score is never coerced
to binary silently.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

OBJECT = "https://storage.googleapis.com/crfm-helm-public/{prefix}display_predictions.json"

# The exact eighteen. Keyed by the scenario spec the bucket uses, valued by the
# short name that becomes the item_id prefix and the `subject` field.
SCENARIOS = {
    "mmlu:subject=abstract_algebra,method=multiple_choice_joint": "mmlu_abstract_algebra",
    "mmlu:subject=college_chemistry,method=multiple_choice_joint": "mmlu_college_chemistry",
    "mmlu:subject=computer_security,method=multiple_choice_joint": "mmlu_computer_security",
    "mmlu:subject=econometrics,method=multiple_choice_joint": "mmlu_econometrics",
    "mmlu:subject=us_foreign_policy,method=multiple_choice_joint": "mmlu_us_foreign_policy",
    "commonsense:dataset=openbookqa,method=multiple_choice_joint": "openbookqa",
    "math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True": "math_algebra",
    "math:subject=counting_and_probability,level=1,use_official_examples=False,use_chain_of_thought=True": "math_counting_and_probability",
    "math:subject=geometry,level=1,use_official_examples=False,use_chain_of_thought=True": "math_geometry",
    "math:subject=intermediate_algebra,level=1,use_official_examples=False,use_chain_of_thought=True": "math_intermediate_algebra",
    "math:subject=number_theory,level=1,use_official_examples=False,use_chain_of_thought=True": "math_number_theory",
    "math:subject=prealgebra,level=1,use_official_examples=False,use_chain_of_thought=True": "math_prealgebra",
    "math:subject=precalculus,level=1,use_official_examples=False,use_chain_of_thought=True": "math_precalculus",
    "legalbench:subset=abercrombie": "legalbench_abercrombie",
    "legalbench:subset=corporate_lobbying": "legalbench_corporate_lobbying",
    "legalbench:subset=function_of_decision_section": "legalbench_function_of_decision_section",
    "legalbench:subset=international_citizenship_questions": "legalbench_international_citizenship_questions",
    "legalbench:subset=proa": "legalbench_proa",
}

METRICS = ("exact_match", "quasi_exact_match", "math_equiv_chain_of_thought")


def version_key(version: str) -> tuple[int, ...]:
    """Semantic order, so v1.10.0 sorts above v1.9.0 rather than below it."""
    return tuple(int(x) for x in re.findall(r"\d+", version))


def rank(run: dict) -> tuple:
    """Sort key picking one run per (scenario, model), highest first.

    Version dominates. The tie-break matters because two runs in the *same*
    version can differ only in their trailing parameters -- `stop=none` and
    nothing -- and picking between them by dict iteration order would make the
    matrix depend on the order the bucket happened to list its objects in.
    Fewer parameters wins, then lexicographic, so the choice is reproducible.
    """
    return (version_key(run["version"]), -len(run.get("run_params", "")), run["prefix"])


def latest_runs(runs: list[dict]) -> list[dict]:
    """One run per (scenario spec, model): the most recently published."""
    best: dict[tuple[str, str], dict] = {}
    for run in runs:
        if run["scenario_spec"] not in SCENARIOS:
            continue
        key = (run["scenario_spec"], run["model"])
        current = best.get(key)
        if current is None or rank(run) > rank(current):
            best[key] = run
    return list(best.values())


def fetch(run: dict) -> tuple[dict, list[dict] | None, str | None]:
    url = OBJECT.format(prefix=run["prefix"])
    try:
        with urllib.request.urlopen(url, timeout=120) as fh:
            return run, json.load(fh), None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return run, None, f"{type(exc).__name__}: {exc}"


def records_from(run: dict, payload: list[dict]) -> tuple[list[dict], str]:
    scenario = SCENARIOS[run["scenario_spec"]]
    metric = None
    for candidate in METRICS:
        if any(candidate in (p.get("stats") or {}) for p in payload):
            metric = candidate
            break
    if metric is None:
        seen = sorted({k for p in payload for k in (p.get("stats") or {})})
        raise SystemExit(
            f"{run['prefix']}: none of {METRICS} present. Found: {seen}. "
            "A missing metric would silently drop this whole scenario, so this "
            "is fatal rather than skipped."
        )

    out = []
    for p in payload:
        stats = p.get("stats") or {}
        if metric not in stats or p.get("instance_id") is None:
            continue
        value = stats[metric]
        if value not in (0, 1, 0.0, 1.0):
            raise SystemExit(
                f"{run['prefix']}: {metric} = {value!r}, which is not binary. "
                "irtcheck never coerces a continuous score silently; fix the "
                "metric choice rather than thresholding here."
            )
        out.append(
            {
                "model_id": run["model"],
                "item_id": f"{scenario}_{p['instance_id']}",
                "correct": int(value),
                "subject": scenario,
            }
        )
    return out, metric


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="helm_runs.json")
    ap.add_argument("-o", "--output", default="helm_responses.jsonl")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    with open(args.runs) as fh:
        every = json.load(fh)["runs"]
    wanted = latest_runs(every)

    missing = set(SCENARIOS) - {r["scenario_spec"] for r in wanted}
    if missing:
        raise SystemExit(
            f"{len(missing)} of the 18 named scenarios are not in {args.runs}: "
            f"{sorted(missing)}. The bucket's contents have drifted; update "
            "SCENARIOS deliberately rather than fetching a different matrix."
        )
    print(f"{len(wanted)} runs to fetch across {len(SCENARIOS)} scenarios", file=sys.stderr)

    errors: list[str] = []
    metrics: dict[str, str] = {}
    per_scenario: dict[str, int] = defaultdict(int)
    models = set()
    written = 0

    with open(args.output, "w") as out, ThreadPoolExecutor(args.workers) as pool:
        for done, (run, payload, error) in enumerate(pool.map(fetch, wanted), start=1):
            if error is not None:
                errors.append(f"{run['prefix']}: {error}")
                continue
            rows, metric = records_from(run, payload)
            metrics[SCENARIOS[run["scenario_spec"]]] = metric
            per_scenario[SCENARIOS[run["scenario_spec"]]] += len(rows)
            models.add(run["model"])
            for row in rows:
                out.write(json.dumps(row) + "\n")
            written += len(rows)
            if done % 100 == 0:
                print(f"  {done}/{len(wanted)} runs, {written:,} responses", file=sys.stderr)

    print(
        f"\n{written:,} responses from {len(models)} models -> {args.output}\n"
        f"{len(errors)} read errors",
        file=sys.stderr,
    )
    for e in errors[:10]:
        print(f"  {e}", file=sys.stderr)
    print("metrics used: " + ", ".join(sorted(set(metrics.values()))), file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
