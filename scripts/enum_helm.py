"""List every HELM Lite run prefix, so a response matrix can be rebuilt.

    python scripts/enum_helm.py -o helm_runs.json

Writes one record per (version, scenario, model) run directory. Nothing here is
part of the installed package: irtcheck never touches the network at runtime,
and these three scripts exist only to rebuild the matrix docs/validation.md
section 2 reports on. They were missing when that section was written, which is
why its numbers could not be re-measured after the interval change in #10.

The bucket is public and unauthenticated. `gs://crfm-helm-public` serves a
plain JSON object-listing API, and a run directory is named

    lite/benchmark_output/runs/<version>/<scenario spec>,model=<model>/

where `<scenario spec>` is itself comma-separated, e.g.
`mmlu:subject=anatomy,method=multiple_choice_joint`. Delimiting on "/" at each
level keeps this to a handful of requests rather than one per object.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from typing import Any

BUCKET = "crfm-helm-public"
ROOT = "lite/benchmark_output/runs/"
LIST = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"


def listing(prefix: str, delimiter: str = "/") -> dict[str, Any]:
    """One page-following call to the object-listing API."""
    out: dict[str, Any] = {"prefixes": [], "items": []}
    token = None
    while True:
        query = {"prefix": prefix, "delimiter": delimiter, "maxResults": "5000"}
        if token:
            query["pageToken"] = token
        with urllib.request.urlopen(f"{LIST}?{urllib.parse.urlencode(query)}") as fh:
            page = json.load(fh)
        out["prefixes"].extend(page.get("prefixes", []))
        out["items"].extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return out


def parse_run(prefix: str) -> dict[str, str] | None:
    """Split a run directory into scenario spec, model, and run parameters.

    `,model=` is the separator, and it is the last one that matters: a scenario
    spec contains commas of its own, so splitting on the first comma would cut
    the spec in half.

    **The model name ends at the next comma, and getting that wrong costs eight
    models.** A run directory may carry parameters *after* the model, as in
    `...,model=amazon_nova-lite-v1:0,stop=none`. Treating the whole tail as the
    model makes `amazon_nova-lite-v1:0` and `amazon_nova-lite-v1:0,stop=none`
    two different respondents, which inflates the matrix from 95 models to 103
    and puts the same model in twice with two abilities. Model ids here never
    contain a comma; run parameters always do.
    """
    name = prefix[len(ROOT) :].rstrip("/")
    version, _, run = name.partition("/")
    if not run or ",model=" not in run:
        return None
    spec, _, tail = run.rpartition(",model=")
    model, _, params = tail.partition(",")
    return {
        "version": version,
        "run": run,
        "scenario_spec": spec,
        # The scenario family is everything before the first ":" or ",", which
        # is what groups `mmlu:subject=anatomy,...` under `mmlu`.
        "scenario": spec.split(":")[0].split(",")[0],
        "model": model,
        # Kept rather than discarded, so a run that differs only by these is
        # visibly a duplicate of the same model instead of a mystery.
        "run_params": params,
        "prefix": prefix,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default="helm_runs.json")
    args = ap.parse_args()

    versions = [p for p in listing(ROOT)["prefixes"]]
    print(f"{len(versions)} published versions", file=sys.stderr)

    runs = []
    for version in versions:
        found = [parse_run(p) for p in listing(version)["prefixes"]]
        found = [r for r in found if r]
        runs.extend(found)
        print(f"  {version}: {len(found)} runs", file=sys.stderr)

    scenarios: dict[str, int] = {}
    models = set()
    for r in runs:
        scenarios[r["scenario"]] = scenarios.get(r["scenario"], 0) + 1
        models.add(r["model"])

    with open(args.output, "w") as fh:
        json.dump({"runs": runs, "scenarios": scenarios}, fh, indent=1)
    print(
        f"\n{len(runs)} runs, {len(models)} distinct models, "
        f"{len(scenarios)} scenario families -> {args.output}",
        file=sys.stderr,
    )
    for name, count in sorted(scenarios.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>5}  {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
