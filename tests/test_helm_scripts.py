"""The HELM rebuild scripts, tested on their parsing rather than the network.

`scripts/` is not part of the installed package — irtcheck never touches the
network at runtime — but these three scripts are what makes the real-data
numbers in docs/validation.md section 2 reproducible, and their absence is why
those numbers could not be re-measured after the interval change. So the parts
that decide *what matrix comes out* are pinned here.

The bug that motivates most of this file: a HELM run directory may carry
parameters after the model name, as in
`...,model=amazon_nova-lite-v1:0,stop=none`. Treating the whole tail as the
model turns one respondent into two, and it did — the first rebuild came out
with 103 models instead of 95, each duplicated model getting its own ability.
Nothing failed; the matrix was just quietly wrong. That is precisely the shape
of bug a test has to hold down, because the output still looks plausible.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name: str):
    """Import a script by path; `scripts/` is deliberately not a package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


enum_helm = load("enum_helm")
fetch_helm = load("fetch_helm")
subsample_helm = load("subsample_helm")

ROOT = "lite/benchmark_output/runs/"


# -- run-directory parsing ---------------------------------------------------


def test_a_plain_run_splits_into_scenario_and_model():
    run = enum_helm.parse_run(f"{ROOT}v1.5.0/legalbench:subset=proa,model=meta_llama-3-70b/")
    assert run["version"] == "v1.5.0"
    assert run["scenario_spec"] == "legalbench:subset=proa"
    assert run["scenario"] == "legalbench"
    assert run["model"] == "meta_llama-3-70b"
    assert run["run_params"] == ""


def test_trailing_run_parameters_are_not_part_of_the_model():
    """The 103-vs-95 bug, pinned.

    `stop=none` is a property of the run, not a different model. Folding it
    into the model id makes one model into two respondents with two abilities,
    which inflates the matrix and is invisible in the output.
    """
    run = enum_helm.parse_run(
        f"{ROOT}v1.13.0/commonsense:dataset=openbookqa,method=multiple_choice_joint"
        ",model=amazon_nova-lite-v1:0,stop=none/"
    )
    assert run["model"] == "amazon_nova-lite-v1:0"
    assert run["run_params"] == "stop=none"
    assert run["scenario_spec"] == "commonsense:dataset=openbookqa,method=multiple_choice_joint"


def test_a_scenario_spec_keeps_its_own_commas():
    """Splitting on the *first* comma would cut the spec in half."""
    spec = (
        "math:subject=counting_and_probability,level=1"
        ",use_official_examples=False,use_chain_of_thought=True"
    )
    run = enum_helm.parse_run(f"{ROOT}v1.9.0/{spec},model=qwen_qwen1.5-7b/")
    assert run["scenario_spec"] == spec
    assert run["scenario"] == "math"
    assert run["model"] == "qwen_qwen1.5-7b"


def test_a_directory_with_no_model_is_skipped():
    assert enum_helm.parse_run(f"{ROOT}v1.0.0/something-else/") is None


# -- version ordering and deduplication --------------------------------------


def test_versions_sort_numerically_not_lexicographically():
    """v1.10.0 is newer than v1.9.0, which string comparison gets backwards."""
    assert fetch_helm.version_key("v1.10.0") > fetch_helm.version_key("v1.9.0")
    assert fetch_helm.version_key("v1.2.0") > fetch_helm.version_key("v1.1.0")
    ordered = sorted(["v1.9.0", "v1.13.0", "v1.0.0", "v1.2.0"], key=fetch_helm.version_key)
    assert ordered == ["v1.0.0", "v1.2.0", "v1.9.0", "v1.13.0"]


def spec_of(name: str) -> str:
    return next(s for s, short in fetch_helm.SCENARIOS.items() if short == name)


def run_for(version: str, scenario: str, model: str, params: str = "") -> dict:
    spec = spec_of(scenario)
    tail = f"{model},{params}" if params else model
    return {
        "version": version,
        "scenario_spec": spec,
        "model": model,
        "run_params": params,
        "prefix": f"{ROOT}{version}/{spec},model={tail}/",
    }


def test_the_latest_version_of_each_run_wins():
    runs = [
        run_for("v1.0.0", "openbookqa", "m"),
        run_for("v1.13.0", "openbookqa", "m"),
        run_for("v1.9.0", "openbookqa", "m"),
    ]
    kept = fetch_helm.latest_runs(runs)
    assert len(kept) == 1
    assert kept[0]["version"] == "v1.13.0"


def test_two_runs_of_one_model_in_one_version_resolve_deterministically():
    """Same scenario, same model, same version, differing only in parameters.

    Whichever is chosen, it must not depend on the order the bucket listed
    them: the matrix would then differ between two runs of the same script.
    """
    plain = run_for("v1.13.0", "openbookqa", "m")
    stopped = run_for("v1.13.0", "openbookqa", "m", "stop=none")
    first = fetch_helm.latest_runs([plain, stopped])
    second = fetch_helm.latest_runs([stopped, plain])
    assert len(first) == len(second) == 1
    assert first[0]["prefix"] == second[0]["prefix"]
    # Fewer parameters wins, so the plain run is the canonical one.
    assert first[0]["run_params"] == ""


def test_scenarios_outside_the_eighteen_are_dropped():
    other = {
        "version": "v1.0.0",
        "scenario_spec": "wmt_14:language_pair=de-en",
        "model": "m",
        "run_params": "",
        "prefix": "x",
    }
    assert fetch_helm.latest_runs([other]) == []


# -- the eighteen scenarios --------------------------------------------------


def test_there_are_exactly_eighteen_named_scenarios():
    """Named, not inferred. Section 2 said "5 MMLU subjects, OpenBookQA, 7 MATH
    level-1 subjects, 5 LegalBench subsets" and left the subsets to guesswork,
    which is how its numbers became unreproducible."""
    assert len(fetch_helm.SCENARIOS) == 18
    families: dict[str, int] = {}
    for spec in fetch_helm.SCENARIOS:
        family = spec.split(":")[0]
        families[family] = families.get(family, 0) + 1
    assert families == {"mmlu": 5, "commonsense": 1, "math": 7, "legalbench": 5}


def test_every_scenario_has_a_distinct_short_name():
    names = list(fetch_helm.SCENARIOS.values())
    assert len(set(names)) == len(names)


def test_the_shortlist_is_twelve_distinct_models():
    assert len(subsample_helm.SHORTLIST) == 12
    assert len(set(subsample_helm.SHORTLIST)) == 12


# -- record construction -----------------------------------------------------


def test_records_carry_the_three_required_fields_and_the_subject():
    run = run_for("v1.13.0", "openbookqa", "meta_llama-3-70b")
    payload = [
        {"instance_id": "id7", "stats": {"exact_match": 1.0, "num_prompt_tokens": 300.0}},
        {"instance_id": "id8", "stats": {"exact_match": 0.0}},
    ]
    rows, metric = fetch_helm.records_from(run, payload)
    assert metric == "exact_match"
    assert rows == [
        {"model_id": "meta_llama-3-70b", "item_id": "openbookqa_id7", "correct": 1,
         "subject": "openbookqa"},
        {"model_id": "meta_llama-3-70b", "item_id": "openbookqa_id8", "correct": 0,
         "subject": "openbookqa"},
    ]


def test_item_ids_are_namespaced_by_scenario():
    """instance_id is only stable *within* a scenario, so the prefix is what
    stops two scenarios' `id0` from being treated as one item."""
    a = fetch_helm.records_from(
        run_for("v1.0.0", "mmlu_econometrics", "m"),
        [{"instance_id": "id0", "stats": {"exact_match": 1}}],
    )[0]
    b = fetch_helm.records_from(
        run_for("v1.0.0", "openbookqa", "m"),
        [{"instance_id": "id0", "stats": {"exact_match": 1}}],
    )[0]
    assert a[0]["item_id"] != b[0]["item_id"]


def test_a_continuous_metric_is_refused_rather_than_thresholded():
    """CLAUDE.md: never coerce a continuous score to binary silently."""
    run = run_for("v1.0.0", "openbookqa", "m")
    with pytest.raises(SystemExit, match="not binary"):
        fetch_helm.records_from(run, [{"instance_id": "id1", "stats": {"exact_match": 0.63}}])


def test_a_scenario_with_no_known_metric_is_fatal_not_skipped():
    """Skipping would silently shrink the matrix by a whole scenario."""
    run = run_for("v1.0.0", "openbookqa", "m")
    with pytest.raises(SystemExit, match="none of"):
        fetch_helm.records_from(run, [{"instance_id": "id1", "stats": {"bleu": 0.4}}])


def test_the_metric_precedence_is_honoured():
    """A scenario logging several metrics must resolve to the first listed, not
    to whichever happens to come out of dict ordering."""
    run = run_for("v1.0.0", "openbookqa", "m")
    payload = [{"instance_id": "id1", "stats": {"quasi_exact_match": 0, "exact_match": 1}}]
    rows, metric = fetch_helm.records_from(run, payload)
    assert metric == fetch_helm.METRICS[0] == "exact_match"
    assert rows[0]["correct"] == 1


def test_an_instance_without_the_metric_is_dropped_not_guessed():
    run = run_for("v1.0.0", "openbookqa", "m")
    payload = [
        {"instance_id": "id1", "stats": {"exact_match": 1}},
        {"instance_id": "id2", "stats": {}},
        {"instance_id": None, "stats": {"exact_match": 1}},
    ]
    rows, _ = fetch_helm.records_from(run, payload)
    assert [r["item_id"] for r in rows] == ["openbookqa_id1"]
