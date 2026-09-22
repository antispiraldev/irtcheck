"""Cut single-subject slices out of the 95-model HELM Lite matrix.

Reads the `helm_responses.jsonl` that `scripts/fetch_helm.py` writes and writes
one JSONL per slice, in irtcheck's input format, beside it:

    citizenship  legalbench_international_citizenship_questions   1,000 items
    openbookqa   openbookqa                                         500 items
    mmlu         the five mmlu_* subjects together                  567 items
    no_citizenship  every scenario except citizenship             2,551 items

The first three are the largest subject groups in the suite; every other
scenario has 135 items or fewer, too few to compare sets of 25 to 400. The
fourth removes the scenario with the answer-format artefact (docs/validation.md
§2b), which is 28% of the suite.

    python studies/helm_selection/slices.py -i helm_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

SLICES = {
    "citizenship": lambda s: s == "legalbench_international_citizenship_questions",
    "openbookqa": lambda s: s == "openbookqa",
    "mmlu": lambda s: s.startswith("mmlu_"),
    "no_citizenship": lambda s: s != "legalbench_international_citizenship_questions",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-i", "--input", type=Path, required=True)
    args = parser.parse_args()
    counts = dict.fromkeys(SLICES, 0)
    with ExitStack() as stack:
        outs = {
            name: stack.enter_context(open(args.input.with_name(f"slice_{name}.jsonl"), "w"))
            for name in SLICES
        }
        for line in stack.enter_context(open(args.input)):
            subject = json.loads(line)["subject"]
            for name, wanted in SLICES.items():
                if wanted(subject):
                    outs[name].write(line)
                    counts[name] += 1
    for name in SLICES:
        print(f"{counts[name]:,} responses -> {args.input.with_name(f'slice_{name}.jsonl')}")


if __name__ == "__main__":
    main()
