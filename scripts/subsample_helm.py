"""Cut the 95-model HELM matrix to the twelve-model shortlist docs report on.

    python scripts/subsample_helm.py -i helm_responses.jsonl -o helm_12.jsonl

Twelve models spanning the observed accuracy range, keeping only items all
twelve answered -- a dense rectangle, which is what a team comparing a
shortlist actually has, and the case irtcheck is designed around.

The twelve are **named** rather than recomputed. docs/validation.md section 2a
lists them with their full-suite accuracies, and re-deriving them by "spacing
twelve evenly across the range" would silently pick a different set as soon as
the bucket publishes another model -- which would quietly make the table
incomparable with the one it is supposed to update.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

# From docs/validation.md section 2a, in its stated accuracy order.
SHORTLIST = [
    "tiiuae_falcon-7b",
    "anthropic_claude-3-haiku-20240307",
    "meta_llama-3.2-11b-vision-instruct-turbo",
    "cohere_command-r",
    "mistralai_mistral-7b-v0.1",
    "qwen_qwen1.5-7b",
    "microsoft_phi-3-small-8k-instruct",
    "google_gemini-2.0-flash-exp",
    "meta_llama-3.2-90b-vision-instruct-turbo",
    "meta_llama-3.1-405b-instruct-turbo",
    "meta_llama-3-70b",
    "google_gemini-1.5-pro-002",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", default="helm_responses.jsonl")
    ap.add_argument("-o", "--output", default="helm_12.jsonl")
    ap.add_argument(
        "--keep-ragged",
        action="store_true",
        help="keep items not all twelve answered, instead of a dense rectangle",
    )
    args = ap.parse_args()

    wanted = set(SHORTLIST)
    rows = []
    answered: dict[str, set[str]] = collections.defaultdict(set)
    with open(args.input) as fh:
        for line in fh:
            r = json.loads(line)
            if r["model_id"] in wanted:
                rows.append(r)
                answered[r["item_id"]].add(r["model_id"])

    present = {r["model_id"] for r in rows}
    if present != wanted:
        raise SystemExit(
            f"{len(wanted - present)} of the twelve named models are not in "
            f"{args.input}: {sorted(wanted - present)}. Section 2a's table "
            "cannot be updated from a different shortlist; decide deliberately."
        )

    if args.keep_ragged:
        keep = set(answered)
    else:
        keep = {item for item, models in answered.items() if len(models) == len(wanted)}

    accuracy: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    written = 0
    with open(args.output, "w") as out:
        for r in rows:
            if r["item_id"] not in keep:
                continue
            out.write(json.dumps(r) + "\n")
            a = accuracy[r["model_id"]]
            a[0] += r["correct"]
            a[1] += 1
            written += 1

    print(
        f"{len(wanted)} models x {len(keep)} items, {written:,} responses "
        f"({written / (len(wanted) * len(keep)):.1%} dense) -> {args.output}",
        file=sys.stderr,
    )
    print(f"{len(answered) - len(keep)} items dropped as not answered by all twelve", file=sys.stderr)
    print("\nfull-suite accuracy on the retained items:", file=sys.stderr)
    for model in SHORTLIST:
        correct, total = accuracy[model]
        print(f"  {correct / total:.3f}  {model}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
