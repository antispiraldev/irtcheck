# Rebuilding the real response matrix

Three scripts that fetch HELM Lite's per-instance predictions and turn them
into irtcheck's input format. They exist so that the real-data numbers in
[`../docs/validation.md`](../docs/validation.md) §2 can be re-measured, which
they could not be when that section was first written.

**Nothing here is part of the installed package.** `irtcheck` never touches the
network at runtime — no API keys, no inference cost — and these scripts are the
one place in the repository that does. They need only the standard library.

```
python scripts/enum_helm.py      -o helm_runs.json        # ~15 requests
python scripts/fetch_helm.py     --runs helm_runs.json -o helm_responses.jsonl
python scripts/subsample_helm.py -i helm_responses.jsonl  -o helm_12.jsonl

irtcheck fit helm_12.jsonl        -o helm12.irt
irtcheck fit helm_responses.jsonl -o helm95.irt
```

`fetch_helm.py` makes 1,710 requests and downloads roughly 350 MB of JSON,
keeping only the two fields per record that are responses. It took about 35
seconds on a home connection with the default 24 workers, and reported 0 read
errors. The bucket, `gs://crfm-helm-public`, is public: no authentication and
no HuggingFace token.

## What is pinned, and why

The eighteen scenarios are **named** in `fetch_helm.py`'s `SCENARIOS`, and the
twelve-model shortlist is **named** in `subsample_helm.py`'s `SHORTLIST`.
Neither is derived at runtime. §2 originally described the matrix as "5 MMLU
subjects, OpenBookQA, 7 MATH level-1 subjects, 5 LegalBench subsets" and left
the exact subsets to be inferred, which was enough to make its numbers
impossible to reproduce later. Deriving the shortlist as "twelve models spaced
across the accuracy range" has the same problem in slower motion: it silently
picks a different twelve as soon as the bucket publishes another model.

`fetch_helm.py` fails rather than proceeding if any named scenario is missing
from the enumeration, and `subsample_helm.py` fails if any named model is
missing. A matrix with the right shape and the wrong contents is the outcome
worth spending an error message on.

`tests/test_helm_scripts.py` covers the parsing, which is where the one real
bug was: a run directory can carry parameters *after* the model name, as in
`...,model=amazon_nova-lite-v1:0,stop=none`, and folding those into the model
id turned 95 models into 103 with eight of them duplicated. The matrix still
looked plausible.

## Provenance of the numbers in docs/validation.md §2

Rebuilt 2026-09-12 from all 14 published `lite` versions, deduplicated to the
latest run per (scenario, model):

```
95 models × 3,551 items, 336,375 responses, 99.7% dense, 0 read errors
metrics used: exact_match, quasi_exact_match, math_equiv_chain_of_thought
every item answered by 93 to 95 of the 95 models
twelve-model cut: 12 × 3,551, 42,612 responses, 100% dense
```

The twelve-model cut matches the original build exactly — same 42,612
responses, and all twelve full-suite accuracies agree to three decimals, from
`tiiuae_falcon-7b` at 0.292 to `google_gemini-1.5-pro-002` at 0.778.

The 95-model matrix is **denser than the original**: 336,375 responses against
323,656, and 93–95 models per item against the 87–95 originally reported. The
scenarios and the model set are identical, so the difference is in how runs
were deduplicated across versions — this build takes the latest published run
for every (scenario, model) pair, which recovers a few runs the first one
missed. It is a rebuild, not a bit-for-bit reproduction, and §2 says so where
it matters.
