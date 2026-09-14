#!/usr/bin/env bash
# The README quickstart, run against an *installed* irtcheck rather than the repo.
#
# The release workflow runs this twice: on the wheel it just built, and again on
# the copy it downloads back from TestPyPI. Both times it runs from an empty
# temporary directory, so `import irtcheck` can only resolve to site-packages —
# a checkout on the path would make this pass on source the wheel never shipped.
#
#   bash .github/release/quickstart.sh 0.1.0
#
# The one argument is the version the installed package must report.
set -euo pipefail

expected="${1:?usage: quickstart.sh <expected version>}"
work="$(mktemp -d)"
cd "$work"

reported="$(irtcheck --version)"
if [ "$reported" != "irtcheck $expected" ]; then
  echo "::error::installed irtcheck reports '$reported', expected 'irtcheck $expected'"
  exit 1
fi
python - <<'EOF'
import pathlib
import irtcheck

location = pathlib.Path(irtcheck.__file__).resolve()
assert "site-packages" in location.parts, f"irtcheck imported from {location}, not an install"
print(f"irtcheck imported from {location.parent}")
EOF

# A small suite in exactly the three-field shape the README asks for, plus the
# prompt variant the quickstart composes into the respondent key.
python - <<'EOF'
import json

from irtcheck.synth import make_truth, responses_from_truth, synthetic_records

truth = make_truth(n_models=8, n_items=120, variants_per_model=2, seed=0)
records = synthetic_records(truth, responses_from_truth(truth, seed=1))
with open("responses.jsonl", "w", encoding="utf-8") as handle:
    for record in records:
        line = {"model_id": record.model_id, "item_id": record.item_id, "correct": record.correct}
        handle.write(json.dumps(line | record.extra) + "\n")
print(f"wrote {len(records)} responses")
EOF

# Fewer SVI steps than the default: this proves the commands run end to end on
# an installed package, and the fitter's accuracy is CI's tests-fit job's claim.
irtcheck fit responses.jsonl --respondent-key model_id,prompt_variant -o suite.irt --epochs 300
irtcheck report suite.irt --limit 5 --html report.html
irtcheck report suite.irt --json > report.json
irtcheck select suite.irt -n 10 -o anchor.json
irtcheck validate suite.irt --sizes 5,10 --epochs 150

python - <<'EOF'
import json
import pathlib

report = json.loads(pathlib.Path("report.json").read_text())
assert report["items"], "report --json produced no items"
html = pathlib.Path("report.html").read_text()
assert "<svg" in html and "http://" not in html.replace("http://www.w3.org", ""), "HTML report is not self-contained"
anchor = json.loads(pathlib.Path("anchor.json").read_text())
print(f"report: {len(report['items'])} items; html {len(html) // 1024} KiB; anchor {len(anchor['item_ids'])} items")
EOF
echo "quickstart OK for irtcheck $expected"
