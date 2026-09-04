"""lm-eval-harness per-sample logs.

**lm-eval only writes per-sample records when it is asked to.** Aggregate runs
keep task-level numbers and nothing else, so a finished run cannot be read
after the fact — it has to be re-run with both flags:

    lm_eval --model hf --model_args pretrained=<model> --tasks <tasks> \\
            --output_path runs/ --log_samples

That produces, per model:

    runs/<model_name_sanitized>/results_<timestamp>.json      aggregates
    runs/<model_name_sanitized>/samples_<task>_<timestamp>.jsonl   one per task

Point irtcheck at one `samples_*.jsonl`, or at the directory holding several of
them (a directory needs `--format lmeval`, because sniffing has to open a file):

    irtcheck fit runs/ --format lmeval -o suite.irt

A sample record — verified against lm-eval 0.4.13's `SampleResult` schema and
against real logs in the wild — looks like:

    {"doc_id": 0, "doc": {...}, "target": "18", "arguments": {...},
     "resps": [...], "filtered_resps": ["18"], "filter": "boxed-match",
     "metrics": ["exact_match"], "doc_hash": "...", "prompt_hash": "...",
     "target_hash": "...", "exact_match": 1.0}

Three things it does *not* carry, each of which this adapter has to solve:

`task`
    Absent. It is in the filename, `samples_<task>_<timestamp>.jsonl`, split
    the way lm-eval's own `get_file_task_name` splits it. Item ids are
    `<task>_<doc_id>` because `doc_id` is a per-task row index and collides
    across tasks — hellaswag's doc 0 and gsm8k's doc 0 are not one item.

`model_id`
    Absent. Taken from, in order: `$IRTCHECK_LMEVAL_MODEL_ID`; the
    `model_name` field of a `results_*.json` sitting beside the samples file;
    the name of the directory containing it, which lm-eval names after the
    sanitized model name. If the file has been moved out of that layout and
    the env var is unset, reading fails rather than inventing a respondent.

one row per (doc, filter)
    `filter` names the answer-extraction filter, and a task with several of
    them (gsm8k: `strict-match` and `flexible-extract`) logs every doc once per
    filter. Those are not separate items — they are the same item scored two
    ways — so the adapter keeps one filter per file (the first one it sees,
    or `$IRTCHECK_LMEVAL_FILTER`) and warns when it drops the others.

Similarly, `metrics` may list several per-sample metrics (`acc` and
`acc_norm`); the adapter prefers `acc`, then `exact_match`, and refuses to pick
between unfamiliar ones. `$IRTCHECK_LMEVAL_METRIC` settles it either way. A
continuous metric is not silently thresholded: it reaches `coerce_correct` and
is refused there.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterator
from pathlib import Path

from irtcheck.io import Reader, register
from irtcheck.records import RecordError, ResponseRecord, build_record

# Consulted in order when a sample logs more than one metric. Everything here
# is binary by construction; anything else is the user's call, not ours.
METRIC_PREFERENCE = ("acc", "exact_match", "em", "acc_norm")

SAMPLES_PREFIX = "samples_"
RESULTS_PREFIX = "results_"


def _task_name(path: Path) -> str:
    """`samples_gsm8k_2026-05-01T12-00-00.000000.jsonl` -> `gsm8k`.

    Mirrors lm-eval's `get_file_task_name`: everything between the first and
    last underscore, which survives task names that contain underscores
    (`mmlu_high_school_biology`) because the timestamp contains none. A file
    renamed out of that shape (people upload these as `samples.jsonl`) keeps
    its stem, so ids stay stable and readable either way.
    """
    stem = path.stem
    if stem.startswith(SAMPLES_PREFIX) and "_" in stem[len(SAMPLES_PREFIX) :]:
        return stem[len(SAMPLES_PREFIX) : stem.rfind("_")]
    if stem.startswith(SAMPLES_PREFIX):
        return stem[len(SAMPLES_PREFIX) :] or stem
    return stem


def _model_from_results(directory: Path) -> str | None:
    """The model name lm-eval recorded in its aggregate results file."""
    for results in sorted(directory.glob(f"{RESULTS_PREFIX}*.json")):
        try:
            data = json.loads(results.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("model_name")
        if isinstance(name, str) and name:
            return name
        config = data.get("config")
        if isinstance(config, dict):
            args = config.get("model_args")
            if isinstance(args, dict):
                for key in ("peft", "delta", "pretrained", "model", "path", "engine"):
                    if isinstance(args.get(key), str) and args[key]:
                        return str(args[key])
            elif isinstance(args, str):
                for key in ("peft", "delta", "pretrained", "model", "path", "engine"):
                    if f"{key}=" in args:
                        return args.split(f"{key}=", 1)[1].split(",")[0]
    return None


def _model_id(path: Path) -> str:
    override = os.environ.get("IRTCHECK_LMEVAL_MODEL_ID")
    if override:
        return override
    directory = path.parent
    from_results = _model_from_results(directory)
    if from_results:
        return from_results
    # lm-eval writes samples into <output_path>/<model_name_sanitized>/, so the
    # directory name is the model name with path-hostile characters replaced.
    name = directory.resolve().name
    if name and name not in (".", "/"):
        return name
    raise RecordError(
        "cannot tell which model produced this log: lm-eval sample records carry "
        "no model id, there is no results_*.json beside the file to read one from, "
        "and the containing directory has no usable name. Set "
        "IRTCHECK_LMEVAL_MODEL_ID, or keep the file in the "
        "<output_path>/<model_name>/ directory lm-eval wrote it to",
        source=str(path),
    )


def _metric_name(raw: dict, source: str, line: int) -> str:
    override = os.environ.get("IRTCHECK_LMEVAL_METRIC")
    metrics = raw.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        # Older writers omitted `metrics`; fall back to whichever numeric keys
        # are not part of the fixed schema.
        fixed = {
            "doc_id", "doc", "target", "arguments", "resps", "filtered_resps",
            "filter", "metrics", "doc_hash", "prompt_hash", "target_hash",
        }  # fmt: skip
        metrics = [k for k, v in raw.items() if k not in fixed and isinstance(v, (int, float))]
    if override:
        if override not in raw:
            raise RecordError(
                f"IRTCHECK_LMEVAL_METRIC={override!r} but this sample logs "
                f"{', '.join(map(str, metrics)) or '<no metrics>'}",
                source=source,
                line=line,
            )
        return override
    if len(metrics) == 1:
        return str(metrics[0])
    for preferred in METRIC_PREFERENCE:
        if preferred in metrics:
            return preferred
    raise RecordError(
        f"sample logs {len(metrics)} metrics ({', '.join(map(str, metrics))}) and none "
        f"is one irtcheck knows to be binary ({', '.join(METRIC_PREFERENCE)}). Choose "
        "one with IRTCHECK_LMEVAL_METRIC",
        source=source,
        line=line,
    )


def _read_samples_file(path: Path) -> Iterator[ResponseRecord]:
    task = _task_name(path)
    model_id = _model_id(path)
    wanted_filter = os.environ.get("IRTCHECK_LMEVAL_FILTER")
    chosen_filter: str | None = wanted_filter
    dropped: set[str] = set()
    kept = 0

    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RecordError(
                    f"line is not valid JSON: {exc.msg}", source=str(path), line=number
                ) from exc
            if not isinstance(raw, dict):
                raise RecordError(
                    f"expected a JSON object, got {type(raw).__name__}",
                    source=str(path),
                    line=number,
                )
            if "doc_id" not in raw:
                raise RecordError(
                    "no `doc_id` — this does not look like an lm-eval sample log. "
                    "lm-eval writes these only under --log_samples",
                    source=str(path),
                    line=number,
                )

            this_filter = raw.get("filter")
            if chosen_filter is None:
                chosen_filter = this_filter
            elif this_filter != chosen_filter:
                dropped.add(str(this_filter))
                continue

            metric = _metric_name(raw, str(path), number)
            doc = raw.get("doc") if isinstance(raw.get("doc"), dict) else {}
            subject = doc.get("subject")

            record = {
                "model_id": model_id,
                "item_id": f"{task}_{raw['doc_id']}",
                "correct": raw.get(metric),
                # The task is the coarsest honest grouping for a per-task log;
                # a doc that names its own subject (MMLU) overrides it.
                "subject": subject if isinstance(subject, str) and subject else task,
                "task": task,
                "doc_id": raw["doc_id"],
                "metric": metric,
            }
            if this_filter is not None:
                record["filter"] = this_filter
            kept += 1
            yield build_record(record, source=str(path), line=number)

    if wanted_filter is not None and kept == 0 and dropped:
        raise RecordError(
            f"IRTCHECK_LMEVAL_FILTER={wanted_filter!r} matched nothing; {path.name} "
            f"was scored with {', '.join(sorted(dropped))}",
            source=str(path),
        )
    if dropped:
        warnings.warn(
            f"{path.name} logs each doc once per filter; kept {chosen_filter!r} and "
            f"skipped {', '.join(sorted(dropped))}. They are the same items scored a "
            "different way, not extra items. Set IRTCHECK_LMEVAL_FILTER to keep a "
            "different one.",
            stacklevel=2,
        )


def _sample_files(directory: Path) -> list[Path]:
    files = sorted(directory.rglob(f"{SAMPLES_PREFIX}*.jsonl"))
    if not files:
        raise RecordError(
            f"no {SAMPLES_PREFIX}*.jsonl files under {directory}. lm-eval writes them "
            "only when run with --log_samples alongside --output_path; a run without "
            "that flag kept aggregate numbers only and has to be re-run",
            source=str(directory),
        )
    return files


def read_lmeval(path: Path) -> Iterator[ResponseRecord]:
    """Read one samples file, or every samples file under a directory."""
    if path.is_dir():
        for file in _sample_files(path):
            yield from _read_samples_file(file)
    else:
        yield from _read_samples_file(path)


def _sniff(path: Path, head: bytes) -> bool:
    for line in head.split(b"\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith(b"{"):
            return False
        try:
            raw = json.loads(stripped)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A sample record is far bigger than the sniff window, so the first
            # line is usually truncated. These keys are near the front of every
            # record lm-eval writes and absent from the reference format.
            return b'"doc_id"' in head and b'"doc"' in head and b'"item_id"' not in head
        if not isinstance(raw, dict):
            return False
        return "doc_id" in raw and "item_id" not in raw and "model_id" not in raw
    return False


register(
    Reader(
        name="lmeval",
        extensions=(".jsonl",),
        sniff=_sniff,
        read=read_lmeval,
        description="lm-eval-harness samples_<task>_<timestamp>.jsonl (--log_samples)",
    )
)
