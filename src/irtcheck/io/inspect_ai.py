"""Inspect (`inspect_ai`) eval logs.

Named `inspect_ai` and not `inspect` on purpose: `inspect` is a stdlib module,
and a module inside this package shadowing it for anything that does a plain
`import inspect` is a trap with no upside. The adapter's `--format` name is
`inspect_ai` for the same reason.

Unlike lm-eval, **Inspect records per-sample scores by default** — there is no
opt-in flag and no re-run. Every `EvalSample` carries `id`, `epoch` and
`scores: dict[str, Score]`, and the log header carries the model and task, so a
log written months ago is readable as-is.

Two on-disk forms, and this adapter reads both where the standard library can:

`.json`
    The whole `EvalLog` as one object: `{"eval": {...}, "samples": [...]}`.
    Written when `--log-format json` is set (or `INSPECT_LOG_FORMAT=json`).

`.eval`
    A zip: `header.json` plus one `samples/<id>_epoch_<n>.json` per sample.
    Older Inspect (0.3.50-era) deflate-compressed those members, which the
    stdlib reads. Current Inspect (0.3.200 onward at least) uses **zstandard**,
    which Python's `zipfile` cannot decompress before 3.14 and which irtcheck
    cannot add a dependency for. Such a file is refused with instructions
    rather than half-read:

        inspect log convert run.eval --to json --output-dir logs-json/
        inspect log dump run.eval > run.json

One log is one model on one task, so the interesting input is usually a
directory of them (a directory needs `--format inspect_ai`, because sniffing
has to open a file):

    irtcheck fit logs/ --format inspect_ai -o suite.irt

Decisions this adapter makes, all of which it will tell you about:

item ids
    `<task>_<sample id>`. Sample ids are unique within a task and routinely
    collide across tasks.

which scorer
    A sample may hold several scores, one per scorer, and no convention says
    which is the eval's headline. With one scorer it is used; with several the
    read fails and names them, and `$IRTCHECK_INSPECT_SCORER` chooses.

score values
    `Score.value` is a str, bool, number, list or dict. `"C"` -> 1 and `"I"`
    -> 0; `"N"` (no answer) -> 0, matching Inspect's own `value_to_float`,
    which is also what the accuracy the user already reports does with it.
    Bools and 0/1 numbers go through `coerce_correct`. Everything else is
    refused: `"P"` (partial credit) and a continuous value both mean a
    threshold, and irtcheck does not pick thresholds.

epochs
    `--epochs 3` scores each sample three times, which is the same item
    answered three times by one model — a duplicate response, not a new item.
    The epoch is kept on the record, so the honest reading is to make the
    repeats pseudo-respondents:

        irtcheck fit logs/ --format inspect_ai --respondent-key model_id,epoch
"""

from __future__ import annotations

import json
import os
import sys
import warnings
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from irtcheck.io import FormatError, Reader, register
from irtcheck.records import RecordError, ResponseRecord, build_record, coerce_correct

HEADER_JSON = "header.json"
SAMPLES_DIR = "samples/"

# zipfile's number for zstandard. Python gained it in 3.14; before that the
# member is unreadable and there is nothing to do but say so precisely.
ZIP_ZSTANDARD = 93


def _supported_compressions() -> set[int]:
    """Compression methods this interpreter's zipfile can inflate.

    Checked before reading rather than after: `zipfile.open()` raises a bare
    NotImplementedError from several frames down, which tells a user nothing
    about `inspect log convert`.
    """
    supported = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
    for name in ("ZIP_BZIP2", "ZIP_LZMA", "ZIP_ZSTANDARD"):
        method = getattr(zipfile, name, None)
        if method is not None:
            supported.add(method)
    return supported


# Inspect's grade letters (inspect_ai.scorer._metric).
CORRECT, INCORRECT, PARTIAL, NOANSWER = "C", "I", "P", "N"


def _score_to_correct(value: Any, *, scorer: str, source: str) -> int:
    if isinstance(value, str):
        letter = value.strip()
        if letter.upper() == CORRECT:
            return 1
        if letter.upper() in (INCORRECT, NOANSWER):
            return 0
        if letter.upper() == PARTIAL:
            raise RecordError(
                f"scorer {scorer!r} returned {PARTIAL!r} (partial credit). A 2PL model "
                "needs a binary response, and irtcheck will not decide whether half "
                "marks count as correct — rescore the eval with a binary scorer",
                source=source,
            )
        return coerce_correct(letter, source=source)
    if isinstance(value, (list, dict)):
        raise RecordError(
            f"scorer {scorer!r} returned a {type(value).__name__} of values, which is "
            "not one graded response. Score the eval with a scorer that returns a "
            "single value, or pick one with IRTCHECK_INSPECT_SCORER",
            source=source,
        )
    return coerce_correct(value, source=source)


def _scorer_name(scores: dict[str, Any], *, source: str, sample_id: Any) -> str:
    override = os.environ.get("IRTCHECK_INSPECT_SCORER")
    if override:
        if override not in scores:
            raise RecordError(
                f"IRTCHECK_INSPECT_SCORER={override!r} but sample {sample_id!r} was "
                f"scored by {', '.join(sorted(scores))}",
                source=source,
            )
        return override
    if len(scores) == 1:
        return next(iter(scores))
    raise RecordError(
        f"sample {sample_id!r} has {len(scores)} scores ({', '.join(sorted(scores))}) "
        "and nothing in the log says which one is the eval's headline. Choose with "
        "IRTCHECK_INSPECT_SCORER",
        source=source,
    )


def _spec(header: dict[str, Any], source: str) -> tuple[str, str]:
    """(model_id, task) from an EvalLog header."""
    spec = header.get("eval")
    if not isinstance(spec, dict):
        raise FormatError(f"no `eval` section in {source}: this is not an Inspect log")
    model = spec.get("model")
    task = spec.get("task")
    if not isinstance(model, str) or not model:
        raise RecordError("the log header records no model", source=source)
    if not isinstance(task, str) or not task:
        task = Path(source).stem
    return model, task


def _records(
    header: dict[str, Any], samples: Iterator[dict[str, Any]], source: str
) -> Iterator[ResponseRecord]:
    model_id, task = _spec(header, source)
    unscored = 0
    for sample in samples:
        if not isinstance(sample, dict):
            raise RecordError(
                f"expected a sample object, got {type(sample).__name__}", source=source
            )
        sample_id = sample.get("id")
        if sample_id is None:
            raise RecordError("sample has no `id`", source=source)
        scores = sample.get("scores")
        if not isinstance(scores, dict) or not scores:
            # A sample that errored carries no score. That is a missing
            # observation, and the matrix is sparse by construction — it is not
            # a zero, and it is not a reason to refuse the other 499 samples.
            unscored += 1
            continue
        scorer = _scorer_name(scores, source=source, sample_id=sample_id)
        score = scores[scorer]
        if not isinstance(score, dict) or "value" not in score:
            raise RecordError(
                f"score {scorer!r} on sample {sample_id!r} has no `value`", source=source
            )
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        subject = metadata.get("subject")
        record = {
            "model_id": model_id,
            "item_id": f"{task}_{sample_id}",
            "correct": _score_to_correct(score["value"], scorer=scorer, source=source),
            "subject": subject if isinstance(subject, str) and subject else task,
            "task": task,
            "sample_id": sample_id,
            "scorer": scorer,
        }
        if sample.get("epoch") is not None:
            record["epoch"] = sample["epoch"]
        yield build_record(record, source=source)
    if unscored:
        warnings.warn(
            f"{unscored} sample(s) in {source} carry no score and were skipped. They "
            "are missing observations, not incorrect answers.",
            stacklevel=2,
        )


def _read_json_log(path: Path) -> Iterator[ResponseRecord]:
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecordError(f"file is not valid JSON: {exc.msg}", source=str(path)) from exc
    if not isinstance(log, dict):
        raise RecordError(f"expected a JSON object, got {type(log).__name__}", source=str(path))
    samples = log.get("samples")
    if samples is None:
        raise RecordError(
            "the log has no `samples` — it was written header-only "
            "(`inspect log dump --header-only`, or a log still being written)",
            source=str(path),
        )
    if not isinstance(samples, list):
        raise RecordError(f"`samples` is a {type(samples).__name__}, not a list", source=str(path))
    yield from _records(log, iter(samples), str(path))


def _read_eval_log(path: Path) -> Iterator[ResponseRecord]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        supported = _supported_compressions()
        unreadable = sorted(
            {i.compress_type for i in archive.infolist() if i.compress_type not in supported}
        )
        if unreadable:
            how = (
                "zstandard-compressed, which Python's zipfile only reads from 3.14 "
                f"(this interpreter is {sys.version_info.major}.{sys.version_info.minor})"
                if unreadable == [ZIP_ZSTANDARD]
                else f"compressed with method(s) {unreadable}, which this Python cannot read"
            )
            raise FormatError(
                f"{path.name} is {how}, and irtcheck will not take a dependency to "
                "read one. Convert it first:\n"
                f"    inspect log convert {path} --to json --output-dir logs-json/\n"
                f"    inspect log dump {path} > {path.stem}.json"
            )
        if HEADER_JSON not in names:
            raise FormatError(
                f"{path.name} is a zip but has no {HEADER_JSON}: not an Inspect .eval log"
            )
        header = json.loads(archive.read(HEADER_JSON))
        sample_members = sorted(
            n for n in names if n.startswith(SAMPLES_DIR) and n.endswith(".json")
        )
        if not sample_members:
            raise RecordError(
                f"{path.name} contains no samples/ members — the run recorded no samples",
                source=str(path),
            )

        def samples() -> Iterator[dict[str, Any]]:
            for name in sample_members:
                yield json.loads(archive.read(name))

        yield from _records(header, samples(), str(path))


def _log_files(directory: Path) -> list[Path]:
    files = []
    for candidate in sorted(directory.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".eval":
            files.append(candidate)
        elif candidate.suffix.lower() == ".json":
            with candidate.open("rb") as handle:
                if _sniff(candidate, handle.read(8192)):
                    files.append(candidate)
    if not files:
        raise FormatError(
            f"no Inspect logs under {directory} (looked for .eval files and .json logs "
            "with an `eval` section)"
        )
    return files


def read_inspect(path: Path) -> Iterator[ResponseRecord]:
    """Read one Inspect log, or every log under a directory."""
    if path.is_dir():
        for file in _log_files(path):
            yield from read_inspect(file)
    elif zipfile.is_zipfile(path):
        yield from _read_eval_log(path)
    else:
        yield from _read_json_log(path)


def _sniff(path: Path, head: bytes) -> bool:
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        except (zipfile.BadZipFile, OSError):
            return False
        return HEADER_JSON in names and any(n.startswith(SAMPLES_DIR) for n in names)
    stripped = head.lstrip()
    if not stripped.startswith(b"{"):
        return False
    # An EvalLog's `eval` section (with the task and model in it) is the first
    # thing after version/status, so it is inside the sniff window even for a
    # log with thousands of samples.
    return b'"eval"' in head and (b'"task"' in head or b'"samples"' in head)


register(
    Reader(
        name="inspect_ai",
        extensions=(".eval", ".json"),
        sniff=_sniff,
        read=read_inspect,
        description="Inspect eval logs (.json, or .eval where the stdlib can unzip it)",
    )
)
