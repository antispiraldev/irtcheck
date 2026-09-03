"""The artifact schema — the contract four wave-1 briefs build against."""

from __future__ import annotations

import gzip
import json

import pytest

from irtcheck.artifact import (
    FLAG_CEILING,
    FLAG_DEAD,
    FLAG_FLOOR,
    FLAG_INSUFFICIENT_DATA,
    SCHEMA_VERSION,
    ArtifactError,
    IrtFit,
    Posterior,
    compute_flags,
)
from irtcheck.synth import synthetic_fit


def test_round_trip_is_byte_identical(tmp_path):
    """The contracts CI job depends on this: an artifact written twice from the
    same fit must produce identical bytes, or 'the numbers changed' becomes
    unanswerable by diff."""
    fit, _ = synthetic_fit(n_models=6, n_items=40, seed=3)
    first = fit.save(tmp_path / "a.irt")
    reloaded = IrtFit.load(first)
    second = reloaded.save(tmp_path / "b.irt")
    assert first.read_bytes() == second.read_bytes()


def test_round_trip_preserves_values(tmp_path):
    fit, _ = synthetic_fit(n_models=5, n_items=30, seed=1)
    reloaded = IrtFit.load(fit.save(tmp_path / "s.irt"))
    assert reloaded.item_ids == fit.item_ids
    assert reloaded.a.mean == pytest.approx(fit.a.mean)
    assert reloaded.derives_from == fit.derives_from
    assert reloaded.flags == fit.flags


def test_schema_version_leads_the_file(tmp_path):
    """A reader must be able to reject an artifact from the future before
    parsing fields it may not understand."""
    fit, _ = synthetic_fit(n_models=4, n_items=10, seed=0)
    path = fit.save(tmp_path / "s.irt")
    with gzip.open(path, "rb") as handle:
        data = json.loads(handle.read())
    assert next(iter(data)) == "schema_version"
    assert data["schema_version"] == SCHEMA_VERSION


def test_a_future_artifact_is_refused_with_a_next_step(tmp_path):
    fit, _ = synthetic_fit(n_models=4, n_items=10, seed=0)
    path = tmp_path / "future.irt"
    data = fit.to_dict()
    data["schema_version"] = SCHEMA_VERSION + 1
    with gzip.GzipFile(filename=path, mode="wb", mtime=0) as handle:
        handle.write(json.dumps(data).encode())
    with pytest.raises(ArtifactError, match="Re-run `irtcheck fit`"):
        IrtFit.load(path)


def test_corrupt_file_is_refused_clearly(tmp_path):
    path = tmp_path / "bad.irt"
    path.write_bytes(b"not gzip at all")
    with pytest.raises(ArtifactError):
        IrtFit.load(path)


# -- validation --------------------------------------------------------------


def test_mismatched_column_lengths_are_caught():
    fit, _ = synthetic_fit(n_models=4, n_items=10, seed=0)
    fit.n_resp = fit.n_resp[:-1]
    with pytest.raises(ArtifactError, match="n_resp has 9 entries"):
        fit.validate()


def test_dead_and_insufficient_data_are_mutually_exclusive():
    """'we are confident it does not discriminate' and 'we cannot tell' are
    different claims, and conflating them is the failure the refusal feature
    exists to prevent."""
    fit, _ = synthetic_fit(n_models=4, n_items=10, seed=0)
    fit.flags[0] = [FLAG_DEAD, FLAG_INSUFFICIENT_DATA]
    with pytest.raises(ArtifactError, match="mutually exclusive"):
        fit.validate()


def test_unknown_flags_are_caught():
    fit, _ = synthetic_fit(n_models=4, n_items=10, seed=0)
    fit.flags[0] = ["probably-fine"]
    with pytest.raises(ArtifactError, match="unknown item flag"):
        fit.validate()


def test_embedded_responses_are_index_checked():
    fit, _ = synthetic_fit(n_models=4, n_items=10, seed=0)
    fit.responses.cols[0] = 9999
    with pytest.raises(ArtifactError, match="item that does not exist"):
        fit.validate()


# -- flag derivation ---------------------------------------------------------


def post(mean, sd):
    return Posterior(
        mean=list(mean),
        sd=list(sd),
        hdi_low=[m - 1.96 * s for m, s in zip(mean, sd, strict=True)],
        hdi_high=[m + 1.96 * s for m, s in zip(mean, sd, strict=True)],
    )


def test_an_interval_spanning_zero_is_insufficient_data_not_dead():
    """This is the small-N case, and it is most of them. With a handful of
    respondents a flat item is indistinguishable from an unmeasured one, and
    the honest answer is 'we cannot tell', not 'it is dead'."""
    a = post([0.10], [0.30])  # interval clearly straddles 0
    flags = compute_flags(a, post([0.0], [0.2]), [0.5], [0.0, 1.0])
    assert flags[0] == [FLAG_INSUFFICIENT_DATA]
    assert FLAG_DEAD not in flags[0]


def test_dead_requires_a_narrow_interval_below_the_threshold():
    a = post([0.10], [0.02])  # tight, and confidently under DEAD_THRESHOLD
    flags = compute_flags(a, post([0.0], [0.2]), [0.5], [0.0, 1.0])
    assert flags[0] == [FLAG_DEAD]


def test_ceiling_and_floor_come_from_observed_rates():
    a = post([1.5, 1.5], [0.05, 0.05])
    b = post([0.0, 0.0], [0.1, 0.1])
    flags = compute_flags(a, b, [1.0, 0.0], [0.0, 1.0])
    assert FLAG_CEILING in flags[0]
    assert FLAG_FLOOR in flags[1]


def test_off_range_marks_an_item_measuring_where_nobody_sits():
    """The finding the information curve is drawn to show: the item may
    discriminate beautifully, just not for anyone in this matrix."""
    a = post([2.0], [0.05])
    b = post([9.0], [0.1])
    flags = compute_flags(a, b, [0.02], [0.0, 1.0])
    assert "off-range" in flags[0]


def test_usable_items_excludes_undecidable_and_degenerate_items():
    fit, _ = synthetic_fit(n_models=5, n_items=60, seed=7)
    usable = set(fit.usable_items())
    for flag in (FLAG_INSUFFICIENT_DATA, FLAG_CEILING, FLAG_FLOOR):
        assert usable.isdisjoint(fit.flagged(flag))
