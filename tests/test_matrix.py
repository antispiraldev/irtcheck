"""Response matrix construction, and the pseudo-respondent provenance that
leave-one-model-out depends on."""

from __future__ import annotations

import numpy as np
import pytest

from irtcheck.matrix import MatrixError, build_matrix, parse_respondent_key
from irtcheck.records import ResponseRecord


def rec(model, item, correct, **extra):
    return ResponseRecord(model_id=model, item_id=item, correct=correct, extra=extra)


def test_build_matrix_shapes_and_order():
    matrix = build_matrix([rec("a", "i1", 1), rec("a", "i2", 0), rec("b", "i1", 1)])
    assert matrix.respondent_ids == ["a", "b"]  # first-seen order
    assert matrix.item_ids == ["i1", "i2"]
    assert matrix.n_responses == 3
    assert matrix.n_real_models == 2


def test_ragged_matrices_are_fine():
    """Not every model answers every item. A fitter that only sees a full
    rectangle has not been tested against what it will get."""
    matrix = build_matrix([rec("a", "i1", 1), rec("a", "i2", 1), rec("b", "i1", 0)])
    assert matrix.density == 3 / 4
    np.testing.assert_array_equal(matrix.item_response_counts(), [2, 1])


def test_duplicate_pairs_raise_rather_than_dedupe():
    with pytest.raises(MatrixError, match="more than once"):
        build_matrix([rec("a", "i1", 1), rec("a", "i1", 0)])


def test_duplicate_message_suggests_a_finer_key():
    with pytest.raises(MatrixError, match="respondent-key"):
        build_matrix([rec("a", "i1", 1), rec("a", "i1", 0)])


def test_empty_input_raises():
    with pytest.raises(MatrixError, match="zero records"):
        build_matrix([])


# -- pseudo-respondents ------------------------------------------------------


def pseudo_records():
    """Two real models, two prompt variants each: four respondents, two models."""
    out = []
    for model in ("a", "b"):
        for variant in ("v1", "v2"):
            for item in ("i1", "i2", "i3"):
                out.append(rec(model, item, 1, prompt_variant=variant))
    return out


def test_pseudo_respondents_inflate_respondents_not_models():
    matrix = build_matrix(pseudo_records(), respondent_key=("model_id", "prompt_variant"))
    assert matrix.n_respondents == 4
    assert matrix.n_real_models == 2  # what belongs in a report header
    assert matrix.derives_from == ["a", "a", "b", "b"]


def test_drop_model_removes_every_variant_of_it():
    """The correctness trap in validate: holding out one row leaks the rest of
    that model's variants into the fit that chooses the anchor set."""
    matrix = build_matrix(pseudo_records(), respondent_key=("model_id", "prompt_variant"))
    held_out = matrix.drop_model("a")
    assert held_out.n_respondents == 2
    assert set(held_out.derives_from) == {"b"}
    assert "a" not in " ".join(held_out.respondent_ids).replace("b", "")


def test_drop_model_rejects_an_unknown_model():
    matrix = build_matrix([rec("a", "i1", 1)])
    with pytest.raises(MatrixError, match="known models: a"):
        matrix.drop_model("nope")


def test_one_respondent_id_claimed_by_two_models_raises():
    """A '|' already inside a field value would silently merge two respondents."""
    records = [rec("a|x", "i1", 1, prompt_variant="y"), rec("a", "i1", 1, prompt_variant="x|y")]
    with pytest.raises(MatrixError, match="does not uniquely identify"):
        build_matrix(records, respondent_key=("model_id", "prompt_variant"))


# -- accuracy / selection ----------------------------------------------------


def test_respondent_accuracy_is_the_full_suite_ground_truth():
    matrix = build_matrix(
        [rec("a", "i1", 1), rec("a", "i2", 1), rec("b", "i1", 1), rec("b", "i2", 0)]
    )
    np.testing.assert_allclose(matrix.respondent_accuracy(), [1.0, 0.5])


def test_select_items_preserves_requested_order():
    matrix = build_matrix([rec("a", f"i{n}", 1) for n in range(4)])
    subset = matrix.select_items(["i3", "i0"])
    assert subset.item_ids == ["i3", "i0"]
    assert subset.n_responses == 2


def test_select_items_rejects_unknown_ids():
    matrix = build_matrix([rec("a", "i1", 1)])
    with pytest.raises(MatrixError, match="not in this matrix"):
        matrix.select_items(["i1", "nope"])


# -- key parsing -------------------------------------------------------------


def test_parse_respondent_key():
    assert parse_respondent_key("model_id, prompt_variant") == ("model_id", "prompt_variant")


def test_respondent_key_must_include_model_id():
    """Otherwise validate cannot hold all of one model's variations out together."""
    with pytest.raises(MatrixError, match="must include model_id"):
        parse_respondent_key("prompt_variant")


def test_respondent_key_rejects_repeats_and_empties():
    with pytest.raises(MatrixError, match="repeats"):
        parse_respondent_key("model_id,model_id")
    with pytest.raises(MatrixError, match="cannot be empty"):
        parse_respondent_key(" , ")
