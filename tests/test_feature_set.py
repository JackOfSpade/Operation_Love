"""`ranker.feature_set`: config, the label loader that builds each feature set, and the model.

`face_crop_v1` must stay exactly today's rows; `face_crop_whole_photo_v1` appends each label's
whole-photo vector and refuses (FeatureSetIncomplete -> NOT READY) while any label lacks one.
"""

import math
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

import operation_love.ranker.evaluate as evaluate_mod
from operation_love import config as c
from operation_love.ranker import threshold as thr
from operation_love.ranker.model import PreferenceModel
from operation_love.ranker.store import SQLiteStore
from operation_love.vision.photo_cards import (
    WHOLE_PHOTO_DIMS,
    WHOLE_PHOTO_FEATURE_VERSION,
)

WHOLE = thr.FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1


class _FeatureStore:
    """Records which loader was used; serves fixed rows."""

    def __init__(self, rows=(), feature_rows=(), stats=None):
        self.rows, self.feature_rows, self.stats = list(rows), list(feature_rows), stats
        self.calls = []

    def load_label_rows(self):
        self.calls.append("load_label_rows")
        return self.rows

    def load_labels(self):
        self.calls.append("load_labels")
        return [row[:2] for row in self.rows]

    def load_label_feature_rows(self, feature_version):
        self.calls.append(("load_label_feature_rows", feature_version))
        return self.feature_rows

    def auto_review_stats(self, **kwargs):
        self.calls.append("auto_review_stats")
        return self.stats


def _photo(value):
    return [value] * WHOLE_PHOTO_DIMS


# ------------------------------------------------------------------ constants
def test_feature_sets_are_the_two_documented_names_with_v1_the_default():
    assert thr.FEATURE_SETS == ("face_crop_v1", "face_crop_whole_photo_v1")
    assert thr.DEFAULT_FEATURE_SET == thr.FEATURE_SET_FACE_CROP_V1 == "face_crop_v1"
    assert c.RankerCfg().feature_set == thr.DEFAULT_FEATURE_SET
    assert WHOLE_PHOTO_FEATURE_VERSION == "whole_photo_v1" and WHOLE_PHOTO_DIMS == 768


@pytest.mark.parametrize("bad", [None, "", "face_crop_v2", "FACE_CROP_V1", 1, ["x"]])
def test_require_feature_set_rejects_anything_unknown(bad):
    with pytest.raises(ValueError, match="feature set"):
        thr.require_feature_set(bad)


# ------------------------------------------------------------------ face_crop_v1: unchanged
def test_v1_is_load_label_rows_untouched_and_never_reads_features():
    rows = [(True, [1.0, 2.0], "manual"), (False, [3.0, 4.0], "auto_review_random")]
    store = _FeatureStore(rows=rows)
    assert thr.fetch_label_rows(store) is store.rows
    assert thr.fetch_label_rows(store, "face_crop_v1") is store.rows
    assert store.calls == ["load_label_rows", "load_label_rows"]

    class PairsOnly:
        def load_labels(self):
            return [(False, [2.0])]

    assert thr.fetch_label_rows(PairsOnly(), "face_crop_v1") == [(False, [2.0])]


def test_an_unknown_feature_set_is_refused_before_any_read():
    store = _FeatureStore()
    with pytest.raises(ValueError, match="feature set"):
        thr.fetch_label_rows(store, "whole_photo_v1")
    assert store.calls == []


# ------------------------------------------------------------------ face_crop_whole_photo_v1
def test_whole_photo_rows_are_v1_then_photo_in_label_order():
    store = _FeatureStore(
        feature_rows=[
            (True, [1.0, 2.0], "manual", _photo(0.5), True),
            (False, [3.0, 4.0], "auto_review_near_cut", _photo(0.125), True),
            (True, [5.0, 6.0], "", _photo(-0.25), True),
        ]
    )
    rows = thr.fetch_label_rows(store, WHOLE)
    assert store.calls == [("load_label_feature_rows", WHOLE_PHOTO_FEATURE_VERSION)]
    assert [(liked, source) for liked, _vec, source in rows] == [
        (True, "manual"),
        (False, "auto_review_near_cut"),
        (True, ""),
    ]
    assert rows[0][1] == [1.0, 2.0, *_photo(0.5)]
    assert rows[1][1] == [3.0, 4.0, *_photo(0.125)]
    assert rows[2][1] == [5.0, 6.0, *_photo(-0.25)]
    assert {len(vec) for _l, vec, _s in rows} == {2 + WHOLE_PHOTO_DIMS}


def test_any_label_without_a_photo_row_raises_feature_set_incomplete():
    store = _FeatureStore(
        feature_rows=[
            (True, [1.0], "manual", _photo(0.5), True),
            (False, [2.0], "manual", None, False),
            (False, [4.0], "manual", None, False),
        ]
    )
    with pytest.raises(thr.FeatureSetIncomplete) as info:
        thr.fetch_label_rows(store, WHOLE)
    exc = info.value
    assert (exc.missing, exc.labels, exc.zero_cards) == (2, 3, 0)
    expected = (
        "2 of 3 labels lack whole-photo features; "
        "run python -m tools.backfill_photo_features"
    )
    assert exc.reason == str(exc) == expected
    assert (
        thr.BACKFILL_PHOTO_FEATURES_COMMAND == "python -m tools.backfill_photo_features"
    )


def test_a_stored_zero_card_row_is_incomplete_never_a_zero_block():
    """RT-2: zero photo cards is an extraction failure, so a zero-card row an earlier build
    stored can never reach a model as v1 ++ zeros; the backfill re-lists and recomputes it."""
    store = _FeatureStore(
        feature_rows=[
            (True, [1.0], "manual", _photo(0.5), True),
            (False, [2.0], "manual", None, False),
            (True, [3.0], "manual", None, True),  # a stored zero-card row
        ]
    )
    with pytest.raises(thr.FeatureSetIncomplete) as info:
        thr.fetch_label_rows(store, WHOLE)
    exc = info.value
    assert (exc.missing, exc.labels, exc.zero_cards) == (2, 3, 1)
    assert exc.reason == (
        "2 of 3 labels lack whole-photo features (1 of them stored with zero photo "
        "cards); run python -m tools.backfill_photo_features"
    )


def test_feature_set_incomplete_survives_pickling_and_validates_counts():
    exc = pickle.loads(pickle.dumps(thr.FeatureSetIncomplete(missing=3, labels=9)))
    assert (exc.missing, exc.labels, str(exc)) == (3, 9, exc.reason)
    assert "3 of 9 labels" in exc.reason
    zero = pickle.loads(
        pickle.dumps(thr.FeatureSetIncomplete(missing=3, labels=9, zero_cards=2))
    )
    assert (zero.missing, zero.labels, zero.zero_cards, str(zero)) == (
        3,
        9,
        2,
        zero.reason,
    )
    with pytest.raises(ValueError):
        thr.FeatureSetIncomplete(missing=-1, labels=2)
    with pytest.raises(ValueError):
        thr.FeatureSetIncomplete(missing=True, labels=2)
    with pytest.raises(ValueError):
        thr.FeatureSetIncomplete(missing=1, labels=2, zero_cards=2)


@pytest.mark.parametrize(
    ("row", "needle"),
    [
        ((True, [1.0], "manual", _photo(0.5)), "5-tuple"),
        ((True, [1.0], "manual", _photo(0.5), 1), "bool"),
        ((True, [1.0], "manual", _photo(0.5), False), "no photo row"),
        ((True, [1.0], "manual", [0.5] * (WHOLE_PHOTO_DIMS + 1), True), "dimension"),
    ],
)
def test_a_malformed_feature_row_is_an_error_not_a_guess(row, needle):
    with pytest.raises(ValueError, match=needle):
        thr.fetch_label_rows(_FeatureStore(feature_rows=[row]), WHOLE)


def test_whole_photo_needs_a_store_that_can_serve_it():
    class RowsOnly:
        def load_label_rows(self):
            return [(True, [1.0], "")]

    with pytest.raises(ValueError, match="load_label_feature_rows"):
        thr.fetch_label_rows(RowsOnly(), WHOLE)


def test_training_inputs_take_the_feature_set_and_stop_before_review_stats():
    stats = {
        "random_reviewed": 0,
        "random_would_like": 0,
        "auto_likes": 0,
        "auto_passes": 0,
    }
    ok = _FeatureStore(
        feature_rows=[(True, [1.0], "manual", _photo(0.5), True)], stats=stats
    )
    rows, live = thr.fetch_training_inputs(ok, WHOLE, log=lambda _m: None)
    assert rows == [(True, [1.0, *_photo(0.5)], "manual")]
    assert live == stats
    incomplete = _FeatureStore(
        feature_rows=[(True, [1.0], "manual", None, False)], stats=stats
    )
    with pytest.raises(thr.FeatureSetIncomplete):
        thr.fetch_training_inputs(incomplete, WHOLE, log=lambda _m: None)
    assert "auto_review_stats" not in incomplete.calls
    # The default stays v1.
    v1 = _FeatureStore(rows=[(True, [1.0], "manual")], stats=stats)
    assert thr.fetch_training_inputs(v1, log=lambda _m: None)[0] == v1.rows


def test_end_to_end_on_a_real_sqlite_store(tmp_path):
    store = SQLiteStore(tmp_path / "s.db")
    try:
        store.add_label("r1", "hinge", True, [0.1, 0.2], profile_id="p1")
        store.add_label("r1", "hinge", False, [0.3, 0.4], profile_id="p2")
        v1 = thr.fetch_label_rows(store)
        assert v1 == store.load_label_rows()
        with pytest.raises(thr.FeatureSetIncomplete, match="2 of 2 labels"):
            thr.fetch_label_rows(store, WHOLE)
        common = {
            "run_id": "r1",
            "app": "hinge",
            "feature_version": WHOLE_PHOTO_FEATURE_VERSION,
            "frames_sha256": "0" * 64,
            "source": "backfill",
        }
        store.add_photo_features(
            **common, profile_id="p1", embedding=_photo(0.5), card_count=2
        )
        with pytest.raises(thr.FeatureSetIncomplete, match="1 of 2 labels"):
            thr.fetch_label_rows(store, WHOLE)
        store.add_photo_features(
            **common, profile_id="p2", embedding=_photo(-0.5), card_count=1
        )
        rows = thr.fetch_label_rows(store, WHOLE)
        assert [(liked, vec[:2], source) for liked, vec, source in rows] == v1
        assert rows[0][1][2:] == _photo(0.5)
        assert rows[1][1][2:] == _photo(-0.5)
        assert thr.fetch_label_rows(store) == v1  # v1 untouched by the feature rows
    finally:
        store.close()


# ------------------------------------------------------------------ the model
def _rows(n, dim, seed=3, signal=0.3):
    """Rows whose first 512 dims are a unit face per row (each its own identity)."""
    rng = np.random.default_rng(seed)
    face = rng.normal(size=(n, 512))
    face /= np.linalg.norm(face, axis=1, keepdims=True)
    tail = rng.normal(size=(n, dim - 512)) / math.sqrt(dim - 512)
    y = rng.random(n) < 0.5
    tail[:, -1] += signal * (2 * y - 1)  # signal in the whole-photo block
    X = np.concatenate([face, tail], axis=1)
    return [(bool(liked), x.tolist(), "manual") for liked, x in zip(y, X, strict=True)]


def test_a_recall_target_model_trains_on_2048_dim_rows_grouping_on_the_face_dims(
    monkeypatch,
):
    real = evaluate_mod.identity_groups
    widths = []

    def spy(face_vectors, *args, **kwargs):
        widths.append(len(face_vectors[0]))
        return real(face_vectors, *args, **kwargs)

    monkeypatch.setattr(evaluate_mod, "identity_groups", spy)
    rows = _rows(120, 1280 + WHOLE_PHOTO_DIMS)
    model = PreferenceModel(40, recall_target=0.9)
    assert model.train(rows) is True, model.not_ready_reason
    assert widths == [
        512
    ]  # identity groups read dims 0-511 only, never the photo block
    decision, p = model.decide(rows[0][1])
    assert decision in {"like", "dislike"} and 0.0 <= p <= 1.0
    with pytest.raises(ValueError, match="dimension 2048"):
        model.decide(rows[0][1][:1280])


def test_mark_not_ready_replaces_a_ready_model_with_the_reason():
    rows = _rows(80, 1280 + WHOLE_PHOTO_DIMS)
    model = PreferenceModel(40, recall_target=0.9)
    assert model.train(rows) and model.ready and model.calibration is not None
    exc = thr.FeatureSetIncomplete(missing=4, labels=81)
    assert model.mark_not_ready(exc.reason, n_labels=exc.labels) is False
    assert not model.ready
    assert model.not_ready_reason == exc.reason
    assert model.n_labels == 81
    assert model.calibration is None and model.max_threshold_lift == 0.0
    with pytest.raises(RuntimeError, match="not ready"):
        model.predict_proba(rows[0][1])
    assert "NOT READY" in thr.format_calibration(model)
    assert exc.reason in thr.format_calibration(model)
    # A later complete training recovers.
    assert model.train(rows) is True and model.not_ready_reason is None


def test_mark_not_ready_keeps_the_label_count_unless_given_and_validates():
    model = PreferenceModel(10)
    model.n_labels = 7
    assert model.mark_not_ready("loader refused") is False
    assert model.n_labels == 7
    for bad in ("", "  ", None, 3):
        with pytest.raises(ValueError):
            model.mark_not_ready(bad)
    for bad in (-1, True, 2.0):
        with pytest.raises(ValueError):
            model.mark_not_ready("x", n_labels=bad)
    for bad in ("", " ", 3):
        with pytest.raises(ValueError):
            model.mark_not_ready("x", kind=bad)


def test_the_not_ready_kind_names_missing_features_until_the_next_train():
    """How an AUTO worker tells missing whole-photo rows from a cold start (both defer)."""
    from operation_love.ranker.decider import mark_feature_set_incomplete

    rows = _rows(80, 1280 + WHOLE_PHOTO_DIMS)
    model = PreferenceModel(40, recall_target=0.9)
    assert model.not_ready_kind is None
    assert mark_feature_set_incomplete(model, thr.FeatureSetIncomplete(1, 81)) is False
    assert (
        model.not_ready_kind
        == thr.FEATURE_SET_INCOMPLETE_KIND
        == "feature_set_incomplete"
    )
    assert model.train(rows[:10]) is False  # a genuine cold start clears it
    assert model.not_ready_kind is None and "min_labels" in model.not_ready_reason
    assert model.mark_not_ready("loader refused") is False
    assert model.not_ready_kind is None


# ------------------------------------------------------------------ config
BASE = {
    "enabled_apps": ["hinge"],
    "mode": "training",
    "storage": {"backend": "sqlite"},
    "opener": {
        "enabled": True,
        "model": "gemini-3.6-flash",
        "thinking": {"gemini-3.6-flash": {}},
    },
    "budget": {
        "run_budget_usd": 5.0,
        "pricing": {"gemini-3.6-flash": {"input": 5, "output": 25}},
    },
}


def _load(d):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(d, f)
    try:
        return c.load(f.name)
    finally:
        Path(f.name).unlink(missing_ok=True)


def test_config_defaults_to_face_crop_v1():
    cfg = _load(BASE)
    c.validate(cfg)
    assert cfg.ranker.feature_set == "face_crop_v1"


@pytest.mark.parametrize("value", ["face_crop_v1", "face_crop_whole_photo_v1"])
def test_config_accepts_both_feature_sets(value):
    cfg = _load({**BASE, "ranker": {"feature_set": value}})
    c.validate(cfg)
    assert cfg.ranker.feature_set == value


@pytest.mark.parametrize(
    "bad", ["whole_photo_v1", "face_crop_v2", "", None, 1, True, ["face_crop_v1"]]
)
def test_config_rejects_an_unknown_or_non_string_feature_set_by_name(bad):
    cfg = _load({**BASE, "ranker": {"feature_set": bad}})
    with pytest.raises(ValueError, match=r"ranker\.feature_set"):
        c.validate(cfg)


def test_shipped_config_yaml_stays_on_face_crop_v1():
    cfg = c.load("config.yaml")
    c.validate(cfg)
    assert cfg.ranker.feature_set == "face_crop_v1"
