"""tools/compare_feature_sets.py: the pre-registered feature-set comparison.

The statistics are checked against hand computations, the rule clause by clause, and the whole
tool end to end on synthetic labels: a whole-photo block that carries real signal (SWITCH) and
one that is pure noise (KEEP).  Synthetic vectors only; no store, network or model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import compare_feature_sets as cfs

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RULE = (
    "switch to `face_crop_whole_photo_v1` only if ALL hold: (1) the corrected-t 95% CI of the "
    "paired ΔAUC has its lower bound > 0; (2) Δ>0 on at least 95% of seeds; (3) the mean nested "
    "dislikes_passed at the configured recall target is not lower than v1's; (4) the mean "
    "nested accepted_recall is not lower than v1's by more than 0.01. Otherwise keep "
    "`face_crop_v1`."
)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _rows(
    *,
    n=200,
    seed=0,
    v1_signal=0.3,
    photo_signal=0.5,
    d1=48,
    d2=16,
    sig_dims=4,
    source="manual",
):
    """(v1_rows, wp_rows): balanced labels; signal in 4 face dims and ``sig_dims`` photo dims."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2 == 0
    sign = np.where(y, 1.0, -1.0)
    face = rng.normal(size=(n, d1))
    face[:, :4] += v1_signal * sign[:, None]
    photo = rng.normal(size=(n, d2))
    photo[:, :sig_dims] += photo_signal * sign[:, None]
    v1 = [(bool(y[i]), face[i].tolist(), source) for i in range(n)]
    wp = [(bool(y[i]), face[i].tolist() + photo[i].tolist(), source) for i in range(n)]
    return v1, wp, photo


# --------------------------------------------------------------------------- statistics


def test_corrected_t_matches_a_hand_computation():
    # k=3 folds x r=2 repetitions; every split holds out 10 of 30 (n_test/n_train = 0.5).
    # mean = 0.12/6 = 0.02; squared deviations 1e-4,0,1e-4,4e-4,4e-4,0 -> s2 = 1e-3/5 = 2e-4;
    # se = sqrt((1/6 + 0.5) * 2e-4) = 0.0115470; t(0.975, df=5) = 2.5706 (tables);
    # half-width = 0.0296825.
    r = cfs.corrected_repeated_cv_t(
        [0.01, 0.02, 0.03, 0.00, 0.04, 0.02], n_test=[10] * 6, n_train=[20] * 6
    )
    assert r["splits"] == 6 and r["df"] == 5
    assert r["mean"] == pytest.approx(0.02, abs=1e-15)
    assert r["variance"] == pytest.approx(2e-4, rel=1e-12)
    assert r["test_train_ratio"] == pytest.approx(0.5)
    assert r["se"] == pytest.approx(0.011547005, abs=1e-9)
    assert r["t_crit"] == pytest.approx(2.5706, abs=1e-4)
    assert r["lower"] == pytest.approx(-0.0096825223, abs=1e-8)
    assert r["upper"] == pytest.approx(0.0496825223, abs=1e-8)
    assert r["t"] == pytest.approx(1.7320508, abs=1e-6)
    # The correction is the point: the naive repeated-CV se, sqrt(s2 / J), is half as wide.
    assert r["se"] == pytest.approx(2.0 * (2e-4 / 6) ** 0.5, rel=1e-9)


def test_corrected_t_averages_the_per_split_size_ratio():
    # Unequal group folds: ratios 10/40 = 0.25 and 30/20 = 1.5 -> 0.875.  deltas 0.0, 0.1:
    # mean 0.05, s2 = 0.005, se = sqrt((1/2 + 0.875) * 0.005) = 0.0829156; t(0.975, 1) = 12.706.
    r = cfs.corrected_repeated_cv_t([0.0, 0.1], n_test=[10, 30], n_train=[40, 20])
    assert r["test_train_ratio"] == pytest.approx(0.875)
    assert r["se"] == pytest.approx(0.0829156198, abs=1e-9)
    assert r["t_crit"] == pytest.approx(12.706, abs=1e-3)
    assert r["lower"] == pytest.approx(-1.0035428, abs=1e-6)
    assert r["upper"] == pytest.approx(1.1035428, abs=1e-6)


def test_corrected_t_zero_variance_is_a_point_interval():
    r = cfs.corrected_repeated_cv_t([0.0] * 10, n_test=[5] * 10, n_train=[20] * 10)
    assert r["lower"] == r["upper"] == r["mean"] == 0.0 and r["t"] is None


@pytest.mark.parametrize(
    ("deltas", "n_test", "n_train"),
    [
        ([0.1], [1], [4]),
        ([0.1, 0.2], [1], [4, 4]),
        ([0.1, float("nan")], [1, 1], [4, 4]),
        ([0.1, 0.2], [0, 1], [4, 4]),
    ],
    ids=["one-split", "ragged-sizes", "nan", "empty-fold"],
)
def test_corrected_t_rejects_unusable_input(deltas, n_test, n_train):
    with pytest.raises(ValueError):
        cfs.corrected_repeated_cv_t(deltas, n_test=n_test, n_train=n_train)


# --------------------------------------------------------------------------- the rule


_PASSING = {
    "ci_lower": 0.004,
    "seeds_positive": 20,
    "seeds": 20,
    "dislikes_passed_v1": 0.355,
    "dislikes_passed_wp": 0.370,
    "recall_v1": 0.913,
    "recall_wp": 0.934,
}


def test_rule_switches_only_when_all_four_clauses_hold():
    verdict = cfs.apply_switch_rule(**_PASSING)
    assert verdict.switch and verdict.clauses == {1: True, 2: True, 3: True, 4: True}


@pytest.mark.parametrize(
    ("change", "clause"),
    [
        ({"ci_lower": 0.0}, 1),
        ({"ci_lower": -0.001}, 1),
        ({"seeds_positive": 18}, 2),
        ({"dislikes_passed_wp": 0.354}, 3),
        ({"recall_wp": 0.9029}, 4),
    ],
    ids=[
        "ci-zero",
        "ci-negative",
        "seeds-18-of-20",
        "fewer-dislikes-passed",
        "recall-drop",
    ],
)
def test_rule_keeps_v1_when_any_one_clause_fails(change, clause):
    verdict = cfs.apply_switch_rule(**{**_PASSING, **change})
    assert not verdict.switch
    assert [k for k, ok in verdict.clauses.items() if not ok] == [clause]


def test_rule_boundaries_are_inclusive_where_the_rule_says_at_least_or_not_lower():
    # 19/20 = 95% is "at least 95%"; equal dislikes_passed is "not lower"; a recall exactly
    # 0.01 lower is not "lower by more than 0.01" (0.95 - 0.94 is 0.010000000000000009 in floats).
    verdict = cfs.apply_switch_rule(
        **{
            **_PASSING,
            "seeds_positive": 19,
            "dislikes_passed_wp": 0.355,
            "recall_v1": 0.95,
            "recall_wp": 0.94,
        }
    )
    assert verdict.switch
    assert not cfs.apply_switch_rule(
        **{**_PASSING, "seeds_positive": 9, "seeds": 10}
    ).clauses[2]


def test_rule_text_is_verbatim_in_the_docstring_and_the_runbook():
    assert cfs.SWITCH_RULE == CONTRACT_RULE
    assert CONTRACT_RULE in _normalized(cfs.__doc__)
    assert CONTRACT_RULE in _normalized((ROOT / "ops" / "RUNBOOK.md").read_text())


# --------------------------------------------------------------------------- pairing


def _flip_liked(rows, i):
    liked, emb, source = rows[i]
    rows[i] = (not liked, emb, source)


@pytest.mark.parametrize(
    "corrupt",
    ["count", "liked", "source", "prefix", "not-longer", "ragged"],
)
def test_pairing_refuses_rows_that_are_not_the_same_labels(corrupt):
    v1, wp, _ = _rows(n=20)
    if corrupt == "count":
        wp = wp[:-1]
    elif corrupt == "liked":
        _flip_liked(wp, 3)
    elif corrupt == "source":
        wp[3] = (wp[3][0], wp[3][1], "auto_review_random")
    elif corrupt == "prefix":
        wp[5] = (wp[5][0], [wp[5][1][0] + 1e-9, *wp[5][1][1:]], wp[5][2])
    elif corrupt == "not-longer":
        wp = [(liked, emb[:48], source) for liked, emb, source in wp]
    elif corrupt == "ragged":
        wp[2] = (wp[2][0], wp[2][1][:-1], wp[2][2])
    with pytest.raises(cfs.ComparisonRefused):
        cfs.pair_rows(v1, wp)


def test_pairing_counts_manual_and_review_labels():
    v1, wp, _ = _rows(n=20)
    for i in (0, 1):
        v1[i] = (v1[i][0], v1[i][1], "auto_review_near_cut")
        wp[i] = (wp[i][0], wp[i][1], "auto_review_near_cut")
    p = cfs.pair_rows(v1, wp)
    assert (p.manual, p.review) == (18, 2)


def test_results_depend_on_the_label_set_not_the_row_order():
    v1, wp, _ = _rows(n=120, seed=3)
    perm = np.random.default_rng(7).permutation(len(v1))
    a = cfs.compare(v1, wp, seeds=2, orders=1, recall_target=0.9)
    b = cfs.compare(
        [v1[i] for i in perm],
        [wp[i] for i in perm],
        seeds=2,
        orders=1,
        recall_target=0.9,
    )
    assert a["cv"] == b["cv"] and a["nested"] == b["nested"]


# --------------------------------------------------------------------------- end to end


def test_a_photo_block_with_real_signal_switches():
    v1, wp, _ = _rows(seed=2)
    r = cfs.compare(v1, wp, seeds=4, orders=3, recall_target=0.9)
    assert r["verdict"] == {
        "switch": True,
        "clauses": {1: True, 2: True, 3: True, 4: True},
    }
    cv = r["cv"]
    assert cv["auc_mean"][cfs.WHOLE_PHOTO] > cv["auc_mean"][cfs.V1]
    assert cv["seeds_positive"] == 4 and cv["corrected_t"]["splits"] == 4 * cfs.N_SPLITS
    assert r["labels"] == 200 and r["likes"] == 100 and r["dims"][cfs.WHOLE_PHOTO] == 64
    assert not r["preregistered_run"]
    assert r["time_forward"] is None
    text = cfs.format_report(r)
    assert "VERDICT: SWITCH" in text and "NOT the pre-registered" in text
    assert "unavailable" in text  # no timestamps were given


def test_a_pure_noise_photo_block_keeps_v1():
    v1, wp, _ = _rows(seed=0, photo_signal=0.0)
    r = cfs.compare(v1, wp, seeds=4, orders=2, recall_target=0.9)
    assert not r["verdict"]["switch"] and not r["verdict"]["clauses"][1]
    assert r["cv"]["corrected_t"]["lower"] <= 0.0
    assert "VERDICT: KEEP face_crop_v1" in cfs.format_report(r)


def test_the_report_prints_aggregates_only():
    v1, wp, photo = _rows(n=120, seed=4)
    times = [float(i) for i in range(120)]
    r = cfs.compare(v1, wp, seeds=2, orders=1, recall_target=0.9, times=times)
    text = cfs.format_report(r)
    assert "train oldest 50% (n=60)" in text and "train oldest 80% (n=96)" in text
    for value in (v1[0][1][0], v1[1][1][5], photo[0][0]):
        assert f"{value:.4f}" not in text and repr(value) not in text
    json.dumps(
        r
    )  # aggregates only: the whole report is plain JSON-able numbers and text


def test_time_forward_tests_only_later_people_never_seen_in_training():
    v1, wp, _ = _rows(n=200, seed=5, photo_signal=1.0)
    # Person 0 is labelled first AND last: the late label shares an identity group with a
    # training label, so it must leave the test side.
    v1[199] = (v1[0][0], list(v1[0][1]), "manual")
    wp[199] = (wp[0][0], list(wp[0][1]), "manual")
    times = [float(i) for i in range(200)]
    p = cfs.pair_rows(v1, wp)
    groups = cfs._identity_groups(p)
    canonical_times = [times[i] for i in p.original_index]
    rows = cfs.time_forward(p, groups, canonical_times)
    assert [row["cut"] for row in rows] == list(cfs.TIME_CUTS)
    repeat = p.original_index.index(199)
    first = p.original_index.index(0)
    assert groups[repeat] == groups[first]
    for row in rows:
        assert row["status"] == "ok"
        assert row["n_train"] == round(row["cut"] * 200)
        later = [i for i in range(200) if canonical_times[i] >= row["n_train"]]
        seen = {groups[i] for i in range(200) if canonical_times[i] < row["n_train"]}
        unseen = [i for i in later if groups[i] not in seen]
        assert repeat in later and repeat not in unseen
        assert row["n_test"] == len(unseen)
        assert row["delta"] == pytest.approx(
            row["auc"][cfs.WHOLE_PHOTO] - row["auc"][cfs.V1]
        )
        assert row["auc"][cfs.WHOLE_PHOTO] > row["auc"][cfs.V1]


# --------------------------------------------------------------------------- label timestamps


def test_label_times_reads_sqlite_created_at_in_the_loaded_row_order(tmp_path):
    from operation_love.ranker.store import SQLiteStore

    store = SQLiteStore(tmp_path / "labels.db")
    try:
        v1, _, _ = _rows(n=6, d1=8)
        for i, (liked, emb, source) in enumerate(v1):
            store.add_label(f"run{i}", "hinge", liked, emb, source, profile_id=f"p{i}")
        loaded = store.load_label_rows()
        times, note = cfs.label_times(store, loaded)
        assert note == "" and len(times) == 6
        order = [next(i for i, r in enumerate(v1) if r[1] == row[1]) for row in loaded]
        by_insertion = [t for _, t in sorted(zip(order, times, strict=True))]
        assert by_insertion == sorted(by_insertion)
        reversed_rows = loaded[::-1]
        times_rev, _ = cfs.label_times(store, reversed_rows)
        assert times_rev == times[::-1]
        stranger = [(True, [9.0] * 8, "manual")]
        assert cfs.label_times(store, stranger) == (
            None,
            "label timestamps do not match the loaded labels",
        )
    finally:
        store.close()


def test_label_times_is_unavailable_for_a_store_it_cannot_query():
    times, note = cfs.label_times(object(), [(True, [1.0], "manual")])
    assert times is None and "cannot report label timestamps" in note


# --------------------------------------------------------------------------- CLI


class FeatureStore:
    """load_label_rows + load_label_feature_rows, so the REAL fetch_label_rows is exercised."""

    def __init__(self, v1_rows, photo_rows):
        self.v1_rows, self.photo_rows = v1_rows, photo_rows
        self.closed = False

    def load_label_rows(self):
        return [(liked, list(emb), source) for liked, emb, source in self.v1_rows]

    def load_label_feature_rows(self, feature_version):
        assert feature_version == "whole_photo_v1"
        return [
            (liked, list(emb), source, photo, has_row)
            for (liked, emb, source), (photo, has_row) in zip(
                self.v1_rows, self.photo_rows, strict=True
            )
        ]

    def close(self):
        self.closed = True


def _cli(monkeypatch, store, recall_target=0.9):
    from operation_love import config as cfg_mod
    from operation_love import ranker

    built = []

    def make_store(cfg, ensure=True):
        built.append(ensure)
        return store

    monkeypatch.setattr(
        cfg_mod,
        "load",
        lambda path: SimpleNamespace(
            ranker=SimpleNamespace(like_recall_target=recall_target)
        ),
    )
    monkeypatch.setattr(ranker, "make_store", make_store)
    return built


def _feature_store(**kwargs):
    v1, _, photo = _rows(d2=768, **kwargs)
    return FeatureStore(v1, [(row.tolist(), True) for row in photo])


def test_cli_switch_exits_zero_through_the_real_loader(monkeypatch, capsys):
    store = _feature_store(seed=1, sig_dims=16)
    built = _cli(monkeypatch, store)
    code = cfs.main(["--config", "x.yaml", "--seeds", "4", "--orders", "3"])
    out = capsys.readouterr().out
    assert code == cfs.EXIT_SWITCH, out
    assert built == [False] and store.closed  # read-only store
    assert "VERDICT: SWITCH" in out and "(816-d)" in out


def test_cli_keep_exits_three(monkeypatch, capsys):
    store = _feature_store(seed=0, photo_signal=0.0)
    _cli(monkeypatch, store)
    assert cfs.main(["--seeds", "3", "--orders", "2"]) == cfs.EXIT_KEEP
    assert "VERDICT: KEEP" in capsys.readouterr().out


def test_cli_refuses_when_labels_lack_whole_photo_features(monkeypatch, capsys):
    store = _feature_store(n=20)
    store.photo_rows[3] = (None, False)
    store.photo_rows[7] = (None, False)
    _cli(monkeypatch, store)
    assert cfs.main([]) == cfs.EXIT_REFUSED
    out = capsys.readouterr().out
    assert (
        "2 of 20 labels lack whole-photo features; run python -m "
        "tools.backfill_photo_features" in out
    )
    assert store.closed


def test_cli_refuses_a_stored_zero_card_label_instead_of_a_zero_block(
    monkeypatch, capsys
):
    """RT-2: zero photo cards is an extraction failure, so a zero-card row an earlier build
    stored is incomplete (the backfill recomputes it), never measured as v1 ++ zeros."""
    store = _feature_store(n=20)
    store.photo_rows[4] = (None, True)
    _cli(monkeypatch, store)
    assert cfs.main([]) == cfs.EXIT_REFUSED
    out = capsys.readouterr().out
    assert "1 of 20 labels lack whole-photo features (1 of them stored with zero" in out
    assert "tools.backfill_photo_features" in out


def test_cli_refuses_mismatched_label_sets(monkeypatch, capsys):
    class Shuffled(FeatureStore):
        def load_label_feature_rows(self, feature_version):
            return super().load_label_feature_rows(feature_version)[::-1]

    store = _feature_store(n=20)
    _cli(monkeypatch, Shuffled(store.v1_rows, store.photo_rows))
    assert cfs.main([]) == cfs.EXIT_REFUSED
    assert "must be the same labels in the same order" in capsys.readouterr().out


def test_cli_refuses_without_a_recall_target(monkeypatch, capsys):
    built = _cli(monkeypatch, _feature_store(n=20), recall_target=None)
    assert cfs.main([]) == cfs.EXIT_REFUSED
    assert built == []  # never even opened the store
    assert "like_recall_target is not set" in capsys.readouterr().out


def test_cli_reports_an_unmeasurable_nested_evaluation_as_a_refusal(
    monkeypatch, capsys
):
    store = _feature_store(n=24)
    _cli(monkeypatch, store)

    def broken(samples, **kwargs):
        return {"status": "insufficient_groups", "message": "too few identities"}

    monkeypatch.setattr("operation_love.ranker.evaluate.evaluate", broken)
    assert cfs.main(["--seeds", "1", "--orders", "1"]) == cfs.EXIT_REFUSED
    assert "too few identities" in capsys.readouterr().out


def test_nested_threads_do_not_change_the_numbers():
    v1, wp, _ = _rows(n=80, seed=6)
    p = cfs.pair_rows(v1, wp)
    serial = cfs.nested_comparison(p, orders=3, recall_target=0.9, jobs=1)
    threaded = cfs.nested_comparison(p, orders=3, recall_target=0.9, jobs=3)
    assert serial == threaded
