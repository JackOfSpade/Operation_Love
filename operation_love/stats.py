"""`python -m operation_love stats` — a quick readout from the store."""

from __future__ import annotations

from . import config as cfg_mod
from .ranker import make_store
from .ranker.decider import format_photo_coverage, mark_feature_set_incomplete
from .ranker.model import PreferenceModel
from .ranker.threshold import (
    DEFAULT_FEATURE_SET,
    FeatureSetIncomplete,
    fetch_training_inputs,
    format_calibration,
    require_feature_set,
)


def _coverage_line(store) -> str | None:
    """One line of whole-photo coverage from ``photo_feature_coverage``, or None."""
    from .vision.photo_cards import WHOLE_PHOTO_FEATURE_VERSION

    coverage = getattr(store, "photo_feature_coverage", None)
    if not callable(coverage):
        return None
    try:
        c = coverage(WHOLE_PHOTO_FEATURE_VERSION)
    except Exception as exc:  # noqa: BLE001 -- a readout line; say why instead of crashing
        return (
            f"Features: whole-photo coverage unavailable ({type(exc).__name__}: {exc})"
        )
    return f"Features: {format_photo_coverage(c)}"


def show(config_path: str = "config.yaml") -> None:
    cfg = cfg_mod.load(config_path)
    cfg_mod.validate(cfg)
    store = make_store(
        cfg, ensure=False
    )  # read-only; don't run DDL/IAM setup just to print a readout
    try:
        feature_set = require_feature_set(
            getattr(cfg.ranker, "feature_set", DEFAULT_FEATURE_SET)
        )
        # (liked, embedding, source) rows when the store records sources; the live review
        # stats feed the recall-target model's false-pass correction (best-effort: None when
        # the store has none or they cannot be read -- a loud message is printed for the latter).
        # Labels lacking whole-photo rows leave the model NOT READY; the v1 rows are then read
        # only for the label counts below.
        incomplete: FeatureSetIncomplete | None = None
        try:
            labels, live = fetch_training_inputs(store, feature_set)
        except FeatureSetIncomplete as exc:
            incomplete = exc
            labels, live = fetch_training_inputs(store, DEFAULT_FEATURE_SET)
        model = PreferenceModel(
            cfg.ranker.min_labels_to_engage,
            cfg.ranker.like_threshold,
            min_per_class=cfg.ranker.min_per_class,
            recall_target=cfg.ranker.like_recall_target,
            contextual_recall_budget=cfg.ranker.contextual_recall_budget,
        )
        if incomplete is None:
            ready = model.train(labels, live_review=live)
        else:
            ready = mark_feature_set_incomplete(model, incomplete)
        liked = sum(1 for row in labels if row[0])

        print(f"Storage : {cfg.storage.backend}")
        print(
            f"Labels  : {len(labels)}  (liked {liked} / passed {len(labels) - liked})"
        )
        if feature_set != DEFAULT_FEATURE_SET:
            print(f"Features: {feature_set}")
            line = _coverage_line(store)
            if line is not None:
                print(line)
        if cfg.ranker.like_recall_target is None:
            threshold_text = f"threshold {cfg.ranker.like_threshold}"
        elif model.calibration is not None:
            threshold_text = f"threshold {model.threshold:.4f}"
        else:
            threshold_text = (
                f"recall target {cfg.ranker.like_recall_target}, uncalibrated"
            )
        print(
            f"Ranker  : ready={ready}  (min {cfg.ranker.min_labels_to_engage}, "
            f"{threshold_text})"
        )
        if cfg.ranker.like_recall_target is not None:
            print(f"          {format_calibration(model)}")
        if not ready:
            need = max(0, cfg.ranker.min_labels_to_engage - len(labels))
            if incomplete is not None:
                print(f"          Not ready: {incomplete.reason}")
            elif need > 0:
                print(
                    f"          Seed ~{need} more decisions in training mode to engage auto mode."
                )
            else:
                # Enough labels but still not ready: say why instead of asking for more.
                print(f"          Not ready: {model.not_ready_reason}")
        if live is not None:
            print(
                "Reviews : "
                f"random {live['random_would_like']}/{live['random_reviewed']} would-have-liked, "
                f"near-cut {live.get('near_cut_would_like', 0)}/"
                f"{live.get('near_cut_reviewed', 0)}; pending "
                f"{live.get('pending_random', 0)} random / "
                f"{live.get('pending_near_cut', 0)} near-cut; "
                f"auto {live['auto_likes']} likes / {live['auto_passes']} passes"
            )
        for app in cfg.enabled_apps:
            print(f"Today (auto): {app}: {store.count_today(app)} swipes")
    finally:
        store.close()
