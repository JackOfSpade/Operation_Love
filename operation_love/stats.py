"""`python -m operation_love stats` — a quick readout from the store."""
from __future__ import annotations

from . import config as cfg_mod
from .ranker import make_store
from .ranker.model import PreferenceModel


def show(config_path: str = "config.yaml") -> None:
    cfg = cfg_mod.load(config_path)
    cfg_mod.validate(cfg)
    store = make_store(cfg)
    try:
        labels = store.load_labels()
        model = PreferenceModel(cfg.ranker.min_labels_to_engage, cfg.ranker.like_threshold,
                                min_per_class=cfg.ranker.min_per_class)
        ready = model.train(labels)
        liked = sum(1 for liked, _ in labels if liked)

        print(f"Storage : {cfg.storage.backend}")
        print(f"Labels  : {len(labels)}  (liked {liked} / passed {len(labels) - liked})")
        print(f"Ranker  : ready={ready}  (min {cfg.ranker.min_labels_to_engage}, "
              f"threshold {cfg.ranker.like_threshold})")
        if not ready:
            need = max(0, cfg.ranker.min_labels_to_engage - len(labels))
            print(f"          Seed ~{need} more swipes in observe mode to engage auto mode.")
        for app in cfg.enabled_apps:
            print(f"Today (auto): {app}: {store.count_today(app)} swipes")
    finally:
        store.close()
