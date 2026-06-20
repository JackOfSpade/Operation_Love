"""Load and validate config.yaml into typed objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .costing import ModelPricing


@dataclass
class RankerCfg:
    embedders: list[str] = field(default_factory=lambda: ["arcface", "clip"])
    like_threshold: float = 0.5
    min_labels_to_engage: int = 40
    retrain_every: int = 10          # retrain live after every N new labels (observe mode)


@dataclass
class QualityCfg:
    enabled: bool = True
    metric: str = "clipiqa"
    min_score: float = 0.30


@dataclass
class OpenerCfg:
    enabled: bool = True
    provider: str = "anthropic"
    model: str = "claude-opus-4-8"
    max_tokens: int = 400
    style: str = ""


@dataclass
class BudgetCfg:
    run_budget_usd: float | None = 5.00
    on_exhausted: str = "stop"  # stop | swipe_without_opener
    pricing: dict[str, ModelPricing] = field(default_factory=dict)


@dataclass
class PacingCfg:
    min_delay_s: float = 2.0
    max_delay_s: float = 6.0


@dataclass
class StorageCfg:
    backend: str = "bigquery"   # bigquery | sqlite
    bigquery: dict = field(default_factory=dict)


@dataclass
class Config:
    enabled_apps: list[str]          # e.g. ["bumble", "hinge"] — run concurrently
    mode: str                        # "observe" (learn from your swipes) | "auto"
    apps: dict                       # per-app options (headless, selectors, ...)
    data_dir: Path
    db_file: Path
    ranker: RankerCfg
    quality_filter: QualityCfg
    opener: OpenerCfg
    budget: BudgetCfg
    pacing: PacingCfg
    storage: StorageCfg


def load(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    paths = raw.get("paths", {})
    b = raw.get("budget", {})
    pricing = {m: ModelPricing.from_dict(d) for m, d in b.get("pricing", {}).items()}
    enabled_apps = raw.get("enabled_apps") or ([raw["app"]] if "app" in raw else ["bumble"])
    return Config(
        enabled_apps=list(enabled_apps),
        mode=raw.get("mode", "observe"),
        apps=raw.get("apps", {}),
        data_dir=Path(paths.get("data_dir", "./data")),
        db_file=Path(paths.get("db_file", "./data/operation_love.db")),
        ranker=RankerCfg(**raw.get("ranker", {})),
        quality_filter=QualityCfg(**raw.get("quality_filter", {})),
        opener=OpenerCfg(**raw.get("opener", {})),
        budget=BudgetCfg(
            run_budget_usd=b.get("run_budget_usd"),
            on_exhausted=b.get("on_exhausted", "stop"),
            pricing=pricing,
        ),
        pacing=PacingCfg(**raw.get("pacing", {})),
        storage=StorageCfg(
            backend=raw.get("storage", {}).get("backend", "bigquery"),
            bigquery=raw.get("storage", {}).get("bigquery", {}),
        ),
    )
