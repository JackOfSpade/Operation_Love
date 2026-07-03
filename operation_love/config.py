"""Load and validate config.yaml into typed objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .costing import ModelPricing


@dataclass
class RankerCfg:
    like_threshold: float = 0.5
    min_labels_to_engage: int = 40
    retrain_every: int = 1           # retrain live after each new observe label
    min_per_class: int = 5           # each class needs at least this many examples to train


@dataclass
class QualityCfg:
    enabled: bool = True
    metric: str = "clipiqa"
    min_score: float = 0.30


@dataclass
class OpenerCfg:
    enabled: bool = True
    model: str = "claude-opus-4-8"
    max_tokens: int = 400
    request_timeout_s: float = 30
    style: str = ""


@dataclass
class BudgetCfg:
    run_budget_usd: float | None = 5.00
    day_budget_usd: float | None = None  # optional daily ceiling across all runs
    on_exhausted: str = "stop"  # stop | swipe_without_opener
    pricing: dict[str, ModelPricing] = field(default_factory=dict)


@dataclass
class PacingCfg:
    swipe_delay_s: float = 3.5        # anchor; human.py adds a log-normal spread around it


@dataclass
class StorageCfg:
    backend: str = "bigquery"   # bigquery | sqlite
    bigquery: dict = field(default_factory=dict)


@dataclass
class Config:
    enabled_apps: list[str]          # e.g. ["bumble", "hinge"] — run concurrently
    mode: str                        # "observe" (learn from your swipes) | "auto"
    apps: dict                       # per-app options (headless, selectors, ...)
    limits: dict                     # auto-mode caps: {max_per_run, max_per_day}
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
        limits=raw.get("limits", {}),
        data_dir=Path(paths.get("data_dir", "./data")),
        db_file=Path(paths.get("db_file", "./data/operation_love.db")),
        ranker=RankerCfg(**raw.get("ranker", {})),
        quality_filter=QualityCfg(**raw.get("quality_filter", {})),
        opener=OpenerCfg(**raw.get("opener", {})),
        budget=BudgetCfg(
            run_budget_usd=b.get("run_budget_usd"),
            day_budget_usd=b.get("day_budget_usd"),
            on_exhausted=b.get("on_exhausted", "stop"),
            pricing=pricing,
        ),
        pacing=PacingCfg(**raw.get("pacing", {})),
        storage=StorageCfg(
            backend=raw.get("storage", {}).get("backend", "bigquery"),
            bigquery=raw.get("storage", {}).get("bigquery", {}),
        ),
    )


_KNOWN_APPS = {"bumble", "hinge"}


def validate(cfg: Config) -> None:
    """Fail fast with a clear message on misconfig (called by the entry points)."""
    if not cfg.enabled_apps:
        raise ValueError("Config: enabled_apps is empty")
    unknown = [a for a in cfg.enabled_apps if a not in _KNOWN_APPS]
    if unknown:
        raise ValueError(f"Config: unknown app(s) {unknown}; supported: {sorted(_KNOWN_APPS)}")
    modes = {cfg.mode} | {(cfg.apps.get(a, {}) or {}).get("mode", cfg.mode) for a in cfg.enabled_apps}
    bad_modes = sorted(m for m in modes if m not in {"observe", "auto"})
    if bad_modes:
        raise ValueError(f"Config: mode must be 'observe' or 'auto' (got {bad_modes})")
    if cfg.storage.backend not in {"bigquery", "sqlite"}:
        raise ValueError(f"Config: storage.backend must be 'bigquery' or 'sqlite' (got {cfg.storage.backend})")
    if cfg.storage.backend == "bigquery" and not cfg.storage.bigquery.get("project_id"):
        raise ValueError("Config: storage.backend=bigquery requires storage.bigquery.project_id")
    if cfg.storage.backend == "bigquery" and not cfg.storage.bigquery.get("photo_bucket"):
        raise ValueError("Config: storage.backend=bigquery requires storage.bigquery.photo_bucket")
    if cfg.opener.enabled and cfg.opener.model not in cfg.budget.pricing:
        raise ValueError(f"Config: opener.model '{cfg.opener.model}' has no entry in budget.pricing")
    if cfg.budget.on_exhausted not in {"stop", "swipe_without_opener"}:
        # Closed enum, like mode/storage.backend above: a typo here would silently fall
        # through to "keep swiping without openers" (OpenerService treats any non-"stop"
        # value that way), quietly disabling the opt-in safety stop. Fail fast instead.
        raise ValueError("Config: budget.on_exhausted must be 'stop' or "
                         f"'swipe_without_opener' (got {cfg.budget.on_exhausted!r})")
    if cfg.ranker.retrain_every < 1:
        raise ValueError(f"Config: ranker.retrain_every must be >= 1 (got {cfg.ranker.retrain_every})")
    lim = cfg.limits
    for key in ("max_per_run", "max_per_day", "max_likes_per_run"):
        val = lim.get(key)
        if val is not None and val <= 0:
            raise ValueError(f"Config: limits.{key} must be > 0 (got {val})")
    ratio = lim.get("target_like_ratio")
    if ratio is not None and not (0 < ratio < 1):
        raise ValueError(f"Config: limits.target_like_ratio must be in (0, 1) (got {ratio})")
