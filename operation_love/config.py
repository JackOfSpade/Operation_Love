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
    retrain_every: int = 1           # retrain live after each new observe label


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


def _section(cls, name: str, raw_section):
    """Construct a config dataclass from its raw.yaml section, turning a typo'd/unknown
    key into a clear ValueError instead of a cryptic TypeError. validate() only catches
    semantic errors (unknown app, bad mode, ...) — this catches a config.yaml that fails
    to even parse into the dataclasses."""
    if not isinstance(raw_section, dict):
        raise ValueError(f"Config: '{name}' section must be a mapping "
                         f"(got {type(raw_section).__name__})")
    try:
        return cls(**raw_section)
    except TypeError as exc:
        raise ValueError(f"Config: invalid '{name}' section ({exc})") from exc


def load(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:                 # empty / comment-only YAML -> defaults everywhere
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config: {path} must be a YAML mapping at the top level "
                         f"(got {type(raw).__name__})")
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
        ranker=_section(RankerCfg, "ranker", raw.get("ranker", {})),
        quality_filter=_section(QualityCfg, "quality_filter", raw.get("quality_filter", {})),
        opener=_section(OpenerCfg, "opener", raw.get("opener", {})),
        budget=BudgetCfg(
            run_budget_usd=b.get("run_budget_usd"),
            on_exhausted=b.get("on_exhausted", "stop"),
            pricing=pricing,
        ),
        pacing=_section(PacingCfg, "pacing", raw.get("pacing", {})),
        storage=StorageCfg(
            backend=raw.get("storage", {}).get("backend", "bigquery"),
            bigquery=raw.get("storage", {}).get("bigquery", {}),
        ),
    )


_KNOWN_APPS = {"bumble", "hinge"}
# supervisor.py only ever constructs AnthropicOpener — opener.provider has no
# pluggable-backend implementation yet, so any other value is a misconfig, not a
# silently-ignored no-op.
_KNOWN_OPENER_PROVIDERS = {"anthropic"}


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
    if cfg.opener.enabled and cfg.opener.provider not in _KNOWN_OPENER_PROVIDERS:
        raise ValueError(f"Config: opener.provider must be one of "
                         f"{sorted(_KNOWN_OPENER_PROVIDERS)} (got {cfg.opener.provider!r})")
