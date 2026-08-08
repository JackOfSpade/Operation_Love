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
    # Tuning anchor: worker.py scales human_motion.think_time_s()'s per-decision delay by
    # (swipe_delay_s / this default), so raising/lowering it slows/speeds up the whole
    # pacing distribution proportionally. 0 disables pacing entirely (no delay, no breaks).
    swipe_delay_s: float = 3.5


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


_BUDGET_KEYS = {"run_budget_usd", "day_budget_usd", "on_exhausted", "pricing"}


def _section(cls, name: str, raw_section):
    """Construct a config dataclass from its raw.yaml section, turning a typo'd/unknown
    key into a clear ValueError instead of a cryptic TypeError. validate() only catches
    semantic errors (unknown app, bad mode, ...) — this catches a config.yaml that fails
    to even parse into the dataclasses.

    A bare `key:` (YAML null) is treated as "section omitted" rather than an error: the
    {} default on raw.get() only covers a MISSING key, and None would otherwise be spread
    as **None a line later."""
    if raw_section is None:
        raw_section = {}
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
    # `or {}` on every hand-built section: the {} default only covers a MISSING key. A key
    # present but written bare (YAML null, e.g. `budget:` with nothing after it) makes
    # .get() return None right past that default, and None is then indexed/spread a few
    # lines below — a raw TypeError instead of the clean ValueError this module owes.
    paths = raw.get("paths", {}) or {}
    b = raw.get("budget", {}) or {}
    unknown = set(b) - _BUDGET_KEYS
    if unknown:
        # budget: is a money control, and it's hand-built with .get() rather than through
        # _section(), so it needs its own typo guard — a typo'd key (run_budget vs
        # run_budget_usd) must fail loudly, not silently yield an unlimited spend cap.
        raise ValueError(f"Config: unknown key(s) under budget: {sorted(unknown)}; "
                         f"supported: {sorted(_BUDGET_KEYS)}")
    pricing = {m: ModelPricing.from_dict(d) for m, d in (b.get("pricing", {}) or {}).items()}
    enabled_apps = raw.get("enabled_apps") or ([raw["app"]] if "app" in raw else ["bumble"])
    storage_raw = raw.get("storage", {}) or {}
    return Config(
        enabled_apps=list(enabled_apps),
        mode=raw.get("mode", "observe"),
        apps=raw.get("apps", {}) or {},
        limits=raw.get("limits", {}) or {},
        data_dir=Path(paths.get("data_dir", "./data")),
        db_file=Path(paths.get("db_file", "./data/operation_love.db")),
        ranker=_section(RankerCfg, "ranker", raw.get("ranker", {})),
        quality_filter=_section(QualityCfg, "quality_filter", raw.get("quality_filter", {})),
        opener=_section(OpenerCfg, "opener", raw.get("opener", {})),
        budget=BudgetCfg(
            run_budget_usd=b.get("run_budget_usd"),
            day_budget_usd=b.get("day_budget_usd"),
            on_exhausted=b.get("on_exhausted", "stop"),
            pricing=pricing,
        ),
        pacing=_section(PacingCfg, "pacing", raw.get("pacing", {})),
        storage=StorageCfg(
            backend=storage_raw.get("backend", "bigquery"),
            bigquery=storage_raw.get("bigquery", {}) or {},
        ),
    )


_KNOWN_APPS = {"bumble", "hinge"}
_LIMITS_KEYS = {"max_per_run", "max_per_day", "max_likes_per_run", "target_like_ratio"}

# worker.py's _pace() scales human_motion.think_time_s()'s WHOLE draw (including its
# shifted-lognormal floor: shift=1.2s for "like"/1.8s for "pass", means ~3.2s/~6.9s) by
# swipe_delay_s / this default. Below this floor the scaled floor drops under ~0.35s and
# the mean under ~1s on the fast tail — no longer distinguishable from scripted,
# machine-speed swiping, the exact behaviour this project's anti-bot design exists to
# prevent (see ops/ANTI-BOT-RESEARCH.md). 0 is a separate, explicit "pacing off" sentinel
# (PacingCfg docstring) and is exempted below, not folded into this floor.
_MIN_SWIPE_DELAY_S = 1.0


def _validate_limits(label: str, lim: dict) -> None:
    """Shared rule set for the global `limits:` block AND any per-app `apps.<app>.limits`
    override (supervisor.py merges them: `{**cfg.limits, **app_cfg.get("limits", {})}`) —
    factored so the two paths can never drift out of sync."""
    unknown = set(lim) - _LIMITS_KEYS
    if unknown:
        raise ValueError(f"Config: unknown key(s) under {label}: {sorted(unknown)}; "
                         f"supported: {sorted(_LIMITS_KEYS)}")
    for key in ("max_per_run", "max_per_day", "max_likes_per_run"):
        val = lim.get(key)
        if val is not None and val <= 0:
            raise ValueError(f"Config: {label}.{key} must be > 0 (got {val})")
    ratio = lim.get("target_like_ratio")
    if ratio is not None and not (0 < ratio < 1):
        raise ValueError(f"Config: {label}.target_like_ratio must be in (0, 1) (got {ratio})")


def validate(cfg: Config) -> None:
    """Fail fast with a clear message on misconfig (called by the entry points)."""
    if not cfg.enabled_apps:
        raise ValueError("Config: enabled_apps is empty")
    unknown = [a for a in cfg.enabled_apps if a not in _KNOWN_APPS]
    if unknown:
        raise ValueError(f"Config: unknown app(s) {unknown}; supported: {sorted(_KNOWN_APPS)}")
    modes = {cfg.mode} | {((cfg.apps or {}).get(a, {}) or {}).get("mode", cfg.mode) for a in cfg.enabled_apps}
    bad_modes = sorted(m for m in modes if m not in {"observe", "auto"})
    if bad_modes:
        raise ValueError(f"Config: mode must be 'observe' or 'auto' (got {bad_modes})")
    if cfg.storage.backend not in {"bigquery", "sqlite"}:
        raise ValueError(f"Config: storage.backend must be 'bigquery' or 'sqlite' (got {cfg.storage.backend})")
    if cfg.storage.backend == "bigquery" and not (cfg.storage.bigquery or {}).get("project_id"):
        raise ValueError("Config: storage.backend=bigquery requires storage.bigquery.project_id")
    if cfg.storage.backend == "bigquery" and not (cfg.storage.bigquery or {}).get("photo_bucket"):
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
    if cfg.pacing.swipe_delay_s != 0 and cfg.pacing.swipe_delay_s < _MIN_SWIPE_DELAY_S:
        # 0 is the documented "pacing off" sentinel (see PacingCfg) and is exempt. Anything
        # else — including negatives — must clear the floor: worker.py's _pace() feeds this
        # straight into threading.Event.wait() as a scale on the whole delay distribution,
        # and Event.wait() treats a negative/near-zero timeout as "return immediately", i.e.
        # machine-speed swiping on a live account with no exception raised. See _MIN_SWIPE_DELAY_S.
        raise ValueError(f"Config: pacing.swipe_delay_s must be 0 (pacing off) or "
                         f">= {_MIN_SWIPE_DELAY_S} (got {cfg.pacing.swipe_delay_s}); smaller "
                         "values scale worker._pace's human-pause distribution down to "
                         "machine-speed swiping")
    _validate_limits("limits", cfg.limits or {})
    for a in cfg.enabled_apps:
        # per-app limits override reaches RateLimiter the same way the global block does
        # (supervisor.py merges them) — validate with the exact same rules, or a per-app
        # max_per_run: 0 silently bypasses the safety cap entirely.
        _validate_limits(f"apps.{a}.limits", ((cfg.apps or {}).get(a, {}) or {}).get("limits", {}) or {})
