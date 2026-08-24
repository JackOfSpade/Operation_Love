"""Load and validate config.yaml into typed objects."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import yaml

from .targeting_policy import (
    HINGE_PHOTO_SELECTION_POLICY_ID, HINGE_SUPERSEDED_PHOTO_SELECTION_POLICY_IDS,
    STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE, STILL_PHOTO_BOUND_CIRCULAR_CHANNEL,
    STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL, STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION,
    StillPhotoAssumptionAcceptance, StillPhotoBoundSummary,
    clear_installed_still_photo_bound, hinge_targeting_unavailable_reason,
    install_accepted_still_photo_assumption, install_verified_still_photo_bound)

from . import platforms
from .costing import ModelPricing
from .bigquery_validation import BIGQUERY_IDENTIFIER_PATTERNS, validate_bigquery_photo_bucket


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
    # gemini is the ONLY supported provider -- the legacy Anthropic/Claude opener path has
    # been removed entirely (not merely defaulted off; see validate()'s provider check).
    # `models` is Gemini's ordered fallback chain; `model` remains the legacy single-model
    # key, retained only as effective_models' fallback when `models` is empty.
    provider: str = "gemini"
    model: str = "gemini-3.6-flash"
    models: list[str] = field(default_factory=list)
    max_tokens: int = 400
    request_timeout_s: float = 30
    style: str = ""
    # Gemini only: model id -> its generationConfig.thinkingConfig dict, passed through
    # verbatim (see GeminiOpener._payload). validate() requires an entry for every model
    # in `models` when provider is gemini -- see _validate_gemini_thinking below for why.
    thinking: dict[str, dict] = field(default_factory=dict)
    # Gemini only: GETs the ListModels endpoint at startup (before make_store()'s slow
    # BigQuery/embedder warmup) to catch a typo'd model id or an invalid key before they
    # cost a full slow startup, or worse, a mid-run permanent 404. True by default; set
    # false for offline development or tests that must never touch the network.
    preflight: bool = True
    # Owner rule (2026-08-10): "I do not want commentless likes. If the response from the
    # AI is bad, redo the prompt... If after 5 attempts it's still a bad response, stop the
    # automation." This is how many times OpenerService re-asks for a rejected AI response
    # (with a correction hint) before giving up and stopping the run. Provider safety-policy
    # blocks are not bad draws and are not retried: AUTO preserves its prior like decision
    # without a comment. There is no general fallback to a bare/commentless like on an
    # opener-capable app (see BudgetCfg's removed on_exhausted for the option this replaced).
    # validate() caps this at
    # _MAX_ATTEMPTS_CEILING (1-15): every attempt is a real, billed, quota-consuming API call,
    # so this can't be left unbounded -- see that constant's docstring for the arithmetic.
    max_attempts: int = 5
    # The ADVISORY (observe-mode) retry budget: a SHORTENED form of max_attempts above, used
    # when the opener is only a suggestion shown to a human rather than text the bot is about
    # to send. Two things differ from the auto path and both argue for fewer attempts. First,
    # nothing irreversible hangs on the outcome: auto normally requires an opener, so it
    # spends the full budget and stops on ordinary generation failures; observe can simply show "no suggestion"
    # and let the human type their own. Second, a human is standing at the phone waiting, so
    # every extra attempt is dead time in front of them, not background work. validate()
    # additionally requires this to be <= max_attempts -- it is the same budget, shortened,
    # never a separate larger one.
    advisory_max_attempts: int = 3
    # Wall-clock ceiling on the ADVISORY retry loop (seconds, monotonic, measured from the
    # top of maybe_opener). advisory_max_attempts alone bounds the COUNT of attempts but not
    # the TIME they take: at the shipped request_timeout_s of 90, three attempts that each
    # stall to the timeout is 270s of a human standing at the phone waiting for a suggestion
    # they could have written themselves in ten. This deadline is checked before STARTING any
    # attempt after the first (it never interrupts an in-flight call, which would waste the
    # spend already committed to it), so it converts that worst case into "one slow attempt,
    # then give up on this profile". validate() caps it at _MAX_ADVISORY_DEADLINE_S.
    advisory_deadline_s: float = 60.0

    @property
    def effective_models(self) -> list[str]:
        """Configured model fallback order, retaining the original singular key."""
        return list(self.models) if self.models else [self.model]


@dataclass
class BudgetCfg:
    run_budget_usd: float | None = 5.00
    day_budget_usd: float | None = None  # optional daily ceiling across all runs
    # `on_exhausted` (stop | swipe_without_opener) lived here until 2026-08-10. The owner
    # ruled out a configurable commentless-like fallback -- "If after 5 attempts it's still a
    # bad response, stop the automation" -- so swipe_without_opener is no longer supported.
    # The separate, hard-coded provider-safety exception is a property of one response class,
    # not a reason to re-add this broad setting. load() below fails
    # loudly if a config still sets it (see the `on_exhausted` check next to _BUDGET_KEYS).
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
    enabled_apps: list[str]          # requested targets; registry enforces coexistence/readiness
    mode: str                        # "observe" (learn from your swipes) | "auto"
    apps: dict                       # per-app device, perception, and action options
    limits: dict                     # auto-mode caps: {max_per_run, max_per_day}
    data_dir: Path
    db_file: Path
    ranker: RankerCfg
    quality_filter: QualityCfg
    opener: OpenerCfg
    budget: BudgetCfg
    pacing: PacingCfg
    storage: StorageCfg


_TOP_LEVEL_KEYS = {
    "enabled_apps", "app", "mode", "apps", "paths", "storage", "ranker",
    "quality_filter", "opener", "budget", "pacing", "limits",
}
_PATHS_KEYS = {"data_dir", "db_file"}
_STORAGE_KEYS = {"backend", "bigquery"}
_BIGQUERY_KEYS = {"project_id", "dataset", "location", "photo_bucket", "flush_every"}
_BUDGET_KEYS = {"run_budget_usd", "day_budget_usd", "pricing"}
_PRICING_KEYS = {"input", "output", "cache_read", "cache_write"}
_PRICING_REQUIRED_KEYS = {"input", "output"}

# AUTO may only consume a Hinge targeting calibration after a separate production OBSERVE run
# has exercised the Worker/hub/store path on that exact device/build.  Keep this separate from
# targeting_calibration: a numeric bound is a perception license, not evidence that the full
# operational pipeline behaved correctly.
_OBSERVE_RELEASE_EVIDENCE_KEYS = {
    "schema_version", "calibration_calibrated_at", "calibration_sha256", "device",
    "hinge_version_name", "frame_size_px", "production_run_reference", "production_run_id",
    "verification_file", "verification_sha256", "verified_at",
}
_OBSERVE_RELEASE_ARTIFACT_KEYS = {
    "schema_version", "kind", "completed", "calibration_calibrated_at",
    "calibration_sha256", "device", "hinge_version_name", "frame_size_px",
    "production_run_reference", "production_run_id", "debug_actions_sha256", "store_persistence_evidence_sha256",
    "provider_store_evidence_sha256", "observe_control_evidence_sha256", "verified_at",
}

# This is intentionally a *different* config/artifact namespace from the long-standing
# supervised-manual release gate above.  An AI-driven OBSERVE release may be useful when the
# owner has explicitly accepted that mode's circular-risk tradeoff, but it must never look like
# a completed human cycle (or silently broaden the manual artifact's meaning).
_AI_OBSERVE_RELEASE_EVIDENCE_KEYS = {
    "schema_version", "acceptance", "calibration_calibrated_at", "calibration_sha256",
    "device", "hinge_version_name", "frame_size_px", "production_run_reference",
    "production_run_id", "verification_file", "verification_sha256", "verified_at",
}
_AI_OBSERVE_RELEASE_ARTIFACT_KEYS = {
    "schema_version", "kind", "completed", "human_ground_truth", "source", "acceptance",
    "calibration_calibrated_at", "calibration_sha256", "device", "hinge_version_name",
    "frame_size_px", "production_run_reference", "production_run_id", "debug_actions_sha256",
    "store_persistence_evidence_sha256", "provider_store_evidence_sha256",
    "automation_provenance_sha256", "independent_review_sha256", "verified_at",
}
_AI_OBSERVE_RELEASE_ACCEPTANCE = "I_ACCEPT_AI_REVIEWED_OBSERVE_RELEASE_RISK"
_AI_OBSERVE_CONTROLLER_KEYS = {"schema_version", "source", "acceptance", "executor"}


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


def _mapping_or_empty(value, name: str) -> dict:
    """Normalize an optional YAML mapping while retaining clean shape errors."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config: '{name}' section must be a mapping "
                         f"(got {type(value).__name__})")
    return dict(value)


def _path_from_raw(paths: Mapping, key: str, default: str) -> Path:
    """Construct one YAML path with a clean error for null/collection/bool values."""
    value = paths.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config: paths.{key} must be a non-empty path string "
                         f"(got {value!r})")
    return Path(value)


def _reject_unknown_keys(section: Mapping, allowed: set[str], name: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        rendered = sorted(repr(key) for key in unknown)
        raise ValueError(f"Config: unknown key(s) under {name}: {rendered}; "
                         f"supported: {sorted(allowed)}")


def _pricing_from_raw(raw_pricing) -> dict[str, ModelPricing]:
    pricing = _mapping_or_empty(raw_pricing, "budget.pricing")
    result: dict[str, ModelPricing] = {}
    for model, record in pricing.items():
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Config: budget.pricing model ids must be non-empty strings")
        values = _mapping_or_empty(record, f"budget.pricing[{model!r}]")
        unknown = set(values) - _PRICING_KEYS
        missing = _PRICING_REQUIRED_KEYS - set(values)
        if unknown or missing:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if unknown:
                details.append(f"unknown {sorted(unknown)}")
            raise ValueError(
                f"Config: budget.pricing[{model!r}] must contain input/output and only "
                f"supported keys {sorted(_PRICING_KEYS)} ({'; '.join(details)})")
        try:
            result[model] = ModelPricing.from_dict(values)
        except ValueError as exc:
            raise ValueError(f"Config: budget.pricing[{model!r}] {exc}") from exc
    return result


def load(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:                 # empty / comment-only YAML -> defaults everywhere
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config: {path} must be a YAML mapping at the top level "
                         f"(got {type(raw).__name__})")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "the top level")

    paths = _mapping_or_empty(raw.get("paths"), "paths")
    _reject_unknown_keys(paths, _PATHS_KEYS, "paths")
    b = _mapping_or_empty(raw.get("budget"), "budget")
    if "on_exhausted" in b:
        # Dedicated, actionable guard -- ahead of the generic unknown-key check below, which
        # would otherwise just say "unknown key(s) under budget: ['on_exhausted']" and leave
        # the reader to guess why. budget.on_exhausted was removed 2026-08-10: the owner
        # ruled out a configurable commentless-like fallback ("if after 5 attempts it's still
        # a bad response, stop the automation"), so swipe_without_opener is no longer supported.
        # The narrow provider-safety exception is hard-coded by response type and does not make
        # this broad setting valid again. This fires even for `on_exhausted: stop` -- not because that value ever
        # did anything wrong, but so nobody keeps a dead key around believing it still
        # controls something; deleting the line changes nothing about how the run behaves.
        raise ValueError(
            "Config: budget.on_exhausted was removed and must be deleted from config.yaml. "
            "A configurable openerless (\"commentless\") fallback is no longer supported "
            "on an opener-capable app: ordinary opener failures stop instead of falling back "
            "to a bare like. Provider safety blocks are handled separately. If your config had "
            "'on_exhausted: stop', nothing about your run's behavior changes -- that was "
            "already the only real outcome; just remove the line. If it had "
            "'on_exhausted: swipe_without_opener', that mode has been removed entirely: "
            "delete the line, and see opener.max_attempts for how many times a rejected "
            "AI response is re-asked before the run stops instead.")
    _reject_unknown_keys(b, _BUDGET_KEYS, "budget")
    pricing = _pricing_from_raw(b.get("pricing"))
    # Legacy singular `app:` key is still honoured. The old "nothing configured" default of
    # ["bumble"] made sense when Bumble meant the (always-live) web app; since Aug 2026 Bumble
    # is an Android target like Hinge and starts out UNCALIBRATED (platforms.py), so defaulting
    # to it would make a from-scratch config.yaml fail check_runnable() on the very first run.
    # Hinge is the one platform the registry ships available by default, so it's the sensible
    # out-of-the-box default now.
    #
    # This used to be a single `raw.get("enabled_apps") or (...)` expression -- and that `or`
    # is exactly the bug an audit found: `[] or default` evaluates to `default`, because an
    # empty list is falsy in Python. An operator who deliberately writes `enabled_apps: []`
    # (or a config-generation bug that emits one) means "run nothing," and got Hinge started
    # against the real phone instead -- silently, because validate()'s own
    # `if not cfg.enabled_apps: raise ValueError(...)` guard (below) never got a chance to
    # fire: load() had already thrown the empty list away and substituted the default before
    # validate() ever saw it. The fix checks PRESENCE (`"enabled_apps" in raw`) rather than
    # truthiness, so an explicit `[]` -- and a bare `enabled_apps:` (YAML null), which is
    # "the key is present but nothing was written," not "the key was never mentioned" -- both
    # pass straight through as `[]` and hit that already-existing, already-actionable guard.
    # Only a genuinely ABSENT key still falls back to today's default.
    if "enabled_apps" in raw and "app" in raw:
        raise ValueError("Config: specify enabled_apps or legacy app, not both")
    if "enabled_apps" in raw:
        configured_apps = raw["enabled_apps"]
        if configured_apps is None:
            enabled_apps = []
        elif not isinstance(configured_apps, list):
            raise ValueError("Config: enabled_apps must be a YAML list of app ids")
        else:
            enabled_apps = list(configured_apps)
    elif "app" in raw:
        legacy_app = raw["app"]
        if not isinstance(legacy_app, str) or not legacy_app.strip():
            raise ValueError("Config: legacy app must be a non-empty app id string")
        enabled_apps = [legacy_app]
    else:
        enabled_apps = ["hinge"]
    apps_raw = _mapping_or_empty(raw.get("apps"), "apps")
    for app, app_cfg in apps_raw.items():
        if not isinstance(app, str) or not app.strip():
            raise ValueError("Config: apps keys must be non-empty app id strings")
        if not isinstance(app_cfg, Mapping):
            raise ValueError(f"Config: apps.{app} must be a mapping "
                             f"(got {type(app_cfg).__name__})")
    storage_raw = _mapping_or_empty(raw.get("storage"), "storage")
    _reject_unknown_keys(storage_raw, _STORAGE_KEYS, "storage")
    bigquery_raw = _mapping_or_empty(storage_raw.get("bigquery"), "storage.bigquery")
    _reject_unknown_keys(bigquery_raw, _BIGQUERY_KEYS, "storage.bigquery")
    limits_raw = {} if raw.get("limits") is None else raw.get("limits")
    return Config(
        enabled_apps=list(enabled_apps),
        mode=raw.get("mode", "observe"),
        apps=apps_raw,
        limits=limits_raw,
        data_dir=_path_from_raw(paths, "data_dir", "./data"),
        db_file=_path_from_raw(paths, "db_file", "./data/operation_love.db"),
        ranker=_section(RankerCfg, "ranker", raw.get("ranker", {})),
        quality_filter=_section(QualityCfg, "quality_filter", raw.get("quality_filter", {})),
        opener=_section(OpenerCfg, "opener", raw.get("opener", {})),
        budget=BudgetCfg(
            # Absence keeps BudgetCfg's safe $5 default. Explicit null still means unlimited,
            # and explicit zero remains a valid immediate-stop/free-tier budget.
            run_budget_usd=b.get("run_budget_usd", 5.00),
            day_budget_usd=b.get("day_budget_usd"),
            pricing=pricing,
        ),
        pacing=_section(PacingCfg, "pacing", raw.get("pacing", {})),
        storage=StorageCfg(
            backend=storage_raw.get("backend", "bigquery"),
            bigquery=bigquery_raw,
        ),
    )


_LIMITS_KEYS = {"max_per_run", "max_per_day", "max_likes_per_run", "target_like_ratio"}

# The closest two *different* Hinge profiles measured 2.565 grey levels apart in the
# sticky-header identity band.  A targeting calibration must choose a strictly smaller
# acceptance ceiling; at or above this value it can call that known foreign pair a match.
_TARGETING_IDENTITY_FALSE_MATCH_DISTANCE = 2.565
# The nearest measured foreign-card acceptance was 14.91.  This is an upper safety cap, not a
# production setting: the configured value must still come from held-out measurements on the
# actual device, but a value at or above a known false accept can never be called calibrated.
_TARGETING_SHEET_FALSE_MATCH_DISTANCE = 14.91
# Keep these local rather than importing HINGE_SPEC: config.py is intentionally below drivers in
# the dependency graph.  They are the exact Hinge spec defaults and are used only to compute the
# effective geometry when config.yaml did not override it.
_TARGETING_HINGE_IDENTITY_BAND = (0.10, 0.048, 0.80, 0.094)
_TARGETING_HINGE_CONTENT_BAND = (0.125, 0.875)
_TARGETING_CALIBRATION_KEYS = {
    "schema_version", "hinge_version_name", "frame_size_px", "composer_layout_id",
    "item_selection_policy_id",
    "identity_match_max_dist", "inline_item_max_dist", "device", "calibrated_at",
    "identity_band", "content_band",
}
# The optional mapping that binds the measured still-photo bound artifact (see
# ops/STILL-PHOTO-DISCRIMINATOR.md section 5).  Its presence, and only its presence, installs
# numbering readiness for this process.  Exact key set on the observe_release_evidence pattern:
# a missing key is an incomplete claim and an unknown key is a claim we are not checking, and
# both must fail loudly rather than be ignored.
_STILL_PHOTO_BOUND_EVIDENCE_KEYS = {
    "artifact_path", "artifact_sha256", "ground_truth_channel", "video_cards", "video_accepts",
    "photo_cards", "photo_false_refusals", "max_video_exact_run_s", "captured_at", "device",
    "hinge_version_name",
}
# Config keys the artifact itself must also carry, with equal values.  artifact_path is a
# repo-local locator and artifact_sha256 is the artifact's own digest, so neither can live
# inside it; everything else is a measurement claim, and the mapping may not claim a number the
# artifact does not carry.
_STILL_PHOTO_BOUND_MIRRORED_INT_KEYS = (
    "video_cards", "video_accepts", "photo_cards", "photo_false_refusals")
_STILL_PHOTO_BOUND_MIRRORED_TEXT_KEYS = (
    "ground_truth_channel", "captured_at", "device", "hinge_version_name")
# Optional, and optional in one direction only: it is REQUIRED on the circular AI-labelled
# channel and FORBIDDEN on the owner-labelled one (see _validate_hinge_still_photo_bound_evidence).
# It is not in the exact key set above because that set is what every bound must carry, and an
# owner-labelled bound accepted no circular risk at all.
_STILL_PHOTO_BOUND_OPTIONAL_EVIDENCE_KEYS = {"accepted_circular_risk"}

# The THIRD readiness channel (owner decision 2026-08-21): a licence granted by a person instead
# of produced by a campaign.  It is a deliberately tiny key set -- there is nothing to bind by
# sha256, no artifact to re-read and no counts to mirror, because nothing was measured.  What it
# does demand is that the decision be attributable: which phone, which Hinge build, when, and
# why, so a reader months later can tell an accepted risk from an accident.
_STILL_PHOTO_ASSUMPTION_KEYS = {
    "acceptance", "accepted_at", "device", "hinge_version_name", "rationale",
}


def _is_finite_number(value: object) -> bool:
    """math.isfinite without leaking OverflowError for enormous Python integers."""
    try:
        return math.isfinite(value)
    except (TypeError, OverflowError):
        return False


def _safe_value_repr(value: object) -> str:
    """Diagnostic repr that also works beyond Python's integer digit safety limit."""
    try:
        return repr(value)
    except ValueError:
        if isinstance(value, int):
            return f"<integer with {value.bit_length()} bits>"
        return f"<{type(value).__name__}>"


def _targeting_band(value, *, key: str, length: int) -> tuple[float, ...]:
    """Return one normalised geometry record or reject an unusable calibration input."""
    rendered = _safe_value_repr(value)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{key} must be a {length}-number array/tuple (got {rendered})")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not _is_finite_number(v)
           for v in value):
        raise ValueError(f"{key} must contain only finite numbers (got {rendered})")
    result = tuple(float(v) for v in value)
    if length == 4:
        x0, y0, x1, y1 = result
        valid = 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
    else:
        y0, y1 = result
        valid = 0.0 <= y0 < y1 <= 1.0
    if not valid:
        raise ValueError(f"{key} must be an ordered normalised band (got {rendered})")
    return result


def _effective_targeting_geometry(app: str, app_cfg: dict) -> tuple[tuple[float, ...],
                                                                       tuple[float, ...]]:
    """The bands the Android driver will actually use for a calibrated targeted action."""
    if app == "hinge":
        identity = app_cfg.get("identity_band", _TARGETING_HINGE_IDENTITY_BAND)
        content = app_cfg.get("content_band", _TARGETING_HINGE_CONTENT_BAND)
    else:
        # Targeted comment-sheet delivery is only implemented for Hinge today.  Requiring an
        # explicit geometry for any future app avoids silently borrowing Hinge's measurements.
        identity = app_cfg.get("identity_band")
        content = app_cfg.get("content_band")
    return (
        _targeting_band(identity, key=f"apps.{app}.identity_band", length=4),
        _targeting_band(content, key=f"apps.{app}.content_band", length=2),
    )


def _validate_targeting_calibration(cfg: Config) -> None:
    """Validate an optional, evidence-backed per-app targeted-like calibration.

    Its absence is deliberately not a config-load error: observe can still run without
    suggestions, and a model-item like is refused by AndroidDriver before touching the phone.
    Once an operator supplies the mapping, though, a partial or typo'd calibration is unsafe and
    must fail at config validation rather than quietly becoming a looser default.
    """
    for app, app_cfg in (cfg.apps or {}).items():
        app_cfg = app_cfg or {}
        if "targeting_calibration" not in app_cfg:
            continue
        calibration = app_cfg["targeting_calibration"]
        if not isinstance(calibration, dict):
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration must be a mapping "
                f"(got {type(calibration).__name__})")
        unknown = set(calibration) - _TARGETING_CALIBRATION_KEYS
        missing = _TARGETING_CALIBRATION_KEYS - set(calibration)
        if unknown or missing:
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if unknown:
                parts.append(f"unknown {sorted(unknown)}")
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration must carry "
                f"{sorted(_TARGETING_CALIBRATION_KEYS)} ({'; '.join(parts)})")
        if type(calibration["schema_version"]) is not int or calibration["schema_version"] != 3:
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration.schema_version must be the exact "
                f"integer 3 (got {_safe_value_repr(calibration['schema_version'])})")
        if calibration["composer_layout_id"] != "hinge_inline_v1":
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration.composer_layout_id must be "
                f"'hinge_inline_v1' (got {calibration['composer_layout_id']!r})")
        policy_id = calibration["item_selection_policy_id"]
        # A retired id gets its own message: it is not a typo, and re-typing the new id over an
        # old mapping would be exactly the wrong fix.  v1 was measured against a selection
        # contract that no longer exists, so only a fresh campaign can license v2.
        if policy_id in HINGE_SUPERSEDED_PHOTO_SELECTION_POLICY_IDS:
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration.item_selection_policy_id {policy_id} "
                f"is superseded by {HINGE_PHOTO_SELECTION_POLICY_ID}; recalibrate under the "
                f"current policy")
        if policy_id != HINGE_PHOTO_SELECTION_POLICY_ID:
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration.item_selection_policy_id must be "
                f"{HINGE_PHOTO_SELECTION_POLICY_ID!r} "
                f"(got {policy_id!r})")
        frame_size = calibration["frame_size_px"]
        if (not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2
                or any(type(v) is not int or v <= 0 for v in frame_size)):
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration.frame_size_px must be two positive "
                f"integers (got {_safe_value_repr(frame_size)})")
        for key in ("identity_match_max_dist", "inline_item_max_dist"):
            value = calibration[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Config: apps.{app}.targeting_calibration.{key} must be a number "
                    f"(got {_safe_value_repr(value)})")
            if not _is_finite_number(value) or value <= 0:
                raise ValueError(
                    f"Config: apps.{app}.targeting_calibration.{key} must be finite and > 0 "
                    f"(got {_safe_value_repr(value)})")
        identity_max = calibration["identity_match_max_dist"]
        if identity_max >= _TARGETING_IDENTITY_FALSE_MATCH_DISTANCE:
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration.identity_match_max_dist must be "
                f"strictly below {_TARGETING_IDENTITY_FALSE_MATCH_DISTANCE}, the known "
                f"different-profile false-match distance (got {_safe_value_repr(identity_max)})")
        sheet_max = calibration["inline_item_max_dist"]
        if sheet_max >= _TARGETING_SHEET_FALSE_MATCH_DISTANCE:
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration.inline_item_max_dist must be "
                f"strictly below {_TARGETING_SHEET_FALSE_MATCH_DISTANCE}, the nearest known "
                f"foreign-card false-match distance (got {_safe_value_repr(sheet_max)})")
        for key in ("device", "calibrated_at", "hinge_version_name"):
            value = calibration[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Config: apps.{app}.targeting_calibration.{key} must be nonempty evidence "
                    f"text (got {value!r})")
        # The measured bounds are only valid for the device that produced them.  `device` is
        # deliberately the exact ADB serial, not a free-form handset description: a description
        # cannot stop a copied calibration from licensing gestures on another phone.  Do not
        # normalise whitespace/case here; this is an identity, so the two config values must be
        # byte-for-byte equal.
        serial = app_cfg.get("serial")
        if not isinstance(serial, str) or not serial:
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration requires a nonempty "
                f"apps.{app}.serial exact ADB device serial")
        if calibration["device"] != serial:
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration.device must exactly equal "
                f"apps.{app}.serial (got {calibration['device']!r} != {serial!r})")
        try:
            calibrated_identity = _targeting_band(
                calibration["identity_band"],
                key=f"apps.{app}.targeting_calibration.identity_band", length=4)
            calibrated_content = _targeting_band(
                calibration["content_band"],
                key=f"apps.{app}.targeting_calibration.content_band", length=2)
            effective_identity, effective_content = _effective_targeting_geometry(app, app_cfg)
        except ValueError as exc:
            raise ValueError(f"Config: {exc}") from exc
        if calibrated_identity != effective_identity:
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration.identity_band must exactly equal "
                f"the effective apps.{app}.identity_band ({calibrated_identity!r} != "
                f"{effective_identity!r})")
        if calibrated_content != effective_content:
            raise ValueError(
                f"Config: apps.{app}.targeting_calibration.content_band must exactly equal "
                f"the effective apps.{app}.content_band ({calibrated_content!r} != "
                f"{effective_content!r})")
        if app == "hinge":
            policy_blocker = hinge_targeting_unavailable_reason()
            if policy_blocker is not None:
                raise ValueError(
                    "Config: apps.hinge.targeting_calibration cannot license numbered "
                    f"targeting because {policy_blocker}. Remove the optional mapping; Hinge "
                    "Observe remains available without targeted suggestions")


def _validate_hinge_still_photo_bound_evidence(cfg: Config) -> None:
    """Install artifact-derived numbering readiness, or leave numbering fail-closed.

    This is the ONLY way ``hinge_targeting_unavailable_reason()`` can start returning None (see
    ops/STILL-PHOTO-DISCRIMINATOR.md section 5).  Readiness is process-global, so the first thing
    this does is drop whatever an earlier ``validate()`` installed: without that, validating a
    config that carries the evidence and then one that does not would leave the second run
    numbering on evidence it never presented.  Absence of the key is not an error (that is the
    normal, shipped state); a present-but-wrong key is always fatal, never a warning.
    """
    clear_installed_still_photo_bound()
    if "hinge" not in cfg.enabled_apps:
        return
    app_cfg = (cfg.apps or {}).get("hinge", {}) or {}
    if "still_photo_bound_evidence" not in app_cfg:
        return
    raw = app_cfg["still_photo_bound_evidence"]
    if not isinstance(raw, dict):
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence must be a mapping "
            f"(got {type(raw).__name__})")
    unknown = (set(raw) - _STILL_PHOTO_BOUND_EVIDENCE_KEYS
               - _STILL_PHOTO_BOUND_OPTIONAL_EVIDENCE_KEYS)
    missing = _STILL_PHOTO_BOUND_EVIDENCE_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence must carry exactly "
            f"{sorted(_STILL_PHOTO_BOUND_EVIDENCE_KEYS)} (missing {sorted(missing)}, "
            f"unknown {sorted(unknown)}), plus "
            f"{sorted(_STILL_PHOTO_BOUND_OPTIONAL_EVIDENCE_KEYS)} on the circular AI-labelled "
            f"{STILL_PHOTO_BOUND_CIRCULAR_CHANNEL} channel only")
    for key in _STILL_PHOTO_BOUND_MIRRORED_TEXT_KEYS + ("artifact_path", "artifact_sha256"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(
                f"Config: apps.hinge.still_photo_bound_evidence.{key} must be nonempty text "
                f"(got {_safe_value_repr(raw[key])})")
    for key in _STILL_PHOTO_BOUND_MIRRORED_INT_KEYS:
        # `type(...) is not int` rather than isinstance: bool is an int subclass, so a stray
        # `video_accepts: true` would otherwise validate as one accept.
        if type(raw[key]) is not int:
            raise ValueError(
                f"Config: apps.hinge.still_photo_bound_evidence.{key} must be an exact integer "
                f"card count (got {_safe_value_repr(raw[key])})")
    run_s = raw["max_video_exact_run_s"]
    if isinstance(run_s, bool) or not isinstance(run_s, Real) or not _is_finite_number(run_s):
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence.max_video_exact_run_s must be a "
            f"finite number of seconds (got {_safe_value_repr(run_s)})")
    # The circular AI-labelled channel (owner decision 2026-08-21) labels its videos with the
    # same mute-glyph matcher whose blind spot the bound is meant to quantify, so it can never
    # claim human ground truth and is licensed only by the owner's recorded acceptance.  The
    # phrase is demanded here AND in the digest-bound artifact below: requiring it in the pasted
    # mapping stops a re-run of the measurement tool from opting the owner in behind their back,
    # and requiring it in the artifact stops a one-line config edit from doing the same.  On the
    # owner channel the key is forbidden outright, so an owner-labelled bound can never be read
    # as having accepted a circularity it does not have.
    is_circular = raw["ground_truth_channel"] == STILL_PHOTO_BOUND_CIRCULAR_CHANNEL
    acceptance = raw.get("accepted_circular_risk")
    if is_circular:
        if acceptance != STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE:
            raise ValueError(
                "Config: apps.hinge.still_photo_bound_evidence.accepted_circular_risk must be "
                f"exactly {STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE!r} because ground_truth_channel "
                f"is {STILL_PHOTO_BOUND_CIRCULAR_CHANNEL!r}, the circular AI-labelled channel "
                "whose video labels come from the same mute-glyph matcher the bound measures "
                f"(got {_safe_value_repr(acceptance)})")
    elif "accepted_circular_risk" in raw:
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence.accepted_circular_risk accepts the "
            f"circular AI-labelled channel {STILL_PHOTO_BOUND_CIRCULAR_CHANNEL!r} and is valid "
            f"only there; this bound declares ground_truth_channel "
            f"{raw['ground_truth_channel']!r}, so remove the key")
    # The measured bound is only valid for the phone it was measured on.  Same reasoning as
    # targeting_calibration.device above: an exact ADB serial, compared byte for byte, is the
    # only thing that stops a copied artifact from licensing numbering on another handset.
    serial = app_cfg.get("serial")
    if not isinstance(serial, str) or not serial:
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence requires a nonempty apps.hinge.serial "
            "exact ADB device serial")
    if raw["device"] != serial:
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence.device must exactly equal "
            f"apps.hinge.serial (got {raw['device']!r} != {serial!r})")

    artifact_path = Path(raw["artifact_path"])
    if artifact_path.is_absolute():
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence.artifact_path must be a "
            "repo-relative local path")
    root = Path.cwd().resolve()
    resolved = (root / artifact_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence.artifact_path escapes the repository"
        ) from exc
    try:
        content = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"Config: apps.hinge.still_photo_bound_evidence artifact is unreadable: {resolved}"
        ) from exc
    if hashlib.sha256(content).hexdigest() != raw["artifact_sha256"]:
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence.artifact_sha256 does not match its "
            "artifact")
    try:
        artifact = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence artifact is not JSON") from exc
    if not isinstance(artifact, dict):
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence artifact must be a JSON object")
    # Deliberately NOT an exact artifact key set: the measurement tool also records per-card
    # verdicts, frame digests and geometry, and pinning its full schema here would make config
    # validation the thing that has to change every time the tool records more evidence.  What
    # is pinned is that every number the mapping quotes is the artifact's own number.
    for key in _STILL_PHOTO_BOUND_MIRRORED_TEXT_KEYS:
        if artifact.get(key) != raw[key]:
            raise ValueError(
                f"Config: apps.hinge.still_photo_bound_evidence artifact disagrees on {key} "
                f"({_safe_value_repr(artifact.get(key))} != {raw[key]!r})")
    for key in _STILL_PHOTO_BOUND_MIRRORED_INT_KEYS:
        if type(artifact.get(key)) is not int or artifact[key] != raw[key]:
            raise ValueError(
                f"Config: apps.hinge.still_photo_bound_evidence artifact disagrees on {key} "
                f"({_safe_value_repr(artifact.get(key))} != {raw[key]!r})")
    artifact_run_s = artifact.get("max_video_exact_run_s")
    if (isinstance(artifact_run_s, bool) or not isinstance(artifact_run_s, Real)
            or not _is_finite_number(artifact_run_s)
            or float(artifact_run_s) != float(run_s)):
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence artifact disagrees on "
            f"max_video_exact_run_s ({_safe_value_repr(artifact_run_s)} != "
            f"{_safe_value_repr(run_s)})")
    # human_ground_truth is deliberately read only from the artifact and is not a config key:
    # it is the one claim nothing downstream can re-derive, so the measurement run has to make
    # it, not the person pasting the mapping.  Which value is demanded is decided by the channel,
    # so a campaign cannot pick its channel in the mapping and its truthfulness in the artifact.
    human_ground_truth = artifact.get("human_ground_truth")
    if is_circular:
        if human_ground_truth is not False:
            raise ValueError(
                "Config: apps.hinge.still_photo_bound_evidence artifact must declare "
                f"human_ground_truth=false on {STILL_PHOTO_BOUND_CIRCULAR_CHANNEL!r}, the "
                "circular AI-labelled channel; claiming human ground truth there is a lie "
                f"(got {_safe_value_repr(human_ground_truth)})")
        artifact_acceptance = artifact.get("accepted_circular_risk")
        if artifact_acceptance != acceptance:
            raise ValueError(
                "Config: apps.hinge.still_photo_bound_evidence artifact disagrees on "
                f"accepted_circular_risk ({_safe_value_repr(artifact_acceptance)} != "
                f"{acceptance!r}); the circular AI-labelled channel "
                f"{STILL_PHOTO_BOUND_CIRCULAR_CHANNEL!r} must be accepted in the measurement "
                "artifact itself, not only in the pasted mapping")
    elif human_ground_truth is not True:
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence artifact must declare "
            f"human_ground_truth=true (got {_safe_value_repr(human_ground_truth)})")
    if artifact.get("ground_truth_channel") not in (
            STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL, STILL_PHOTO_BOUND_CIRCULAR_CHANNEL):
        raise ValueError(
            "Config: apps.hinge.still_photo_bound_evidence artifact must record "
            f"ground_truth_channel {STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL!r} (owner labelled) "
            f"or {STILL_PHOTO_BOUND_CIRCULAR_CHANNEL!r} (circular AI labelled, opted in through "
            "accepted_circular_risk)")
    try:
        install_verified_still_photo_bound(StillPhotoBoundSummary(
            ground_truth_channel=raw["ground_truth_channel"],
            # The artifact's own value, proven exactly True or exactly False just above; never a
            # constant, so the summary records which channel actually produced the labels.
            human_ground_truth=human_ground_truth,
            video_cards=raw["video_cards"],
            video_accepts=raw["video_accepts"],
            photo_cards=raw["photo_cards"],
            photo_false_refusals=raw["photo_false_refusals"],
            max_video_exact_run_s=float(run_s),
            artifact_sha256=raw["artifact_sha256"],
            device=raw["device"],
            hinge_version_name=raw["hinge_version_name"],
            accepted_circular_risk=acceptance if is_circular else None,
        ))
    except ValueError as exc:
        raise ValueError(
            f"Config: apps.hinge.still_photo_bound_evidence is not a shippable bound: {exc}"
        ) from exc


def _validate_hinge_still_photo_assumption_acceptance(cfg: Config) -> None:
    """Install the owner's centered-autoplay assumption as the numbering licence, or leave it off.

    Deliberately does NOT clear readiness on entry, unlike the measured validator: the two share
    one slot and one lifecycle, and clearing here would wipe a bound the measured pass had just
    installed.  ``_validate_hinge_still_photo_readiness`` owns the reset for both, and the
    mutual-exclusion check there guarantees at most one of the two keys is ever present.
    """
    if "hinge" not in cfg.enabled_apps:
        return
    app_cfg = (cfg.apps or {}).get("hinge", {}) or {}
    if "still_photo_assumption_acceptance" not in app_cfg:
        return
    raw = app_cfg["still_photo_assumption_acceptance"]
    if not isinstance(raw, dict):
        raise ValueError(
            "Config: apps.hinge.still_photo_assumption_acceptance must be a mapping "
            f"(got {type(raw).__name__})")
    unknown = set(raw) - _STILL_PHOTO_ASSUMPTION_KEYS
    missing = _STILL_PHOTO_ASSUMPTION_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(
            "Config: apps.hinge.still_photo_assumption_acceptance must carry exactly "
            f"{sorted(_STILL_PHOTO_ASSUMPTION_KEYS)} (missing {sorted(missing)}, "
            f"unknown {sorted(unknown)})")
    for key in sorted(_STILL_PHOTO_ASSUMPTION_KEYS):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(
                f"Config: apps.hinge.still_photo_assumption_acceptance.{key} must be nonempty "
                f"text (got {_safe_value_repr(raw[key])})")
    if raw["acceptance"] != STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION:
        # Byte-equal, same reasoning as the circular acceptance above: a sentence naming what is
        # being accepted cannot be typed by mistake, and a near-miss must not half-license it.
        raise ValueError(
            "Config: apps.hinge.still_photo_assumption_acceptance.acceptance must be exactly "
            f"{STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION!r}; this channel ships numbering with NO "
            "measured video false-accept rate, so the acceptance is the whole licence "
            f"(got {_safe_value_repr(raw['acceptance'])})")
    # An assumption about how Hinge autoplays is an assumption about ONE app build on ONE phone.
    # Same exact-serial rule as the measured bound: a copied config block must not silently
    # license numbering on a handset the owner never looked at.
    serial = app_cfg.get("serial")
    if not isinstance(serial, str) or not serial:
        raise ValueError(
            "Config: apps.hinge.still_photo_assumption_acceptance requires a nonempty "
            "apps.hinge.serial exact ADB device serial")
    if raw["device"] != serial:
        raise ValueError(
            "Config: apps.hinge.still_photo_assumption_acceptance.device must exactly equal "
            f"apps.hinge.serial (got {raw['device']!r} != {serial!r})")
    try:
        install_accepted_still_photo_assumption(StillPhotoAssumptionAcceptance(
            acceptance=raw["acceptance"],
            accepted_at=raw["accepted_at"],
            device=raw["device"],
            hinge_version_name=raw["hinge_version_name"],
            rationale=raw["rationale"],
        ))
    except ValueError as exc:
        raise ValueError(
            f"Config: apps.hinge.still_photo_assumption_acceptance is not installable: {exc}"
        ) from exc


def _validate_hinge_still_photo_readiness(cfg: Config) -> None:
    """Install AT MOST ONE numbering licence: the measured bound, or the accepted assumption.

    Readiness is process-global and single-slotted, so the reset lives here, before anything can
    raise.  The mutual-exclusion check runs before either installer for the same reason: a config
    carrying both keys must leave readiness OFF, not install one of them and then reject the run.
    """
    clear_installed_still_photo_bound()
    if "hinge" in cfg.enabled_apps:
        app_cfg = (cfg.apps or {}).get("hinge", {}) or {}
        if ("still_photo_bound_evidence" in app_cfg
                and "still_photo_assumption_acceptance" in app_cfg):
            raise ValueError(
                "Config: apps.hinge.still_photo_bound_evidence and "
                "apps.hinge.still_photo_assumption_acceptance are mutually exclusive; numbering "
                "accepts exactly one licence. A measured held-out bound must not be shadowed by "
                "an unmeasured assumption, and an assumption must not be dressed up as a "
                "measurement: keep the evidence and delete the acceptance, or delete the "
                "evidence to ship on the assumption alone")
    # Order matters only in that the measured pass also performs the reset for a direct caller
    # (the measurement tool's own tests call it standalone); it is idempotent here.
    _validate_hinge_still_photo_bound_evidence(cfg)
    _validate_hinge_still_photo_assumption_acceptance(cfg)


def _canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _validate_hinge_auto_release_evidence(cfg: Config) -> None:
    """Require a verified production-OBSERVE artifact before Hinge AUTO can start.

    This runs only for an enabled Hinge app whose effective mode is AUTO.  Observe deliberately
    remains allowed with a targeting calibration but without this evidence so the required
    production validation can actually be performed.
    """
    if "hinge" not in cfg.enabled_apps:
        return
    app_cfg = (cfg.apps or {}).get("hinge", {}) or {}
    if app_cfg.get("mode", cfg.mode) != "auto":
        return
    policy_blocker = hinge_targeting_unavailable_reason()
    if policy_blocker is not None:
        raise ValueError(
            "Config: Hinge AUTO is blocked because numbered still-photo targeting cannot "
            f"be licensed: {policy_blocker}. Observe remains available without targeted "
            "suggestions")
    calibration = app_cfg.get("targeting_calibration")
    if not isinstance(calibration, dict):
        raise ValueError(
            "Config: Hinge AUTO requires apps.hinge.targeting_calibration and separately "
            "verified apps.hinge.observe_release_evidence; run production OBSERVE first")
    raw = app_cfg.get("observe_release_evidence")
    if not isinstance(raw, dict):
        raise ValueError(
            "Config: Hinge AUTO is blocked until apps.hinge.observe_release_evidence is an "
            "exact verified production-OBSERVE mapping")
    unknown = set(raw) - _OBSERVE_RELEASE_EVIDENCE_KEYS
    missing = _OBSERVE_RELEASE_EVIDENCE_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(
            "Config: apps.hinge.observe_release_evidence must carry exactly "
            f"{sorted(_OBSERVE_RELEASE_EVIDENCE_KEYS)} (missing {sorted(missing)}, "
            f"unknown {sorted(unknown)})")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Config: apps.hinge.observe_release_evidence.schema_version must be integer 1")
    expected_sha = _canonical_sha256(calibration)
    if raw["calibration_calibrated_at"] != calibration.get("calibrated_at"):
        raise ValueError("Config: observe_release_evidence.calibration_calibrated_at does not bind this calibration")
    if raw["calibration_sha256"] != expected_sha:
        raise ValueError("Config: observe_release_evidence.calibration_sha256 does not bind this calibration")
    for key in ("device", "hinge_version_name"):
        if raw[key] != calibration.get(key):
            raise ValueError(f"Config: observe_release_evidence.{key} does not bind this calibration")
    if raw["frame_size_px"] != calibration.get("frame_size_px"):
        raise ValueError("Config: observe_release_evidence.frame_size_px does not bind this calibration")
    for key in ("production_run_reference", "production_run_id", "verification_file", "verification_sha256", "verified_at"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"Config: observe_release_evidence.{key} must be nonempty text")
    if Path(raw["production_run_reference"]).name != raw["production_run_id"]:
        raise ValueError("Config: observe_release_evidence production_run_reference does not bind production_run_id")
    evidence_path = Path(raw["verification_file"])
    if evidence_path.is_absolute():
        raise ValueError("Config: observe_release_evidence.verification_file must be a repo-relative local path")
    root = Path.cwd().resolve()
    resolved = (root / evidence_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Config: observe_release_evidence.verification_file escapes the repository") from exc
    try:
        content = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"Config: observe_release_evidence verification artifact is unreadable: {resolved}") from exc
    if hashlib.sha256(content).hexdigest() != raw["verification_sha256"]:
        raise ValueError("Config: observe_release_evidence.verification_sha256 does not match its artifact")
    try:
        artifact = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Config: observe_release_evidence artifact is not JSON") from exc
    if not isinstance(artifact, dict) or set(artifact) != _OBSERVE_RELEASE_ARTIFACT_KEYS:
        raise ValueError("Config: observe_release_evidence artifact has an invalid exact schema")
    if artifact["schema_version"] != 1 or artifact["kind"] != "hinge_production_observe_release":
        raise ValueError("Config: observe_release_evidence artifact has unsupported schema/kind")
    if artifact["completed"] is not True:
        raise ValueError("Config: observe_release_evidence artifact is incomplete")
    for key in ("calibration_calibrated_at", "calibration_sha256", "device",
                "hinge_version_name", "frame_size_px", "production_run_reference",
                "production_run_id", "verified_at"):
        if artifact[key] != raw[key]:
            raise ValueError(f"Config: observe_release_evidence artifact disagrees on {key}")
    for key in ("debug_actions_sha256", "store_persistence_evidence_sha256",
                "provider_store_evidence_sha256", "observe_control_evidence_sha256"):
        value = artifact[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Config: observe_release_evidence artifact {key} is not a SHA-256 digest")


def _validate_hinge_ai_reviewed_auto_release_evidence(cfg: Config) -> None:
    """Validate the explicitly accepted, non-manual Hinge OBSERVE release artifact.

    This does not relax or reinterpret ``observe_release_evidence``.  It validates the
    separately named artifact emitted by ``tools.hinge_observe_ai_release`` and rejects any
    attempt to describe automated evidence as human ground truth.
    """
    app_cfg = (cfg.apps or {}).get("hinge", {}) or {}
    raw = app_cfg.get("ai_reviewed_observe_release_evidence")
    if not isinstance(raw, dict):
        raise ValueError(
            "Config: Hinge AUTO is blocked until ai_reviewed_observe_release_evidence is an "
            "exact explicitly accepted AI-reviewed production-OBSERVE mapping")
    unknown = set(raw) - _AI_OBSERVE_RELEASE_EVIDENCE_KEYS
    missing = _AI_OBSERVE_RELEASE_EVIDENCE_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(
            "Config: apps.hinge.ai_reviewed_observe_release_evidence must carry exactly "
            f"{sorted(_AI_OBSERVE_RELEASE_EVIDENCE_KEYS)} (missing {sorted(missing)}, "
            f"unknown {sorted(unknown)})")
    if raw["schema_version"] != 1:
        raise ValueError("Config: ai_reviewed_observe_release_evidence.schema_version must be integer 1")
    if raw["acceptance"] != _AI_OBSERVE_RELEASE_ACCEPTANCE:
        raise ValueError("Config: AI-reviewed Hinge AUTO release requires the exact explicit acceptance token")
    calibration = app_cfg.get("targeting_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Config: Hinge AUTO requires apps.hinge.targeting_calibration before release evidence")
    if raw["calibration_calibrated_at"] != calibration.get("calibrated_at"):
        raise ValueError("Config: AI-reviewed release evidence does not bind this calibration timestamp")
    if raw["calibration_sha256"] != _canonical_sha256(calibration):
        raise ValueError("Config: AI-reviewed release evidence does not bind this calibration hash")
    for key in ("device", "hinge_version_name", "frame_size_px"):
        if raw[key] != calibration.get(key):
            raise ValueError(f"Config: AI-reviewed release evidence {key} does not bind this calibration")
    for key in ("production_run_reference", "production_run_id", "verification_file",
                "verification_sha256", "verified_at"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"Config: ai_reviewed_observe_release_evidence.{key} must be nonempty text")
    if Path(raw["production_run_reference"]).name != raw["production_run_id"]:
        raise ValueError("Config: AI-reviewed release production_run_reference does not bind production_run_id")
    evidence_path = Path(raw["verification_file"])
    if evidence_path.is_absolute():
        raise ValueError("Config: AI-reviewed release verification_file must be a repo-relative local path")
    root = Path.cwd().resolve()
    resolved = (root / evidence_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Config: AI-reviewed release verification_file escapes the repository") from exc
    try:
        content = resolved.read_bytes()
    except OSError as exc:
        raise ValueError("Config: AI-reviewed release verification artifact is unreadable") from exc
    if hashlib.sha256(content).hexdigest() != raw["verification_sha256"]:
        raise ValueError("Config: AI-reviewed release verification_sha256 does not match its artifact")
    try:
        artifact = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Config: AI-reviewed release artifact is not JSON") from exc
    if not isinstance(artifact, dict) or set(artifact) != _AI_OBSERVE_RELEASE_ARTIFACT_KEYS:
        raise ValueError("Config: AI-reviewed release artifact has an invalid exact schema")
    if artifact["schema_version"] != 1 or artifact["kind"] != "hinge_ai_reviewed_production_observe_release":
        raise ValueError("Config: AI-reviewed release artifact has unsupported schema/kind")
    if artifact["completed"] is not True:
        raise ValueError("Config: AI-reviewed release artifact is incomplete")
    if artifact["human_ground_truth"] is not False:
        raise ValueError("Config: AI-reviewed release artifact must honestly declare human_ground_truth=false")
    if artifact["source"] not in {"external_ai_review", "automation"}:
        raise ValueError("Config: AI-reviewed release artifact source must be external_ai_review or automation")
    for key in ("acceptance", "calibration_calibrated_at", "calibration_sha256", "device",
                "hinge_version_name", "frame_size_px", "production_run_reference",
                "production_run_id", "verified_at"):
        if artifact[key] != raw[key]:
            raise ValueError(f"Config: AI-reviewed release artifact disagrees on {key}")
    for key in ("debug_actions_sha256", "store_persistence_evidence_sha256",
                "provider_store_evidence_sha256", "automation_provenance_sha256",
                "independent_review_sha256"):
        value = artifact[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Config: AI-reviewed release artifact {key} is not a SHA-256 digest")


def _validate_hinge_auto_release_gate(cfg: Config) -> None:
    """Select exactly one release-gate provenance mode; legacy manual validator stays intact."""
    if "hinge" not in cfg.enabled_apps:
        return
    app_cfg = (cfg.apps or {}).get("hinge", {}) or {}
    if app_cfg.get("mode", cfg.mode) != "auto":
        return
    manual = app_cfg.get("observe_release_evidence")
    ai_reviewed = app_cfg.get("ai_reviewed_observe_release_evidence")
    if manual is not None and ai_reviewed is not None:
        raise ValueError(
            "Config: Hinge AUTO requires exactly one release gate: manual observe_release_evidence "
            "or explicitly accepted ai_reviewed_observe_release_evidence, never both")
    if ai_reviewed is not None:
        _validate_hinge_ai_reviewed_auto_release_evidence(cfg)
    else:
        _validate_hinge_auto_release_evidence(cfg)


def _validate_hinge_ai_observe_controller(cfg: Config) -> None:
    """Require an explicit controller declaration before OBSERVE stores non-manual labels."""
    if "hinge" not in cfg.enabled_apps:
        return
    app_cfg = (cfg.apps or {}).get("hinge", {}) or {}
    source = app_cfg.get("observe_evidence_source", "manual")
    if source == "manual":
        return
    if source not in {"external_ai_review", "automation"}:
        raise ValueError("Config: apps.hinge.observe_evidence_source must be manual, external_ai_review, or automation")
    if app_cfg.get("mode", cfg.mode) != "observe":
        raise ValueError("Config: non-manual Hinge OBSERVE evidence source is allowed only in mode observe")
    controller = app_cfg.get("ai_reviewed_observe_controller")
    if not isinstance(controller, dict) or set(controller) != _AI_OBSERVE_CONTROLLER_KEYS:
        raise ValueError("Config: non-manual Hinge OBSERVE requires an exact ai_reviewed_observe_controller mapping")
    if controller.get("schema_version") != 1 or controller.get("source") != source:
        raise ValueError("Config: ai_reviewed_observe_controller must bind schema version and evidence source")
    if controller.get("acceptance") != _AI_OBSERVE_RELEASE_ACCEPTANCE:
        raise ValueError("Config: non-manual Hinge OBSERVE requires the exact explicit acceptance token")
    executor = controller.get("executor")
    if not isinstance(executor, dict) or set(executor) != {"model", "id", "version", "process"}:
        raise ValueError("Config: ai_reviewed_observe_controller.executor must carry exact model/id/version/process")
    if any(not isinstance(executor.get(k), str) or not executor[k].strip() for k in executor):
        raise ValueError("Config: ai_reviewed_observe_controller.executor values must be nonempty text")

# worker.py's _pace() scales human_motion.think_time_s()'s WHOLE draw (including its
# shifted-lognormal floor: shift=1.2s for "like"/1.8s for "pass", means ~3.2s/~6.9s) by
# swipe_delay_s / this default. Below this floor the scaled floor drops under ~0.35s and
# the mean under ~1s on the fast tail — no longer distinguishable from scripted,
# machine-speed swiping, the exact behaviour this project's anti-bot design exists to
# prevent (see ops/ANTI-BOT-RESEARCH.md). 0 is a separate, explicit "pacing off" sentinel
# (PacingCfg docstring) and is exempted below, not folded into this floor.
_MIN_SWIPE_DELAY_S = 1.0
# A one-hour tuning anchor is already orders of magnitude beyond plausible human pacing and
# still leaves ample headroom when Worker scales its random pause before Event.wait(). Without
# any ceiling, values such as 1e300 pass finite-number validation and overflow the platform
# timeout conversion instead of producing a controlled wait.
_MAX_SWIPE_DELAY_S = 3600.0
# Android read cadence is repeated once per capture, so bound both operands: values above
# these are operational mistakes, not useful tuning. A tenth of a second is the smallest
# deliberate dwell; one minute per frame and 100 frames are already extremely conservative
# diagnostic ceilings while remaining far below Event.wait's platform overflow range.
_MIN_ANDROID_DWELL_S = 0.1
_MAX_ANDROID_DWELL_S = 60.0
_MAX_ANDROID_SCROLL_CAPTURES = 100
# Each candidate beyond the first spends a real navigation hop plus a full two-burst dwell, so
# this ceiling is far tighter than scroll_captures' -- a value this high is already several times
# more cards than any real Hinge profile carries.
_MAX_STILL_PHOTO_DWELL_CANDIDATES = 20


def _require_bool(value, label: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"Config: {label} must be true or false (got {value!r})")


def _require_nonempty_text(value, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config: {label} must be a non-empty string (got {value!r})")


def _require_positive_int(value, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"Config: {label} must be a positive integer (got {_safe_value_repr(value)})")


def _require_finite_real(value, label: str, *, minimum: float | None = None,
                         maximum: float | None = None) -> None:
    rendered = _safe_value_repr(value)
    if isinstance(value, bool) or not isinstance(value, Real) or not _is_finite_number(value):
        raise ValueError(f"Config: {label} must be a finite number (got {rendered})")
    if minimum is not None and value < minimum:
        raise ValueError(f"Config: {label} must be >= {minimum} (got {rendered})")
    if maximum is not None and value > maximum:
        raise ValueError(f"Config: {label} must be <= {maximum} (got {rendered})")


def normalize_swipe_delay_s(value: object) -> float:
    """Return the validated pacing anchor shared by config and direct Workers."""
    _require_finite_real(value, "pacing.swipe_delay_s", maximum=_MAX_SWIPE_DELAY_S)
    if value != 0 and value < _MIN_SWIPE_DELAY_S:
        raise ValueError(f"Config: pacing.swipe_delay_s must be 0 (pacing off) or "
                         f">= {_MIN_SWIPE_DELAY_S} (got {value}); smaller values scale "
                         "worker._pace's human-pause distribution down to machine-speed "
                         "swiping")
    return float(value)


def _validate_config_shape(cfg: Config) -> None:
    if not isinstance(cfg.enabled_apps, list):
        raise ValueError("Config: enabled_apps must be a YAML list of app ids")
    if any(not isinstance(app, str) or not app.strip() for app in cfg.enabled_apps):
        raise ValueError("Config: enabled_apps must contain only non-empty app id strings")
    if len(set(cfg.enabled_apps)) != len(cfg.enabled_apps):
        raise ValueError("Config: enabled_apps must not contain duplicate app ids")
    if not isinstance(cfg.mode, str):
        raise ValueError(f"Config: mode must be a string (got {cfg.mode!r})")
    if not isinstance(cfg.apps, Mapping):
        raise ValueError(f"Config: apps must be a mapping (got {type(cfg.apps).__name__})")
    for app, app_cfg in cfg.apps.items():
        if not isinstance(app, str) or not app.strip():
            raise ValueError("Config: apps keys must be non-empty app id strings")
        if app not in platforms.KNOWN_APPS:
            raise ValueError(
                f"Config: unknown app block apps.{app}; supported: "
                f"{sorted(platforms.KNOWN_APPS)}")
        if not isinstance(app_cfg, Mapping):
            raise ValueError(f"Config: apps.{app} must be a mapping "
                             f"(got {type(app_cfg).__name__})")
        if "mode" in app_cfg and not isinstance(app_cfg["mode"], str):
            raise ValueError(f"Config: apps.{app}.mode must be a string "
                             f"(got {app_cfg['mode']!r})")
        if "mode" in app_cfg and app_cfg["mode"] not in {"observe", "auto"}:
            raise ValueError(
                f"Config: apps.{app}.mode must be 'observe' or 'auto' "
                f"(got {app_cfg['mode']!r})")


def _validate_core_scalars(cfg: Config) -> None:
    _require_finite_real(cfg.ranker.like_threshold, "ranker.like_threshold",
                         minimum=0, maximum=1)
    for name in ("min_labels_to_engage", "retrain_every", "min_per_class"):
        _require_positive_int(getattr(cfg.ranker, name), f"ranker.{name}")

    _require_bool(cfg.quality_filter.enabled, "quality_filter.enabled")
    _require_nonempty_text(cfg.quality_filter.metric, "quality_filter.metric")
    _require_finite_real(cfg.quality_filter.min_score, "quality_filter.min_score",
                         minimum=0, maximum=1)

    normalize_swipe_delay_s(cfg.pacing.swipe_delay_s)


def _validate_opener_scalars(opener: OpenerCfg) -> None:
    _require_bool(opener.enabled, "opener.enabled")
    _require_bool(opener.preflight, "opener.preflight")
    _require_nonempty_text(opener.provider, "opener.provider")
    _require_nonempty_text(opener.model, "opener.model")
    if opener.model != opener.model.strip():
        raise ValueError(
            f"Config: opener.model must not contain surrounding whitespace "
            f"(got {opener.model!r})")
    if not isinstance(opener.style, str):
        raise ValueError(f"Config: opener.style must be a string (got {opener.style!r})")
    _require_positive_int(opener.max_tokens, "opener.max_tokens")
    if not isinstance(opener.models, list):
        raise ValueError("Config: opener.models must be a YAML list of model ids")
    if any(not isinstance(model, str) or not model.strip() for model in opener.models):
        raise ValueError("Config: opener.models must contain only non-empty model ids")
    if any(model != model.strip() for model in opener.models):
        raise ValueError(
            "Config: opener.models model ids must not contain surrounding whitespace")
    if len(set(opener.models)) != len(opener.models):
        raise ValueError("Config: opener.models must not contain duplicate model ids")


def _validate_budget(cfg: Config) -> None:
    for name in ("run_budget_usd", "day_budget_usd"):
        value = getattr(cfg.budget, name)
        if value is not None:
            _require_finite_real(value, f"budget.{name}", minimum=0)
    if not isinstance(cfg.budget.pricing, Mapping):
        raise ValueError("Config: budget.pricing must be a mapping")
    for model, pricing in cfg.budget.pricing.items():
        _require_nonempty_text(model, "budget.pricing model id")
        if not isinstance(pricing, ModelPricing):
            raise ValueError(f"Config: budget.pricing[{model!r}] must be a pricing mapping")


def _validate_storage(cfg: Config) -> None:
    if not isinstance(cfg.storage.backend, str):
        raise ValueError(f"Config: storage.backend must be a string "
                         f"(got {cfg.storage.backend!r})")
    if cfg.storage.backend not in {"bigquery", "sqlite"}:
        raise ValueError(f"Config: storage.backend must be 'bigquery' or 'sqlite' "
                         f"(got {cfg.storage.backend})")
    if not isinstance(cfg.storage.bigquery, Mapping):
        raise ValueError("Config: storage.bigquery must be a mapping")
    _reject_unknown_keys(cfg.storage.bigquery, _BIGQUERY_KEYS, "storage.bigquery")
    if "flush_every" in cfg.storage.bigquery:
        _require_positive_int(cfg.storage.bigquery["flush_every"],
                              "storage.bigquery.flush_every")
    if cfg.storage.backend == "bigquery":
        for key in ("project_id", "photo_bucket"):
            _require_nonempty_text(cfg.storage.bigquery.get(key), f"storage.bigquery.{key}")
        try:
            validate_bigquery_photo_bucket(cfg.storage.bigquery["photo_bucket"])
        except ValueError as exc:
            raise ValueError(f"Config: storage.bigquery.photo_bucket is invalid: {exc}") from exc
        # BigQueryStore's long-standing public defaults remain valid when omitted. Validate
        # the values that will actually be interpolated into SQL, including those defaults.
        effective = {
            "project_id": cfg.storage.bigquery["project_id"],
            "dataset": cfg.storage.bigquery.get("dataset", "operation_love"),
            "location": cfg.storage.bigquery.get("location", "US"),
        }
        _require_nonempty_text(effective["dataset"], "storage.bigquery.dataset")
        _require_nonempty_text(effective["location"], "storage.bigquery.location")
        for key, pattern in BIGQUERY_IDENTIFIER_PATTERNS.items():
            value = effective[key]
            if pattern.fullmatch(value) is None:
                raise ValueError(
                    f"Config: storage.bigquery.{key} contains unsupported characters "
                    f"(got {value!r})")


def _validate_limits(label: str, lim: Mapping) -> None:
    """Shared rule set for the global `limits:` block AND any per-app `apps.<app>.limits`
    override (supervisor.py merges them: `{**cfg.limits, **app_cfg.get("limits", {})}`) —
    factored so the two paths can never drift out of sync."""
    if not isinstance(lim, Mapping):
        raise ValueError(f"Config: {label} must be a mapping "
                         f"(got {type(lim).__name__})")
    unknown = set(lim) - _LIMITS_KEYS
    if unknown:
        raise ValueError(f"Config: unknown key(s) under {label}: {sorted(unknown)}; "
                         f"supported: {sorted(_LIMITS_KEYS)}")
    for key in ("max_per_run", "max_per_day", "max_likes_per_run"):
        val = lim.get(key)
        if val is not None:
            _require_positive_int(val, f"{label}.{key}")
    ratio = lim.get("target_like_ratio")
    if ratio is not None:
        _require_finite_real(ratio, f"{label}.target_like_ratio")
        if not 0 < ratio < 1:
            raise ValueError(
                f"Config: {label}.target_like_ratio must be in (0, 1) (got {ratio})")


# Gemini's generationConfig.thinkingConfig recognizes exactly these two keys -- the field
# name differs by model family (thinkingLevel on the 3.x line, thinkingBudget on 2.5), and
# sending the wrong one, or an unrecognized key, is a guaranteed 400 on every opener call.
_THINKING_KEYS = {"thinkingLevel", "thinkingBudget"}
_THINKING_LEVELS = {"minimal", "low", "medium", "high"}


def _validate_gemini_thinking(opener: OpenerCfg) -> None:
    """Gemini-only: every model in opener.models needs an explicit opener.thinking entry.

    Thinking is ON BY DEFAULT for nearly every free-tier model in this project's cascade
    (all but gemini-2.5-flash-lite), and thought tokens are billed against opener.max_tokens
    -- a model silently running at its (often "high") default thinking level can burn the
    whole token budget and return no opener text at all (see GeminiOpener._parse's
    MAX_TOKENS diagnostic). Requiring an entry means that risk is always a deliberate choice,
    never an oversight. An explicit empty dict {} is the sanctioned way to say "use this
    model's server default" and must pass.

    Also validates each entry's shape here rather than letting it fail live: an unrecognized
    key, a bad thinkingLevel value, or a non-int thinkingBudget is a guaranteed 400 on every
    single opener call once a run starts, so it's worth catching at config-load time instead.
    """
    # `thinking:` left bare in YAML parses to None, and OpenerCfg is built generically via
    # cls(**raw_section) so nothing coerces it first. Normalize locally (rather than mutating
    # the config) and reject any non-mapping outright: without this, a blanked-out thinking
    # block crashed validate() with a raw TypeError instead of the actionable ValueError
    # every other optional block in this file degrades to. Note a None/empty mapping still
    # fails the `missing` check below whenever any model is configured, so an enabled opener
    # can never reach the API with thinking unset.
    thinking = opener.thinking or {}
    if not isinstance(thinking, dict):
        raise ValueError(
            f"Config: opener.thinking must be a mapping of model id -> thinkingConfig "
            f"(got {type(opener.thinking).__name__})")
    missing = [m for m in opener.effective_models if m not in thinking]
    if missing:
        raise ValueError(
            f"Config: opener.thinking is missing an entry for {missing!r}. Every Gemini "
            "model in opener.models needs an explicit opener.thinking entry -- pass {} to "
            "deliberately use that model's server-default thinking level, or a "
            "{thinkingLevel: ...} / {thinkingBudget: ...} mapping to override it. Thinking "
            "is on by default for nearly every free-tier model and is billed against "
            "opener.max_tokens, so an unset entry risks silently truncating every opener.")
    configured_models = set(opener.effective_models)
    unknown_models = set(thinking) - configured_models
    if unknown_models:
        raise ValueError(
            f"Config: opener.thinking contains unconfigured model id(s) "
            f"{sorted(unknown_models, key=repr)!r}; entries must exactly match "
            "opener.models (or the legacy opener.model).")
    for model, entry in thinking.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Config: opener.thinking[{model!r}] must be a mapping (got "
                f"{type(entry).__name__})")
        unknown = set(entry) - _THINKING_KEYS
        if unknown:
            raise ValueError(
                f"Config: opener.thinking[{model!r}] has unknown key(s) {sorted(unknown)}; "
                f"Gemini's generationConfig.thinkingConfig only recognizes "
                f"{sorted(_THINKING_KEYS)} -- a typo here is a 400 on every opener call.")
        if "thinkingLevel" in entry and entry["thinkingLevel"] not in _THINKING_LEVELS:
            raise ValueError(
                f"Config: opener.thinking[{model!r}].thinkingLevel must be one of "
                f"{sorted(_THINKING_LEVELS)} (got {entry['thinkingLevel']!r})")
        if "thinkingBudget" in entry:
            budget = entry["thinkingBudget"]
            # bool is a subclass of int in Python, so a bare `isinstance(budget, int)` would
            # wave `thinkingBudget: true` straight through to the API as JSON `true` -- a 400
            # on every opener call, which is exactly what this validator exists to prevent.
            # Negative budgets are rejected for the same reason: 0 disables thinking and
            # positive values cap it, so anything below 0 is meaningless to the API.
            if isinstance(budget, bool) or not isinstance(budget, int):
                raise ValueError(
                    f"Config: opener.thinking[{model!r}].thinkingBudget must be an integer "
                    f"(got {_safe_value_repr(budget)})")
            if budget < 0:
                raise ValueError(
                    f"Config: opener.thinking[{model!r}].thinkingBudget must be >= 0 "
                    f"(0 disables thinking; got {_safe_value_repr(budget)})")


def _validate_verification(cfg: Config) -> None:
    """`halt_on_error: false` is not permitted for an app running in AUTO mode.

    That flag does more than its name suggests: it gates the driver's post-action checks
    (`_verify_progress`, `_verify_like_landed`) entirely, not just what happens after one
    fails. With it off, a like whose "Send Like" tap missed returns normally, the worker
    records a decision for it, and nothing raises -- so the halt-on-unexpected path never
    engages either. The run keeps swiping while its record of what it did drifts from what
    actually happened, which corrupts the taste model, not merely the run.

    In OBSERVE mode that is tolerable: a human is driving, and the checks mostly guard
    against the bot's own missed taps. In AUTO mode nobody is watching, so unverified
    autonomous actions are exactly the thing that must not be silently switchable.

    Until now this key had no validation at all -- no type check, no enum, no warning --
    so a stale or copy-pasted config block could disable verification invisibly.
    """
    for app, app_cfg in (cfg.apps or {}).items():
        app_cfg = app_cfg or {}
        if "halt_on_error" not in app_cfg:
            continue
        value = app_cfg["halt_on_error"]
        if not isinstance(value, bool):
            raise ValueError(
                f"Config: apps.{app}.halt_on_error must be true or false (got {value!r})")
        mode = app_cfg.get("mode", cfg.mode)
        if value is False and mode == "auto":
            raise ValueError(
                f"Config: apps.{app}.halt_on_error=false is not allowed with mode='auto'. "
                f"It disables the post-action checks that confirm a like or pass actually "
                f"landed, so unsent actions would be recorded as sent and the run would "
                f"keep swiping. Set it true, or run this app in observe mode.")


# Every Android app's config block (apps.<app>) may set a `coords` mapping (each entry an
# [x, y] pair, FRACTIONS of the screen, 0..1) and assorted `*_frac` knobs (read_scroll_frac
# today; matched by suffix, not by name, so a future one is covered for free). Neither was
# validated anywhere before this: an out-of-range value -- a typo like 1.30 for 0.130, or a
# raw pixel written where a fraction was meant -- used to reach hinge.py's
# _assert_tap_allowed as the only backstop, and only AFTER a driver session was already open
# on a real phone. This is the config-load-time half of a two-part fix; the other half is
# AndroidAppSpec.__post_init__ (android_spec.py), which validates the same shapes for a
# spec's own hardcoded defaults. Both are needed because they catch different authors' typos
# at different times: a bad literal baked into HINGE_SPEC/BUMBLE_SPEC is a code-review-time
# mistake (android_spec.py's job), while a bad value under `apps.<app>.coords` in config.yaml
# is an OPERATOR's mistake made after the code shipped. Scoped to KIND_ANDROID platforms
# only, so future non-Android platform settings remain outside this safety check.
def _validate_android_fractions(cfg: Config) -> None:
    from .drivers.adb import validate_android_package_id

    for app, app_cfg in (cfg.apps or {}).items():
        if app not in platforms.KNOWN_APPS or platforms.get(app).kind != platforms.KIND_ANDROID:
            continue          # not a known Android app -- nothing here to validate
        app_cfg = app_cfg or {}
        if "package" in app_cfg:
            try:
                validate_android_package_id(
                    app_cfg["package"], context=f"apps.{app}.package")
            except ValueError as exc:
                raise ValueError(f"Config: {exc}") from exc
        for key in ("adb_path", "debug_dir"):
            if key in app_cfg:
                _require_nonempty_text(app_cfg[key], f"apps.{app}.{key}")
        if "serial" in app_cfg:
            serial = app_cfg["serial"]
            if serial is not None:
                _require_nonempty_text(serial, f"apps.{app}.serial")
        for key in ("debug_log", "observe_touch_watch", "observe_name_ocr"):
            if key in app_cfg:
                _require_bool(app_cfg[key], f"apps.{app}.{key}")
        if "touch_backend" in app_cfg:
            backend = app_cfg["touch_backend"]
            if not isinstance(backend, str) or backend not in {
                    "auto", "uhid", "adb", "uhid_persistent"}:
                raise ValueError(
                    f"Config: apps.{app}.touch_backend must be auto, uhid, adb, or "
                    f"uhid_persistent (got {backend!r})")
        if "scroll_captures" in app_cfg:
            _require_positive_int(app_cfg["scroll_captures"],
                                  f"apps.{app}.scroll_captures")
            if app_cfg["scroll_captures"] > _MAX_ANDROID_SCROLL_CAPTURES:
                raise ValueError(
                    f"Config: apps.{app}.scroll_captures must not exceed "
                    f"{_MAX_ANDROID_SCROLL_CAPTURES} "
                    f"(got {_safe_value_repr(app_cfg['scroll_captures'])})")
        # 2026-08-23: how many cards ONE capture's still-photo dwell (C2/C3) walks a real
        # navigation hop out to, beyond the free card the read already left the phone parked on.
        # 1 reproduces the pre-walk driver exactly. Each extra candidate spends a live
        # navigate_to_item climb, a fresh two-burst dwell and a measured walk back to the entry,
        # so -- same reasoning as scroll_captures -- an unbounded value is an operational mistake
        # rather than useful tuning.
        if "still_photo_dwell_candidates" in app_cfg:
            _require_positive_int(app_cfg["still_photo_dwell_candidates"],
                                  f"apps.{app}.still_photo_dwell_candidates")
            if app_cfg["still_photo_dwell_candidates"] > _MAX_STILL_PHOTO_DWELL_CANDIDATES:
                raise ValueError(
                    f"Config: apps.{app}.still_photo_dwell_candidates must not exceed "
                    f"{_MAX_STILL_PHOTO_DWELL_CANDIDATES} "
                    f"(got {_safe_value_repr(app_cfg['still_photo_dwell_candidates'])})")
        if "dwell_s" in app_cfg:
            _require_finite_real(app_cfg["dwell_s"], f"apps.{app}.dwell_s",
                                 minimum=_MIN_ANDROID_DWELL_S,
                                 maximum=_MAX_ANDROID_DWELL_S)
        if "change_threshold" in app_cfg:
            _require_finite_real(app_cfg["change_threshold"],
                                 f"apps.{app}.change_threshold")
            if not 0 < app_cfg["change_threshold"] <= 255:
                raise ValueError(
                    f"Config: apps.{app}.change_threshold must be in (0, 255] "
                    f"(got {app_cfg['change_threshold']!r})")
        # A missing/null block means "no coordinate overrides". Other falsy values are
        # supplied malformed configuration, not an alternate spelling of an empty mapping.
        coords = app_cfg.get("coords")
        if coords is None:
            coords = {}
        if not isinstance(coords, dict):
            raise ValueError(
                f"Config: apps.{app}.coords must be a mapping of name -> [x, y] "
                f"(got {type(coords).__name__})")
        for key, value in coords.items():
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
                raise ValueError(
                    f"Config: apps.{app}.coords.{key} must be a [x, y] pair of numbers "
                    f"(got {_safe_value_repr(value)})")
            for axis, v in zip("xy", value, strict=True):
                if not (_is_finite_number(v) and 0.0 <= v <= 1.0):
                    raise ValueError(
                        f"Config: apps.{app}.coords.{key} "
                        f"{axis}={_safe_value_repr(v)} must be in 0..1 -- "
                        f"coords are FRACTIONS of the screen, never pixels. An out-of-range "
                        f"value is never a legitimate tap target: the real touch transport "
                        f"clamps it onto a screen edge instead of failing, which can land "
                        f"inside a forbidden zone undetected (see hinge.py's "
                        f"_assert_tap_allowed).")
        for key, value in app_cfg.items():
            if not key.endswith("_frac"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Config: apps.{app}.{key} must be a number "
                    f"(got {_safe_value_repr(value)})")
            if not (_is_finite_number(value) and 0.0 <= value <= 1.0):
                raise ValueError(
                    f"Config: apps.{app}.{key} must be in 0..1 (a fraction of the screen) -- "
                    f"got {_safe_value_repr(value)}. This knob feeds a touch-down coordinate "
                    "computation "
                    f"(operation_love/drivers/hinge.py's AndroidDriver); an out-of-range "
                    f"value clamps onto a screen edge on a real device instead of failing, "
                    f"which can land inside a forbidden zone undetected.")


# opener.max_attempts, opener.request_timeout_s and opener.advisory_deadline_s sanity
# ceilings. All bound the SAME underlying risk -- an opener misconfig turning into a
# real-money, real-quota, real-time runaway on a single profile -- so they're derived
# together and cross-referenced below.
#
# Every opener.max_attempts retry, and every model GeminiOpener.generate() tries within a
# single attempt, is a real, billed, quota-consuming API call (see OpenerService.get_opener's
# docstring: "EVERY attempt is a real billed call, retries included"). Two independent damage
# vectors follow, and the ceiling has to cover both:
#
#   1. QUOTA: generate() returns as soon as ANY model answers with HTTP 2xx (opener.py's
#      `return self._parse(response, model)`), even when that response then fails to parse
#      into a usable opener. A rejected-content retry is therefore NOT a capacity signal and
#      does NOT advance the cascade -- the SAME model (ordinarily the first configured, and
#      in the shipped config.yaml cascade one of the three 20-requests/day Flash models) gets
#      re-hit on every retry for a stubborn profile. An uncapped max_attempts can burn an
#      entire day's quota of that one 20-RPD model -- the scarcest resource in a cascade that
#      otherwise has two models at 500 RPD and two Gemma models at 14,400 RPD -- on a SINGLE
#      profile, starving every other profile that needs that model for the rest of the day.
#
#   2. WALL CLOCK: request_timeout_s is the only thing bounding one already-in-flight HTTP
#      call (opener.py's should_stop comment spells out the existing worst-case formula for
#      the shipped 7-model cascade: max_attempts x len(models) x request_timeout_s = 5 x 7 x
#      90s = 3,150s, ~52 minutes, at today's default). Raising max_attempts without a ceiling
#      raises that product without limit: max_attempts=10000 (the value an audit found
#      `validate()` accepted) is 10000 x 7 x 90s = 6,300,000s, ~1,750 hours -- weeks, not
#      "hours," of a single profile silently wedging the worker. request_timeout_s itself was
#      equally unbounded before this change: a huge value here defeats any max_attempts cap on
#      its own, since ONE call could then hang indefinitely regardless of how few retries are
#      allowed.
#
# _MAX_ATTEMPTS_CEILING = 15: triples the owner's own stated reference point ("if after 5
# attempts it's still a bad response, stop the automation") while keeping the QUOTA worst
# case (vector 1 above) to at most 15 of a 20-RPD model's daily 20 requests -- 75% of one
# model's entire day, still leaving 5 requests of headroom for whatever other profiles queue
# up that same day, and never so large that one stubborn profile could exhaust it alone
# without the run itself first noticing and stopping (a real Gemini day-quota 429 on that
# model blacklists it for the run well before 15 further retries could occur against it).
# 10-20 was the expected range for this ceiling; 15 sits in the upper half so genuine
# resilience gains (3x today's default) are still available without approaching the point
# where a single profile's retries alone could exhaust a model's whole day.
#
# _MAX_REQUEST_TIMEOUT_S = 180.0: exactly 2x the MEASURED (not guessed) 90s worst case for a
# real 8-screenshot Hinge profile (see OpenerCfg's request_timeout_s comment in config.yaml --
# 30s was observed to time out mid-request live). Doubling the measured worst case is generous
# headroom for a slower network or a larger future payload while still keeping any ONE stuck
# call bounded to a human-scale 3 minutes rather than an unbounded stall. Paired with the
# ceiling above, the documented worst-case formula for the shipped 7-model cascade becomes 15
# x 7 x 180s = 18,900s, ~5.25 hours -- still bounded and firmly worse-than-typical, but no
# longer capable of the multi-day stalls an unbounded request_timeout_s previously allowed.
#
# _MAX_ADVISORY_DEADLINE_S = 300.0: the third vector, and the only one whose cost is paid in
# HUMAN time rather than quota or background wall clock. The advisory path exists to put a
# suggested opener in front of a person who is standing at the phone with the profile open,
# waiting to type it. That is precisely why the deadline exists at all: the two ceilings above
# bound a runaway to hours, which is fine for a background worker and useless here, because
# advisory_max_attempts x request_timeout_s on the SHIPPED config is already 3 x 90s = 270s of
# a human staring at a phone. Five minutes is deliberately far PAST the point where a
# suggestion is still worth waiting for (nobody stands at a phone for five minutes rather than
# type their own sentence), so the ceiling is not a target -- it is the outer bound past which
# the value can no longer be a considered trade-off, only a typo or a misunderstanding of what
# this knob is for. Anything genuinely useful lives an order of magnitude below it, and the
# shipped 60.0 is a fifth of the ceiling.
_MAX_ATTEMPTS_CEILING = 15
_MAX_REQUEST_TIMEOUT_S = 180.0
_MAX_ADVISORY_DEADLINE_S = 300.0


def validate(cfg: Config) -> None:
    """Fail fast with a clear message on misconfig (called by the entry points)."""
    # Numbered-targeting readiness is process-global, and only the evidence check below can
    # turn it on.  Clear it here, before anything can raise, so a config that fails validation
    # for an unrelated reason (or one that simply omits the evidence key) can never inherit a
    # licence installed by an earlier validate() in this process.
    clear_installed_still_photo_bound()
    _validate_config_shape(cfg)
    if not cfg.enabled_apps:
        raise ValueError("Config: enabled_apps is empty")
    unknown = [a for a in cfg.enabled_apps if a not in platforms.KNOWN_APPS]
    if unknown:
        raise ValueError(f"Config: unknown app(s) {unknown}; supported: {sorted(platforms.KNOWN_APPS)}")
    # Registry-level guard: STRUCTURAL only (check_selection, not check_runnable) -- two
    # Android platforms requested together (one physical phone, one foreground app at a
    # time). Deliberately does NOT reject an unavailable platform (uncalibrated Android
    # target, or a future target with no live service behind it): availability is a property of
    # the world that changes without the config file changing, and writing Bumble's
    # coordinates into config.yaml is exactly how Bumble GETS calibrated -- a config file
    # merely naming an uncalibrated platform must still load. The availability gate lives
    # at start time instead: supervisor.run() and HubState.start() both call
    # platforms.check_runnable(), the version of this check that also looks at availability.
    incoherent = platforms.check_selection(cfg.enabled_apps)
    if incoherent:
        raise ValueError(incoherent)
    modes = [cfg.mode]
    modes.extend((cfg.apps.get(a, {}) or {}).get("mode", cfg.mode)
                 for a in cfg.enabled_apps)
    bad_modes = [mode for mode in modes if mode not in {"observe", "auto"}]
    if bad_modes:
        raise ValueError(f"Config: mode must be 'observe' or 'auto' (got {bad_modes})")
    _validate_core_scalars(cfg)
    _validate_opener_scalars(cfg.opener)
    _validate_budget(cfg)
    _validate_storage(cfg)
    _validate_verification(cfg)
    _validate_android_fractions(cfg)
    # Must run BEFORE the calibration and release gates: those read
    # hinge_targeting_unavailable_reason(), which this call is what answers.  It covers BOTH
    # readiness channels (measured bound, accepted assumption) and refuses a config that
    # configures both.
    _validate_hinge_still_photo_readiness(cfg)
    _validate_targeting_calibration(cfg)
    _validate_hinge_ai_observe_controller(cfg)
    _validate_hinge_auto_release_gate(cfg)
    if cfg.opener.provider != "gemini":
        # Not merely "unsupported" -- the Anthropic/Claude opener path was deleted from the
        # codebase outright (operation_love/opener/opener.py no longer defines
        # AnthropicOpener at all), so any other value here can never be a live fallback.
        # This must fail loudly rather than silently degrade: a stray/typo'd/legacy
        # `provider: anthropic` in a config file is exactly the kind of dead-code
        # reactivation the project's fail-loud philosophy exists to catch at load time,
        # not mid-run.
        raise ValueError(
            "Config: opener.provider must be 'gemini' -- the Anthropic/Claude opener path "
            "has been removed entirely; openers run on Gemini or the run fails loudly "
            f"(got {cfg.opener.provider!r})")
    if cfg.opener.enabled:
        missing_pricing = [m for m in cfg.opener.effective_models if m not in cfg.budget.pricing]
        if missing_pricing:
            # Mention the original field too: callers with legacy single-model configs
            # receive the same useful diagnostic they did before the fallback chain.
            label = "opener.model" if not cfg.opener.models else "opener.models"
            raise ValueError(f"Config: {label} {missing_pricing!r} has no entry in budget.pricing")
    if cfg.opener.enabled:
        # provider is unconditionally "gemini" by this point -- the check above already
        # raised for any other value -- so this is Gemini's thinking-config validation,
        # not a branch on provider.
        _validate_gemini_thinking(cfg.opener)
    # opener.max_attempts: how many times OpenerService re-asks for a rejected AI response
    # (owner rule, 2026-08-10) before giving up and stopping the run. bool is a subclass of
    # int in Python, so a bare isinstance(x, int) check would wave `max_attempts: true`
    # through as 1 attempt with no warning -- the same trap opener.thinking's thinkingBudget
    # guards against above. Must be >= 1: a 0-or-negative value would mean "never even try",
    # silently skipping the opener on every profile without ever asking Gemini once.
    if isinstance(cfg.opener.max_attempts, bool) or not isinstance(cfg.opener.max_attempts, int):
        raise ValueError(
            f"Config: opener.max_attempts must be an integer "
            f"(got {_safe_value_repr(cfg.opener.max_attempts)}). "
            "This is how many times a rejected AI response is re-asked (with a correction "
            "hint) before the run stops rather than send a commentless like.")
    if cfg.opener.max_attempts < 1:
        raise ValueError(
            f"Config: opener.max_attempts must be >= 1 "
            f"(got {_safe_value_repr(cfg.opener.max_attempts)}). "
            "It must allow at least one real attempt at generating an opener before the "
            "run can decide the response is unusable and stop.")
    if cfg.opener.max_attempts > _MAX_ATTEMPTS_CEILING:
        # An audit found max_attempts had NO upper bound: `max_attempts: 10000` passed this
        # function cleanly. Every attempt is a real, billed API call, and a rejected-content
        # retry does not advance GeminiOpener's model cascade (it lands on the same model
        # again -- see _MAX_ATTEMPTS_CEILING's docstring above), so an uncapped value can burn
        # an entire day's 20-request quota of this project's best models on ONE stubborn
        # profile, and/or (combined with request_timeout_s) hang that profile for hundreds of
        # hours. See _MAX_ATTEMPTS_CEILING above for the exact arithmetic behind this number.
        raise ValueError(
            f"Config: opener.max_attempts must be between 1 and {_MAX_ATTEMPTS_CEILING} "
            f"(got {_safe_value_repr(cfg.opener.max_attempts)}). Each attempt is a real, "
            "billed API call "
            "against a small daily quota (as few as 20 requests/day for the best models in "
            "this project's cascade), and consecutive retries for one profile ordinarily hit "
            "the SAME model rather than advancing through the fallback chain, so an uncapped "
            f"value can exhaust a whole model's day on a single stubborn profile. "
            f"{_MAX_ATTEMPTS_CEILING} already triples the owner's own reference point of 5 "
            "(\"if after 5 attempts it's still a bad response, stop the automation\"). If you "
            "genuinely need more resilience than that, raising this further just re-asks a "
            "setup that has already shown it's systemically broken -- instead, find out WHY "
            "so many consecutive attempts are being rejected (check the run's debug log for "
            "the rejection reasons, and reconsider opener.style or the model's opener.thinking "
            "level) rather than spend more of the daily quota re-asking the same broken setup.")
    # opener.advisory_max_attempts: the same retry budget as max_attempts above, SHORTENED for
    # the advisory (observe-mode) path, where the opener is a suggestion shown to a human and
    # not text the bot is about to send. Same bool-before-int trap and same floor as
    # max_attempts (a 0-or-negative value means "never even try", which would silently make
    # observe mode show no suggestion on every profile without ever asking Gemini once), and
    # the same billed-call ceiling, since an advisory attempt costs exactly what an auto
    # attempt costs. These checks deliberately run AFTER max_attempts' own three above: the
    # cross-field check at the end compares the two, and comparing against an already invalid
    # max_attempts (a string, a bool, a negative) would produce a confusing message about the
    # wrong setting -- or a TypeError -- instead of naming the key actually at fault.
    if isinstance(cfg.opener.advisory_max_attempts, bool) or not isinstance(cfg.opener.advisory_max_attempts, int):
        raise ValueError(
            f"Config: opener.advisory_max_attempts must be an integer (got "
            f"{_safe_value_repr(cfg.opener.advisory_max_attempts)}). This is how many times "
            "an observe-mode "
            "opener SUGGESTION is re-asked before the profile is skipped with no suggestion.")
    if cfg.opener.advisory_max_attempts < 1:
        raise ValueError(
            f"Config: opener.advisory_max_attempts must be >= 1 (got "
            f"{_safe_value_repr(cfg.opener.advisory_max_attempts)}). It must allow at least "
            "one real attempt at "
            "generating a suggestion; 0 would silently show no opener on every observe profile "
            "without ever asking Gemini once. To turn openers off entirely, set "
            "opener.enabled: false instead -- that is the explicit, visible way to say it.")
    if cfg.opener.advisory_max_attempts > _MAX_ATTEMPTS_CEILING:
        raise ValueError(
            f"Config: opener.advisory_max_attempts must be between 1 and "
            f"{_MAX_ATTEMPTS_CEILING} "
            f"(got {_safe_value_repr(cfg.opener.advisory_max_attempts)}). An advisory "
            "attempt is a real, billed API call against the same small daily quota as an auto "
            "attempt (as few as 20 requests/day for the best models in this project's "
            "cascade), so it carries the identical ceiling. Lower this rather than raise it: a "
            "human is standing at the phone waiting for the suggestion, and if that many "
            "consecutive attempts are being rejected the setup is systemically broken -- check "
            "the run's debug log for the rejection reasons, and reconsider opener.style or the "
            "model's opener.thinking level, rather than spend more quota re-asking it.")
    if cfg.opener.advisory_max_attempts > cfg.opener.max_attempts:
        # Cross-field: advisory is a SHORTENED form of the same budget, never a bigger one.
        # OpenerService takes min(max_attempts, advisory_max_attempts) at runtime, so a larger
        # value here would not do what it says -- it would be silently clamped, leaving a
        # config file that reads as one policy and behaves as another. Fail loudly instead.
        raise ValueError(
            "Config: opener.advisory_max_attempts "
            f"({_safe_value_repr(cfg.opener.advisory_max_attempts)}) must be <= "
            f"opener.max_attempts ({_safe_value_repr(cfg.opener.max_attempts)}). Advisory is "
            "a SHORTENED "
            "form of the SAME retry budget -- the observe-mode path gives up sooner because a "
            "human is standing at the phone waiting and nothing irreversible depends on the "
            "result -- so it can never exceed the full budget it is a shortening of. Either "
            "lower opener.advisory_max_attempts or raise opener.max_attempts.")
    # opener.advisory_deadline_s: wall-clock ceiling on the advisory retry loop, the only
    # bound on how long a HUMAN waits at the phone for a suggested opener. bool-before-numeric
    # for the same int-subclass reason as the settings above (`advisory_deadline_s: true` must
    # not silently become a 1.0 second deadline that kills every retry). Must be > 0: a 0 or
    # negative deadline is already expired before the first retry is even considered, which
    # would make opener.advisory_max_attempts dead config -- if one attempt is what you want,
    # say it directly with advisory_max_attempts: 1. See _MAX_ADVISORY_DEADLINE_S above for the
    # ceiling's derivation; note it bounds a different resource (a person's patience) than
    # request_timeout_s does (one HTTP call), which is why it is a separate knob and not that
    # one reused.
    if isinstance(cfg.opener.advisory_deadline_s, bool) or not isinstance(cfg.opener.advisory_deadline_s, (int, float)):
        raise ValueError(
            f"Config: opener.advisory_deadline_s must be a number of seconds (got "
            f"{_safe_value_repr(cfg.opener.advisory_deadline_s)}).")
    if not (0 < cfg.opener.advisory_deadline_s <= _MAX_ADVISORY_DEADLINE_S):
        raise ValueError(
            f"Config: opener.advisory_deadline_s must be > 0 and <= "
            f"{_MAX_ADVISORY_DEADLINE_S} "
            f"(got {_safe_value_repr(cfg.opener.advisory_deadline_s)}). This is the "
            "only thing bounding how long a human stands at the phone waiting for a suggested "
            "opener: opener.advisory_max_attempts bounds the COUNT of advisory attempts but "
            "not their duration, so without this a run at the shipped opener.request_timeout_s "
            f"of 90 could make that person wait 3 x 90s. {_MAX_ADVISORY_DEADLINE_S} is already "
            "far past the point where a suggestion is worth waiting for rather than typing "
            "your own, so if you need longer the thing to fix is the latency or the model, not "
            "the patience budget. A value of 0 or less is not a shorter deadline but a "
            "permanently expired one -- to allow exactly one attempt, set "
            "opener.advisory_max_attempts: 1.")
    # opener.request_timeout_s: the only bound on how long a single opener API call can run
    # (see GeminiOpener.generate()'s transport call). bool-before-numeric for the same reason
    # as max_attempts above (bool is an int subclass; `request_timeout_s: true` must not
    # silently become 1.0 second). Must be > 0: 0 or a negative value is not a meaningful
    # timeout (Python's socket layer treats them as "non-blocking" / raises outright, not as
    # "wait longer"), so this is a real misconfiguration to catch at load time rather than let
    # crash unpredictably mid-run. See _MAX_REQUEST_TIMEOUT_S above for the ceiling's arithmetic
    # -- an unbounded request_timeout_s would undo opener.max_attempts' own ceiling, since a
    # single stuck call could still hang a profile indefinitely regardless of how few retries
    # are allowed.
    if isinstance(cfg.opener.request_timeout_s, bool) or not isinstance(cfg.opener.request_timeout_s, (int, float)):
        raise ValueError(
            f"Config: opener.request_timeout_s must be a number of seconds (got "
            f"{_safe_value_repr(cfg.opener.request_timeout_s)}).")
    if not (0 < cfg.opener.request_timeout_s <= _MAX_REQUEST_TIMEOUT_S):
        raise ValueError(
            f"Config: opener.request_timeout_s must be > 0 and <= {_MAX_REQUEST_TIMEOUT_S} "
            f"(got {_safe_value_repr(cfg.opener.request_timeout_s)}). This is the only thing "
            "bounding how long a "
            "single opener API call can run, and an unbounded value here would defeat "
            "opener.max_attempts' own ceiling by letting one stalled call hang a profile "
            f"indefinitely no matter how few retries are allowed. {_MAX_REQUEST_TIMEOUT_S} is "
            "2x the MEASURED (not guessed) 90s worst case for a real request -- generous "
            "headroom for a slow network or a large payload, without allowing an effectively "
            "unbounded stall.")
    # opener.max_tokens deliberately has NO ceiling here, unlike the two settings above.
    # Checked and rejected as a candidate during the same audit that added the bounds above:
    # (1) output tokens (which this budgets, including thought tokens -- see
    # _validate_gemini_thinking) are FREE on the Google free tier this project is pinned to
    # (see config.yaml's budget.pricing comment), so a large value has no cost vector distinct
    # from what's already guarded; (2) a real dollar cost, if paid pricing is ever enabled, is
    # already bounded by budget.run_budget_usd, which OpenerService re-checks after EVERY
    # attempt's spend is recorded -- max_tokens does not let spend evade that check; (3) wall
    # clock is already bounded by request_timeout_s above regardless of max_tokens' value, a
    # slow/huge generation just times out and cascades like any other stall. A too-LOW value
    # (e.g. 1) is real misconfiguration, but it fails loud and fast, not silently: every attempt
    # hits GeminiOpener._parse's MAX_TOKENS diagnostic (a normal OpenerParseError), which is
    # already bounded by opener.max_attempts above, so it stops the run within the same
    # already-enforced ceiling rather than opening a new unbounded harm vector.
    _validate_limits("limits", cfg.limits)
    for a, app_cfg in cfg.apps.items():
        # per-app limits override reaches RateLimiter the same way the global block does
        # (supervisor.py merges them) — validate with the exact same rules, or a per-app
        # max_per_run: 0 silently bypasses the safety cap entirely.
        app_limits = (app_cfg or {}).get("limits")
        _validate_limits(f"apps.{a}.limits", {} if app_limits is None else app_limits)
