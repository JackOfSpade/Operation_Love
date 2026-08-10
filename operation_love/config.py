"""Load and validate config.yaml into typed objects."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import platforms
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
    # (with a correction hint) before giving up and stopping the run -- there is no fallback
    # to a bare/commentless like on an opener-capable app (see BudgetCfg's removed
    # on_exhausted for the option this replaced). validate() caps this at
    # _MAX_ATTEMPTS_CEILING (1-15): every attempt is a real, billed, quota-consuming API call,
    # so this can't be left unbounded -- see that constant's docstring for the arithmetic.
    max_attempts: int = 5

    @property
    def effective_models(self) -> list[str]:
        """Configured model fallback order, retaining the original singular key."""
        return list(self.models) if self.models else [self.model]


@dataclass
class BudgetCfg:
    run_budget_usd: float | None = 5.00
    day_budget_usd: float | None = None  # optional daily ceiling across all runs
    # `on_exhausted` (stop | swipe_without_opener) lived here until 2026-08-10. The owner
    # ruled out commentless likes entirely -- "If after 5 attempts it's still a bad
    # response, stop the automation" -- so swipe_without_opener is no longer a supported
    # behavior at all, not merely a non-default option: stopping is now the ONLY outcome
    # when an opener cannot be produced. Do not re-add this field; load() below fails
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


_BUDGET_KEYS = {"run_budget_usd", "day_budget_usd", "pricing"}


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
    if "on_exhausted" in b:
        # Dedicated, actionable guard -- ahead of the generic unknown-key check below, which
        # would otherwise just say "unknown key(s) under budget: ['on_exhausted']" and leave
        # the reader to guess why. budget.on_exhausted was removed 2026-08-10: the owner
        # ruled out commentless likes entirely ("if after 5 attempts it's still a bad
        # response, stop the automation"), so swipe_without_opener is no longer a supported
        # behavior, and stopping is now the run's only response to an opener that can't be
        # produced. This fires even for `on_exhausted: stop` -- not because that value ever
        # did anything wrong, but so nobody keeps a dead key around believing it still
        # controls something; deleting the line changes nothing about how the run behaves.
        raise ValueError(
            "Config: budget.on_exhausted was removed and must be deleted from config.yaml. "
            "Openerless (\"commentless\") likes are no longer supported on an "
            "opener-capable app: the run now always stops when an opener cannot be "
            "produced, instead of falling back to a bare like. If your config had "
            "'on_exhausted: stop', nothing about your run's behavior changes -- that was "
            "already the only real outcome; just remove the line. If it had "
            "'on_exhausted: swipe_without_opener', that mode has been removed entirely: "
            "delete the line, and see opener.max_attempts for how many times a rejected "
            "AI response is re-asked before the run stops instead.")
    unknown = set(b) - _BUDGET_KEYS
    if unknown:
        # budget: is a money control, and it's hand-built with .get() rather than through
        # _section(), so it needs its own typo guard — a typo'd key (run_budget vs
        # run_budget_usd) must fail loudly, not silently yield an unlimited spend cap.
        raise ValueError(f"Config: unknown key(s) under budget: {sorted(unknown)}; "
                         f"supported: {sorted(_BUDGET_KEYS)}")
    pricing = {m: ModelPricing.from_dict(d) for m, d in (b.get("pricing", {}) or {}).items()}
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
    if "enabled_apps" in raw:
        enabled_apps = list(raw["enabled_apps"] or [])
    elif "app" in raw:
        enabled_apps = [raw["app"]]
    else:
        enabled_apps = ["hinge"]
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
            pricing=pricing,
        ),
        pacing=_section(PacingCfg, "pacing", raw.get("pacing", {})),
        storage=StorageCfg(
            backend=storage_raw.get("backend", "bigquery"),
            bigquery=storage_raw.get("bigquery", {}) or {},
        ),
    )


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
                    f"(got {budget!r})")
            if budget < 0:
                raise ValueError(
                    f"Config: opener.thinking[{model!r}].thinkingBudget must be >= 0 "
                    f"(0 disables thinking; got {budget!r})")


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
    for app in cfg.enabled_apps:
        app_cfg = (cfg.apps or {}).get(app, {}) or {}
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
# is an OPERATOR's mistake made after the code shipped -- exactly Bumble's current state,
# where every coordinate is an explicit placeholder awaiting a human typing real numbers in
# (see BUMBLE_SPEC's module docstring). Scoped to KIND_ANDROID platforms only: a web app's
# config (e.g. bumble_web's CSS `selectors`) has no coords/*_frac concept, and validating it
# here would be a category error, not a safety net.
def _validate_android_fractions(cfg: Config) -> None:
    for app, app_cfg in (cfg.apps or {}).items():
        if app not in platforms.KNOWN_APPS or platforms.get(app).kind != platforms.KIND_ANDROID:
            continue          # not a known Android app -- nothing here to validate
        app_cfg = app_cfg or {}
        coords = app_cfg.get("coords") or {}
        if not isinstance(coords, dict):
            raise ValueError(
                f"Config: apps.{app}.coords must be a mapping of name -> [x, y] "
                f"(got {type(coords).__name__})")
        for key, value in coords.items():
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
                raise ValueError(
                    f"Config: apps.{app}.coords.{key} must be a [x, y] pair of numbers "
                    f"(got {value!r})")
            for axis, v in zip("xy", value):
                if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                    raise ValueError(
                        f"Config: apps.{app}.coords.{key} {axis}={v!r} must be in 0..1 -- "
                        f"coords are FRACTIONS of the screen, never pixels. An out-of-range "
                        f"value is never a legitimate tap target: the real touch transport "
                        f"clamps it onto a screen edge instead of failing, which can land "
                        f"inside a forbidden zone undetected (see hinge.py's "
                        f"_assert_tap_allowed).")
        for key, value in app_cfg.items():
            if not key.endswith("_frac"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Config: apps.{app}.{key} must be a number (got {value!r})")
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(
                    f"Config: apps.{app}.{key} must be in 0..1 (a fraction of the screen) -- "
                    f"got {value!r}. This knob feeds a touch-down coordinate computation "
                    f"(operation_love/drivers/hinge.py's AndroidDriver); an out-of-range "
                    f"value clamps onto a screen edge on a real device instead of failing, "
                    f"which can land inside a forbidden zone undetected.")


# opener.max_attempts and opener.request_timeout_s sanity ceilings. Both bound the SAME
# underlying risk -- an opener misconfig turning into a real-money, real-quota, real-time
# runaway on a single profile -- so they're derived together and cross-referenced below.
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
_MAX_ATTEMPTS_CEILING = 15
_MAX_REQUEST_TIMEOUT_S = 180.0


def validate(cfg: Config) -> None:
    """Fail fast with a clear message on misconfig (called by the entry points)."""
    if not cfg.enabled_apps:
        raise ValueError("Config: enabled_apps is empty")
    unknown = [a for a in cfg.enabled_apps if a not in platforms.KNOWN_APPS]
    if unknown:
        raise ValueError(f"Config: unknown app(s) {unknown}; supported: {sorted(platforms.KNOWN_APPS)}")
    # Registry-level guard: STRUCTURAL only (check_selection, not check_runnable) -- two
    # Android platforms requested together (one physical phone, one foreground app at a
    # time). Deliberately does NOT reject an unavailable platform (uncalibrated Android
    # target, or a web target with no live site behind it): availability is a property of
    # the world that changes without the config file changing, and writing Bumble's
    # coordinates into config.yaml is exactly how Bumble GETS calibrated -- a config file
    # merely naming an uncalibrated platform must still load. The availability gate lives
    # at start time instead: supervisor.run() and HubState.start() both call
    # platforms.check_runnable(), the version of this check that also looks at availability.
    incoherent = platforms.check_selection(cfg.enabled_apps)
    if incoherent:
        raise ValueError(incoherent)
    modes = {cfg.mode} | {((cfg.apps or {}).get(a, {}) or {}).get("mode", cfg.mode) for a in cfg.enabled_apps}
    bad_modes = sorted(m for m in modes if m not in {"observe", "auto"})
    if bad_modes:
        raise ValueError(f"Config: mode must be 'observe' or 'auto' (got {bad_modes})")
    _validate_verification(cfg)
    _validate_android_fractions(cfg)
    if cfg.storage.backend not in {"bigquery", "sqlite"}:
        raise ValueError(f"Config: storage.backend must be 'bigquery' or 'sqlite' (got {cfg.storage.backend})")
    if cfg.storage.backend == "bigquery" and not (cfg.storage.bigquery or {}).get("project_id"):
        raise ValueError("Config: storage.backend=bigquery requires storage.bigquery.project_id")
    if cfg.storage.backend == "bigquery" and not (cfg.storage.bigquery or {}).get("photo_bucket"):
        raise ValueError("Config: storage.backend=bigquery requires storage.bigquery.photo_bucket")
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
    if not isinstance(cfg.opener.models, list):
        raise ValueError("Config: opener.models must be a YAML list of model ids")
    if not cfg.opener.effective_models or any(not isinstance(m, str) or not m for m in cfg.opener.effective_models):
        raise ValueError("Config: opener.models must contain one or more non-empty model ids")
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
            f"Config: opener.max_attempts must be an integer (got {cfg.opener.max_attempts!r}). "
            "This is how many times a rejected AI response is re-asked (with a correction "
            "hint) before the run stops rather than send a commentless like.")
    if cfg.opener.max_attempts < 1:
        raise ValueError(
            f"Config: opener.max_attempts must be >= 1 (got {cfg.opener.max_attempts}). "
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
            f"(got {cfg.opener.max_attempts}). Each attempt is a real, billed API call "
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
            f"{cfg.opener.request_timeout_s!r}).")
    if not (0 < cfg.opener.request_timeout_s <= _MAX_REQUEST_TIMEOUT_S):
        raise ValueError(
            f"Config: opener.request_timeout_s must be > 0 and <= {_MAX_REQUEST_TIMEOUT_S} "
            f"(got {cfg.opener.request_timeout_s}). This is the only thing bounding how long a "
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
