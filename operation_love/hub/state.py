"""HubState — owns the (at most one) active run and exposes a JSON-able snapshot.

Also tracks live hub-page (browser tab) liveness so the server can shut itself
down once the last browser tab goes away, except at a live Training approval
checkpoint that must remain available for the owner to reopen and decide.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import io
import math
import re
import threading
import time
from dataclasses import dataclass

from .. import config as cfg_mod
from .. import platforms
from .. import supervisor
from ..ranker.decider import format_photo_coverage, mark_feature_set_incomplete
from ..ranker.threshold import (
    DEFAULT_FEATURE_SET,
    FeatureSetIncomplete,
    fetch_label_rows,
    require_feature_set,
)
from ..training_actions import TrainingActionBridge

# Chrome (and others) throttle setInterval in a hidden tab to ~once/minute after 5min hidden —
# very plausible during a real run (owner watching the phone or another window, or the display
# asleep). The stale window has to clear that worst case with margin, or a throttled tab looks
# "gone" and stop()s a live run out from under the owner.
_BROWSER_CLIENT_STALE_S = 120.0
_CLOSED_BROWSER_CLIENT_TTL_S = 30.0
_EVAL_COLD_WAIT_S = (
    60.0  # bound on a cold-eval waiter so a dead computer thread can't hang it
)


_MAX_BROWSER_CLIENT_ID_LENGTH = 128

# --- AUTO pass review -----------------------------------------------------------
# The three pools the review card draws from: a uniform sample of what AUTO really passed (whose
# answers alone measure the live false-pass rate), the passes that scored closest to the cut
# (whose answers are the most informative labels) and a uniform sample of what AUTO liked.
# Mirrors the store's `kind` vocabulary.
AUTO_REVIEW_KIND_LIKES = "likes"
AUTO_REVIEW_PASS_KINDS = ("random", "near_cut")
AUTO_REVIEW_KINDS = (AUTO_REVIEW_KIND_LIKES, *AUTO_REVIEW_PASS_KINDS)
# A pass id is the worker's per-card lineage id, `uuid.uuid4().hex`: exactly 32 lowercase hex
# digits.  Anything else cannot name a stored pass, and refusing it here -- before it reaches a
# store query, a GCS object path or a response header -- is the whole path-traversal defence.
_AUTO_REVIEW_PROFILE_ID_RE = re.compile(r"[0-9a-f]{32}")
# Photos per archived pass are bounded by the driver's capture ceiling (single digits to low
# tens); a three-digit index is already nonsense, and bounding it keeps a hostile query string
# from becoming an arbitrary-size integer.
_MAX_AUTO_REVIEW_PHOTO_INDEX = 999
_MAX_AUTO_REVIEW_NAME_CHARS = 200
# The hub-side recomputation of the model calibration (idle hub, nothing live to read it from)
# is a full fit plus grouped OOF scoring, so it is cached and refreshed in the background.
_REVIEW_CALIBRATION_MAX_AGE_S = 300.0
# The card header's "N waiting" count is a store query (BigQuery when that is the backend), and
# every open tab polls it, so one answer is shared for this long; an answer or an undo through
# this hub drops it at once.
_REVIEW_WAITING_MAX_AGE_S = 15.0
_REVIEW_WAITING_KEYS = ("likes", "random", "near_cut", "total")
# The photo cards cut from an archived profile's frames for the review card.  Cutting them is
# seconds of CPU, so the last few profiles' JPEGs stay in memory (the one shown, the one the
# page prefetched, one an undo may bring back) and are never written to disk.
_REVIEW_CARDS_CACHE_PROFILES = 4
_REVIEW_CARDS_CONCURRENT_CUTS = 2
_REVIEW_CARD_JPEG_QUALITY = 88
_REVIEW_PHOTO_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


def validate_browser_client_id(value: object) -> tuple[bool, str | None, str | None]:
    """Validate the opaque per-tab identifier accepted by the liveness API.

    Browser IDs are dictionary keys for the lifetime of the page. Rejecting malformed and
    needlessly large values prevents unhashable JSON values from becoming handler errors and
    bounds the memory retained by a local client that never closes its tab.
    """
    if not isinstance(value, str) or not value.strip():
        return False, None, "browser client id must be a non-empty string"
    if len(value) > _MAX_BROWSER_CLIENT_ID_LENGTH:
        return (
            False,
            None,
            (
                f"browser client id must not exceed {_MAX_BROWSER_CLIENT_ID_LENGTH} characters"
            ),
        )
    return True, value, None


def validate_max_per_run(value: object) -> tuple[bool, int | None, str | None]:
    """Normalize the optional Hub run-cap override without coercing malformed JSON.

    ``None`` delegates to config, zero explicitly removes a configured cap for this run,
    and a positive integer sets a temporary cap. Booleans are rejected even though Python
    treats them as integers.
    """
    try:
        return True, supervisor.normalize_max_per_run(value), None
    except ValueError as exc:
        return False, None, str(exc)


def validate_apps(value: object) -> tuple[bool, list[str] | None, str | None]:
    """Validate the optional API app-id list before selection logic calls ``len`` on it."""
    if value is None:
        return True, None, None
    if not isinstance(value, list):
        return False, None, "apps must be null or a list of app ids"
    if any(not isinstance(app, str) or not app.strip() for app in value):
        return False, None, "apps must contain only non-empty app ids"
    if len(set(value)) != len(value):
        return False, None, "apps must not contain duplicate app ids"
    return True, list(value), None


def validate_stop_after_seconds(value: object) -> tuple[bool, int | None, str | None]:
    """Normalize the hub's optional local timed-stop duration.

    ``None`` and zero deliberately mean unlimited.  Do not coerce strings, floats, or
    booleans here: accepting a surprising JSON value for an operation that can stop a
    live run makes a typo look like a successful configuration.
    """
    if value is None:
        return True, None, None
    if type(value) is int and value == 0:
        return True, None, None
    if type(value) is not int or value < 1:
        return False, None, "stop_after_seconds must be null, 0, or a positive integer"
    # Event.wait (used by threading.Timer) rejects an excessively large timeout on some
    # platforms.  Keep API validation deterministic instead of letting a background thread
    # crash after Start appeared to succeed.
    if value > threading.TIMEOUT_MAX:
        return (
            False,
            None,
            f"stop_after_seconds must not exceed {int(threading.TIMEOUT_MAX)}",
        )
    return True, value, None


def validate_auto_review_kind(value: object) -> tuple[bool, str | None, str | None]:
    """Validate the review pool selector (``likes``, ``random`` or ``near_cut``), no coercion."""
    if not isinstance(value, str) or value not in AUTO_REVIEW_KINDS:
        return False, None, f"kind must be one of {', '.join(AUTO_REVIEW_KINDS)}"
    return True, value, None


def validate_auto_review_profile_id(
    value: object,
) -> tuple[bool, str | None, str | None]:
    """Accept only a worker-issued pass id: exactly 32 lowercase hex digits.

    ``re.fullmatch`` (not ``$``, which tolerates a trailing newline) so the accepted set is
    exactly the ids ``uuid.uuid4().hex`` produces and nothing a path or query could smuggle.
    """
    if not isinstance(value, str) or not _AUTO_REVIEW_PROFILE_ID_RE.fullmatch(value):
        return False, None, "profile_id must be a 32-character lowercase hex id"
    return True, value, None


def validate_auto_review_index(value: object) -> tuple[bool, int | None, str | None]:
    """Accept a photo index as a real int or a plain digit string (a query parameter)."""
    if isinstance(value, bool):
        return False, None, "index must be a non-negative integer"
    if isinstance(value, str):
        # isascii: str.isdigit() is true for e.g. superscript digits that int() then rejects.
        if not (value.isascii() and value.isdigit()) or len(value) > 4:
            return False, None, "index must be a non-negative integer"
        value = int(value)
    if not isinstance(value, int) or not 0 <= value <= _MAX_AUTO_REVIEW_PHOTO_INDEX:
        return False, None, "index must be a non-negative integer"
    return True, value, None


def validate_auto_review_answer(
    body: object,
) -> tuple[bool, tuple[str, bool] | None, str | None]:
    """Validate an answer body: a pass id and a REAL JSON boolean ``would_like``.

    Truthy strings, 0/1 and null are all refused.  An answer becomes a training label, so the
    one ambiguity a client could introduce (``"false"`` is truthy) is closed at the boundary
    instead of being interpreted.
    """
    if not isinstance(body, dict):
        return False, None, "request body must be a JSON object"
    valid_id, profile_id, id_error = validate_auto_review_profile_id(
        body.get("profile_id")
    )
    if not valid_id or profile_id is None:
        return False, None, id_error
    would_like = body.get("would_like")
    if type(would_like) is not bool:
        return False, None, "would_like must be a JSON boolean (true or false)"
    return True, (profile_id, would_like), None


def _json_safe(value: object) -> object:
    """Make a store/model payload strictly JSON-serialisable.

    ``json.dumps`` happily writes ``NaN``/``Infinity``, which are not JSON: the page's
    ``fetch().json()`` would throw and the whole card would die over one bad number.  Non-finite
    floats become ``None``; numpy scalars are unwrapped; anything unrecognised becomes ``str``.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)  # numpy scalar
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:  # noqa: BLE001 -- fall through to the string form
            pass
    return str(value)


class _ReviewUnavailable(Exception):
    """The store in use cannot serve AUTO pass review (a missing capability, not a failure)."""


def _require_review_methods(store: object, names: tuple[str, ...]) -> None:
    missing = [n for n in names if not callable(getattr(store, n, None))]
    if missing:
        raise _ReviewUnavailable(
            f"{type(store).__name__} does not support auto pass review "
            f"(missing {', '.join(missing)})"
        )


@dataclass(frozen=True)
class _ReviewCards:
    """One archived profile as the review card shows it: its photo cards as JPEGs, in order.

    ``fallback`` is why the full screenshots are shown instead (``None`` when there are cards);
    ``frame_count`` is how many archived frames the cards were cut from.
    """

    images: tuple[bytes, ...]
    sizes: tuple[tuple[int, int], ...]  # (width, height) of each card, in pixels
    frame_count: int
    fallback: str | None = None


def _encode_review_card(pixels) -> bytes:
    """One card's RGB pixels as a JPEG (display only; the bytes never leave memory)."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(
        buffer, format="JPEG", quality=_REVIEW_CARD_JPEG_QUALITY
    )
    return buffer.getvalue()


def _cut_review_cards(frames: list[bytes]) -> _ReviewCards:
    """The photo cards of one archived profile, or the reason to show its screenshots instead.

    The cards are cut by the same `extract_photo_cards` the ranker's whole-photo feature uses,
    with the like button left in (the owner sees the photos as Hinge showed them).  Any failure
    reported by PhotoCardsError falls back to screenshots; unexpected failures propagate.
    """
    if not frames:
        return _ReviewCards((), (), 0, "this profile has no stored screenshots")
    from ..vision.photo_cards import NoPhotoCards, PhotoCardsError, extract_photo_cards

    try:
        result = extract_photo_cards(frames, paint_like_button=False)
        if not result.cards:
            raise NoPhotoCards(
                f"no photo card was found in {len(frames)} archived frame(s)"
            )
        images = tuple(_encode_review_card(card.pixels) for card in result.cards)
        sizes = tuple(
            (int(card.pixels.shape[1]), int(card.pixels.shape[0]))
            for card in result.cards
        )
    except PhotoCardsError as exc:
        return _ReviewCards((), (), len(frames), f"{type(exc).__name__}: {exc}")
    return _ReviewCards(images, sizes, len(frames))


def _clean_review_waiting(value: object) -> dict | None:
    """The store's waiting counts as exact non-negative ints, or ``None`` when malformed."""
    if not isinstance(value, dict):
        return None
    clean = {}
    for key in _REVIEW_WAITING_KEYS:
        count = value.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        clean[key] = count
    if clean["total"] != sum(
        clean[key] for key in _REVIEW_WAITING_KEYS if key != "total"
    ):
        return None
    return clean


def _sniff_review_image_type(data: object) -> str | None:
    """The image media type of a stored pass photo, or ``None`` when it is not a known image.

    The bytes are served with `nosniff` and a fixed type chosen HERE rather than echoed from
    anything stored, so a corrupt or foreign object in the photo bucket is refused instead of
    being delivered to the browser under an image content type.
    """
    if not isinstance(data, (bytes, bytearray)):
        return None
    head = bytes(data[:12])
    for signature, media_type in _REVIEW_PHOTO_SIGNATURES:
        if head.startswith(signature):
            return media_type
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def _selected_platform(enabled_apps: list[str]) -> "platforms.Platform":
    """Which platform the hub should show pre-selected: the FIRST entry of enabled_apps, if
    it names a real registry id. If enabled_apps is empty or names something the registry
    doesn't know (a stale/hand-edited config.yaml), fall back to the first AVAILABLE
    platform so the hub doesn't default to a selection that would just fail on Start; if
    nothing is available at all, fall back to the first platform overall."""
    first = enabled_apps[0] if enabled_apps else None
    if first in platforms.KNOWN_APPS and platforms.get(first).available:
        return platforms.get(first)
    available = [p for p in platforms.all_platforms() if p.available]
    if available:
        return available[0]
    return platforms.all_platforms()[0]


class HubState:
    """Owns the (at most one) active run and exposes a JSON-able snapshot."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._run_generation = 0
        self._timed_stop_timer: threading.Timer | None = None
        self._timed_stop_deadline: float | None = None
        self._timed_stop_duration_seconds: int | None = None
        self._status = None  # RunStatus, captured via on_status
        self._error: str | None = None
        self._eval: dict | None = None  # cached model-quality CV (eval_snapshot)
        self._eval_at: float = 0.0
        self._eval_labels: int | None = (
            None  # label count when _eval was computed (cadence gate)
        )
        self._eval_refreshing = False
        self._eval_cold_event: threading.Event | None = None  # cold-start single-flight
        self._live_store = (
            None  # the running supervisor's store (live in-memory labels)
        )
        self._live_cfg = None  # the active run's EFFECTIVE Config (shutdown prices its own waits off it)
        self._opener_service = None  # the running supervisor's OpenerService (recent_openers_snapshot for the bug report)
        # Plain, detached copies from the most recently completed run.  The live service owns
        # locks and references to a store which supervisor.run() closes during shutdown, so the
        # hub must never keep that object merely to make a later bug report nicer.  Freeze its
        # two diagnostic rings just before dropping the reference instead.
        self._completed_openers: list[dict] = []
        self._completed_opener_rejections: list[dict] = []
        self._browser_clients: dict[str, float] = {}
        self._closed_browser_clients: dict[str, float] = {}
        self._browser_shutdown_requested = False
        self._browser_stale_watch_active = False
        # Storage mutations run outside this lock because they may involve BigQuery. This flag
        # serializes them with start(), so persisted labels cannot change underneath a newly
        # created in-memory model.
        self._training_data_mutating = False
        self._training_actions = TrainingActionBridge()
        # --- AUTO pass review (hub card) ---------------------------------------------
        # The running supervisor's PreferenceModel, captured when its Worker is bound, so the
        # card can show the calibration the live bot is actually using.  Cleared with the run.
        self._live_model = None
        # Plain-data calibration frozen when the last run in THIS hub process finished; shown
        # only while nothing fresher exists.
        self._last_calibration: dict | None = None
        # Hub-side recomputation for an idle hub (see _review_calibration).
        self._calibration_cache: dict | None = None
        self._calibration_error: str | None = None
        self._calibration_attempt_at = 0.0
        self._calibration_dirty = False
        self._calibration_refreshing = False
        # An idle hub opens its OWN store for review requests.  The browser fetches a pass's
        # photos in parallel, so concurrent requests SHARE one short-lived store (a BigQuery
        # client costs far more than a photo read) and the last one out closes it, which is
        # what flushes any buffered row.
        self._review_store = None
        self._review_store_users = 0
        self._review_store_lock = threading.Lock()
        # Serialises answers: a double-click or two open tabs must not race two label writes
        # for one pass (the store also refuses a second answer; this keeps the refusal clean).
        self._review_answer_lock = threading.Lock()
        # The header's waiting counts, shared by every tab for _REVIEW_WAITING_MAX_AGE_S.  The
        # generation is bumped by an answer or an undo, so a query that started before one can
        # never re-cache the count it made stale.
        self._review_waiting_cache: tuple[float, dict] | None = None
        self._review_waiting_generation = 0
        # Photo cards per archived profile (LRU, newest last) and the cuts in flight, so the
        # page's prefetch and its display of the same profile cut it once.
        self._review_cards_lock = threading.Lock()
        self._review_cards_cache: collections.OrderedDict[str, _ReviewCards] = (
            collections.OrderedDict()
        )
        self._review_cards_inflight: dict[str, threading.Event] = {}
        self._review_cards_slots = threading.BoundedSemaphore(
            _REVIEW_CARDS_CONCURRENT_CUTS
        )

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        mode: str | None = None,
        apps=None,
        max_per_run: int | None = None,
        stop_after_seconds: int | None = None,
    ) -> tuple[bool, str]:
        valid_cap, max_per_run, cap_error = validate_max_per_run(max_per_run)
        if not valid_cap:
            return False, cap_error or "invalid max-per-run override"
        valid_timeout, stop_after_seconds, timeout_error = validate_stop_after_seconds(
            stop_after_seconds
        )
        if not valid_timeout:
            return False, timeout_error or "invalid timed-stop duration"
        valid_apps, apps, apps_error = validate_apps(apps)
        if not valid_apps:
            return False, apps_error or "invalid app selection"
        with self._lock:
            if self.is_running():
                return False, "a run is already active"
            if self._training_data_mutating:
                return (
                    False,
                    "training data is being updated — wait for it to finish before starting",
                )
            if apps is not None and len(apps) == 0:
                # chosenApps() sends [] when every app checkbox is unchecked. The shared
                # supervisor gate also rejects this, but the Hub can give the owner a clearer
                # selection-specific message. apps=None (no override argument at all) still
                # falls through to config, unchanged.
                return (
                    False,
                    "no apps selected — check at least one app before starting",
                )
            try:
                # Validate the complete EFFECTIVE config synchronously. Registry readiness
                # alone is insufficient: a platform can implement Auto while config-level
                # release evidence still deliberately blocks it. This shared helper also
                # resolves apps=None through config.yaml and honors per-app mode overrides,
                # exactly as supervisor.run() does in its direct-caller backstop.
                effective_cfg = supervisor.load_effective_config(
                    self.config_path, mode=mode, enabled_apps=apps
                )
                # A mode chosen in the Hub is an explicit operator instruction, not merely a
                # suggestion for config.yaml.  Per-app overrides remain useful for unattended
                # CLI/config runs, but silently changing an explicit Hub mode selection is a
                # dangerous control-surface mismatch. Refuse before the
                # background thread/device setup; the operator can select the shown mode or
                # remove the per-app override deliberately.
                if mode is not None and effective_cfg is not None:
                    effective_modes = {
                        app: ((effective_cfg.apps or {}).get(app, {}) or {}).get(
                            "mode", effective_cfg.mode
                        )
                        for app in effective_cfg.enabled_apps
                    }
                    mismatched = {
                        app: actual
                        for app, actual in effective_modes.items()
                        if actual != mode
                    }
                    if mismatched:
                        details = ", ".join(
                            f"{app}={actual}"
                            for app, actual in sorted(mismatched.items())
                        )
                        return False, (
                            f"Hub requested {mode!r}, but per-app mode override(s) resolve to "
                            f"{details}; choose that mode explicitly or remove the override"
                        )
            except Exception as exc:  # noqa: BLE001 — configuration/load errors are user-facing
                return False, str(exc)
            # Treat background allocation as a transaction. A completed run's status/error
            # remains the Hub's last diagnostic snapshot unless both the optional timer and
            # the replacement run thread start successfully.
            previous_run = (
                self._run_generation,
                self._thread,
                self._stop,
                self._status,
                self._error,
                self._live_store,
                self._live_cfg,
                self._opener_service,
                self._live_model,
            )

            def _restore_failed_start() -> None:
                self._cancel_timed_stop_locked(clear=True)
                (
                    self._run_generation,
                    self._thread,
                    self._stop,
                    self._status,
                    self._error,
                    self._live_store,
                    self._live_cfg,
                    self._opener_service,
                    self._live_model,
                ) = previous_run

            # No prior timer should survive into a replacement run.  The generation check in
            # its callback is a second guard for a callback that was already queued when it
            # was cancelled.
            self._cancel_timed_stop_locked(clear=True)
            self._run_generation += 1
            generation = self._run_generation
            self._stop = threading.Event()
            self._status = None
            self._error = None
            self._live_store = None
            # Same call supervisor.run() makes on its own thread a moment later, so this IS the
            # config that run's shutdown will price its worker join and archive grace from.
            # Kept (not just validated and dropped) because the Hub's own shutdown wait has to
            # be composed from those same per-run terms -- see server._run_archive_ceiling_s.
            self._live_cfg = effective_cfg
            self._opener_service = None
            self._live_model = None  # captured again when this run's Worker is bound
            stop = self._stop

            if stop_after_seconds is not None:
                self._timed_stop_duration_seconds = stop_after_seconds
                self._timed_stop_deadline = time.monotonic() + stop_after_seconds
                timer = threading.Timer(
                    stop_after_seconds,
                    self._timed_stop_elapsed,
                    args=(generation, stop),
                )
                # A timed stop is strictly a run helper.  It must not keep the local hub
                # process alive after a normal shutdown or a test teardown.
                timer.daemon = True
                self._timed_stop_timer = timer

            def _capture(st):
                with self._lock:
                    self._status = st

            def _capture_store(store):
                with self._lock:
                    self._live_store = store

            def _capture_opener_service(svc):
                with self._lock:
                    self._opener_service = svc

            def _target():
                try:
                    supervisor.run(
                        self.config_path,
                        stop_event=stop,
                        on_status=_capture,
                        on_store=_capture_store,
                        on_opener_service=_capture_opener_service,
                        on_worker=self._bind_training_worker,
                        mode=mode,
                        enabled_apps=apps,
                        max_per_run=max_per_run,
                    )
                except (Exception, SystemExit) as exc:  # noqa: BLE001
                    # supervisor.run raises SystemExit (a BaseException, not Exception) for a
                    # startup failure it treats as fatal (e.g. missing cloud deps) — catch it
                    # here too so it surfaces in self._error like any other failure, instead of
                    # threading's default SystemExit handling silently swallowing it. Don't
                    # widen this to bare BaseException: that would also eat KeyboardInterrupt /
                    # genuine interpreter shutdown.
                    with self._lock:
                        self._error = f"{type(exc).__name__}: {exc}"
                finally:
                    # Snapshot before forgetting the live service, but retain only data.  In
                    # particular, do not keep an OpenerService after supervisor has closed its
                    # store: a completed-run report must be useful without reading a torn-down
                    # object (or retaining its locks/client/store).
                    self._freeze_completed_opener_telemetry()
                    self._freeze_completed_calibration()
                    with self._lock:
                        if generation == self._run_generation and stop is self._stop:
                            # Completion (including a supervisor-side stop) makes the timer
                            # irrelevant.  Clear it so a completed snapshot is unambiguously
                            # inactive, and so it cannot later wake up a new run.
                            self._cancel_timed_stop_locked(clear=True)
                        self._live_store = None  # supervisor closed it on exit; don't read a dead store
                        self._live_cfg = (
                            None  # no run left to price a shutdown wait for
                        )
                        self._opener_service = None  # same reason: don't read a torn-down object after the supervisor tore the run down
                        self._live_model = None  # frozen into plain data above; don't keep the model object

            self._thread = threading.Thread(target=_target, name="hub-run", daemon=True)
            # Start the optional timer first while holding _lock. Its callback also needs
            # _lock, so it cannot observe a half-started run. If timer allocation/start fails,
            # no worker exists yet. If the worker start then fails, cancellation plus the stop
            # identity reset makes even an already-queued timer callback inert.
            if self._timed_stop_timer is not None:
                try:
                    self._timed_stop_timer.start()
                except Exception as exc:  # noqa: BLE001 — user-facing lifecycle refusal
                    _restore_failed_start()
                    return False, (
                        f"could not start timed-stop timer: {type(exc).__name__}: {exc}"
                    )
            try:
                self._thread.start()
            except Exception as exc:  # noqa: BLE001 — never leave half-started Hub state
                _restore_failed_start()
                return False, f"could not start run thread: {type(exc).__name__}: {exc}"
            # Only a successfully started run supersedes the last completed run's diagnostic
            # paper trail. While active, reports must not present prior-run opener text as if
            # it belonged to the card currently on screen.
            self._completed_openers = []
            self._completed_opener_rejections = []
            return True, "started"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            stop = self._stop
            # Liveness, not merely "an Event object exists": self._stop is assigned in start()
            # and never reset, so after a run ends on its own (deck exhausted, rate limit, an
            # error halt) a gate on `stop is None` alone would keep answering (True,
            # "stopping") forever, i.e. reporting that a stop it never performed had worked.
            if stop is None or not self.is_running():
                # No run has ever started (or the last one already finished and cleared
                # self._stop -- start() only replaces it, never resets it to None itself,
                # but a fresh HubState/never-started hub has it unset). Pre-fix this always
                # returned (True, "stopping") here too, so a stray /api/stop with nothing
                # active looked like it worked.
                return False, "no run is active"
            stop.set()
            # A manual stop wins over the scheduled one.  In addition to cancelling the
            # wait, clearing the status prevents a stale countdown from surviving the
            # shutdown tail.
            self._cancel_timed_stop_locked(clear=True)
        # Printed (not just returned in the API response) so Stop lands in the SAME place
        # every OTHER shutdown trigger already announces itself: the tab-close path
        # (server.py's _schedule_shutdown_if_tab_stayed_closed), SIGINT
        # (supervisor.py's _install_signal_handlers), and a Stop-during-startup abort
        # (supervisor.py's _abort_startup) all print a line. Stop via the hub button was the
        # one silent path -- on the terminal AND on the hub's own live-log panel, since
        # bugreport.install_log_capture tees stdout into the ring /api/logs reads, so an
        # operator watching only the browser tab (not a terminal) saw nothing happen for the
        # whole worker-join tail. The decision to actually discard an in-flight swipe is
        # worker.py's, not this line's (see its stop_event re-checks) -- this just says so.
        print(
            "Hub: stop requested -- waiting for the active run to finish what it's doing. "
            "A swipe made from now on will NOT be recorded."
        )
        return True, "stopping"

    def _timed_stop_elapsed(self, generation: int, stop: threading.Event) -> None:
        """Request a stop only for the exact run that created this timer."""
        with self._lock:
            if (
                generation != self._run_generation
                or stop is not self._stop
                or not self.is_running()
                or stop.is_set()
            ):
                return
            stop.set()
            # Keep the duration/deadline until the run's thread finishes, so the active
            # stopping snapshot can show a truthful zero-second countdown.  The callback is
            # the timer thread itself, so there is nothing left to cancel.
            self._timed_stop_timer = None
            duration = self._timed_stop_duration_seconds
        print(
            "Hub: timed stop reached after "
            f"{duration} seconds -- waiting for the active run to finish what it's doing. "
            "A swipe made from now on will NOT be recorded."
        )

    def _cancel_timed_stop_locked(self, *, clear: bool) -> None:
        """Cancel the scheduled callback.  Caller must hold ``self._lock``."""
        timer = self._timed_stop_timer
        self._timed_stop_timer = None
        if timer is not None:
            timer.cancel()
        if clear:
            self._timed_stop_deadline = None
            self._timed_stop_duration_seconds = None

    def wait_for_run(self, timeout: float | None = None) -> bool:
        """Block until the active run's thread finishes (or `timeout` elapses).
        Returns True if the thread is no longer alive (already stopped, or just
        finished); False if `timeout` elapsed while it was still running."""
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return True

    def live_store(self):
        """The running supervisor's store, or None when no run holds one.

        Exposed for the shutdown path (server._wait_for_run_shutdown), which must decide
        whether the run is still inside a registered archive write before it abandons a
        decision that already landed on the phone. Read under the lock because the run thread
        both installs and clears this reference; the caller only probes it, never closes it --
        the supervisor owns its lifecycle.
        """
        with self._lock:
            return self._live_store

    def live_run_cfg(self):
        """The EFFECTIVE config of the run currently in flight, or None when idle.

        The shutdown wait is priced per-run rather than from a module constant: the archive
        deadline the supervisor grants scales with THIS run's scroll_captures, and config
        validation admits values far above the Training review ceiling.
        """
        with self._lock:
            return self._live_cfg

    def browser_client_opened(self, client_id: str | None) -> bool:
        """Mark a hub page as alive. Return True when a stale-client watch should start."""
        valid, client_id, _error = validate_browser_client_id(client_id)
        if not valid or client_id is None:
            return False
        now = time.monotonic()
        with self._lock:
            self._prune_closed_browser_clients_locked(now)
            if client_id in self._closed_browser_clients:
                return False
            self._browser_clients[client_id] = now
            self._browser_shutdown_requested = False
            if self._browser_stale_watch_active:
                return False
            self._browser_stale_watch_active = True
            return True

    def browser_client_ping(self, client_id: str | None) -> bool:
        """Refresh liveness, including a suspended page returning after stale expiry."""
        return self.browser_client_opened(client_id)

    def browser_client_closed(self, client_id: str | None) -> bool:
        """Return True once, when the last known hub page has gone away."""
        valid, client_id, _error = validate_browser_client_id(client_id)
        if not valid or client_id is None:
            return False
        now = time.monotonic()
        with self._lock:
            self._prune_closed_browser_clients_locked(now)
            self._browser_clients.pop(client_id, None)
            self._closed_browser_clients[client_id] = now
            if self._browser_clients or self._browser_shutdown_requested:
                return False
            self._browser_stale_watch_active = False
            self._browser_shutdown_requested = True
            return True

    def browser_close_shutdown_disposition(self) -> str:
        """Classify the delayed last-tab-close check without exposing run internals.

        The server calls this after its short reload grace.  Keeping the client check and the
        approval-boundary check together prevents a stale close callback from stopping a run
        after another page has reopened, and makes the decision boundary durable if a preview
        tab is unloaded.  A Training worker deliberately pauses at ``waiting_approval`` with a
        verified composer/checkpoint still live; losing the browser must not turn that pause
        into a Stop because the only safe way forward is for the owner to reopen the Hub and
        choose, or explicitly press Stop.
        """
        with self._lock:
            if self._browser_clients:
                return "client_reopened"
            if not self.is_running() or self._stop is None or self._stop.is_set():
                return "shutdown"

            status = self._status
            if status is not None:
                try:
                    apps = status.snapshot().get("apps", {})
                except Exception:  # noqa: BLE001 -- lifecycle must retain normal shutdown fallback
                    apps = {}
                if not isinstance(apps, dict):
                    apps = {}
                if any(
                    isinstance(app_status, dict)
                    and app_status.get("mode") == "training"
                    and app_status.get("state") == "waiting_approval"
                    for app_status in apps.values()
                ):
                    return "preserve_training_approval"

            # The bridge is authoritative over the exact-card lifecycle: it covers the small
            # publish-before-status hand-off, and keeps the Hub/run alive after a browser has
            # queued a choice while the worker is claiming or durably recording it.  In all of
            # those phases the reviewed profile is still live and must not be abandoned because
            # the submitting tab disappeared.
            if self._training_actions.has_live_checkpoint():
                return "preserve_training_approval"
            return "shutdown"

    def browser_stale_watch_start_failed(self) -> None:
        """Release the stale-watch claim after its watcher thread could not start.

        The next heartbeat can then retry. Otherwise the live page would be permanently marked
        as watched, while a dropped close beacon could leave the local hub running forever.
        """
        with self._lock:
            self._browser_stale_watch_active = False

    def has_browser_clients(self) -> bool:
        with self._lock:
            return bool(self._browser_clients)

    def browser_stale_watch_active(self) -> bool:
        with self._lock:
            return self._browser_stale_watch_active

    def expire_stale_browser_clients(self) -> bool:
        """Drop hub pages that stopped heartbeating.

        Missing heartbeats are not evidence of a closed tab: browsers suspend background
        pages and laptops sleep. Only an explicit close beacon may request shutdown.
        Keep the return value for the watcher's existing protocol, but never request exit.
        """
        now = time.monotonic()
        cutoff = now - _BROWSER_CLIENT_STALE_S
        with self._lock:
            self._browser_clients = {
                client_id: seen_at
                for client_id, seen_at in self._browser_clients.items()
                if seen_at >= cutoff
            }
            if self._browser_clients:
                return False
            self._browser_stale_watch_active = False
            return False

    def _prune_closed_browser_clients_locked(self, now: float) -> None:
        cutoff = now - _CLOSED_BROWSER_CLIENT_TTL_S
        for client_id, closed_at in list(self._closed_browser_clients.items()):
            if closed_at < cutoff:
                self._closed_browser_clients.pop(client_id, None)

    def snapshot(self) -> dict:
        with self._lock:
            running = self.is_running()
            status = self._status
            error = self._error
            stop_pending = self._stop is not None and self._stop.is_set()
            timed_stop_duration = self._timed_stop_duration_seconds if running else None
            timed_stop_deadline = self._timed_stop_deadline if running else None
        snap = status.snapshot() if status else None
        if snap is not None and running and stop_pending:
            # RunStatus.stopping is written by supervisor.run()'s shutdown finally, which is
            # only reached once the run has fully STARTED. A Stop pressed during startup
            # (loading labels, training the ranker, warming the ML models) is handled by
            # _abort_startup instead and never passes through that finally — so the page saw
            # running=true / stopping=false and kept the Stop button live and the pill on
            # "loading ML models" for the whole warmup, exactly the "did my click do anything?"
            # gap the stopping state exists to close. Derived here, at read time, rather than
            # adding a second writer: the stop Event IS the authority on "a stop is pending",
            # and this needs no clearing logic (running goes false when the thread ends).
            snap["stopping"] = True
        return {
            "running": running,
            "error": error,
            "status": snap,
            "timed_stop": (
                {
                    "duration_seconds": timed_stop_duration,
                    "remaining_seconds": max(
                        0, math.ceil(timed_stop_deadline - time.monotonic())
                    ),
                }
                if timed_stop_duration is not None and timed_stop_deadline is not None
                else None
            ),
        }

    def training_action_snapshot(
        self, *, run_id: str | None = None, app: str | None = None
    ) -> dict:
        return self._training_actions.snapshot(run_id=run_id, app=app)

    def training_profile_review_image(
        self, *, run_id: str, app: str, profile_token: str, index: int
    ) -> bytes | None:
        return self._training_actions.profile_review_image(
            run_id=run_id, app=app, profile_token=profile_token, index=index
        )

    def submit_training_action(self, body: dict) -> tuple[bool, dict, int]:
        return self._training_actions.submit(body)

    def record_training_browser_notification(self, body: dict) -> tuple[bool, str]:
        """Accept bounded delivery telemetry only for the currently published checkpoint."""
        run_id = body.get("run_id")
        app = body.get("app")
        profile_token = body.get("profile_token")
        outcome = body.get("notification")
        if not self._training_actions.has_checkpoint(
            run_id=run_id, app=app, profile_token=profile_token
        ):
            return False, "no matching live training checkpoint"
        from ..notifications import record_training_browser_notification

        if not isinstance(outcome, str) or not record_training_browser_notification(
            outcome
        ):
            return False, "invalid browser notification outcome"
        return True, "recorded"

    def _training_store_mutation(self, method_name: str) -> tuple[bool, dict | str]:
        """Run an explicit training-set mutation only while no model is live.

        A running supervisor keeps labels and a fitted model in memory.  Altering the
        persisted set underneath it would make the hub claim a reset while that old model
        still decides profiles, so the local control surface refuses until Stop completes.
        """
        with self._lock:
            if self.is_running():
                return False, "stop the active run before changing training data"
            if self._training_data_mutating:
                return False, "a training-data change is already in progress"
            self._training_data_mutating = True
        try:
            from ..ranker import make_store

            cfg = cfg_mod.load(self.config_path)
            store = make_store(cfg)
            try:
                result = getattr(store, method_name)()
            finally:
                store.close()
            with self._lock:
                self._eval = None
                self._eval_at = 0.0
                self._eval_labels = None
            return True, result
        except Exception as exc:  # noqa: BLE001 - surface storage errors to the local operator
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._training_data_mutating = False

    def remove_latest_training_label(self) -> tuple[bool, dict | str]:
        ok, result = self._training_store_mutation("remove_latest_training_label")
        if not ok:
            return False, result
        if result is None:
            return False, "there is no saved training label to remove"
        name = str(result.get("profile_name") or "<name unavailable>")
        print(f"Training label removed for profile: {name}")
        return True, {"profile_name": name}

    # --- AUTO pass review ------------------------------------------------------------
    # The card works with NO run live: it opens its own store exactly the way the model-quality
    # card does (make_store(cfg, ensure=False)).  With a run live it uses that run's store
    # instead, for the same freshness reason as the eval card -- passes the worker archived
    # seconds ago may still sit in the live store's buffer, invisible to a second connection.
    def _live_review_store(self):
        with self._lock:
            return self._live_store if self.is_running() else None

    @contextlib.contextmanager
    def _leased_review_store(self):
        """Lend out a short-lived store of the hub's own, shared by overlapping requests.

        The browser fetches every photo of a pass in parallel; opening (and authenticating) a
        BigQuery client per photo would dwarf the read itself.  Requests that overlap in time
        therefore share one store, and the LAST one out closes it -- ``close()`` is what flushes
        a buffered row, so nothing is left open between interactions.
        """
        from ..ranker import make_store

        with self._review_store_lock:
            if self._review_store is None:
                cfg = cfg_mod.load(self.config_path)
                self._review_store = make_store(cfg, ensure=False)
            self._review_store_users += 1
            store = self._review_store
        try:
            yield store
        finally:
            closing = None
            with self._review_store_lock:
                self._review_store_users -= 1
                if self._review_store_users == 0:
                    closing, self._review_store = self._review_store, None
            if closing is not None:
                try:
                    closing.close()
                except Exception as exc:  # noqa: BLE001 -- cleanup; writes flush explicitly first
                    print(
                        f"Hub: closing the auto-review store failed ({type(exc).__name__}: {exc})"
                    )

    @staticmethod
    def _flush_review_store(store) -> None:
        """Make a just-written review row durable BEFORE the owner is told it was saved."""
        flush = getattr(store, "flush", None)
        if callable(flush):
            flush()

    def _with_review_store(self, work, *, write: bool = False):
        """Run ``work(store)`` against the live run's store, else a leased one of our own.

        A READ that fails on the live store retries on a fresh one: the run may have ended
        and its supervisor closed that store between our lookup and the call.  A WRITE never
        retries -- after an unknown partial failure a second attempt could double-record, and
        the store itself refuses a second answer for one pass.  Writes are flushed before
        returning so a success reply means the label is durable, not merely buffered.
        """
        live = self._live_review_store()
        if live is not None:
            try:
                result = work(live)
            except _ReviewUnavailable:
                raise
            except Exception:
                if write:
                    raise
            else:
                if write:
                    self._flush_review_store(live)
                return result
        with self._leased_review_store() as store:
            result = work(store)
            if write:
                self._flush_review_store(store)
            return result

    @staticmethod
    def _clean_review_item(item: object) -> dict | None:
        """Whitelist and sanity-check one archived pass before it reaches the page.

        ``None`` means the store handed back something that is not a reviewable pass; the
        caller reports that as a store fault rather than rendering half a card.
        """
        if not isinstance(item, dict):
            return None
        valid, profile_id, _ = validate_auto_review_profile_id(item.get("profile_id"))
        if not valid or profile_id is None:
            return None
        photo_count = item.get("photo_count")
        if isinstance(photo_count, bool) or not isinstance(photo_count, int):
            return None
        if not 0 <= photo_count <= _MAX_AUTO_REVIEW_PHOTO_INDEX + 1:
            return None

        def _number(key: str) -> float | None:
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value) if math.isfinite(value) else None

        return {
            "profile_id": profile_id,
            "app": str(item.get("app") or "")[:32],
            "run_id": str(item.get("run_id") or "")[:64],
            "created_at": _number("created_at"),
            "photo_count": photo_count,
            "profile_name": str(item.get("profile_name") or "")[
                :_MAX_AUTO_REVIEW_NAME_CHARS
            ],
        }

    @staticmethod
    def _review_pool_order(kind: str) -> list[str]:
        """The pools ``auto_review_next`` tries, in order, when the owner asks for ``kind``.

        A pass pool falls back to the other pass pool before the likes, and the likes to the
        random passes before the near-cut ones, so likes and passes keep alternating for as
        long as both exist.
        """
        if kind == AUTO_REVIEW_KIND_LIKES:
            return [AUTO_REVIEW_KIND_LIKES, *AUTO_REVIEW_PASS_KINDS]
        return [
            kind,
            *(k for k in AUTO_REVIEW_PASS_KINDS if k != kind),
            AUTO_REVIEW_KIND_LIKES,
        ]

    def auto_review_next(self, kind: str) -> tuple[int, dict]:
        """The next unreviewed archived profile, preferring ``kind``, else any non-empty pool.

        The fallback lives here (one store session, one round trip) rather than in the page:
        when one pool is empty the owner should still be offered something from another.
        """
        order = self._review_pool_order(kind)

        def work(store):
            _require_review_methods(
                store,
                (
                    "next_auto_pass_for_review",
                    "auto_pass_photo",
                    "record_auto_pass_review",
                ),
            )
            for candidate in order:
                item = store.next_auto_pass_for_review(kind=candidate)
                if item is not None:
                    return item
            return None

        try:
            item = self._with_review_store(work)
        except _ReviewUnavailable as exc:
            return 200, {"available": False, "reason": str(exc), "item": None}
        except Exception as exc:  # noqa: BLE001 -- surface storage errors to the local operator
            return 502, {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "item": None,
            }
        # Pool names imply the bot's decision. Return only neutral profile metadata until
        # the owner has answered; selection details stay inside the store/hub.
        payload: dict = {"available": True, "item": None}
        if item is None:
            return 200, payload
        clean = self._clean_review_item(item)
        if clean is None:
            return 502, {
                "available": False,
                "reason": "the store returned a malformed archived pass",
                "item": None,
            }
        payload["item"] = clean
        return 200, payload

    def auto_review_image(
        self, profile_id: str, index: int
    ) -> tuple[int, bytes | None, str]:
        """One archived pass photo as ``(status, bytes | None, media type or message)``."""

        def work(store):
            _require_review_methods(store, ("auto_pass_photo",))
            return store.auto_pass_photo(profile_id=profile_id, index=index)

        try:
            data = self._with_review_store(work)
        except _ReviewUnavailable as exc:
            return 501, None, str(exc)
        except ValueError as exc:
            return 404, None, str(exc) or "auto pass image not found"
        except Exception as exc:  # noqa: BLE001 -- surface storage errors to the local operator
            return 502, None, f"{type(exc).__name__}: {exc}"
        if data is None:
            return 404, None, "auto pass image not found"
        media_type = _sniff_review_image_type(data)
        if media_type is None:
            return 502, None, "the stored auto pass photo is not a recognised image"
        return 200, bytes(data), media_type

    # --- the archived profile as its photo cards -------------------------------------------
    def _load_review_frames(self, profile_id: str) -> list[bytes]:
        """Every archived frame of one profile, in scroll order; store errors propagate."""

        def work(store):
            _require_review_methods(store, ("auto_pass_photo",))
            frames: list[bytes] = []
            for index in range(_MAX_AUTO_REVIEW_PHOTO_INDEX + 1):
                data = store.auto_pass_photo(profile_id=profile_id, index=index)
                if data is None:
                    break
                frames.append(bytes(data))
            return frames

        return self._with_review_store(work)

    def _review_cards(self, profile_id: str) -> _ReviewCards:
        """The profile's cards from the cache, else cut once and cached.

        A request that finds the same profile already being cut (the page's prefetch, then its
        display) waits for that cut instead of starting a second one; if the cut failed, the
        waiter tries itself.  A store failure is raised and never cached; a fallback is.
        """
        while True:
            with self._review_cards_lock:
                cached = self._review_cards_cache.get(profile_id)
                if cached is not None:
                    self._review_cards_cache.move_to_end(profile_id)
                    return cached
                pending = self._review_cards_inflight.get(profile_id)
                if pending is None:
                    pending = threading.Event()
                    self._review_cards_inflight[profile_id] = pending
                    break
            pending.wait()  # the cutter sets it in its `finally`, success or not
        cards = None
        try:
            # Different profiles may arrive from several tabs at once. Bound their CPU
            # and transient image memory as well as the completed-profile LRU below.
            with self._review_cards_slots:
                cards = _cut_review_cards(self._load_review_frames(profile_id))
            return cards
        finally:
            with self._review_cards_lock:
                if self._review_cards_inflight.get(profile_id) is pending:
                    del self._review_cards_inflight[profile_id]
                if cards is not None:
                    self._review_cards_cache[profile_id] = cards
                    self._review_cards_cache.move_to_end(profile_id)
                    while len(self._review_cards_cache) > _REVIEW_CARDS_CACHE_PROFILES:
                        self._review_cards_cache.popitem(last=False)
            pending.set()

    def auto_review_cards(self, profile_id: str) -> tuple[int, dict]:
        """One archived profile as its sequence of photo cards, for the review card.

        The payload lists each card's size; ``auto_review_card_image`` serves its JPEG.  When the
        cards cannot be cut, ``fallback`` says why and the page shows the full screenshots.
        """
        try:
            cards = self._review_cards(profile_id)
        except _ReviewUnavailable as exc:
            return 200, {"available": False, "reason": str(exc)}
        except ValueError as exc:
            return 404, {
                "available": False,
                "reason": str(exc) or "archived profile not found",
            }
        except Exception as exc:  # noqa: BLE001 -- surface storage errors to the local operator
            return 502, {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        return 200, {
            "available": True,
            "reason": None,
            "profile_id": profile_id,
            "frame_count": cards.frame_count,
            "cards": [
                {"index": index, "width": width, "height": height}
                for index, (width, height) in enumerate(cards.sizes)
            ],
            "fallback": cards.fallback,
        }

    def auto_review_card_image(
        self, profile_id: str, index: int
    ) -> tuple[int, bytes | None, str]:
        """One photo card as ``(status, JPEG bytes | None, media type or message)``."""
        try:
            cards = self._review_cards(profile_id)
        except _ReviewUnavailable as exc:
            return 501, None, str(exc)
        except ValueError as exc:
            return 404, None, str(exc) or "archived profile not found"
        except Exception as exc:  # noqa: BLE001 -- surface storage errors to the local operator
            return 502, None, f"{type(exc).__name__}: {exc}"
        if not 0 <= index < len(cards.images):
            return 404, None, "photo card not found"
        return 200, cards.images[index], "image/jpeg"

    # --- how many profiles wait for an answer ------------------------------------------------
    def _invalidate_review_waiting(self) -> None:
        with self._lock:
            self._review_waiting_cache = None
            self._review_waiting_generation += 1

    def auto_review_waiting_snapshot(self) -> tuple[int, dict]:
        """Unreviewed archived profiles per pool (``likes``, ``random``, ``near_cut``, ``total``).

        A passive count for the card header: nothing about it stops or slows a run.
        """
        with self._lock:
            cached = self._review_waiting_cache
            generation = self._review_waiting_generation
        if (
            cached is not None
            and time.monotonic() - cached[0] < _REVIEW_WAITING_MAX_AGE_S
        ):
            return 200, {"available": True, "reason": None, "waiting": dict(cached[1])}

        def work(store):
            _require_review_methods(store, ("auto_review_waiting",))
            return store.auto_review_waiting(app=None)

        try:
            raw = self._with_review_store(work)
        except _ReviewUnavailable as exc:
            return 200, {"available": False, "reason": str(exc), "waiting": None}
        except Exception as exc:  # noqa: BLE001 -- surface storage errors to the local operator
            return 502, {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "waiting": None,
            }
        waiting = _clean_review_waiting(raw)
        if waiting is None:
            return 502, {
                "available": False,
                "reason": "the store returned malformed waiting counts",
                "waiting": None,
            }
        with self._lock:
            if self._review_waiting_generation == generation:
                self._review_waiting_cache = (time.monotonic(), dict(waiting))
        return 200, {"available": True, "reason": None, "waiting": waiting}

    def submit_auto_review_answer(
        self, body: object
    ) -> tuple[bool, dict | None, str, int]:
        """Record the owner's answer to one archived pass; it becomes a training label.

        Returns ``(ok, result, message, http status)``.  Nothing is written unless the pass id
        is a real worker-issued id and ``would_like`` is a real boolean.
        """
        valid, parsed, error = validate_auto_review_answer(body)
        if not valid or parsed is None:
            return False, None, error or "invalid answer", 400
        profile_id, would_like = parsed
        with self._lock:
            if self._training_data_mutating:
                return (
                    False,
                    None,
                    "training data is being updated -- try again in a moment",
                    409,
                )

        def work(store):
            _require_review_methods(store, ("record_auto_pass_review",))
            return store.record_auto_pass_review(
                profile_id=profile_id, would_like=would_like
            )

        with self._review_answer_lock:
            try:
                result = self._with_review_store(work, write=True)
            except _ReviewUnavailable as exc:
                return False, None, str(exc), 501
            except ValueError as exc:
                # Unknown pass or one already answered (another tab, a double click).  `retired`
                # tells the page this card can never be answered, so it moves on instead of
                # leaving the owner stuck on it.
                self._invalidate_review_waiting()
                return (
                    False,
                    {"retired": True},
                    str(exc) or "this pass cannot be reviewed",
                    409,
                )
            except Exception as exc:  # noqa: BLE001 -- surface storage errors to the local operator
                print(
                    f"Hub: recording an auto-review answer failed ({type(exc).__name__}: {exc})"
                )
                return False, None, f"{type(exc).__name__}: {exc}", 500
            with self._lock:
                # The answer moves the live false-pass evidence and the training set.
                self._calibration_dirty = True
            self._invalidate_review_waiting()
        safe = _json_safe(result)
        return True, safe if isinstance(safe, dict) else {}, "recorded", 200

    def undo_auto_review_answer(self) -> tuple[bool, dict | None, str, int]:
        """Undo the newest answer to an archived pass: its label and review row are deleted.

        The profile is unreviewed again, so a mis-tapped Like or Dislike can be corrected
        instead of living on in the training set (and, for a random-sample pass, in the
        false-pass numerator).
        Goes through the same store path (and the same answer lock) as an answer, so it sees the
        answer just given even when that is still buffered in the live run's store.

        Returns ``(ok, result, message, http status)``; ``result`` names the undone answer.
        """
        with self._lock:
            if self._training_data_mutating:
                return (
                    False,
                    None,
                    "training data is being updated -- try again in a moment",
                    409,
                )

        def work(store):
            _require_review_methods(store, ("undo_latest_auto_review",))
            return store.undo_latest_auto_review()

        with self._review_answer_lock:
            try:
                result = self._with_review_store(work, write=True)
            except _ReviewUnavailable as exc:
                return False, None, str(exc), 501
            except Exception as exc:  # noqa: BLE001 -- surface storage errors to the local operator
                print(
                    f"Hub: undoing an auto-review answer failed ({type(exc).__name__}: {exc})"
                )
                return False, None, f"{type(exc).__name__}: {exc}", 500
            if result is None:
                return False, None, "there is no review answer to undo", 404
            with self._lock:
                # The undone answer leaves the live false-pass evidence and the training set.
                self._calibration_dirty = True
            self._invalidate_review_waiting()
        safe = _json_safe(result)
        print(
            "Hub: undid the auto-review answer for "
            f"{(safe.get('profile_name') if isinstance(safe, dict) else '') or '<name unavailable>'}"
        )
        return True, safe if isinstance(safe, dict) else {}, "undone", 200

    # --- review calibration (the "expected" side of the false-pass comparison) -------------
    def _freeze_completed_calibration(self) -> None:
        """Copy the finished run's calibration into plain data before the model is dropped."""
        with self._lock:
            model = self._live_model
        if model is None:
            return
        try:
            calibration = model.calibration
        except Exception:  # noqa: BLE001 -- diagnostic only; never block run teardown
            return
        if isinstance(calibration, dict):
            with self._lock:
                self._last_calibration = copy.deepcopy(calibration)

    def _compute_review_calibration(self) -> None:
        """Train a throwaway model exactly as a run would and keep its calibration (bg thread)."""
        calibration = None
        error: str | None = None
        try:
            from ..ranker.model import PreferenceModel

            cfg = cfg_mod.load(self.config_path)
            ranker_cfg = cfg.ranker
            feature_set = require_feature_set(
                getattr(ranker_cfg, "feature_set", DEFAULT_FEATURE_SET)
            )

            def work(store):
                if feature_set == DEFAULT_FEATURE_SET:
                    rows_fn = getattr(store, "load_label_rows", None)
                    rows = rows_fn() if callable(rows_fn) else store.load_labels()
                else:
                    rows = fetch_label_rows(store, feature_set=feature_set)
                stats_fn = getattr(store, "auto_review_stats", None)
                live_review = None
                if callable(stats_fn):
                    try:
                        live_review = stats_fn(app=None)
                    except Exception as exc:  # noqa: BLE001 -- reviews are an add-on
                        print(
                            "Hub: auto-review stats unavailable for the calibration "
                            f"({type(exc).__name__}: {exc}); showing it without live evidence."
                        )
                return rows, live_review

            # Labels lacking whole-photo rows leave the model NOT READY (the reason names the
            # count and the backfill command), exactly as a run would.
            incomplete = None
            try:
                rows, live_review = self._with_review_store(work)
            except FeatureSetIncomplete as exc:
                rows, live_review, incomplete = None, None, exc
            recall_target = getattr(ranker_cfg, "like_recall_target", None)
            model = PreferenceModel(
                min_labels=ranker_cfg.min_labels_to_engage,
                threshold=ranker_cfg.like_threshold,
                min_per_class=ranker_cfg.min_per_class,
                recall_target=recall_target,
                contextual_recall_budget=getattr(
                    ranker_cfg, "contextual_recall_budget", 0.02
                ),
            )
            if incomplete is None:
                model.train(rows, live_review=live_review)
            else:
                mark_feature_set_incomplete(model, incomplete)
            calibration = model.calibration
            if calibration is None:
                error = (
                    "ranker.like_recall_target is not set, so there is no recall "
                    "calibration to show"
                    if recall_target is None
                    else (model.not_ready_reason or "the ranker is not ready")
                )
        except Exception as exc:  # noqa: BLE001 -- the card must degrade, never raise
            calibration, error = None, f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._calibration_cache = (
                copy.deepcopy(calibration) if calibration is not None else None
            )
            self._calibration_error = error
            self._calibration_attempt_at = time.time()

    def _start_review_calibration_refresh(self) -> bool:
        with self._lock:
            if self._calibration_refreshing:
                return True
            self._calibration_refreshing = True
            self._calibration_dirty = False

        def _target():
            try:
                self._compute_review_calibration()
            finally:
                with self._lock:
                    self._calibration_refreshing = False

        try:
            threading.Thread(
                target=_target, name="hub-review-calibration", daemon=True
            ).start()
        except Exception:  # noqa: BLE001 -- a failed spawn must not wedge the single-flight guard
            with self._lock:
                self._calibration_refreshing = False
            return False
        return True

    def _review_calibration(self) -> tuple[dict | None, str | None, bool, str | None]:
        """``(calibration, source, pending, error)`` for the review card.

        Preference order: the running bot's own model (what it is actually using right now),
        then a hub-side recomputation, then the snapshot frozen when the last run in this
        process ended.  The recomputation is a full fit, so it runs in the background and the
        card shows ``pending`` (or the older figures) until it lands.
        """
        with self._lock:
            model = self._live_model if self.is_running() else None
            cached = self._calibration_cache
            error = self._calibration_error
            attempt_at = self._calibration_attempt_at
            dirty = self._calibration_dirty
            refreshing = self._calibration_refreshing
            last = self._last_calibration
        if model is not None:
            try:
                live = model.calibration
            except Exception as exc:  # noqa: BLE001 -- a mid-retrain read must not break the card
                return None, None, False, f"{type(exc).__name__}: {exc}"
            if isinstance(live, dict):
                return live, "live_run", False, None
            if getattr(model, "recall_target", None) is None:
                return (
                    None,
                    None,
                    False,
                    "the running ranker uses a fixed threshold (ranker.like_recall_target is not set)",
                )
            return (
                None,
                None,
                False,
                getattr(model, "not_ready_reason", None)
                or ("the running ranker has no calibration yet"),
            )
        fresh = (
            attempt_at > 0.0
            and not dirty
            and (time.time() - attempt_at) < _REVIEW_CALIBRATION_MAX_AGE_S
        )
        if not fresh and not refreshing:
            refreshing = self._start_review_calibration_refresh()
        if cached is not None:
            return cached, "computed", False, None
        if last is not None:
            return last, "last_run", False, None
        return None, None, bool(refreshing), error

    @staticmethod
    def _review_summary(store_stats: dict, calibration: dict | None) -> dict:
        """The measured-vs-expected false-pass comparison, computed once server-side."""
        from ..ranker.threshold import MIN_RANDOM_REVIEWS, wilson_lower

        def _count(key: str) -> int:
            value = store_stats.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return 0
            return value

        reviewed = _count("random_reviewed")
        would_like = min(_count("random_would_like"), reviewed)
        measured = would_like / reviewed if reviewed else None
        lower = wilson_lower(would_like, reviewed) if reviewed else None
        expected = (
            calibration.get("expected_false_pass_rate")
            if isinstance(calibration, dict)
            else None
        )
        if isinstance(expected, bool) or not isinstance(expected, (int, float)):
            expected = None
        if reviewed < MIN_RANDOM_REVIEWS:
            verdict = "insufficient"
        elif expected is None:
            verdict = "unknown"
        elif lower is not None and lower > expected:
            verdict = "above"
        else:
            verdict = "within"
        return {
            "random_reviewed": reviewed,
            "random_would_like": would_like,
            "measured_false_pass_rate": measured,
            "observed_false_pass_lower": lower,
            "expected_false_pass_rate": expected,
            "min_random_reviews": MIN_RANDOM_REVIEWS,
            "verdict": verdict,
        }

    @staticmethod
    def _review_likes_summary(store_stats: dict) -> dict | None:
        """The owner's agreement with AUTO's likes (display only), or ``None`` without one.

        Read from the store's separate ``likes`` section, which nothing in the false-pass
        comparison above or in ``threshold.live_adjustment`` reads.
        """
        likes = store_stats.get("likes")
        if not isinstance(likes, dict):
            return None

        def _count(key: str) -> int:
            value = likes.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return 0
            return value

        reviewed = _count("reviewed")
        return {
            "archived": _count("archived"),
            "reviewed": reviewed,
            "agreed": min(_count("agreed"), reviewed),
        }

    def auto_review_stats_snapshot(self) -> tuple[int, dict]:
        """Store counts plus the model calibration, for the review card's header."""
        result: dict = {
            "available": False,
            "reason": None,
            "store": None,
            "calibration": None,
            "calibration_source": None,
            "calibration_pending": False,
            "calibration_error": None,
            "summary": None,
            "likes": None,
        }

        def work(store):
            _require_review_methods(store, ("auto_review_stats",))
            return store.auto_review_stats(app=None)

        try:
            store_stats = self._with_review_store(work)
        except _ReviewUnavailable as exc:
            result["reason"] = str(exc)
            return 200, result
        except Exception as exc:  # noqa: BLE001 -- surface storage errors to the local operator
            result["reason"] = f"{type(exc).__name__}: {exc}"
            return 502, result
        if not isinstance(store_stats, dict):
            result["reason"] = "the store returned malformed auto-review stats"
            return 502, result
        calibration, source, pending, error = self._review_calibration()
        safe_calibration = _json_safe(calibration) if calibration is not None else None
        result.update(
            available=True,
            store=_json_safe(store_stats),
            calibration=safe_calibration,
            calibration_source=source,
            calibration_pending=pending,
            calibration_error=error,
            summary=_json_safe(self._review_summary(store_stats, calibration)),
            likes=self._review_likes_summary(store_stats),
        )
        return 200, result

    def _bind_training_worker(self, worker) -> None:
        # Called by supervisor before Thread.start(), so no hub request can observe a half-bound
        # worker.  The worker subsequently registers itself at run entry as a harmless idempotent
        # backstop for direct/test construction.
        # Do NOT recompute training_action_supported here. Worker.__init__ already derived it
        # from the same mode/app/driver, and a second copy of that predicate running LAST means a
        # future change to the worker's version is silently clobbered back -- every checkpoint
        # would then be refused by publish_checkpoint's authorization guard with nothing in this
        # file to explain it.
        worker.training_action_bridge = self._training_actions
        self._training_actions.register(worker)
        # The model every Worker of this run shares.  Duck-typed and best-effort: a decider
        # without one (a fake, a third-party Decider) simply leaves the review card without a
        # live calibration.
        model = getattr(getattr(worker, "decider", None), "model", None)
        if model is not None:
            with self._lock:
                self._live_model = model

    @staticmethod
    def _service_snapshot(service, method_name: str) -> list[dict]:
        """Return a detached diagnostic ring from ``service``, or [] on any bad hand-off.

        OpenerService itself returns a list while holding its own lock.  ``deepcopy`` here is
        deliberate: its entries contain nested telemetry lists, and the completed-run cache
        must not share any mutable object with an object about to be torn down.
        """
        snapshot_fn = getattr(service, method_name, None)
        if snapshot_fn is None:
            return []
        try:
            rows = snapshot_fn()
            if not isinstance(rows, list) or not all(
                isinstance(row, dict) for row in rows
            ):
                return []
            return copy.deepcopy(rows)
        except Exception:  # noqa: BLE001 — diagnostics must never break the hub
            return []

    def _freeze_completed_opener_telemetry(self) -> None:
        """Detach final opener/rejection snapshots without retaining the live service."""
        with self._lock:
            service = self._opener_service
        openers = self._service_snapshot(service, "recent_openers_snapshot")
        rejections = self._service_snapshot(service, "recent_rejections_snapshot")
        with self._lock:
            # start() cannot replace a still-running target thread, but retain this identity
            # guard so a future lifecycle change cannot publish one run's rows into another.
            if self._opener_service is service:
                self._completed_openers = openers
                self._completed_opener_rejections = rejections

    def recent_openers(self) -> list[dict]:
        """Committed opener records from the active or most recently completed run.

        A completed run uses the detached snapshot made during HubState's shutdown hand-off;
        this method never reads a torn-down OpenerService.
        """
        with self._lock:
            svc = self._opener_service
            completed = copy.deepcopy(self._completed_openers)
        return (
            self._service_snapshot(svc, "recent_openers_snapshot")
            if svc is not None
            else completed
        )

    def recent_opener_rejections(self) -> list[dict]:
        """Committed rejected-opener records from the active or last completed run.

        Same detached-snapshot contract as recent_openers(), using OpenerService's rejection
        ring rather than retaining a shut-down service object.
        """
        with self._lock:
            svc = self._opener_service
            completed = copy.deepcopy(self._completed_opener_rejections)
        return (
            self._service_snapshot(svc, "recent_rejections_snapshot")
            if svc is not None
            else completed
        )

    def config_defaults(self) -> dict:
        try:
            cfg = cfg_mod.load(self.config_path)

            def _mode_status(app: str, mode: str) -> tuple[bool, str | None]:
                try:
                    # Use the shared startup gate. A fresh process has no installed Hinge
                    # still-photo licence until config validation reads its accepted
                    # assumption/bound; probing the registry first would reject a valid AUTO
                    # config merely because the picker examined Auto before a validating path.
                    # load_effective_config validates, then invokes the registry before any
                    # construction, so it is the authoritative readiness answer.
                    effective = supervisor.load_effective_config(
                        self.config_path, mode=mode, enabled_apps=[app]
                    )
                except Exception as exc:  # noqa: BLE001 — exact start reason is picker help
                    return False, str(exc)
                # ``load_effective_config`` intentionally retains per-app precedence for
                # unattended CLI runs.  The Hub, however, refuses an explicit mode click that
                # would resolve to another mode in start(). Reflect that exact refusal in the
                # picker instead of advertising an option that Start will immediately reject.
                actual = (
                    (getattr(effective, "apps", {}) or {}).get(app, {}) or {}
                ).get("mode", getattr(effective, "mode", None))
                if actual != mode:
                    return False, (
                        f"Hub requested {mode!r}, but per-app mode override resolves to "
                        f"{app}={actual}; choose that mode explicitly or remove the override"
                    )
                return True, None

            def _platform_payload(p: "platforms.Platform") -> dict:
                status = {
                    mode: _mode_status(p.app, mode) for mode in ("training", "auto")
                }
                return {
                    "app": p.app,
                    "label": p.label,
                    "available": p.available,
                    "reason": p.reason,
                    # Keep the established boolean contract and add diagnostic detail beside it.
                    "modes": {
                        mode: available for mode, (available, _) in status.items()
                    },
                    "mode_reasons": {
                        mode: reason for mode, (_, reason) in status.items()
                    },
                }

            # The hub is an operational control surface, not a calibration console. Do not
            # send unavailable apps to its picker: a dimmed option still looks selectable and
            # made Bumble appear usable before its safety calibration was complete.
            kinds = [
                {
                    "kind": kind,
                    "label": platforms.KIND_LABELS[kind],
                    "platforms": [
                        _platform_payload(p)
                        for p in platforms.for_kind(kind)
                        if p.available
                    ],
                }
                for kind in platforms.kinds()
                if any(p.available for p in platforms.for_kind(kind))
            ]
            pending = [
                {"app": p.app, "label": p.label, "reason": p.reason}
                for p in platforms.all_platforms()
                if not p.available
            ]
            selected = _selected_platform(cfg.enabled_apps)
            return {
                "mode": cfg.mode,
                "backend": cfg.storage.backend,
                "enabled_apps": cfg.enabled_apps,
                "all_apps": [p.app for p in platforms.all_platforms()],
                "kinds": kinds,
                "pending_platforms": pending,
                "selected": {"kind": selected.kind, "app": selected.app},
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def eval_snapshot(self, every: int = 5, max_age: float = 300.0) -> dict:
        """Leakage-free, identity-grouped CV of the ranker, for the GUI's model-quality
        card. Recomputed only when `every` new labels have been recorded since the last
        run (read from the live in-memory label counter — no extra store/BigQuery read
        just to check the gate), with `max_age` seconds as an idle fallback. Once a
        cached card exists, threshold refreshes run in the background so the UI can show
        every/every progress until the new metrics land. Never raises."""
        every = max(1, every)
        with self._lock:
            cached, at, base, status = (
                self._eval,
                self._eval_at,
                self._eval_labels,
                self._status,
            )
        live = getattr(status, "labels", None) if status is not None else None

        fresh_enough = cached is not None and (time.time() - at) < max_age
        if live is not None and base is not None:
            enough_new = (live - base) >= every
        else:
            enough_new = cached is None  # no live counter -> lean on max_age
        if cached is not None:
            if enough_new:
                self._start_eval_refresh(every)
                return self._attach_refresh(
                    cached,
                    every,
                    live,
                    base,
                    status,
                    training=self._live_training_mix(status),
                )
            if fresh_enough:
                return self._attach_refresh(
                    cached,
                    every,
                    live,
                    base,
                    status,
                    training=self._live_training_mix(status),
                )

        return self._compute_eval_snapshot_singleflight(every)

    def _compute_eval_snapshot_singleflight(self, every: int) -> dict:
        """Cold path (no cached eval yet — e.g. hub just started): the page polls /api/eval
        every 5s and ThreadingHTTPServer gives every request its own thread, so without a
        guard here N overlapping cold polls each run a full grouped CV (and, on BigQuery,
        each opens its own read-only store). Make concurrent cold callers share ONE
        computation: the first caller computes, everyone else just waits on it and reads
        the result it left behind. Same never-raises contract as eval_snapshot()."""
        with self._lock:
            event = self._eval_cold_event
            if event is None:
                event = self._eval_cold_event = threading.Event()
                is_computer = True
            else:
                is_computer = False
        if not is_computer:
            finished = event.wait(timeout=_EVAL_COLD_WAIT_S)
            with self._lock:
                cached, computed_at, status = (
                    self._eval,
                    self._eval_labels,
                    self._status,
                )
            live = getattr(status, "labels", None) if status is not None else None
            if not finished or cached is None:
                # Either the wait timed out, or the computing thread died via a BaseException
                # (e.g. SystemExit) that _compute_eval_snapshot's `except Exception` doesn't
                # catch, so it never populated self._eval before its `finally: event.set()`
                # ran. Degrade to the documented error dict instead of handing None to
                # _attach_refresh, which would break eval_snapshot's "never raises" contract.
                cached = {
                    "status": "error",
                    "message": "eval computation did not complete",
                    "labels": None,
                    "identities": None,
                    "folds": 0,
                    "roc_auc": None,
                    "pr_auc": None,
                    "brier": None,
                    "base_rate": 0.0,
                    "like_threshold": None,
                    "accepted_recall": None,
                    "false_dislike_rate": None,
                    "confusion": None,
                }
            return self._attach_refresh(
                cached,
                every,
                live,
                computed_at,
                status,
                training=self._live_training_mix(status),
            )
        try:
            return self._compute_eval_snapshot(every)
        finally:
            event.set()  # wake waiters first...
            with self._lock:
                self._eval_cold_event = (
                    None  # ...then clear the slot for the next cold start
                )

    def _start_eval_refresh(self, every: int) -> None:
        with self._lock:
            if self._eval_refreshing:
                return
            self._eval_refreshing = True

        def _target():
            try:
                self._compute_eval_snapshot(every)
            finally:
                with self._lock:
                    self._eval_refreshing = False

        try:
            threading.Thread(
                target=_target, name="hub-eval-refresh", daemon=True
            ).start()
        except Exception:  # noqa: BLE001 — preserve eval_snapshot's never-raise contract
            # A failed Thread.start() runs none of _target's finally block.  Clear this guard
            # ourselves so a transient resource failure does not make every later eval request
            # believe a refresh is still running forever.
            with self._lock:
                self._eval_refreshing = False

    def _compute_eval_snapshot(self, every: int) -> dict:
        with self._lock:
            base, status, live_store = self._eval_labels, self._status, self._live_store
        live = getattr(status, "labels", None) if status is not None else None
        running = self.is_running()
        # Set where a FeatureSetIncomplete is caught, while its store is still open to read the
        # coverage line from; the except clause below turns it into the card's message.
        incomplete_message: str | None = None
        try:
            from ..ranker import make_store
            from ..ranker.evaluate import evaluate

            cfg = cfg_mod.load(self.config_path)
            feature_set = require_feature_set(
                getattr(
                    getattr(cfg, "ranker", None), "feature_set", DEFAULT_FEATURE_SET
                )
            )
            # FRESHNESS: when a run is live, read its in-memory store so the card sees
            # EVERY swipe (incl. the unflushed buffer) — not a re-query of BigQuery, which
            # only returns committed rows and lags behind the streaming buffer. When idle,
            # a fresh read-only store is correct (shutdown already flushed everything).
            samples = None
            if running and live_store is not None:
                try:
                    samples = self._eval_label_pairs(live_store, feature_set)
                except FeatureSetIncomplete as exc:
                    incomplete_message = self._feature_set_incomplete_message(
                        exc, live_store
                    )
                    raise
                except Exception:  # noqa: BLE001 — run ended + store closed mid-read; fall back
                    samples = None
            if samples is None:
                store = make_store(
                    cfg, ensure=False
                )  # read-only; don't run DDL just to eval
                try:
                    samples = self._eval_label_pairs(store, feature_set)
                except FeatureSetIncomplete as exc:
                    incomplete_message = self._feature_set_incomplete_message(
                        exc, store
                    )
                    raise
                finally:
                    store.close()
                live_store = None  # don't reuse a dead store for the chart
            # This evaluates the ranker's configured base threshold. The stored dataset has
            # labels and embeddings, but not the live profile/session state required to replay
            # AutoSessionPolicy's additional conservative demotions, so this result is useful
            # baseline evidence rather than an AUTO release decision.
            ranker_cfg = getattr(cfg, "ranker", None)
            like_threshold, threshold_source, recall_target = self._eval_threshold(
                ranker_cfg, samples
            )
            if recall_target is None or threshold_source == "uncalibrated":
                result = evaluate(samples, like_threshold=like_threshold)
            else:
                # Under a recall target the cutoff is picked from out-of-fold scores of these
                # same labels, so scoring them at that cutoff would report "recall >= target" for
                # ANY ranker.  evaluate() re-runs the calibration inside every fold instead and
                # grades the held-out people (see its docstring).
                result = evaluate(
                    samples, like_threshold=like_threshold, recall_target=recall_target
                )
            result = {
                **result,
                "threshold_source": threshold_source,
                "recall_target": recall_target,
            }
            if threshold_source == "uncalibrated" and result.get("status") == "ok":
                # A recall-target ranker's configured like_threshold is only a placeholder until
                # it calibrates.  Recall and false-dislike numbers measured AT the placeholder
                # describe a threshold the bot never uses, so the card refuses to show them.
                result = {
                    **result,
                    "status": "uncalibrated",
                    "message": (
                        "The ranker has no calibrated threshold yet (ranker.like_recall_target "
                        f"{recall_target}), so recall at the placeholder like_threshold would be "
                        "misleading."
                    ),
                    "accepted_recall": None,
                    "false_dislike_rate": None,
                    "confusion": None,
                }
            training = self._training_mix(samples)
            # Gate baseline must use the SAME counter the gate compares against: the live
            # swipe counter (status.labels), not len(samples). Using len(samples) would lag
            # status.labels by the worker's unflushed buffer and re-fire the gate every poll.
            computed_at = live if live is not None else len(samples)
        except Exception as exc:  # noqa: BLE001
            # Labels lacking whole-photo rows are the ranker's NOT READY state, not a fault:
            # the card says how many and which command fills them.
            incomplete = isinstance(exc, FeatureSetIncomplete)
            result = {
                "status": "incomplete_features" if incomplete else "error",
                "message": (
                    (incomplete_message or exc.reason)
                    if incomplete
                    else f"{type(exc).__name__}: {exc}"
                ),
                "labels": None,
                "identities": None,
                "folds": 0,
                "roc_auc": None,
                "pr_auc": None,
                "brier": None,
                "base_rate": 0.0,
                "like_threshold": None,
                "accepted_recall": None,
                "false_dislike_rate": None,
                "confusion": None,
            }
            computed_at = live if live is not None else base
            training = None
        with self._lock:
            self._eval, self._eval_at, self._eval_labels = (
                result,
                time.time(),
                computed_at,
            )
            status = self._status
        live = getattr(status, "labels", None) if status is not None else None
        return self._attach_refresh(
            result, every, live, computed_at, status, training=training
        )

    def _eval_threshold(self, ranker_cfg, samples) -> tuple[float, str, float | None]:
        """``(threshold, source, recall_target)`` the model-quality card should evaluate at.

        Without ``ranker.like_recall_target`` that is the configured ``like_threshold`` (the
        legacy behavior, unchanged).  WITH a recall target the configured threshold is only a
        placeholder, and the bot deploys the model's calibrated cutoff instead -- evaluating at
        the placeholder would report an accepted-recall / false-dislike pair for a threshold the
        bot never uses.  Preference order for the calibrated cutoff:

        * ``"calibrated"``: the running bot's own threshold, else the hub's cached or last-run
          calibration (read passively: this never starts the background model fit).  A cached or
          last-run calibration is only used when it was made for the CURRENT target: after the
          owner edits ``like_recall_target`` with the hub still open it describes a different
          bot, and would otherwise be shown as the cutoff "calibrated to keep" the new target.
          The running bot's own calibration is authoritative and brings its own target with it.
        * ``"oof_estimate"``: nothing calibrated is on hand (idle hub, first card poll), so
          derive it from this card's own out-of-fold scores on the same manual labels -- the
          identical procedure the model uses, minus the review labels and live correction.
        * ``"uncalibrated"``: calibration is impossible (too little data); the caller must not
          present recall figures measured at the placeholder.
        """
        configured = getattr(ranker_cfg, "like_threshold", 0.5)
        target = getattr(ranker_cfg, "like_recall_target", None)
        if target is None:
            return configured, "configured", None
        deployed = self._passive_calibrated_threshold(target)
        if deployed is not None:
            cutoff, calibrated_target = deployed
            return cutoff, "calibrated", calibrated_target
        derived = self._oof_recall_threshold(samples, target)
        if derived is not None:
            return derived, "oof_estimate", target
        return configured, "uncalibrated", target

    def _passive_calibrated_threshold(
        self, target: float
    ) -> tuple[float, float] | None:
        """``(cutoff, the target it was calibrated for)`` already on hand, else ``None``.

        The live model's own calibration wins; otherwise the cached or last run's, but only if
        it was calibrated for ``target`` (the config's CURRENT ``like_recall_target``): a
        calibration without a ``recall_target``, or with a different one, is somebody else's
        cutoff and is ignored, so the caller falls through to its own estimate.
        """
        with self._lock:
            model = self._live_model if self.is_running() else None
            cached, last = self._calibration_cache, self._last_calibration

        def number(value) -> float | None:
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                return float(value)
            return None

        def cutoff(calibration, *, expect: float | None) -> tuple[float, float] | None:
            if not isinstance(calibration, dict):
                return None
            threshold = number(calibration.get("threshold"))
            if threshold is None or not 0.0 < threshold < 1.0:
                return None
            calibrated_for = number(calibration.get("recall_target"))
            if expect is not None and (
                calibrated_for is None or abs(calibrated_for - expect) > 1e-9
            ):
                return None
            return threshold, (calibrated_for if calibrated_for is not None else target)

        if model is not None:
            # The running bot's own model is authoritative -- it IS what the bot deploys, so
            # its cutoff is shown with the target it was built for.  If it has no calibration
            # (yet), an older cached cutoff is not what it deploys, so none is substituted.
            try:
                return cutoff(model.calibration, expect=None)
            except Exception:  # noqa: BLE001 -- a mid-retrain read must not break the card
                return None
        found = cutoff(cached, expect=target)
        return found if found is not None else cutoff(last, expect=target)

    @staticmethod
    def _oof_recall_threshold(samples, target: float) -> float | None:
        """The cutoff holding ``target`` recall on these labels' out-of-fold scores, or None."""
        try:
            import numpy as np

            from ..ranker.threshold import (
                min_positives_for_target,
                oof_scores,
                recall_threshold,
            )

            X = np.asarray([row[1] for row in samples], dtype=float)
            y = np.asarray([1 if row[0] else 0 for row in samples], dtype=int)
            if y.size < 10 or not y.any() or y.all():
                return None
            # The model refuses to calibrate on fewer manual likes than the target can be
            # measured from (it is NOT ready), so an estimate here would describe a cutoff the
            # bot never uses.
            if int(y.sum()) < min_positives_for_target(target):
                return None
            oof = oof_scores(X, y)
            if oof is None:
                return None
            return float(recall_threshold(oof[y == 1].tolist(), target))
        except Exception as exc:  # noqa: BLE001 -- the card degrades to "uncalibrated"
            print(
                "Hub: could not derive a recall-target threshold for the model-quality card "
                f"({type(exc).__name__}: {exc})."
            )
            return None

    @staticmethod
    def _eval_label_pairs(store, feature_set: str = DEFAULT_FEATURE_SET) -> list:
        """The labels the model-quality card may judge the ranker on: MANUAL ones only.

        An answer to an archived AUTO pass (source ``auto_review...``) is a good training label
        but a biased evaluation one: every reviewed profile is something the model had already
        PASSED, so scoring the model on them would count its own past misses as the whole
        population (and its correct passes as extra easy negatives).  Stores that predate the
        review archive have no ``load_label_rows``; their ``load_labels`` already is manual-only.

        Rows are built for ``feature_set`` (``fetch_label_rows``), so the card grades the
        feature set the bot trains on; a ``FeatureSetIncomplete`` propagates.
        """
        from ..ranker.threshold import is_calibration_source

        if feature_set == DEFAULT_FEATURE_SET:
            rows_fn = getattr(store, "load_label_rows", None)
            if not callable(rows_fn):
                return store.load_labels()
            rows = rows_fn()
        else:
            rows = fetch_label_rows(store, feature_set=feature_set)
        return [
            (row[0], row[1])
            for row in rows
            if len(row) < 3 or is_calibration_source(row[2])
        ]

    @staticmethod
    def _feature_set_incomplete_message(exc: BaseException, store) -> str:
        """The not-ready reason plus the store's whole-photo coverage line, when it has one."""
        reason = exc.reason
        coverage_fn = getattr(store, "photo_feature_coverage", None)
        if not callable(coverage_fn):
            return reason
        try:
            from ..vision.photo_cards import WHOLE_PHOTO_FEATURE_VERSION

            coverage = format_photo_coverage(coverage_fn(WHOLE_PHOTO_FEATURE_VERSION))
        except Exception as cov_exc:  # noqa: BLE001 -- the reason alone still says what to do
            coverage = (
                "whole-photo coverage unavailable "
                f"({type(cov_exc).__name__}: {cov_exc})"
            )
        return f"The ranker is not ready: {reason}. Coverage: {coverage}."

    @staticmethod
    def _training_mix(samples) -> dict:
        """Return current label totals for the hub without running another CV."""
        rows = list(samples or [])
        # Stores always return (liked, embedding) pairs. Keep the display helper tolerant of
        # a malformed/testing row nevertheless: count the record but do not guess its class.
        likes = sum(
            1 for row in rows if isinstance(row, tuple) and row and bool(row[0])
        )
        return {"labels": len(rows), "likes": likes, "passes": len(rows) - likes}

    def _live_training_mix(self, status) -> dict | None:
        """Read the running store's cached labels, including its unflushed buffer.

        This keeps the count that confirms a just-made swipe current while the grouped CV
        refresh remains deliberately throttled. BigQueryStore serves this from its in-memory
        cache after startup, so polling does not issue a BigQuery query every five seconds.
        """
        if (
            not bool(getattr(status, "running", False))
            or getattr(status, "mode", None) != "training"
        ):
            return None
        with self._lock:
            live_store = self._live_store
        if live_store is None:
            return None
        try:
            return self._training_mix(self._eval_label_pairs(live_store))
        except Exception:  # noqa: BLE001 - the run may close the store between poll and read
            return None

    @staticmethod
    def _attach_refresh(
        result: dict, every: int, live, base, status, training: dict | None = None
    ) -> dict:
        """Return a shallow copy of the cached/fresh eval with a `refresh` countdown
        attached (the cached dict itself stays refresh-free so the count stays live)."""
        every = max(1, every)
        running = (
            bool(getattr(status, "running", False)) if status is not None else False
        )
        mode = getattr(status, "mode", None) if status is not None else None
        if live is None or base is None or not running:
            refresh = {
                "every": every,
                "remaining": None,
                "since": None,
                "live": False,
                "mode": mode,
            }
        else:
            since = max(0, int(live) - int(base))
            refresh = {
                "every": every,
                "since": since,
                "remaining": max(0, every - since) or every,
                "live": True,
                "mode": mode,
            }
        attached = {**result, "refresh": refresh}
        if training is not None:
            attached["training"] = training
        return attached
