"""Cross-layer readiness for Hinge's numbered still-photo targeting policy.

Keep this module dependency-free so configuration, drivers, and offline tools can all consume the
same fail-closed fact.  The current vision signals can reject visible/high-motion videos, but none
positively distinguishes a still photo from a paused or low-motion video.

Readiness is deliberately NOT a hand-editable module constant (it used to be
``HINGE_POSITIVE_STILL_PHOTO_DISCRIMINATOR_READY = False``).  A boolean that a single-line diff
can flip is exactly the wrong shape for a gate whose whole justification is a measured held-out
false-accept bound: nothing in the code could tell an earned `True` from a wished-for one.  It is
now installed process-locally by ``config.validate()`` from a sha256-bound measurement artifact
(ops/STILL-PHOTO-DISCRIMINATOR.md section 5), so turning numbering on requires producing, binding
and passing that evidence.  No installation means the original blocker text, byte for byte.

There are now THREE ways a readiness licence can be earned, and they are deliberately not
interchangeable: the owner-labelled measured bound, the owner-accepted circular AI-labelled
measured bound, and (owner decision 2026-08-21) an accepted centered-autoplay ASSUMPTION that
was never measured at all.  All three make numbering behave identically, which is exactly why
``still_photo_licence_provenance()`` exists: the difference can only survive to a human in
words, so consumers print it.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import threading


HINGE_PHOTO_SELECTION_POLICY_ID = "hinge_photos_only_v2"
# v1 licensed numbering on rejectors alone.  v2 is the C1-C4 selection contract (existing
# rejectors + dwell byte-exactness + per-frame complete mute screen + a verified bound artifact),
# so a mapping calibrated under v1 measured a different thing and cannot be reinstalled.  It gets
# its own refusal message everywhere: the operator's fix is a recalibration campaign, not a typo
# correction, and a generic "unsupported id" reads like the latter.
HINGE_SUPERSEDED_PHOTO_SELECTION_POLICY_IDS = frozenset({"hinge_photos_only_v1"})
HINGE_TARGETING_UNAVAILABLE_REASON = (
    "positive still-photo discriminator unavailable: photographic pixels, low signature drift, "
    "and an absent auto-hiding mute control do not exclude a paused/static video"
)

# --- What makes a bound artifact shippable (ops/STILL-PHOTO-DISCRIMINATOR.md section 4) ------
# Single source of truth: config validation and the offline measurement tool both import these,
# so a campaign can never be measured against looser numbers than the ones that gate it.
#
# 60 owner-labelled video cards with zero accepts is the Rule of Three: the 95% upper bound on
# the false-accept rate is 3/60 = 5%.  (300 cards would buy 1%; 60 is the smallest corpus whose
# bound is worth quoting.)  Counts are per CARD, never per frame pair, because frames from one
# card are not independent trials.
STILL_PHOTO_BOUND_MIN_VIDEO_CARDS = 60
STILL_PHOTO_BOUND_REQUIRED_VIDEO_ACCEPTS = 0
# The still-photo side bounds usefulness rather than safety: a discriminator that refuses most
# real photos is safe and useless, and would quietly turn numbering back off in production.
STILL_PHOTO_BOUND_MIN_PHOTO_CARDS = 60
STILL_PHOTO_BOUND_MAX_PHOTO_FALSE_REFUSAL_FRAC = 0.05
# The label channel that breaks the ground-truth circularity: every "proven video" already on
# disk exists because the mute matcher fired, so a bound measured against the matcher's own
# output structurally excludes the videos whose control never rendered, which is precisely the
# false-accept population.  Only owner tap-to-play labels count.
STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL = "owner_tap_to_play_v1"
# The second, explicitly opted-in label channel (owner decision 2026-08-21).  Its video labels
# come from the same mute-glyph matcher whose blind spot the bound is supposed to quantify, so a
# bound measured on it CANNOT see the false-accept population it structurally excludes: videos
# whose control never rendered.  That is the circularity from section 2 of the design doc, and it
# is not fixed here, it is recorded.  The channel exists so the owner can trade a weaker licence
# for a campaign that does not need a human at the phone for every card; the owner-labelled
# channel above stays the preferred one and is unchanged.
STILL_PHOTO_BOUND_CIRCULAR_CHANNEL = "ai_mute_glyph_circular_v1"
# Installing the circular channel requires this phrase byte for byte, in the summary AND in the
# artifact.  A boolean flag would be flippable by accident and unreadable in a diff; a sentence
# that names what is being accepted cannot be typed by mistake and tells a config reader exactly
# what the risk was.
STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE = "I_ACCEPT_CIRCULAR_AI_LABELED_STILL_PHOTO_BOUND"
# The dwell window W must exceed the worst byte-exact run observed on held-out videos by this
# factor.  Consumed by the measurement tool when it proposes W; recorded here so the window and
# the bound that licenses it can never drift apart.
STILL_PHOTO_DWELL_WINDOW_SAFETY_FACTOR = 3.0
# --- The THIRD readiness channel: an accepted assumption, not a measurement -----------
# Owner decision 2026-08-21, made after the risk was raised twice and reaffirmed: measuring the
# held-out video false-accept rate costs ~420 profiles and ~420 real passes, which the owner
# judges not worth paying.  The owner instead DIRECTS the assumption that a video moved to the
# centre of the screen is playing, and therefore that the deterministic motion test can see it.
# That is a licence granted by a person, not evidence produced by a campaign, so it gets its own
# channel rather than being smuggled in as a bound with invented numbers: nothing downstream may
# ever be able to read an assumption as a measurement.  Same shape as the circular acceptance
# above -- a sentence that names what is being accepted, because a boolean would be flippable by
# accident and unreadable in a diff.
STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION = "I_ACCEPT_UNMEASURED_CENTERED_AUTOPLAY_ASSUMPTION"
# The two channel tags a licence can carry.  Consumers branch on these rather than on the type of
# the underlying record, so "which channel licensed this run" stays one explicit string.
STILL_PHOTO_LICENCE_MEASURED = "measured"
STILL_PHOTO_LICENCE_ASSUMPTION = "assumption"
# The honesty strings.  They live here, next to the acceptance phrase, so the operator-facing
# wording and the licence that produces it can never drift apart across worker/hub/tooling.
STILL_PHOTO_ASSUMPTION_PROVENANCE = (
    "UNMEASURED: centered-autoplay assumption accepted by the owner; no video false-accept "
    "rate has been measured")
# What the operator sees at the top of every run that is licensed this way.  Deliberately states
# the enabled capability AND the missing measurement in one sentence: a reader who skims the
# first half must not come away thinking a bound exists.
STILL_PHOTO_ASSUMPTION_OPERATOR_NOTICE = (
    "targeted suggestions enabled under an UNMEASURED assumption (centered autoplay); "
    "no video false-accept rate has been measured")

_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class StillPhotoBoundSummary:
    """The measured held-out bound that licenses numbering, as installed by config validation.

    Frozen and compared by value: re-installing the identical summary is a no-op, while a second
    DIFFERENT summary is refused rather than silently swapped in (see
    ``install_verified_still_photo_bound``).
    """

    ground_truth_channel: str
    human_ground_truth: bool
    video_cards: int
    video_accepts: int
    photo_cards: int
    photo_false_refusals: int
    max_video_exact_run_s: float
    artifact_sha256: str
    device: str
    hinge_version_name: str
    # Set only on STILL_PHOTO_BOUND_CIRCULAR_CHANNEL, where it carries the owner's acceptance
    # phrase.  Defaulted so every existing owner-labelled construction stays valid unchanged, and
    # kept as a field (rather than inferred from the channel) so the accepted risk travels with
    # the installed summary instead of having to be re-derived by every reader.
    accepted_circular_risk: str | None = None


@dataclass(frozen=True)
class StillPhotoAssumptionAcceptance:
    """The owner's accepted centered-autoplay assumption, as installed by config validation.

    This records a DECISION, not a measurement, and carries no counts on purpose: there is no
    corpus, no accept rate and no artifact digest behind it, so there is nothing here that a
    later reader could mistake for evidence.  ``rationale`` is mandatory and free text because
    the one thing worth preserving about an assumption is why a person accepted it.
    """

    acceptance: str
    accepted_at: str
    device: str
    hinge_version_name: str
    rationale: str


@dataclass(frozen=True)
class StillPhotoLicence:
    """Which channel licensed numbering readiness right now, plus the record that did it.

    Readiness has exactly ONE slot, and this tag is how a consumer tells the two channels apart
    without type-sniffing the record.  Keeping the record attached (rather than only the tag)
    means a consumer that wants the counts, the device or the owner's rationale does not have to
    reach back into a second accessor and hope it refers to the same installation.
    """

    channel: str
    record: StillPhotoBoundSummary | StillPhotoAssumptionAcceptance

    @property
    def measured(self) -> bool:
        return self.channel == STILL_PHOTO_LICENCE_MEASURED

    @property
    def assumed(self) -> bool:
        return self.channel == STILL_PHOTO_LICENCE_ASSUMPTION


# ONE slot for all channels.  A measured bound and an accepted assumption make incompatible
# claims about the same gate, so they must not be able to coexist: whichever was installed
# second would otherwise decide what the provenance string says, and that would depend on
# validation order rather than on what the owner configured.
_installed_still_photo_licence: StillPhotoLicence | None = None

# Config validation deliberately clears and then installs this process default.  Those are a
# single state transition, and a supervisor must capture the resulting immutable record before
# another validation can begin its own transition.  The lock is intentionally only for that
# small config/readiness transaction; live workers use ContextVar snapshots and never take it.
_still_photo_licence_validation_lock = threading.RLock()


@contextmanager
def still_photo_licence_validation_transaction():
    """Serialize config validation with its immediately-following run snapshot."""
    with _still_photo_licence_validation_lock:
        yield


def _serialise_still_photo_licence_mutation(function):
    """Keep direct installer/reset callers out of a config snapshot transaction too."""
    @wraps(function)
    def locked(*args, **kwargs):
        with _still_photo_licence_validation_lock:
            return function(*args, **kwargs)
    return locked

# A configuration check is allowed to replace the process-default licence: that is how a new
# run becomes licensed. It must not, however, rewrite the answer for a worker which already
# opened a live Hinge session. The Hub evaluates config again when a page is opened (and an
# operator can edit that file while a run is live), so consulting only the mutable process slot
# made a harmless /api/config request capable of stopping the active worker between validation's
# clear and reinstall steps. Workers enter this context with the immutable record present at
# their own startup; ordinary callers continue to see the current process default.
_NO_RUN_LICENCE = object()
_run_still_photo_licence: ContextVar[object] = ContextVar(
    "operation_love_run_still_photo_licence", default=_NO_RUN_LICENCE)


def _effective_still_photo_licence() -> StillPhotoLicence | None:
    contextual = _run_still_photo_licence.get()
    if contextual is _NO_RUN_LICENCE:
        return _installed_still_photo_licence
    # Only this module installs the context value. Keep the defensive assertion local so a
    # malformed future caller cannot turn an arbitrary truthy object into readiness.
    return contextual if isinstance(contextual, StillPhotoLicence) else None


def _installed_process_default_still_photo_licence() -> StillPhotoLicence | None:
    """Return the mutable process-default slot, never a worker's ContextVar snapshot.

    This is deliberately private: validation's locked clear/install/snapshot transaction is
    the only consumer. Runtime readiness must continue to use `_effective_still_photo_licence`.
    """
    return _installed_still_photo_licence


@contextmanager
def use_run_still_photo_licence(licence: StillPhotoLicence | None):
    """Freeze one worker's still-photo readiness for its complete live session.

    ``None`` is an intentional, fail-closed snapshot: a worker created without a licence may
    not become licensed merely because another thread later validates a different config.
    Context variables are thread-local, so concurrent Hub/config work continues to use the
    mutable default without affecting the active worker.
    """
    if licence is not None and not isinstance(licence, StillPhotoLicence):
        raise ValueError("run still-photo licence must be a StillPhotoLicence or None")
    token = _run_still_photo_licence.set(licence)
    try:
        yield
    finally:
        _run_still_photo_licence.reset(token)


def _refuse_second_licence_channel(installed: StillPhotoLicence, incoming: str) -> None:
    """Refuse a licence from the OTHER channel: readiness accepts one licence at a time.

    Named channels in the message, not types: the operator's fix is to remove one of two config
    keys, and the message has to be readable enough to say which pair is in conflict.
    """
    if installed.channel == incoming:
        return
    raise ValueError(
        f"a still-photo readiness licence from the {installed.channel!r} channel is already "
        f"installed in this process and the {incoming!r} channel cannot be installed beside it; "
        "numbering accepts exactly one licence at a time, so clear the installed one first")


def _exact_int(value: object) -> bool:
    """True only for a real int.  ``bool`` is an int subclass, so ``True`` would count as 1."""
    return type(value) is int


def _finite_real(value: object) -> bool:
    """Finite int/float, without importing math (this module stays stdlib-minimal)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and float("-inf") < value < float("inf")


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@_serialise_still_photo_licence_mutation
def install_verified_still_photo_bound(summary: StillPhotoBoundSummary) -> None:
    """Install the process-local numbering licence, or refuse with the exact reason it fails.

    Every numeric check here is a threshold from section 4 of the design doc, and they are
    identical on both label channels: the circular channel buys a cheaper campaign, never a
    smaller corpus or a looser accept rate.  What differs is the claim each channel is allowed to
    make about its labels.  The point of validating at the install boundary (rather than trusting
    the caller) is that config validation, the measurement tool and any future consumer all have
    to clear the same bar, so there is no "internal" path that can install a weaker bound.
    """
    global _installed_still_photo_licence
    if not isinstance(summary, StillPhotoBoundSummary):
        raise ValueError(
            f"still-photo bound must be a StillPhotoBoundSummary (got {type(summary).__name__})")
    circular = summary.ground_truth_channel == STILL_PHOTO_BOUND_CIRCULAR_CHANNEL
    if not circular and summary.ground_truth_channel != STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL:
        raise ValueError(
            f"ground_truth_channel must be {STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL!r}, the owner "
            f"tap-to-play label channel, or {STILL_PHOTO_BOUND_CIRCULAR_CHANNEL!r}, the "
            f"explicitly opted-in circular AI-labelled channel (got "
            f"{summary.ground_truth_channel!r})")
    if circular:
        # The circular channel is licensed by the owner's acceptance, never by a ground-truth
        # claim.  `is not False` rather than `not ...` for the same reason the owner branch uses
        # `is not True`: 0 or "" would otherwise pass as an honest denial.
        if summary.human_ground_truth is not False:
            raise ValueError(
                f"human_ground_truth must be exactly False on {STILL_PHOTO_BOUND_CIRCULAR_CHANNEL!r}; "
                "its labels come from the mute-glyph matcher, so claiming human ground truth "
                f"there is a lie (got {summary.human_ground_truth!r})")
        if summary.accepted_circular_risk != STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE:
            raise ValueError(
                f"accepted_circular_risk must be exactly {STILL_PHOTO_BOUND_CIRCULAR_ACCEPTANCE!r} "
                f"to install the circular AI-labelled channel {STILL_PHOTO_BOUND_CIRCULAR_CHANNEL!r} "
                f"(got {summary.accepted_circular_risk!r})")
    else:
        # `is not True` rather than `not ...`: a truthy string or 1 would otherwise let an artifact
        # claim human ground truth it does not have, which is the one claim nothing downstream can
        # re-derive.
        if summary.human_ground_truth is not True:
            raise ValueError(
                "human_ground_truth must be exactly True; a bound labelled by the mute matcher "
                f"cannot license numbering (got {summary.human_ground_truth!r})")
        # An owner-labelled bound accepted nothing, so it must not carry an acceptance that a
        # later reader could mistake for one.
        if summary.accepted_circular_risk is not None:
            raise ValueError(
                "accepted_circular_risk must be None on the owner channel "
                f"{STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL!r}; only "
                f"{STILL_PHOTO_BOUND_CIRCULAR_CHANNEL!r} accepts circular AI labels "
                f"(got {summary.accepted_circular_risk!r})")
    if not _exact_int(summary.video_cards) or summary.video_cards < STILL_PHOTO_BOUND_MIN_VIDEO_CARDS:
        raise ValueError(
            f"video_cards must be an integer >= {STILL_PHOTO_BOUND_MIN_VIDEO_CARDS} owner-labelled "
            f"video cards (got {summary.video_cards!r})")
    if not _exact_int(summary.video_accepts) or summary.video_accepts < 0:
        raise ValueError(f"video_accepts must be a non-negative integer (got {summary.video_accepts!r})")
    if summary.video_accepts != STILL_PHOTO_BOUND_REQUIRED_VIDEO_ACCEPTS:
        raise ValueError(
            f"video_accepts must be exactly {STILL_PHOTO_BOUND_REQUIRED_VIDEO_ACCEPTS}; a single "
            f"held-out video accept means the bound is not licensed (got {summary.video_accepts!r})")
    if not _exact_int(summary.photo_cards) or summary.photo_cards < STILL_PHOTO_BOUND_MIN_PHOTO_CARDS:
        raise ValueError(
            f"photo_cards must be an integer >= {STILL_PHOTO_BOUND_MIN_PHOTO_CARDS} owner-labelled "
            f"still photos (got {summary.photo_cards!r})")
    if not _exact_int(summary.photo_false_refusals) or summary.photo_false_refusals < 0:
        raise ValueError(
            f"photo_false_refusals must be a non-negative integer (got {summary.photo_false_refusals!r})")
    ceiling = STILL_PHOTO_BOUND_MAX_PHOTO_FALSE_REFUSAL_FRAC * summary.photo_cards
    if summary.photo_false_refusals > ceiling:
        raise ValueError(
            f"photo_false_refusals {summary.photo_false_refusals} exceeds "
            f"{STILL_PHOTO_BOUND_MAX_PHOTO_FALSE_REFUSAL_FRAC:.0%} of {summary.photo_cards} "
            f"owner-labelled still photos ({ceiling:g})")
    if not _finite_real(summary.max_video_exact_run_s) or summary.max_video_exact_run_s < 0:
        raise ValueError(
            "max_video_exact_run_s must be a finite number of seconds >= 0 "
            f"(got {summary.max_video_exact_run_s!r})")
    digest = summary.artifact_sha256
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in _SHA256_HEX_DIGITS for c in digest)):
        raise ValueError(f"artifact_sha256 must be 64 lowercase hex digits (got {digest!r})")
    if not _nonempty_text(summary.device):
        raise ValueError(f"device must be the nonempty ADB serial the bound was measured on (got {summary.device!r})")
    if not _nonempty_text(summary.hinge_version_name):
        raise ValueError(
            f"hinge_version_name must be the nonempty Hinge build the bound was measured on "
            f"(got {summary.hinge_version_name!r})")

    installed = _installed_still_photo_licence
    if installed is not None:
        # An accepted assumption is not a weaker bound, it is a different KIND of licence, so it
        # is refused here rather than overwritten (see _refuse_second_licence_channel).
        _refuse_second_licence_channel(installed, STILL_PHOTO_LICENCE_MEASURED)
        if installed.record != summary:
            # Two different bounds in one process means two different measurement campaigns are
            # claiming the same licence.  Refuse rather than let the later one win silently:
            # which artifact is live would then depend on validation order.
            raise ValueError(
                "a different verified still-photo bound is already installed in this process; "
                "clear it before installing another")
    _installed_still_photo_licence = StillPhotoLicence(
        channel=STILL_PHOTO_LICENCE_MEASURED, record=summary)


@_serialise_still_photo_licence_mutation
def install_accepted_still_photo_assumption(acceptance: StillPhotoAssumptionAcceptance) -> None:
    """Install the owner's centered-autoplay assumption as the numbering licence.

    Validated at the install boundary for the same reason the measured installer is: every
    caller, config validation included, has to clear the same bar, so there is no "internal"
    path that can license numbering with a half-filled acceptance.  What is checked is only what
    an assumption can be checked for -- the exact phrase, and that every field a later reader
    would need in order to understand what was accepted is actually present.  There is nothing
    numeric to check, and inventing a number here would be the exact dishonesty this channel is
    designed to avoid.
    """
    global _installed_still_photo_licence
    if not isinstance(acceptance, StillPhotoAssumptionAcceptance):
        raise ValueError(
            "still-photo assumption must be a StillPhotoAssumptionAcceptance "
            f"(got {type(acceptance).__name__})")
    if acceptance.acceptance != STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION:
        raise ValueError(
            f"acceptance must be exactly {STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION!r} to license "
            "numbering on the unmeasured centered-autoplay assumption "
            f"(got {acceptance.acceptance!r})")
    if not _nonempty_text(acceptance.accepted_at):
        raise ValueError(
            "accepted_at must be nonempty text recording when the owner accepted the assumption "
            f"(got {acceptance.accepted_at!r})")
    if not _nonempty_text(acceptance.device):
        raise ValueError(
            "device must be the nonempty ADB serial the assumption was accepted for "
            f"(got {acceptance.device!r})")
    if not _nonempty_text(acceptance.hinge_version_name):
        raise ValueError(
            "hinge_version_name must be the nonempty Hinge build the assumption was accepted for "
            f"(got {acceptance.hinge_version_name!r})")
    if not _nonempty_text(acceptance.rationale):
        # The whole value of an assumption channel is that the reason is on record.  An empty
        # rationale would leave a licence nobody can audit, which is indistinguishable from a
        # wished-for one -- the same failure the hand-editable boolean had.
        raise ValueError(
            "rationale must be the nonempty reason the owner accepted this assumption; an "
            f"unexplained assumption is not auditable (got {acceptance.rationale!r})")

    installed = _installed_still_photo_licence
    if installed is not None:
        _refuse_second_licence_channel(installed, STILL_PHOTO_LICENCE_ASSUMPTION)
        if installed.record != acceptance:
            # Same no-silent-swap rule as the measured installer: two different acceptances in
            # one process means two different decisions claim the same gate.
            raise ValueError(
                "a different accepted still-photo assumption is already installed in this "
                "process; clear it before installing another")
    _installed_still_photo_licence = StillPhotoLicence(
        channel=STILL_PHOTO_LICENCE_ASSUMPTION, record=acceptance)


def installed_still_photo_bound() -> StillPhotoBoundSummary | None:
    """The verified MEASURED bound licensing numbering right now, or ``None``.

    Deliberately still answers ``None`` while an accepted assumption holds the licence: every
    existing caller of this function asks it for measured facts (corpus counts, the artifact
    digest, the dwell window the campaign observed), and an assumption has none of those.  Ask
    ``installed_still_photo_licence()`` for "is numbering licensed at all".
    """
    licence = _effective_still_photo_licence()
    if licence is None or not licence.measured:
        return None
    return licence.record


def installed_still_photo_licence() -> StillPhotoLicence | None:
    """The licence that makes numbering available right now, or ``None`` while none is installed.

    This is the one accessor that answers "is numbering licensed", across both channels.
    """
    return _effective_still_photo_licence()


@_serialise_still_photo_licence_mutation
def clear_installed_still_photo_bound() -> None:
    """Drop any installed licence, returning numbering to its fail-closed default.

    Clears BOTH channels: there is a single slot, so this is simply "no licence".  Safe to call
    from anywhere -- it can only make the system MORE restrictive.  Config validation calls it
    first thing so a run that validates a config without an evidence or acceptance key can never
    inherit readiness from an earlier validate() in the same process.
    """
    global _installed_still_photo_licence
    _installed_still_photo_licence = None


def _reset_installed_still_photo_bound_for_tests() -> None:
    """Tests only: drop the process-local licence so one test cannot license another."""
    clear_installed_still_photo_bound()


def still_photo_licence_provenance() -> str | None:
    """A short human string naming WHAT licensed numbering, or ``None`` while nothing does.

    Consumers print this.  It exists because the two channels are not interchangeable and the
    difference is invisible in behaviour: numbering looks identical either way, so the only
    place the distinction can survive to an operator is in words.  The assumption wording leads
    with UNMEASURED for that reason -- a reader who reads three words must not come away
    believing a false-accept rate exists.
    """
    licence = _effective_still_photo_licence()
    if licence is None:
        return None
    if licence.assumed:
        return STILL_PHOTO_ASSUMPTION_PROVENANCE
    bound = licence.record
    text = (f"measured held-out bound ({bound.video_cards} video cards, "
            f"{bound.video_accepts} accepts)")
    if bound.ground_truth_channel == STILL_PHOTO_BOUND_CIRCULAR_CHANNEL:
        # Section 5a of the design doc, said out loud rather than left to the channel id: on the
        # circular channel the label and the accept rule read the same pixels, so quoting the
        # accept count without that clause overstates what the campaign measured.
        text += (" on the circular AI-labelled channel, where the accept count is zero by "
                 "construction")
    return text


def still_photo_licence_operator_notice() -> str | None:
    """The run-level line an operator must see, or ``None`` when nothing needs saying.

    Only the assumption channel produces one: a measured bound is the state the whole design
    doc assumes, so announcing it every run would be noise, and noise is what teaches people to
    stop reading notices.  An unmeasured licence is news every single run.
    """
    licence = _effective_still_photo_licence()
    if licence is None or not licence.assumed:
        return None
    return STILL_PHOTO_ASSUMPTION_OPERATOR_NOTICE


def hinge_targeting_unavailable_reason() -> str | None:
    """Return the actionable policy blocker, or ``None`` while EITHER channel licenses numbering."""
    if _effective_still_photo_licence() is not None:
        return None
    return HINGE_TARGETING_UNAVAILABLE_REASON


# The two operator next steps for an absent `apps.hinge.targeting_calibration`, kept here beside
# the licence they branch on so the terminal notice and the hub's fine print cannot drift.
#
# WHY THIS IS A BRANCH AND NOT A SENTENCE (bug report 2026-08-22).  Both surfaces used to state
# the still-photo prerequisite unconditionally, which was true only while nothing licensed
# numbering.  Once the owner's centered-autoplay acceptance installed a licence, that sentence
# started naming a prerequisite the operator had ALREADY satisfied and never named the one step
# that would actually restore suggestions -- so a run whose only remaining gate was "capture the
# calibration" read as "you are still blocked upstream, do not run the capture", and the missing
# calibration stayed missing.  Guidance that outlives its own precondition is a defect, so this
# is derived from the installed licence rather than written down.
# Each carries its OWN doc pointer, and consumers print them unprefixed. A consumer that added
# "per ops/RUNBOOK.md, ..." in front would read the reference twice in the calibrate branch and
# not at all if a future branch omitted it -- the pointer belongs to the step, not to the surface.
TARGETING_SETUP_NEXT_STEP_BLOCKED = (
    "per ops/RUNBOOK.md section 2, positive still-photo proof must exist before a fresh "
    "calibration can enable suggestions")
TARGETING_SETUP_NEXT_STEP_CALIBRATE = (
    "still-photo numbering is licensed, so the one remaining step is the per-device targeting "
    "calibration campaign in ops/RUNBOOK.md section 2: capture both splits, measure them, and "
    "paste the emitted apps.hinge.targeting_calibration block into config.yaml")


def targeting_setup_next_step() -> str:
    """The ONE next action that would restore Hinge's numbered suggestions, given today's state.

    Never returns None: this is only ever consulted while a targeting blocker is already being
    reported, and "there is a blocker but no next step" is precisely the dead end the bug report
    described.  It answers for the still-photo licence only; the calibration mapping itself is
    config data that this dependency-free module deliberately cannot see, so the calibrate branch
    is the correct answer for BOTH "licence installed, calibration absent" and "licence installed,
    calibration present but rejected" -- in each case the operator's move is a fresh campaign.
    """
    if hinge_targeting_unavailable_reason() is not None:
        return TARGETING_SETUP_NEXT_STEP_BLOCKED
    return TARGETING_SETUP_NEXT_STEP_CALIBRATE
