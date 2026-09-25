"""Shared, budget-aware opener generation across all app workers.

One OpenerService is shared by every worker (Bumble, Hinge, ...) so the per-run
spend cap is GLOBAL, not per-app. Thread-safe. Handles the ways opener generation
ends for a run: the configured run budget, max_attempts consecutive bad AI responses
for one profile (see maybe_opener's retry loop below), the whole Gemini model cascade
falling through (GeminiCapacityExhausted -- every configured model out of per-day
quota, retired/404-gone, or some mix of those plus transient per-minute throttling;
see opener.py's GeminiOpener.generate()), and a permanent provider failure (bad/missing
API key, no access to the Gemini API at all, or -- only after several back-to-back
malformed-request errors -- a broken request schema/params) -- each flips the
service to disabled and asks the supervisor to stop all workers.

THE OWNER'S RULE (this is why there is no "swipe without an opener" mode anymore): a
commentless like is normally not acceptable. If the AI's response is bad, redo the prompt --
spending more usage is fine. The narrow exception is a provider safety-policy block: AUTO has
already decided to like that profile, so it may proceed without a comment rather than retrying
or stopping. Only if a profile's ordinary response is STILL bad after
max_attempts tries does that mean something is actually wrong, and only then does the
whole run stop. The old budget.on_exhausted="swipe_without_opener" configuration was
removed outright rather than left dormant -- the same reasoning that removed the
Anthropic provider path applies here: a setting that can silently reintroduce a banned
behavior is worse than no setting at all. Every exhaustion path below sets stop_requested
and disables further opener generation for the rest of the run.

Note what is NOT here: a single model id being retired (HTTP 404) is no longer a
service-wide, permanent failure. GeminiOpener.generate() retires just that one model
and cascades to the next configured model on its own -- see _permanent_reason's
docstring below for why this function deliberately does not classify a 404.

A LONE HTTP 400 is treated as transient, not permanent: an opener client embeds
that specific profile's own captured photos in every request, so a single 400
can be caused by a profile-specific payload
problem (a truncated/corrupt PNG from a flaky `adb screencap`, an oversized
image, too many photos) rather than by the request itself being malformed. Only
a run of consecutive 400s (see _BAD_REQUEST_LATCH_THRESHOLD) -- which a genuine
schema/param bug produces deterministically, since it re-fires on literally the
next call regardless of that call's content -- latches the service permanently
disabled.

A per-call failure is handled differently depending on whether re-asking can plausibly
help. Most OpenerParseError responses (a billed response that didn't parse into a usable
opener) are RETRIED, up to self.max_attempts times, with the model told what was wrong on each
retry -- see maybe_opener's docstring for the full loop and why: a parse failure is usually
stochastic (the model drew a bad sample), so a re-ask genuinely has a decent chance of
producing something usable. A prompt or response blocked by Gemini's safety policy is the
narrow exception: it is recorded and skipped for that profile without retrying or disabling the
run, because resending the same captured content cannot safely improve it. AUTO may still carry
out its already-made like decision without a comment. A non-parse
OpenerError (for example, a corrupt photo that can't even be encoded) is NOT retried -- it is
almost always specific to THAT
profile's content in a way a re-ask cannot fix (the same corrupt bytes go in again), so
THIS SERVICE just returns None and stays enabled for the next profile. Either way, every
billed attempt has its spend recorded before degrading. What a None return means to the
caller is the caller's decision, not this service's: OBSERVE mode swipes without an
opener (a human already made the call), but AUTO mode on an opener-capable app (Hinge)
does not -- see worker.py's _auto_loop, which refuses to send ANY like with no opener on
such an app, whether the None came from a narrow per-call path or from full exhaustion
below. last_skip_reason exists to give that AUTO guard an accurate, per-call reason for
exactly the narrow (non-exhausting) case (see its docstring).

A run of consecutive TRANSIENT failures (see _TRANSIENT_LATCH_THRESHOLD) is its own latch,
alongside the 400 streak above: an unclassified exception (a bare timeout, a connection
blip, a provider 5xx that GeminiAPIError didn't already classify) or a per-profile
non-parse OpenerError (see opener.py's item B corrupt-capture fix) is,
individually, exactly the kind of thing that must NOT kill a run over one unlucky profile.
But left completely unbounded, EVERY such failure used to fall through to the same silent
"swiping without an opener" branch forever, with the service left enabled and nothing on the
hub to show it: a systemic bug in our own request building, a dead network, or a
provider-wide outage would turn every remaining profile this run into a bare like with no
opener and no visible sign anything was wrong -- directly against this project's "fail
loudly, never silently degrade" rule. A short streak (not a lone failure, since one-off
blips are normal and must not halt the run) now latches the service exactly like the 400
streak does.
"""
from __future__ import annotations

import inspect
import json
import os
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ..costing import CostTracker
from ..perception.capture import Profile
from .opener import (
    FIRST_ITEM_INDEX,
    INDEX_SPACE_MODEL_ITEMS,
    INDEX_SPACE_PROFILE_PHOTOS,
    ITEM_INDEX_ABSENT,
    GeminiAPIError,
    GeminiCapacityExhausted,
    ItemRequest,
    OpenerAborted,
    OpenerClient,
    OpenerError,
    OpenerParseError,
    REASON_BAD_REQUEST,
    REASON_OPENER_ERROR,
    REASON_PROMPT_BLOCKED,
    REASON_RESPONSE_BLOCKED,
    REASON_TRANSIENT_ERROR,
    _leading_ngram,
    prompt_stamp,
)
from .replay_corpus import prune_replay_corpus, write_replay_capture

# The two `openers.decision` values discard_opener() below writes, spelled ONCE here because
# they are read outside this module -- worker.py chooses between them at every abandonment site,
# and tools/opener_corpus_report.py buckets on them -- and because this project has already been
# bitten by one rule spelled twice in two places that then disagreed (two regexes for the same
# permitted opener move carrying different noun sets). DECISION_REPLAY in replay_corpus.py is
# the established shape for a marker like this: it lives in the installed package precisely so
# both the writer and the offline reporting tool can import the one string rather than re-type
# it. The other value in this vocabulary, "like", belongs to commit_opener and is not a discard
# reason, so it is deliberately not spelled here.
#
# NEVER_SENT: generated, billed, and abandoned with nothing typed or sent -- an explicit Dislike
# is the other non-sent case and carries its own "dislike" from Training, see discard_opener.
DECISION_NEVER_SENT = "never_sent"
# SEND_UNVERIFIED (2026-09-15): the draft physically WENT OUT and we cannot prove what happened
# to it. `driver.like()` is not atomic -- HingeDriver types the opener, taps Send Like, and only
# THEN runs its verification (_verify_like_landed, and in Training the deck-advance proof) -- so
# a raise from that post-send window used to be recorded as NEVER_SENT, asserting in the one
# durable table the corpus report trusts that a message a real person may have received was only
# ever a draft. This is the third value that window needs, and it is deliberately not "like":
# nothing verified a landed Like, and "like" is the exact literal Store.joined_opener_outcomes
# tests for when deciding which opener may own an owner-observed outcome. A row carrying this
# value is therefore excluded from outcome attribution automatically, by the same predicate that
# excludes every other non-"like" decision, which is the correct answer -- an unconfirmed send
# must not be credited with a match somebody else's opener earned.
DECISION_SEND_UNVERIFIED = "send_unverified"


@dataclass
class OpenerPick:
    """An opener plus which profile item it is about, so the driver can attach the comment to
    the RIGHT photo/prompt instead of always the first.

    `index` CHANGED MEANING on 2026-08-12 and was NOT renamed, so read this before using it
    (ops/OPENER-REDESIGN.md 5.1/5.3/5.7):

      before: 0-based index into the profile CAPTURE ORDER -- i.e. into the raw scroll frames
              the request was built from, which is not an item space at all (one card appears
              in several frames, one frame can hold two cards).
      now:    the MODEL ITEM INDEX, 1-based, straight off OpenerResult.item_index. It counts
              the numbered items the model was shown, and it is one answer to two questions:
              the item the opener is about AND the item to like.

    Both are ints, both are small, and neither carries its base in its type, which is exactly
    the class of bug doc 5.3 exists to prevent -- so no consumer may assume the old meaning
    still holds, and none may quietly add or subtract one to make an old consumer fit.

    `index_space` is what stops that from being a comment nobody reads: it says WHICH LIST
    `index` counts, is set from the request the opener client actually built, and
    `capture_order_index` below is the ONLY sanctioned way to turn this pick into something a
    driver can act on. It returns None whenever no sound conversion exists, which a driver must
    treat as "I was not told which item" rather than as any particular item.

    ITEM_INDEX_ABSENT (0) means the model gave no usable item number. Under the 1-based
    contract that is out of band by construction and can never be confused with a real pick --
    but note that it IS a legal value in the driver's own 0-based capture-order space, which is
    precisely why `capture_order_index` maps it to None instead of passing the zero along.

    `referenced` is the model's own one-line statement of WHICH profile detail the opener is
    about (OpenerResult.referenced, echoed here verbatim). It is not used for targeting --
    that's what `index` is for -- it exists so the hub can show the operator what the
    suggestion is supposed to be about, which is precisely how a mismatch between the opener
    and the item the comment actually hangs under becomes visible instead of silent -- the
    failure this whole opener-to-item binding exists to prevent."""
    text: str
    index: int = ITEM_INDEX_ABSENT
    referenced: str = ""
    # The model's own free-text words for what this opener is DOING ("guessing where the ridge
    # is", "teasing her about the cold"), echoed verbatim from OpenerResult.angle. PURE
    # TELEMETRY: nothing in this service, the worker, or any driver branches on it, and it is
    # deliberately not an enum anywhere in the pipeline (ops/OPENER-REDESIGN.md 3.5) -- a closed
    # set would force the model to pick a move from a menu and shoehorn the opener into it,
    # which is exactly the awkwardness the move list is written to avoid. It exists so the
    # persisted `openers` rows can eventually answer which SHAPES correlate with matches, a
    # question this project currently cannot ask at all. Trailing, with a "" default, so every
    # existing OpenerPick(...) construction (worker.py, the tests' fakes) keeps working
    # untouched and a client that never populates the field degrades to "" rather than raising.
    angle: str = ""
    # The model's own short description of the ITEM it picked, echoed verbatim from
    # OpenerResult.item_description. Carried always (doc 5.7), regardless of caller, so every
    # request stays the same shape.
    #
    # Distinct from `referenced`: that is the DETAIL the opener reacts to, this is the ITEM it
    # was picked from. Doc 5.8 uses this one, coarsely (photo vs written prompt), to check our
    # own crop at `index` against what the model thought it chose BEFORE anything is tapped.
    # That check is a later workflow; nothing branches on this today.
    item_description: str = ""
    # WHICH LIST `index` COUNTS -- one of opener.INDEX_SPACE_*, echoed from
    # OpenerResult.index_space, which generate() sets from the request shape it actually built.
    #
    # Trailing with a default so every existing OpenerPick(...) construction keeps working, and
    # the default is the UNTRANSLATABLE space on purpose (same reasoning as
    # OpenerResult.index_space): a pick built by a fake or by some future call site that never
    # set this must not have its number converted into a tap on somebody's phone. Fail-safe
    # means "no target", not "target 1".
    index_space: str = INDEX_SPACE_MODEL_ITEMS
    # Private, one-run staging envelope.  AUTO and Observe both commit it only after the
    # driver's like boundary returns: a generated opener is not evidence of a landed action.
    _staged_record: "_StagedOpenerRecord | None" = field(
        default=None, repr=False, compare=False)

    @property
    def capture_order_index(self) -> int | None:
        """This pick as a 0-based index into the DRIVER's capture order, or None.

        This is the one sanctioned crossing between the model's index space and the driver's,
        and it exists because on 2026-08-12 the two stopped agreeing while nothing in the types
        said so. `Driver.like(item_index=...)` counts the frames the driver captured for this
        profile; `index` counts the numbered images the model was SENT. Handing one to the
        other unconverted is not a near miss -- an in-range 1-based item number resolves to a
        real, adjacent, WRONG frame and the driver reports it as on-target, so the comment
        lands one card down with full confidence and nothing detects it.

        None means "no sound conversion exists", and a driver must read it as "I was not told
        which item", never as an index. It is NOT the same as 0, which is a perfectly legal
        first frame in the driver's space -- collapsing the two is how ITEM_INDEX_ABSENT ("the
        model could not pick") turned into a confident like of item 1.

        The three cases:

        * ITEM_INDEX_ABSENT -> None. Doc 5.3: "treat a missing table as a hard stop, never as a
          reason to fall back to a fixed coordinate." And a hard stop is now literally what it
          is: worker._auto_loop refuses to call the driver at all with an untranslatable pick,
          and the driver itself refuses an opener carrying no item index
          (hinge._like_comment_sheet). Until 2026-08-12 this state instead tapped the first heart
          and repaired the message against whatever the sheet turned out to show; that repair
          hatch is removed, because rewriting the TEXT does not undo spending the LIKE on an item
          the model never chose (doc 5.6, never substitute).

        * INDEX_SPACE_PROFILE_PHOTOS -> index - FIRST_ITEM_INDEX. The numbered images WERE the
          profile's scroll frames, sent in capture order, so model item k is frame k-1. This is
          a derivation, not a fudge: the request was built from `profile.photos`, the driver
          captured `profile.photos` and keeps `_current_sigs` index-aligned with it (see
          hinge._capture_current, pinned by
          tests/test_hinge_observe.py::test_capture_keeps_sigs_index_aligned_with_photos), and
          the same Profile object is what the worker holds across both calls. The frames are
          poor "items" -- that is what doc 5.2 replaces with crops -- but the correspondence
          itself is exact.

        * INDEX_SPACE_MODEL_ITEMS -> None, and it STAYS None now that doc 5.5's counting
          navigation exists. Item k is the k-th SELECTABLE item, which the driver reaches by
          walking to its HEART ORDINAL -- not to a captured frame -- so there is no capture-order
          value that names it and inventing one is the bug this property was written to prevent.
          What changed on 2026-08-12 is what a CALLER does with the None: `worker._auto_loop` no
          longer reads it as "this pick cannot be acted on", it passes `pick.index` to
          `driver.like(model_item_index=...)` instead and lets the driver navigate. Also the
          default, so an unknown or unrecognised space refuses too -- and there a None really
          does mean "cannot be acted on", which is why the worker branches on `index_space`
          rather than on this being None.
        """
        if self.index == ITEM_INDEX_ABSENT:
            return None
        if self.index_space == INDEX_SPACE_PROFILE_PHOTOS:
            return self.index - FIRST_ITEM_INDEX
        return None


@dataclass(frozen=True)
class _StagedOpenerRecord:
    """An uncommitted, profile-attributable opener row for one pending like."""
    run_id: str
    app: str
    model: str
    opener: str
    referenced: str
    angle: str
    item_description: str
    recent_entry: dict
    # The digest of the prompt era this draft was GENERATED under (opener.py's prompt_stamp).
    # Carried on the staged record rather than read from the service at commit time so a draft
    # can never be attributed to a prompt it was not generated under: staging is what separates
    # the two moments, and a long-lived service could in principle be handed a different style
    # between them. Last field, and passed by KEYWORD at the construction site, so a later edit
    # to the positional prefix cannot silently bind it to `recent_entry`.
    prompt_sha256: str


# How many consecutive provider HTTP 400s to require before treating the failure
# as permanent and latching the service disabled. A genuinely broken request (bad schema,
# bad param) is deterministic: it re-fires on the very next call no matter which profile's
# content is sent, so it still latches within a couple of profiles. A profile-specific
# payload problem (corrupt screenshot, oversized image) is very unlikely to repeat back to
# back across unrelated profiles' photos, so requiring a streak protects it from a single
# unlucky capture while still catching a systemic problem quickly.
_BAD_REQUEST_LATCH_THRESHOLD = 3

# How many consecutive TRANSIENT opener failures (an unclassified exception, or a per-profile
# OpenerError -- see _consecutive_transient_failures below for exactly what counts) to allow
# before latching the service disabled, mirroring _BAD_REQUEST_LATCH_THRESHOLD's tradeoff.
# Deliberately small, and deliberately not 1: a single timeout, one dropped connection, or
# one bad photo is ordinary and must not halt an otherwise-healthy run (that's the whole
# reason these failures degrade instead of raising in the first place). But an audit of this
# project demonstrated 50 consecutive bare RuntimeErrors sailing through with the service
# still enabled and nothing on the hub showing anything was wrong -- every one of those was a
# live profile that got a bare like with no opener. A short streak is enough to tell "one
# unlucky call" apart from "the pipeline itself is broken" (a systemic problem re-fires on
# essentially every subsequent call, exactly like the 400 case) while still absorbing an
# occasional real blip.
_TRANSIENT_LATCH_THRESHOLD = 3

# How many of the most recent SUCCESSFUL opener generations OpenerService.recent_openers keeps
# (see its docstring in __init__) for a bug report to inspect after the fact. Small and fixed:
# this is a live debugging aid, not the permanent record (that's self.store.record_opener), so
# it only needs to cover roughly the last screenful of activity, not the whole run.
_RECENT_OPENERS = 12

# Same idea as _RECENT_OPENERS, but for REJECTED attempts (OpenerParseError) -- see
# self.recent_rejections' docstring in __init__. Kept as its own ring buffer/constant rather
# than folded into recent_openers: a rejection is not a shorter-lived variant of a success,
# it is the other outcome entirely, and a bug report needs to show both without one crowding
# the other out of a shared, fixed-size buffer.
_RECENT_REJECTIONS = 12
_MAX_ATTEMPTS = 15

# How many LEADING words the entropy guard compares between openers -- see
# OpenerService._leading_ngram_collision / _apply_entropy_guard below and
# ops/OPENER-REDESIGN.md 3.6. Four is wide enough to catch a recurring OPENING FORMULA
# ("based on that ridgeline", "i bet you were") and narrow enough not to fire on two openers
# that merely share a word or two of ordinary English ("that view ...", "that husky ..."),
# which is not a fingerprint. Compared against self.recent_openers, whose own cap
# (_RECENT_OPENERS) therefore also decides how far back the guard can see.
_ENTROPY_NGRAM_WORDS = 4

# --- Opener-rejection dead-letter (see _write_opener_rejection_deadletter's docstring) ---
#
# Bounds for the LOCAL dead-letter file below. This is a diagnostic scratch file, not the
# durable record (that is still self.store.record_opener_rejection) -- it only exists to
# survive the ONE piece of evidence a failed store write would otherwise lose: the exception
# text. An unbounded append-only file would defeat that same purpose over time (a multi-day
# BigQuery outage would turn "helpful diagnostic" into "an ever-growing file nobody wants to
# open"), so it is capped on two independent axes, oldest entries dropped first on either:
#   - _DEADLETTER_MAX_ENTRIES (200): generous headroom over any single bad run (max_attempts
#     tops out at _MAX_ATTEMPTS=15 per profile, so 200 covers well over a dozen consecutive
#     doomed profiles) while staying small enough to read by eye without paging.
#   - _DEADLETTER_MAX_BYTES (2 MiB): the hard backstop, checked AFTER the per-field truncation
#     below, so 200 entries that each happened to carry a pathological multi-KB provider error
#     still cannot blow past a sane file size.
# The newest failure is the one relevant to whatever incident someone is currently debugging,
# which is why both bounds evict the oldest entries rather than the newest.
_DEADLETTER_MAX_ENTRIES = 200
_DEADLETTER_MAX_BYTES = 2 * 1024 * 1024

# Cap on any SINGLE large text field written into one dead-letter entry (reason, raw_opener,
# the formatted traceback). A single pathological value -- e.g. the multi-KB HTML error body
# some providers put in an exception message -- must not by itself blow the whole-file byte
# ceiling above and evict every older entry in one write. 4000 chars comfortably holds a
# realistic provider error message and a full multi-frame traceback; anything longer is
# truncated with an explicit marker so a reader can tell "this is all there is" from "this was
# cut short".
_DEADLETTER_FIELD_LIMIT = 4000
_DEADLETTER_TRUNCATION_MARKER = "...<deadletter-truncated>"

# The four maybe_opener() branches that call self.store.record_opener_rejection, in the order
# they appear below -- passed as the `branch` argument to
# _write_opener_rejection_deadletter so a reader can tell which failure produced a given
# entry without re-deriving it from reason_code (REASON_OPENER_ERROR/REASON_BAD_REQUEST/
# REASON_TRANSIENT_ERROR are shared with other telemetry; OpenerParseError has no single
# REASON_* of its own since e.reason_code is model-supplied).
#
# (2026-09-17) A fifth branch value, "shutdown_flush", is not one of these four -- it is
# written from OUTSIDE this service, by supervisor.py's
# _deadletter_stranded_opener_rejections, for rows that never raised here at all. See
# _write_opener_rejection_deadletter's own "THE flush_every DEPENDENCY" section below for why
# that second call site exists.


def _deadletter_truncate(value: str | None, limit: int = _DEADLETTER_FIELD_LIMIT) -> str | None:
    """Bound one text field for the dead-letter entry (see _DEADLETTER_FIELD_LIMIT). None
    passes through unchanged -- raw_opener is legitimately None for three of the four
    branches, and that is a meaningful "no raw text was ever parsed", not a value to coerce
    into a string first."""
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[:limit] + _DEADLETTER_TRUNCATION_MARKER


def _rejection_identity_key(*, run_id: str, app: str, model: str, attempt: object,
                            reason_code: str, reason: str, raw_opener: str | None,
                            prompt_sha256: str | None) -> tuple:
    """(2026-09-17) Stable, in-run identity for one opener_rejections row, so
    OpenerService can tell supervisor.py's shutdown sweep "I already dead-lettered this row
    myself" -- see OpenerService.is_rejection_row_deadlettered and
    supervisor._deadletter_stranded_opener_rejections, the two places this is compared.

    Built from exactly the eight fields BOTH sides of that comparison actually have: every
    per-call `except Exception as store_exc:` site in maybe_opener() below calls this with
    the same run_id/app/model/attempt/reason_code/reason/raw_opener/prompt_sha256 it just
    passed to record_opener_rejection AND to _write_opener_rejection_deadletter, and
    BigQueryStore.pending_opener_rejections() (ranker/bigquery_store.py) snapshots a
    buffered row built from those identical eight values.

    Deliberately EXCLUDES `created_at`: BigQueryStore.record_opener_rejection assigns that
    timestamp itself, internally, only once the row is appended to its buffer -- the
    exception handler here never gets it back (record_opener_rejection returns nothing), so
    a key that depended on it could never match between "the call that raised" and "the
    still-buffered row a later snapshot sees". The two call sites can only ever agree on
    what the CALLER supplied, which is exactly this tuple.

    Content-identity, not a nonce -- the same philosophy ranker/bigquery_store.py's own
    _row_id already uses for BigQuery's streaming dedup (a row the caller cannot tell apart
    from one already seen IS, by that convention, the same logical row). Two genuinely
    different rejections would need identical run_id, app, model, attempt, reason_code,
    reason text, raw_opener text, AND prompt era to produce the same key here -- and
    `attempt` plus `reason_code` alone already separate the four maybe_opener() branches and
    every retry within one profile's OpenerParseError loop, since a profile's own retry hint
    (folded into the model's next raw response) makes two attempts' raw_opener/reason
    diverge in practice even when the profile is the same. Nothing downstream is asked to
    consider this cryptographically unique; it only has to avoid colliding across DISTINCT
    real failures, and the existing latch thresholds (_BAD_REQUEST_LATCH_THRESHOLD,
    _TRANSIENT_LATCH_THRESHOLD, max_attempts) keep the set of keys any one run ever produces
    small enough that this holds in practice.
    """
    attempt_int = attempt if isinstance(attempt, int) and not isinstance(attempt, bool) else 0
    return (run_id, app, model, attempt_int, reason_code, reason, raw_opener, prompt_sha256)


def _write_opener_rejection_deadletter(path: str | None, branch: str, store: object, *,
                                       run_id: str, app: str, model: str, attempt: int,
                                       reason_code: str, reason: str,
                                       raw_opener: str | None, prompt_sha256: str,
                                       exc: Exception) -> None:
    """Best-effort, LOCAL record of one opener-rejection row that FAILED to reach
    self.store.record_opener_rejection, kept so a store write that raises is diagnosable
    afterwards instead of silently lost.

    WHAT THIS WAS BUILT FOR, AND WHAT THAT TURNED OUT NOT TO BE. It shipped 2026-09-14 against a
    diagnosed incident: production's BigQuery `opener_rejections` table sat at ZERO rows across
    its entire history, read at the time as a silent failure on the BigQuery WIRE. That root
    cause is REFUTED. See ops/OPENER-REDESIGN.md's `Addendum -- 2026-09-15 (a)`, which corrects
    `Addendum -- 2026-09-14 (c)` (both entries stay as written; that file is append-only):
      - The "159 billed rejections" that ruled the boring explanation out was never a rejection
        count. It is `COUNT(DISTINCT run_id)` over the WHOLE `spend` table -- 159 distinct run
        ids across 446 spend rows, most of them runs whose openers committed normally -- carried
        out of one query and into a sentence about another.
      - The signature it rested on (runs with spend rows and zero `openers` rows) is the
        ORDINARY STAGED-COMMIT LIFECYCLE, not a lost rejection: maybe_opener's `record_spend`
        is unconditional, its `record_opener` is gated on `if not staged_for_action:`, and a
        staged draft reaches `openers` only through commit_opener (after a landed Like) or
        discard_opener (on an explicit Dislike or refusal). The corrected cohort is 76 runs and
        105 spend rows, 2026-08-14 through 2026-09-09, and 0 of those 76 ever Liked anything --
        precisely the cohort the staging lifecycle is built to leave out of `openers`.
      - The table is not structurally unable to accept rows. A real rejection row from live run
        `aed1a870d740` landed in BigQuery on 2026-09-14 (`hinge`, `gemini-3.5-flash`, attempt 1,
        `unconfirmed_location_followup`) carrying exactly the freeform `reason`/`raw_opener`
        text the wire was suspected over, with no wire fix shipped in between.

    SO THIS IS A STANDING, QUIET DIAGNOSTIC, not evidence of a diagnosed wire failure. It is
    kept deliberately after that correction, because the HAZARD it covers is untouched by it:
    all four rejection call sites below still swallow store exceptions into a `print()` that
    production captures nowhere, and BigQueryStore._flush_table still keeps a failed batch
    buffered quietly until five consecutive failures. Delete this and the next GENUINE wire
    failure produces the same nothing the empty table did. Its silence is also a measurement in
    its own right: no dead-letter file existing beside the configured data_dir is what let
    2026-09-15 (a) state that rejection rows do reach the store. A diagnostic is not refuted by
    reporting "no problem"; that is the only other answer it was ever able to give.

    THIS DOES NOT FIX THE WRITE. It cannot from here: this function runs only after the store
    call has already raised. All this keeps is the evidence a human needs to root-cause an
    occurrence, in a place that does not depend on someone watching the live console at the
    exact moment it happens.

    WHAT WAS DELIBERATELY NOT DONE. No retry of the store call (a write that just failed on
    the wire is not obviously fixed by asking again, and this is a diagnostic path, not a
    delivery-guarantee mechanism). No queue-and-flush-later, no second network call of any
    kind, no alerting/paging integration. This is a local file and nothing else -- the
    smallest thing that preserves the missing evidence without adding a new way for THIS path
    to fail.

    WHAT TO DO WHEN THIS FILE HAS ENTRIES: read `exc_type` / `exc_str` first, then
    `cause_type` / `cause_str` -- BigQuery client-library errors routinely wrap the actual
    reason (a malformed row, an auth/quota failure, a schema mismatch the SDK detected
    client-side) in `exc.__cause__` and leave a generic message on the outer exception, so the
    cause fields are often where the real answer lives, not `exc_str`. `traceback` is the
    full picture if those are not enough. `store_class` says instantly whether this was the
    BigQuery or SQLite backend (this file should, in practice, only ever fill up with the
    former). Entries here are a NEW incident, not a continuation of the empty-table one that
    2026-09-15 (a) closed -- so once the actual wire-level cause is known, record it the way
    this project records every other one: append a fresh dated addendum to
    ops/OPENER-REDESIGN.md stating what was found and what, if anything, ships to fix it. Do
    not edit the 2026-09-06 (c) or 2026-09-14 (c) entries; that file is a historical decisions
    log and corrections go at the end.

    THE flush_every DEPENDENCY (2026-09-17). Every call site below reaches this function from
    inside an `except Exception as store_exc:` wrapped around ITS OWN synchronous
    self.store.record_opener_rejection(...) call, so each one only ever sees a wire failure
    that happens to raise AT THAT CALL. For BigQueryStore, a call only raises when
    storage.bigquery.flush_every makes THIS row the one that triggers a flush -- see
    BigQueryStore._maybe_flush, which flushes only once the buffer reaches flush_every rows.
    config.yaml currently pins flush_every to 1 (every call flushes immediately, so this path
    fires reliably), but at the constructor/`make_store` DEFAULT of 25, most calls merely
    append to the buffer and return cleanly -- the real wire failure then surfaces much later,
    at store.flush(), by which point this function's per-row context (reason_code, reason,
    raw_opener, ...) is long out of scope and unrecoverable from the exception alone.
    supervisor.py's shutdown path now recovers whatever opener_rejections rows are still
    buffered at THAT point too (_deadletter_stranded_opener_rejections, which calls this same
    function per surviving row with branch="shutdown_flush" -- see its own docstring), but
    only for a run that actually reaches its own shutdown flush. A process killed before that
    point (or one that never gets there at all) still loses any buffered-but-unflushed
    rejection with nothing written here. Raising flush_every enlarges that unprotected window;
    config.yaml's own flush_every comment states this same dependency from the operator's side
    -- keep the two in agreement if either changes.

    NEVER RAISES. This function is called from inside an `except Exception as store_exc:`
    block that already gave up on persisting the real record; it must not turn a diagnostic
    for a failed write into a second failure that changes maybe_opener's control flow. The
    entire body below is one try/except that swallows everything, including a failure of the
    swallowing itself doing nothing more than nothing -- no re-raise, no further I/O, no
    second print.
    """
    if not path:
        return
    try:
        cause = exc.__cause__
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "branch": branch,
            "store_class": type(store).__name__,
            "row": {
                "run_id": run_id,
                "app": app,
                "model": model,
                "attempt": attempt,
                "reason_code": reason_code,
                "reason": _deadletter_truncate(reason),
                "raw_opener": _deadletter_truncate(raw_opener),
                "prompt_sha256": prompt_sha256,
            },
            "exc_type": type(exc).__name__,
            "exc_str": _deadletter_truncate(str(exc)),
            "exc_repr": _deadletter_truncate(_safe_repr(exc)),
            "cause_type": type(cause).__name__ if cause is not None else None,
            "cause_str": _deadletter_truncate(str(cause)) if cause is not None else None,
            "traceback": _deadletter_truncate(traceback.format_exc()),
        }
        line = json.dumps(entry, default=str)

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [stripped for stripped in (raw.strip() for raw in f) if stripped]
        except FileNotFoundError:
            lines = []
        lines.append(line)

        # Bound 1: entry count, oldest dropped first.
        if len(lines) > _DEADLETTER_MAX_ENTRIES:
            lines = lines[-_DEADLETTER_MAX_ENTRIES:]
        # Bound 2: total bytes, oldest dropped first, stopping short of an empty file (the
        # newest entry is kept even if it alone exceeds the ceiling -- truncation above
        # already keeps any one entry small, so this is a last-resort backstop, not the
        # normal case).
        while len(lines) > 1 and sum(len(ln) + 1 for ln in lines) > _DEADLETTER_MAX_BYTES:
            lines = lines[1:]

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:  # noqa: BLE001 -- see docstring: a diagnostic for a failed write must
        pass          # never itself raise, so this swallows unconditionally.


def _safe_repr(value: object) -> str:
    """Represent rejected direct-call values without integer-string-limit failures."""
    try:
        return repr(value)
    except (ValueError, OverflowError):
        return f"<{type(value).__name__}>"


def _record_staged_opener(store, record: "_StagedOpenerRecord", pick: OpenerPick, *,
                          profile_id: str, decision: str, decision_source: str,
                          decision_created_at: object | None,
                          pre_send_evidence: dict[str, object] | None = None,
                          profile_key: str = "") -> None:
    """Persist a committed draft with modern lineage when the store declares support for it.

    OpenerService has long accepted small duck-typed stores.  The action-lineage columns are a
    backward-compatible schema extension, not a reason for such a store to turn a real Like into
    a failed worker run.  Signature inspection (rather than a TypeError fallback) keeps a genuine
    store implementation error observable.

    That single inspection now backs THREE independent capability probes: the action-lineage
    columns, `prompt_sha256` (2026-09-05 (b)), and `profile_key` (2026-09-06).  They are probed
    separately, not together, because they are separate extensions that shipped on different
    dates, so a duck-typed store may have any combination of them -- see the rationale comment
    on `accepts_stamp` below, which applies identically to `accepts_profile_key`.
    """
    sink = store.record_opener
    try:
        parameters = inspect.signature(sink).parameters.values()
    except (TypeError, ValueError):
        accepts_lineage = True
        accepts_stamp = True
        accepts_profile_key = True
    else:
        names = {parameter.name for parameter in parameters}
        var_keyword = any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                          for parameter in parameters)
        accepts_lineage = (var_keyword
                           or {"profile_id", "decision", "decision_source",
                               "decision_created_at", "model_item_index"}.issubset(names))
        # ORTHOGONAL to the lineage probe above, deliberately: the action-lineage columns and
        # the prompt-era stamp (prompt_sha256, 2026-09-05 (b)) are separate backward-compatible
        # schema extensions that shipped on different dates, so a duck-typed store can perfectly
        # well have one and not the other. Folding the stamp into `accepts_lineage` would drop
        # it for a store that accepts it but predates the lineage columns, and would send it to
        # a store that has the lineage columns but not the stamp -- a TypeError turning a real
        # Like into a failed worker run, which is exactly what this probe exists to prevent.
        accepts_stamp = var_keyword or "prompt_sha256" in names
        # Same reasoning again, for `profile_key` (ranker/profile_key.py, 2026-09-06): a
        # separate backward-compatible extension that shipped on its own date, so it gets its
        # own probe rather than joining either of the two above.
        accepts_profile_key = var_keyword or "profile_key" in names
    stamp = {"prompt_sha256": record.prompt_sha256} if accepts_stamp else {}
    if accepts_profile_key:
        stamp["profile_key"] = profile_key
    if accepts_lineage:
        sink(record.run_id, record.app, record.model, record.opener,
             record.referenced, record.angle, record.item_description,
             profile_id=profile_id, decision=decision, decision_source=decision_source,
             decision_created_at=decision_created_at,
             model_item_index=(pick.index if pick.index != ITEM_INDEX_ABSENT else None),
             **stamp)
    else:
        sink(record.run_id, record.app, record.model, record.opener,
             record.referenced, record.angle, record.item_description, **stamp)

    evidence_sink = getattr(store, "record_opener_send_evidence", None)
    if pre_send_evidence is not None and callable(evidence_sink):
        evidence_sink(
            record.run_id, record.app, record.opener, profile_id=profile_id,
            decision_source=decision_source, decision_created_at=decision_created_at,
            model_item_index=(pick.index if pick.index != ITEM_INDEX_ABSENT else None),
            evidence=pre_send_evidence,
        )


def _is_invalid_gemini_api_key(exc: Exception) -> bool:
    """True if exc is Gemini's actual invalid/revoked API key signal: HTTP 400
    INVALID_ARGUMENT with a message like "API key not valid. Please pass a valid API key."
    (some responses instead carry the machine-readable reason API_KEY_INVALID). Detected on
    the message/body, deliberately NOT on the bare 400 status: an ordinary malformed-payload
    400 (bad schema, a profile's own corrupt screenshot) must keep its existing 3-strikes
    streak behaviour (see _BAD_REQUEST_LATCH_THRESHOLD) rather than being misdiagnosed as a
    dead key, and a real dead key must not instead take 3 profiles to latch and get reported
    as a generic "malformed request" streak."""
    if not isinstance(exc, GeminiAPIError) or exc.http_code != 400:
        return False
    haystack = f"{exc.message} {exc.status}".lower()
    return "api key not valid" in haystack or "api_key_invalid" in haystack


def _permanent_reason(exc: Exception) -> str | None:
    """Classify a provider error as PERMANENT (will fail identically on every remaining
    profile this run: bad/missing key, no API access at all) vs. TRANSIENT (rate limit,
    5xx, connection blip -- worth retrying on the next profile). Returns an operator-facing
    explanation of what's wrong, or None if exc isn't a recognized permanent failure.
    Deliberately does NOT classify an ordinary HTTP 400 -- see _is_bad_request and
    OpenerService._consecutive_bad_requests, since a single 400 can be caused by that
    profile's own captured photos rather than by a systemic request problem. The one 400
    that IS classified here is an invalid/revoked Gemini API key (see
    _is_invalid_gemini_api_key), which Google reports as 400 INVALID_ARGUMENT rather than
    401 -- that specific failure is exactly as permanent as a 401 would be, so it must
    latch immediately rather than wait out the bad-request streak.

    Deliberately does NOT classify HTTP 404 either, even though a bad/typo'd model id looks
    just as permanent at a glance. It isn't the same kind of permanent: a 404 means ONE
    configured model id is retired or unavailable to this account, and says nothing about
    the rest of the configured cascade -- EMPIRICALLY MEASURED against the real API,
    gemini-2.5-flash and gemini-2.5-flash-lite both still appear in ListModels with
    generateContent support, yet 404 the instant generateContent is actually called (see
    GeminiOpener.preflight's docstring). Latching the WHOLE service disabled over one
    retired model id would have silently killed every other configured model too.
    GeminiOpener.generate() owns this instead: it retires just the offending model and
    cascades to the next configured one on its own (see its 404 handling), so a bare 404
    GeminiAPIError never reaches this function anymore. Only once the ENTIRE cascade has
    fallen through (whatever the mix of per-day/404/transient reasons) does generate() raise
    GeminiCapacityExhausted -- a different exception, handled by maybe_opener()'s own
    isinstance check before it ever gets here, not by this function.

    Gemini is the only opener provider (the Anthropic/Claude path was removed entirely), so
    this only ever classifies GeminiAPIError; anything else isn't a recognized permanent
    failure."""
    if isinstance(exc, GeminiAPIError):
        if _is_invalid_gemini_api_key(exc):
            return "GEMINI_API_KEY is not valid or has been revoked"
        if exc.http_code == 401:
            return "GEMINI_API_KEY is missing or invalid"
        if exc.http_code == 403:
            return "GEMINI_API_KEY does not have permission to use the Gemini API"
        return None
    return None


def _is_bad_request(exc: Exception) -> bool:
    """True if exc is Gemini's HTTP 400 error. Split out from _permanent_reason because a
    400 is only sometimes permanent -- see _BAD_REQUEST_LATCH_THRESHOLD. Callers must check
    _permanent_reason() FIRST: an invalid-key 400 (_is_invalid_gemini_api_key) is also a 400
    here, and must latch on the first occurrence, not join this streak."""
    return isinstance(exc, GeminiAPIError) and exc.http_code == 400


def _display_cost(cost: float | None) -> str:
    """Format known spend without pretending an unpriceable billed call cost $0."""
    return f"${cost:.4f}" if cost is not None else "an unknown amount"


class OpenerService:
    @property
    def last_skip_reason(self) -> str | None:
        """The current caller thread's most recent non-exhausting failure reason.

        One service is shared by concurrent app workers.  Keeping this per thread preserves the
        public attribute used by callers while preventing worker B from overwriting worker A's
        just-returned reason in the small window before A publishes its status.
        """
        return getattr(self._call_outcome, "last_skip_reason", None)

    @last_skip_reason.setter
    def last_skip_reason(self, value: str | None) -> None:
        self._call_outcome.last_skip_reason = value

    @property
    def last_skip_allows_commentless_like(self) -> bool:
        """Whether this thread's latest call permits AUTO to use its existing like decision.

        True only for a Gemini content-policy block.  This structured, thread-local outcome
        keeps worker.py from parsing operator-facing message text and from confusing every
        other per-profile opener failure with the one exception the owner allows.
        """
        return bool(getattr(self._call_outcome, "allows_commentless_like", False))

    @last_skip_allows_commentless_like.setter
    def last_skip_allows_commentless_like(self, value: bool) -> None:
        self._call_outcome.allows_commentless_like = bool(value)

    def __init__(self, client: OpenerClient | None, tracker: CostTracker, store,
                 style: str, max_attempts: int = 5, *,
                 replay_corpus_dir: str | None = None,
                 replay_corpus_max_captures: int = 0,
                 replay_corpus_max_age_days: int = 0,
                 deadletter_path: str | None = None):
        # BUG 2 (adversarial audit): max_attempts=0 (or negative) made range(1, max_attempts+1)
        # empty, so maybe_opener()'s retry loop body never ran at all -- 0 API calls, disabled
        # stayed False, stop_requested stayed False, no reason was ever recorded. That is
        # exactly the silent-forever degradation this whole file exists to prevent: a service
        # that looks perfectly healthy on the hub while producing zero openers, forever.
        # config.validate() already keeps this out of the shipped app, but OpenerService is a
        # public class constructed directly all over the tests (and by any future direct call
        # site), so the invariant belongs here too, not only at the config layer. `bool` is
        # rejected explicitly even though it IS an int subclass in Python (True == 1) --
        # accepting it would silently treat OpenerService(..., max_attempts=True) as
        # max_attempts=1, a confusing coincidence rather than a real configuration.
        if (type(max_attempts) is not int
                or not 1 <= max_attempts <= _MAX_ATTEMPTS):
            raise ValueError(
                f"OpenerService max_attempts must be an int in [1, {_MAX_ATTEMPTS}], "
                f"got {_safe_repr(max_attempts)}")
        if not isinstance(style, str):
            raise ValueError(
                f"OpenerService style must be a string, got {_safe_repr(style)}")
        self.client = client
        self.tracker = tracker
        self.store = store
        self.style = style
        # ops/OPENER-REDESIGN.md 5.2/5.7's replay corpus (opener/replay_corpus.py): a local,
        # gitignored directory every generated ItemRequest is ALSO persisted to, verbatim, so a
        # future prompt revision can be measured against real historical requests without a
        # fresh live batch. OFF (None) unless config.yaml's opener.replay_corpus_enabled turns
        # it on (see supervisor.py's construction site) -- this writes real people's photos to
        # disk, so it is the owner's explicit, opt-in choice, never a silent default. See
        # maybe_opener's own capture call for the best-effort discipline around it.
        self.replay_corpus_dir = replay_corpus_dir
        # Retention bounds for the replay corpus above -- passed straight through to
        # `prune_replay_corpus` after every successful capture (see
        # `_capture_replay_corpus`). 0/0 here (the class default, matching `replay_corpus_dir`'s
        # own None-by-default) means "unlimited unless a caller says otherwise" -- config.yaml's
        # opener.replay_corpus_max_captures / opener.replay_corpus_max_age_days are what actually
        # supply bounded values in a real run (see supervisor.py's construction site); a test or
        # script constructing OpenerService directly is never surprised by pruning it didn't ask
        # for, exactly like replay_corpus_dir itself defaulting to disabled.
        self.replay_corpus_max_captures = replay_corpus_max_captures
        self.replay_corpus_max_age_days = replay_corpus_max_age_days
        # Local dead-letter file for a record_opener_rejection call that raised -- see
        # _write_opener_rejection_deadletter's docstring above for what it captures, and for why
        # it is a STANDING QUIET diagnostic rather than the fix for a diagnosed wire failure (the
        # empty-BigQuery-table root cause it was built for is refuted: ops/OPENER-REDESIGN.md's
        # `Addendum -- 2026-09-15 (a)`). Mirrors
        # replay_corpus_dir's own shape exactly: OFF (None) unless a caller passes a real path,
        # rather than this class silently picking one under the caller's cwd. None is the
        # correct default for a direct construction (tests, scripts) -- it disables the write
        # outright, so nothing is ever touched unless someone deliberately turns it on.
        #
        # WIRED, UNCONDITIONALLY AND WITHOUT A CONFIG FLAG, at supervisor.py's construction site
        # (it passes str(cfg.data_dir / "opener_rejection_deadletter.jsonl") on every startup,
        # following db_file's own convention of deriving from cfg.data_dir rather than a second
        # independent "./data/..." literal). So a real run DOES produce
        # data/opener_rejection_deadletter.jsonl the moment a rejection write raises -- that file
        # is the one place the captured exception text lives. Read it before proposing any wire
        # theory; an ABSENT file is itself a finding, not proof the diagnostic is off, and that is
        # not hypothetical: its absence across every run since it was wired is half of what let
        # ops/OPENER-REDESIGN.md's `Addendum -- 2026-09-15 (a)` establish that rejection rows do
        # reach the store (the live BigQuery row from run aed1a870d740 is the other half).
        # The None default above therefore describes DIRECT construction only (tests,
        # scripts); it is never what a real run gets. (An earlier version of this paragraph said
        # the wiring had not landed yet: it landed in the same commit that added this parameter,
        # and the claim that this diagnostic "is inert in a real run" was false the day it was
        # written -- which is exactly the sentence that would have stopped someone from looking
        # for the one file holding the captured exception.)
        self._deadletter_path = deadletter_path
        # The prompt era every row this service writes is stamped with. Computed ONCE here
        # because all three inputs (this style text, opener.py's _SYSTEM, and _SCHEMA) are
        # fixed for the life of the process -- the style is read from config at supervisor
        # startup and a running process keeps it until restarted -- so re-hashing per profile
        # would buy nothing. See prompt_stamp for what the digest covers and excludes.
        self.prompt_sha256 = prompt_stamp(style)
        # How many times maybe_opener() will re-ask for ONE profile after an unusable
        # (but billed) response before giving up on that profile and stopping the whole
        # run -- see maybe_opener's docstring for the full retry loop. THE OWNER'S RULE:
        # a bad response is retried, never sent and never silently swapped for a bare
        # like; this many consecutive failures in a row is what turns "one-off bad luck"
        # into "something is actually wrong".
        self.max_attempts = max_attempts
        self.disabled = client is None
        self.stop_requested = False     # set once exhaustion stops the whole run
        # Human-readable cause of the FIRST exhaustion (see _exhaust) -- e.g. "run budget
        # reached" or "all configured Gemini models exhausted their free-tier quota; no
        # opener capacity remains". Every worker sharing this service reads the same value,
        # so whichever app happened to trigger it, every worker's hub status shows the one
        # true original cause rather than each other's follow-on symptoms (e.g. a second
        # worker's own maybe_opener() call failing merely because the service is already
        # disabled). None until the service has actually exhausted once.
        self.exhausted_reason: str | None = None
        # Per-thread outcome storage is required even though maybe_opener itself is serialized:
        # after it releases the service lock, another worker can complete and overwrite a shared
        # scalar before the first worker reads its reason. The property above keeps the existing
        # caller API but binds the value to the thread that made the call.
        self._call_outcome = threading.local()
        # Human-readable cause of THIS THREAD'S last maybe_opener() call that returned None WITHOUT
        # exhausting the service -- an unparseable response (OpenerParseError), a per-profile
        # OpenerError, a single sub-latch HTTP 400, or a single sub-latch transient failure
        # (see maybe_opener). Unlike exhausted_reason this deliberately keeps the LAST outcome
        # in each caller thread, not one first-writer-wins run-lifetime value: exhausted_reason answers "why
        # did the service ever stop serving openers at all", which every worker must agree on
        # forever after; this answers "why did THIS specific call just fail", which is allowed
        # (expected, even) to change from one profile to the next while the service otherwise
        # stays healthy. It exists so an AUTO-mode worker that refuses to send a like with no
        # opener (see worker.py's _auto_loop) can report the actual cause instead of a generic
        # "no opener" line. Cleared on every SUCCESSFUL call (see maybe_opener's success
        # branch) so a stale reason from three profiles ago never lingers and gets misread as
        # describing the current one. Deliberately left untouched by the disabled-at-entry
        # short-circuit and by every path that calls _exhaust(): those are global-exhaustion
        # causes, already carried by exhausted_reason, and must not be shadowed by whatever
        # per-call symptom happened to trigger that exhaustion.
        self.last_skip_reason = None
        self.last_skip_allows_commentless_like = False
        self._consecutive_bad_requests = 0   # streak of back-to-back 400s; see
                                              # _BAD_REQUEST_LATCH_THRESHOLD
        # Streak of back-to-back TRANSIENT failures (an unclassified exception, or a
        # per-profile OpenerError -- see _register_transient_failure for exactly what counts
        # and _TRANSIENT_LATCH_THRESHOLD for why); reset on any outcome that proves the
        # opener pipeline actually works.
        self._consecutive_transient_failures = 0
        # Ring buffer of the most recent COMMITTED opener records (AUTO/Training generations,
        # plus Observe suggestions only after a confirmed Like; see recent_openers_snapshot),
        # mirroring self.store.record_opener's permanent per-run record. WHY THIS EXISTS: a
        # bug report that says only "accounted_provider_results=1" cannot
        # tell you whether that opener was about the right photo. Recording the model's own
        # `referenced` string alongside the MODEL ITEM INDEX the pick targeted (OpenerPick.index)
        # is exactly the evidence needed to diagnose an out-of-place opener after the fact --
        # this is that item-binding's own paper trail. (Historic note, 2026-09-02: this comment
        # used to say "whether the call was anchored (see maybe_opener's anchor parameter)".
        # maybe_opener has no such parameter -- the anchor-SCREENSHOT request shape was removed
        # project-wide; what survives is the item binding named here.) Guarded by
        # self._lock like every other piece of mutable state on this instance; maybe_opener
        # already runs its whole body under that lock, so the append there needs no extra
        # locking -- only recent_openers_snapshot (a reader that may run on a different
        # thread, e.g. while building a bug report) takes the lock itself.
        self.recent_openers: deque[dict] = deque(maxlen=_RECENT_OPENERS)
        # Privacy-preserving entropy scratch space.  Advisory suggestions are drafts until a
        # real decision lands, so they must not enter the diagnostic/durable opener trail.
        # The entropy guard still needs to avoid a run of identical outbound openings, though;
        # retain only its normalized leading n-gram, in memory, for this process lifetime.
        # It has no profile id, item, model, full opener, or store write.
        self._recent_opening_ngrams: deque[str] = deque(maxlen=_RECENT_OPENERS)
        # Ring buffer of the most recent REJECTED opener attempts (OpenerParseError), mirroring
        # recent_openers above but for the other outcome. WHY THIS EXISTS: self.store.
        # record_opener_rejection is the durable per-run AUTO record, but a bug report (see
        # bugreport.py's Recent opener rejections section) needs this data WITHOUT a BigQuery
        # round-trip, exactly like recent_openers exists so the report doesn't have to query
        # self.store.record_opener's table either. Appended in the OpenerParseError branch of
        # maybe_opener below for every rejected attempt (including the final one that exhausts
        # retries), regardless of whether persistence succeeds.
        # OpenerError/HTTP-400/transient failures record a durable store row (see their branches
        # below) but deliberately do NOT append here: this buffer's shape is pinned to
        # OpenerParseError's fields (reason_code/raw_opener from a PARSED response), and those
        # three failure kinds have neither.
        # "Attempt" is exact: an unusable ENTROPY REGENERATION draw (see _apply_entropy_guard)
        # is deliberately absent from both this buffer and the store's ledger, because that
        # draw never gated a send, and counting it would inflate the very guard-firing rate
        # these records exist to measure.
        # Guarded by self._lock exactly like recent_openers (maybe_opener already runs its
        # whole body under that lock).
        self.recent_rejections: deque[dict] = deque(maxlen=_RECENT_REJECTIONS)
        # (2026-09-17) Identity of every opener_rejections row THIS service has already
        # dead-lettered via a per-call `except Exception as store_exc:` site below (see
        # _rejection_identity_key's own docstring for exactly what "identity" means here).
        # Read exclusively by is_rejection_row_deadlettered, which
        # supervisor._deadletter_stranded_opener_rejections consults before writing a
        # SECOND dead-letter entry for a row that is still sitting in the store's buffer at
        # shutdown -- a row already covered here got there because a FAILED flush leaves the
        # whole batch buffered (ranker/bigquery_store.py's own _flush_table comment), not
        # because it was never seen. Without this set, that shutdown sweep cannot tell "this
        # row already has its one entry" apart from "this row was never dead-lettered at
        # all", and doubled every count.
        #
        # Unbounded on purpose, unlike recent_openers/recent_rejections above: this is not a
        # display ring buffer, it is the actual dedup key set, and dropping an old entry
        # would silently reopen the double-count bug for a row that happens to still be
        # stuck in the buffer many profiles later. Safe to leave unbounded in practice
        # because the same latch thresholds that bound recent_rejections' realistic size
        # (_BAD_REQUEST_LATCH_THRESHOLD, _TRANSIENT_LATCH_THRESHOLD, max_attempts) bound how
        # many distinct per-call dead-letter writes one run can ever produce before
        # _exhaust() disables the service outright.
        self._deadlettered_rejection_keys: set[tuple] = set()
        self._lock = threading.RLock()

    def _register_transient_failure(self, exc: Exception) -> bool:
        """Count one more consecutive transient-class failure and, once
        _TRANSIENT_LATCH_THRESHOLD is reached, latch the service disabled and return True
        (so the caller can skip its own "swiping without" print -- _exhaust() already prints
        a clear stop line, and printing both would just be noise). Called from both places
        that can produce this class of failure (see maybe_opener): the final catch-all branch
        (an unclassified exception -- timeout, connection blip, an uncaught 5xx) and the
        OpenerError branch (a per-profile content problem, e.g. a refusal or -- see
        opener.py's item B fix -- a corrupt captured photo).

        Records exc so the eventual _exhaust() reason names the LAST failure actually seen,
        not just a bare count -- an operator staring at "3 consecutive opener failures" with
        no detail has to go spelunking in old log lines for the one that matters.
        """
        self._consecutive_transient_failures += 1
        if self._consecutive_transient_failures >= _TRANSIENT_LATCH_THRESHOLD:
            last = f"{type(exc).__name__}: {exc}"
            self._exhaust(
                f"{self._consecutive_transient_failures} consecutive opener calls failed in a "
                f"row without ever producing a usable opener (most recent: {last}) -- this "
                "looks like a systemic problem (a dead network, a provider outage, our own "
                "request building, or a corrupted capture pipeline), not one-off bad luck, so "
                "the run is stopping rather than silently sending bare likes with no opener "
                "for the rest of it",
            )
            return True
        return False

    def _leading_ngram_collision(self, opener: str) -> str:
        """The leading n-gram this opener SHARES with one already produced this run, or "" when
        it opens with words we have not used yet. Pure lookup over a bounded, in-memory n-gram
        scratch buffer; it decides nothing on its own, _apply_entropy_guard below owns every
        consequence.

        Deliberately NOT filtered by app: the fingerprint this guard exists to avoid is "every
        message this account sends opens the same way", and neither a woman comparing
        screenshots with a friend nor an anti-bot heuristic cares which worker produced a line.

        The n-gram is recomputed from each entry's stored text rather than cached in the entry:
        the buffer is bounded at _RECENT_OPENERS, so this is a dozen string compares, and
        keeping that dict schema stable matters more than the microseconds -- bugreport.py
        renders those same entries.
        """
        ngram = _leading_ngram(opener, _ENTROPY_NGRAM_WORDS)
        if not ngram:
            return ""
        if ngram in self._recent_opening_ngrams:
            return ngram
        return ""

    def _record_billed_draw(self, run_id: str, model: str, usage, *, note: str) -> None:
        """Track + persist the spend of a billed opener call whose text is NOT the one being
        sent.

        Spend is billing telemetry, rather than a preference/card decision.  It remains durable
        for every real provider call, including a discarded draft, so run/day budgets never
        under-report actual usage.

        Deliberately does NOT call _exhaust on an unpriceable model, unlike every other
        tracker.record() call site in this file -- the entropy guard's contract is that it can
        never stop a run (see _apply_entropy_guard). Nothing is lost by that: the draw we KEEP
        is recorded a few lines later in maybe_opener's success branch, essentially always from
        the same model id, and that call site does exhaust on the identical KeyError. The only
        gap is the freak case where the cascade changed models between the two draws and only
        the discarded one is unpriceable, which is why that path still prints a loud warning
        rather than passing silently.
        """
        try:
            cost = self.tracker.record(model, usage)
        except KeyError:
            print(f"Warning: no budget.pricing entry for model '{model}'; the {note} was "
                  "billed but its spend cannot be tracked")
            cost = None
        except Exception as e:  # noqa: BLE001
            # Broader than the main path's KeyError-only guard, deliberately: this helper is
            # reached ONLY from the entropy guard, which is holding a perfectly good opener at
            # the time. Letting an accounting oddity (a result with no usage, a tracker that
            # raises something new) escape from here would turn a cosmetic style check into a
            # failed generation, which is precisely the inversion the guard must never cause.
            print(f"Warning: could not track the spend of the {note} "
                  f"({type(e).__name__}: {e})")
            cost = None
        try:
            self.store.record_spend(run_id, model, usage, cost)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: failed to persist the spend record for the {note} "
                  f"({_display_cost(cost)}): {e}")

    def _apply_entropy_guard(self, run_id: str, profile: Profile, result, *,
                             items: ItemRequest | None,
                             should_stop: Callable[[], bool] | None,
                             skip_models: frozenset[str]) -> tuple[object, str, bool]:
        """THE ENTROPY GUARD (ops/OPENER-REDESIGN.md 3.6, whose stated rationale is half stale
        -- read it with the 2026-09-05 addendum at the end of that file). Given a parsed, usable
        opener, return `(result_to_send, colliding_ngram, regenerated)`: either the result
        handed in, or a second draw taken because the first one opened with words already sent
        this run.

        WHY IT EXISTS: shortening the openers compresses the output space, which is what makes
        collisions likely at all (SAY IT ONCE shortens them further), but the style block has
        shipped no examples since the 2026-08-16 de-templating pass, so copying prompt copy is
        NOT the risk. The guard exists because a minimal thinking model collapses onto a
        favorite construction on its own (measured live 2026-08-11: 5 of 5 openers led with the
        same hedge), and because across a burner account
        sending uncapped volume, a run of messages that all open "Based on that X, I'm going to
        guess ..." is simultaneously a bot fingerprint and genuinely embarrassing if two matches
        ever compare screenshots. The check is a plain string comparison of leading n-grams (see
        _leading_ngram_collision) -- no semantics, no classifier, no second judge, and no
        constraint whatsoever on WHAT the model is allowed to write. It only ever asks for
        another draw.

        THREE THINGS IT MUST NEVER DO, and they are the whole design:
          1. Never raise. Every failure of the extra call is swallowed and the original opener
             is kept, because the original was already good enough to send.
          2. Never reject, never _exhaust(), never touch stop_requested or either latch counter.
             A stylistic near-miss is not evidence that the provider, the prompt, or the request
             pipeline is unhealthy, and stopping a run over one would be wildly disproportionate.
          3. Never consume the per-profile attempt budget. Note precisely what that means
             structurally: the call site is lexically INSIDE maybe_opener's
             `for attempt in range(...)` loop, on the success path, but that path returns
             unconditionally, so the extra draw can never produce another `attempt` iteration
             and can never push a profile toward the max_attempts exhaustion that stops the
             whole run. The budget exemption comes from the guard living on a returning path
             and never re-entering the loop, not from its line number. Anyone moving this call
             must preserve that property, not just its position. Doc 3.6 calls the requirement
             out explicitly: a hard rejection here "can stop a run over a stylistic near-miss,
             which is disproportionate".

        ONE extra draw, never a loop: if the second draw collides too, it is ACCEPTED and
        logged. A near-identical opener that gets sent is a much smaller problem than a retry
        storm, a stalled profile, or the sunk cost of a third billed call, and an unbounded
        "keep asking until it is different" loop is a spend hole with no ceiling.
        """
        text = str(getattr(result, "opener", "") or "")
        collision = self._leading_ngram_collision(text)
        if not collision:
            return result, "", False

        # Below here we KNOW we would like another draw. Two reasons not to actually take one,
        # both about spending money we should not spend; in each case the original opener is
        # perfectly sendable, so this degrades to "ship the repetitive one" rather than to any
        # kind of failure.
        if should_stop is not None and should_stop():
            print(f"Opener: this draft repeats an earlier opening this run "
                  f"(\"{collision}\"), but the run is stopping -- sending it as is rather "
                  "than spending on another draw.")
            return result, collision, False
        if self.tracker.budget_reached():
            print(f"Opener: this draft repeats an earlier opening this run "
                  f"(\"{collision}\"), but the run budget is reached -- sending it as is "
                  "rather than spending on another draw.")
            return result, collision, False

        print(f"Opener: this draft opens with words already sent this run (\"{collision}\"); "
              "asking once for a different opening. This extra call is deliberately EXEMPT "
              "from the per-profile attempt budget -- see _apply_entropy_guard.")
        # Written to be read by the MODEL, so: no em dash, no hyphen, plain ASCII, and no
        # suggestion that the previous draft was bad. It was not; it was fine and merely
        # familiar. Telling the model it was "rejected" (the wording the real retry path uses,
        # where the text genuinely was unusable) would push it to change the wrong things.
        retry_hint = (
            f"Your previous draft was fine, but it opened with the same words as an opener "
            f"already sent to someone else recently: \"{collision}\". Write a different "
            f"opener for this profile that does not open with those words. Every other rule "
            f"is unchanged."
        )
        try:
            # `items` rides along because the second draw is the
            # SAME request with a different retry_hint. Dropping it here would silently switch
            # the regeneration to the raw-frame shape -- the model would be answering about a
            # different set of images, in a different index space, and whichever draw survived
            # would carry the other one's numbering. The entropy guard is about the opening
            # WORDS and must change nothing else about the request. `run_id` rides along too,
            # unchanged from this method's own parameter -- this extra draw belongs to the same
            # run as the draft it is replacing, so any cascade print it triggers must carry the
            # same `Run {run_id}: ` tag, not go unattributed.
            generate_kwargs = dict(items=items, should_stop=should_stop,
                                   skip_models=skip_models, run_id=run_id)
            second = self.client.generate(profile, self.style, retry_hint=retry_hint,
                                          **generate_kwargs)
        except OpenerParseError as e:
            # Billed but unusable. Record the spend (real money, see _record_billed_draw) and
            # keep the original opener. Deliberately NOT written to the opener_rejections
            # ledger: that table answers "how often do the deterministic SEND guards reject an
            # attempt", and this draw was never gating a send -- counting it there would inflate
            # exactly the statistic it exists to measure. The console line below plus the
            # `entropy_collision` field on the ring-buffer entry are this path's paper trail.
            self._record_billed_draw(run_id, e.model, e.usage,
                                     note="rejected entropy regeneration draw")
            print(f"Opener: the entropy regeneration came back unusable ({e}); keeping the "
                  "original opener, which was already good enough to send.")
            return result, collision, False
        except Exception as e:  # noqa: BLE001
            # Anything else at all: a stop signal mid-cascade (OpenerAborted), a per-profile
            # OpenerError, an HTTP failure, capacity exhaustion, a fake client that cannot cope
            # with a second call. NONE of them move a latch counter, set last_skip_reason, or
            # exhaust: we are holding a perfectly good opener, so there is no failure to report
            # to anyone -- only a cosmetic improvement we did not get.
            print(f"Opener: the entropy regeneration failed ({type(e).__name__}: {e}); "
                  "keeping the original opener, which was already good enough to send.")
            return result, collision, False

        second_text = str(getattr(second, "opener", "") or "")
        if not second_text:
            # This second call was real and billed, exactly like the first -- record it before
            # anything else. Defensive in practice: a real GeminiOpener._parse() raises
            # REASON_EMPTY_AFTER_SANITIZE on an empty opener, so only a fake or future client
            # can land `second` here with blank text. But _record_billed_draw's own invariant
            # ("every real call being recorded exactly once") is unconditional, not contingent
            # on which client made the call, so it is recorded anyway.
            self._record_billed_draw(run_id, getattr(second, "model", "") or "",
                                     getattr(second, "usage", None),
                                     note="empty entropy regeneration draw")
            # A client that returns a result with no text at all would turn a cosmetic guard
            # into a commentless like, the one outcome this whole file exists to prevent.
            print("Opener: the entropy regeneration returned no opener text; keeping the "
                  "original opener.")
            return result, collision, False

        # ITEM BINDING OUTRANKS THE OPENING WORDS. This guard is about the first few WORDS and
        # nothing else, but the draw it hands back also carries the item number the worker will
        # target. A second draw whose item_index came back ABSENT (opener.py collapses any
        # out-of-range/odd number to ABSENT rather than raising, exactly as _parse documents),
        # or that counts a different index_space than the draft being discarded, would trade a
        # sendable AND targetable opener for one worker.py can only answer by setting
        # stop_reason and stop_event -- i.e. the guard would stop the run over a stylistic
        # near-miss through a layer below it, which is precisely what invariant 2 above forbids.
        # Keeping the first draft keeps its TEXT and its INDEX together, so this is the opposite
        # of substituting the liked item. Same shape as the empty-text branch above: record the
        # discarded second draw's spend here, and leave the survivor's own recording to
        # maybe_opener's normal post-guard site.
        result_index = getattr(result, "item_index", ITEM_INDEX_ABSENT)
        second_index = getattr(second, "item_index", ITEM_INDEX_ABSENT)
        if result_index != ITEM_INDEX_ABSENT and (
                second_index == ITEM_INDEX_ABSENT
                or getattr(second, "index_space", "") != getattr(result, "index_space", "")):
            self._record_billed_draw(run_id, getattr(second, "model", "") or "",
                                     getattr(second, "usage", None),
                                     note="entropy regeneration lost its item binding")
            print("Opener: the entropy regeneration came back without the item binding the "
                  "original draft had (item_index "
                  f"{result_index} in {getattr(result, 'index_space', '') or 'an unstated space'}"
                  f" -> {second_index} in "
                  f"{getattr(second, 'index_space', '') or 'an unstated space'}); discarding "
                  "the redraw and keeping the original opener, which was already good enough "
                  "to send and still knows which item it is about.")
            return result, collision, False

        # The first draft is being thrown away, but it was BILLED. Record it here so the money
        # is accounted for exactly once; maybe_opener records the surviving draw itself.
        self._record_billed_draw(run_id, getattr(result, "model", "") or "",
                                 getattr(result, "usage", None),
                                 note="entropy regenerated opener draft")
        again = self._leading_ngram_collision(second_text)
        if again:
            # ONE extra draw only -- accept and be loud about it. (A second draw that repeats
            # the FIRST draft's opening is covered by this same check: the first draft collided
            # with the buffer, so anything sharing its n-gram collides with the buffer too.)
            print(f"Opener: the regenerated opener still opens with \"{again}\"; SENDING it "
                  "anyway. The guard asks once and never loops -- a near-repeat that goes out "
                  "is a smaller problem than a retry storm or an unbounded spend.")
        return second, collision, True

    def _capture_replay_corpus(self, items: ItemRequest) -> None:
        """Best-effort persistence of this profile's opener evidence to the local replay
        corpus (opener/replay_corpus.py, ops/OPENER-REDESIGN.md 5.2/5.7). The corpus keeps the
        exact model-visible request inputs (name, numbered items, truncation) and retains
        withheld unnumbered context crops as a forensic superset; current Gemini generation
        and replay deliberately omit that context. Thus a future prompt revision can replay
        real historical requests without a fresh live batch. A no-op unless
        `self.replay_corpus_dir` is set (opener.replay_corpus_enabled in config.yaml, DEFAULT
        DISABLED -- see OpenerService.__init__).

        REAL PEOPLE'S PHOTOS: `write_replay_capture` itself already never raises (every failure
        comes back as `ReplayWriteResult(ok=False, ...)`, see that function's own SAFE TO FAIL
        section) -- this wraps the call in try/except anyway, matching every other best-effort
        telemetry write in this file (record_spend, record_opener, record_opener_rejection),
        so a caller here never has to reason about whether THIS particular best-effort write
        follows the rule differently from the others.

        RETENTION runs right here too, AFTER a successful capture only: a failed write means
        there is nothing new on disk to bound, and re-pruning on every failed attempt would just
        be wasted directory-walk work for no benefit. `prune_replay_corpus` never raises on its
        own (same SAFE TO FAIL contract as `write_replay_capture`, see that function's
        docstring) but gets the exact same defensive try/except wrapping as the write above
        anyway, for the same reason: this call site must never be the one place that assumes a
        best-effort helper's own contract instead of enforcing it locally too.
        """
        if not self.replay_corpus_dir:
            return
        captured_ok = False
        try:
            result = write_replay_capture(
                self.replay_corpus_dir, items=items.items, name=items.name,
                context=items.context, truncated=items.truncated,
                prompt_sha256=self.prompt_sha256)
            captured_ok = result.ok
            if not result.ok:
                print(f"Warning: opener replay-corpus capture failed: {result.error}")
        except Exception as exc:  # noqa: BLE001 -- telemetry must never break opener generation
            print(f"Warning: opener replay-corpus capture raised unexpectedly: {exc}")

        if not captured_ok:
            return
        try:
            prune_result = prune_replay_corpus(
                self.replay_corpus_dir,
                max_captures=self.replay_corpus_max_captures,
                max_age_days=self.replay_corpus_max_age_days)
            if not prune_result.ok:
                print(f"Warning: opener replay-corpus prune failed: {prune_result.error}")
            elif prune_result.removed:
                print(f"Opener: replay-corpus prune removed {len(prune_result.removed)} "
                      f"capture(s), kept {prune_result.kept}, from {self.replay_corpus_dir}")
        except Exception as exc:  # noqa: BLE001 -- telemetry must never break opener generation
            print(f"Warning: opener replay-corpus prune raised unexpectedly: {exc}")

    def maybe_opener(self, run_id: str, app: str, profile: Profile, *,
                      items: ItemRequest | None = None,
                      should_stop: Callable[[], bool] | None = None,
                      stage: bool = False) -> "OpenerPick | None":
        """Return an OpenerPick (text + the MODEL ITEM INDEX the opener is about and the item
        to like -- 1-based, see OpenerPick), or None (disabled / budget
        reached / out of credit / permanent provider error / every retry attempt used up /
        a per-profile OpenerError / a single sub-latch 400 or transient failure / the run
        stopping via should_stop).

        items (default None): ops/OPENER-REDESIGN.md 5.2/5.7's item-crop request shape -- one
        cropped image per numbered profile item, her name as text, and the capture's truncation
        flag, built by opener.ItemRequest.from_profile() from what the driver enumerated.
        Unnumbered context crops remain attached to the value only for replay and forensic
        diagnostics; they are deliberately withheld from Gemini generation. When present it
        REPLACES profile.photos as the model's view of her, which is the whole point: image k
        IS item k, so the number the model answers with means something. Forwarded to
        self.client.generate(...) unconditionally below: a client that silently dropped this
        kwarg would fall back
        to sending raw scroll frames, where one card appears in several frames and one frame can
        hold two cards, and the returned item number would be confidently meaningless. A
        TypeError out of the call is the correct failure there.

        Every caller passes `items`; the same crops are reused verbatim on every attempt for
        this profile, so only `retry_hint` changes after a rejected response.

        run_id: forwarded to self.client.generate(...) on every attempt (and to the entropy
        guard's own extra draw, which reuses this same run_id -- see _apply_entropy_guard) so
        GeminiOpener can prefix its non-fatal cascade print()s with `Run {run_id}: `. This is
        the ONLY thing run_id does for generate() -- it plays no role in this method's own
        retry/latch/exhaustion logic -- but it is what lets bugreport.py's completion verdict
        tell THIS run's recovered provider failures apart from an earlier run's, sharing the
        same hub process's one never-cleared log ring (see bugreport._lines_for_run and
        _run_completion_assessment_md).

        should_stop (BUG 1, adversarial audit): a cheap, non-blocking "is the run stopping?"
        check -- in practice worker.py's threading.Event.is_set for the shared stop flag.
        Threaded into every self.client.generate(...) call below (so GeminiOpener can abort
        BETWEEN models mid-cascade -- see its own should_stop docstring) and also checked
        directly by this method BETWEEN RETRY ATTEMPTS, since a fake/simple OpenerClient
        cannot be relied on to implement its own should_stop cascade the way GeminiOpener
        does. Either check raising/observing a stop returns None IMMEDIATELY: no further
        retries, no _exhaust() call, and neither latch counter moves -- a deliberate
        shutdown is not evidence the provider or this service is unhealthy, and treating it
        as one would poison exhausted_reason with a misleading "provider failed" story for
        an operator who just clicked Stop. Pre-fix, nothing on this path ever consulted a
        stop signal at all: this method holds self._lock across every attempt (up to
        max_attempts), each of which could cascade across every configured model at
        request_timeout_s apiece, so a Stop click could be silently ignored for as long as
        max_attempts * len(models) * request_timeout_s (5 * 7 * 90s = 3150s, ~52 minutes,
        against the shipped config) while this service's lock stayed held, blocking every
        other worker sharing it.

        THE OWNER'S RULE: a commentless like is normally not acceptable. If the AI's response
        is bad, redo the prompt -- spending more usage is fine. The narrow exception is a
        provider safety-policy block in AUTO: the ranker has already decided to like that
        profile, so the worker may carry out that decision without a comment. Most billed responses that
        didn't parse into a usable opener (OpenerParseError) are RETRIED for THIS profile,
        up to self.max_attempts times, each retry telling the model what was wrong with its
        previous attempt (retry_hint) so a re-ask can actually do better -- this is worth
        doing because a parse failure is usually stochastic (the same request often
        succeeds on a re-ask), unlike a deterministic failure (see below). Gemini safety
        blocks are deliberately not resent: the current profile is skipped with a visible
        reason while the next profile remains eligible. Only once
        max_attempts consecutive attempts have ALL failed does that stop looking like
        one-off bad luck and start looking like something actually broken (a bad prompt, a
        degenerate model, a schema bug) -- at that point the whole run stops via
        _exhaust(), exactly like the owner's instruction: "if after 5 attempts it's still a
        bad response, stop the automation -- that means something is wrong."

        ONE further billed call can happen per profile, and it is NOT an attempt: after a
        successful parse, the entropy guard (ops/OPENER-REDESIGN.md 3.6, implemented in
        _apply_entropy_guard) may ask once more when the opener opens with the same words as
        one already produced this run. It sits deliberately OUTSIDE the attempt loop, so a
        stylistic near-miss can never push a profile toward the max_attempts exhaustion that
        stops the run; it never raises, never rejects, and never latches anything. Its spend
        is recorded either way. See _apply_entropy_guard for the full reasoning.

        Every OTHER kind of per-call failure is deliberately NOT retried:
          - OpenerError (not a parse error) is almost always deterministic -- e.g. a
            corrupt captured photo that fails to decode while fitting the request to the
            inline-size budget (see opener.py's _fit_images_to_budget). Re-asking would
            resend the exact same bad payload and fail identically, so retrying it here
            would be pure wasted spend; this SERVICE just returns None and stays enabled
            for the next profile.
          - An HTTP-level failure (429/404/5xx/a transport error) is not retried HERE at
            all: GeminiOpener.generate() already cascades those across every configured
            model internally before it ever raises to this method (see its class
            docstring) -- by the time an exception reaches here, the whole model cascade
            has already run its course for this call, so retrying THAT would just repeat
            work generate() already did.

        The budget check, every provider call for this profile (including retries), and
        spend recording are all serialized under one lock acquisition, so concurrent app
        workers cannot interleave with a retry sequence and overspend the shared run
        budget. The budget is also re-checked after EVERY attempt's spend is recorded
        (not just once, before the first attempt): the owner's retry rule never overrides
        the run budget, so a budget crossed mid-retry stops the retries immediately with
        "run budget reached" rather than continuing to spend on attempt after attempt.

        Every path that returns None because of a global exhaustion (budget reached, a
        latch tripped, a permanent provider error, GeminiCapacityExhausted, every retry
        attempt failing) goes through _exhaust(), which records exhausted_reason and, for
        AUTO, sets stop_requested. Every OTHER path that returns None (a per-profile OpenerError, or
        a single sub-latch 400/transient failure) instead records last_skip_reason: this
        call failed, but the service is still enabled and expects to succeed again next
        profile. Callers that must never send a like with no opener (see worker.py's
        _auto_loop) need to distinguish the two, since only the second kind leaves
        self.disabled False. A should_stop-triggered None joins the SECOND group (per-call,
        service stays enabled): the run stopping says nothing about whether the opener
        pipeline itself is healthy, so it must not be reported or counted like one.
        """
        with self._lock:
            # Per-call permission, never sticky. A successful call or any unrelated failure
            # after a safety-blocked profile must restore the ordinary no-bare-like rule.
            self.last_skip_allows_commentless_like = False
            if self.disabled:
                return None
            if self.tracker.budget_reached():
                self._exhaust("run budget reached")
                return None

            effective_max_attempts = self.max_attempts
            # Which models have already produced an unusable response FOR THIS PROFILE, so a
            # retry can steer GeminiOpener's cascade away from re-hitting the same (often
            # scarcest-quota) model that just failed -- see opener.py's generate() skip_models
            # docstring. Local to this call/profile by construction (a fresh call stack frame
            # per maybe_opener() invocation): it can never leak into the NEXT profile's call,
            # which gets its own empty set here.
            failed_models: set[str] = set()

            # ops/OPENER-REDESIGN.md 5.2/5.7's replay corpus: capture the EXACT request inputs
            # right here, once per profile call, before any attempt is made -- not per retry
            # (the same `items` is reused verbatim across every attempt for this profile, see
            # this method's own docstring, so capturing again on a retry would just re-detect
            # "already captured" via write_replay_capture's own content-derived idempotency).
            # Placed AFTER the disabled/budget-reached checks above: a call that returns None
            # there never builds a request at all, so capturing one would record content that
            # was never actually about to be sent.
            if items is not None:
                self._capture_replay_corpus(items)

            retry_hint = ""   # "" means first attempt; a retry fills this in below
            for attempt in range(1, effective_max_attempts + 1):
                if should_stop is not None and should_stop():
                    # Checked BETWEEN retry attempts (including before the very first one),
                    # independent of whatever the client itself does internally -- a simple/
                    # fake OpenerClient cannot be relied on to implement GeminiOpener's own
                    # per-model should_stop cascade, so this method's own retry loop must not
                    # depend on that. Returns immediately: no _exhaust(), no latch movement,
                    # no retry -- this is a deliberate shutdown, not a failure of any kind
                    # (see this method's should_stop docstring paragraph).
                    self.last_skip_reason = (
                        "opener call skipped before this attempt: the run is stopping "
                        "(should_stop signaled), not a provider failure"
                    )
                    return None
                try:
                    # items is forwarded unconditionally, and the
                    # cost of getting it wrong is higher: a client that quietly ignored it would
                    # send profile.photos instead, and the model's item_index would then count
                    # scroll frames while every consumer downstream reads it as an item number
                    # (doc 5.2/5.7). Loud TypeError over silent renumbering.
                    generate_kwargs = dict(
                        items=items,
                        should_stop=should_stop,
                        skip_models=frozenset(failed_models),
                        run_id=run_id,
                    )
                    result = self.client.generate(profile, self.style, retry_hint=retry_hint,
                                                  **generate_kwargs)
                except OpenerAborted as e:
                    # The client itself aborted mid-call (e.g. GeminiOpener's cascade caught a
                    # stop signal BETWEEN models, after already issuing at least one request
                    # this attempt). Same contract as the loop-top check above: return
                    # immediately, no retry, no _exhaust(), no latch movement -- see BUG 1's
                    # fix rationale and OpenerAborted's own docstring for why this must never
                    # be treated like an ordinary OpenerError.
                    self.last_skip_reason = f"opener call aborted mid request: {e}"
                    return None
                except OpenerParseError as e:
                    # The call reached the API and was billed (it returned usage) even though
                    # the body didn't parse into a usable opener. Record the spend like a
                    # normal call -- EVERY attempt is a real billed call, retries included, so
                    # this happens on each pass through this branch, not just the first.
                    # Reaching the API at all (billed usage came back) proves the request shape
                    # itself is fine, so this breaks any streak of bad-request failures too --
                    # and, for the same reason, any streak of transient failures: a real
                    # end-to-end round trip through transport/auth/request-building just
                    # succeeded, which is exactly the "pipeline works" signal that latch resets
                    # on (see _TRANSIENT_LATCH_THRESHOLD).
                    self._consecutive_bad_requests = 0
                    self._consecutive_transient_failures = 0
                    # D: this model just produced an unusable response FOR THIS PROFILE -- steer
                    # any further retry attempt's cascade away from re-hitting it first (see
                    # opener.py's generate() skip_models docstring). Reset to an empty set at the
                    # top of every maybe_opener() call, so this never survives past the current
                    # profile.
                    failed_models.add(e.model)
                    # Record WHY this attempt produced no opener before any of the exhaustion
                    # branches below have a chance to overwrite it with a global cause -- this is
                    # a per-call reason, not a global one (see last_skip_reason's docstring), and
                    # must describe what actually happened here (an unparseable body) regardless
                    # of whether this also turns out to be the attempt that exhausts the run.
                    self.last_skip_reason = (
                        f"opener response could not be parsed into a usable opener "
                        f"(attempt {attempt}/{effective_max_attempts}): {e}"
                    )
                    try:
                        cost = self.tracker.record(e.model, e.usage)
                    except KeyError:
                        # Same guard as the success path: the API echoed a model with no
                        # budget.pricing entry, so this (real, billed) spend can't be tracked.
                        self._exhaust(f"no budget.pricing entry for model '{e.model}'; "
                                      "spend can no longer be tracked")
                        cost = None
                    try:
                        self.store.record_spend(run_id, e.model, e.usage, cost)
                    except Exception as store_exc:  # noqa: BLE001
                        print(f"Warning: failed to persist opener spend record "
                              f"({_display_cost(cost)}): {store_exc}")
                    attempt_outcome = (
                        "safety-blocked" if e.reason_code in
                        (REASON_PROMPT_BLOCKED, REASON_RESPONSE_BLOCKED) else "unparseable"
                    )
                    result_noun = "provider result" if attempt_outcome == "safety-blocked" else "response"
                    print(f"Opener attempt {attempt}/{effective_max_attempts}: {attempt_outcome} "
                          f"{result_noun} (billed {_display_cost(cost)}): {e}")
                    # Durable record of EVERY rejected attempt, not just successes (see
                    # ranker/bigquery_store.py's opener_rejections table and opener.py's
                    # OpenerParseError docstring for reason_code/raw_opener semantics).
                    # Recorded HERE, before the retry/exhaust decision below, so the FINAL
                    # attempt -- the one that calls _exhaust() and stops the run -- is
                    # captured too, not only the attempts that go on to a further retry.
                    # Without this, only SUCCEEDED openers were ever persisted, so there was
                    # no way to ask how often any guard fires or whether the deterministic
                    # detectors (scaffolding, sentence-count, ...) are too strict or too
                    # loose. Guarded exactly like record_spend just above: a store outage
                    # must never take down opener generation, the one invariant this whole
                    # file exists to protect.
                    try:
                        self.store.record_opener_rejection(
                            run_id, app, e.model, attempt, e.reason_code, str(e),
                            e.raw_opener, prompt_sha256=self.prompt_sha256)
                    except Exception as store_exc:  # noqa: BLE001
                        print(f"Warning: failed to persist opener rejection record: {store_exc}")
                        _write_opener_rejection_deadletter(
                            self._deadletter_path, "parse_error", self.store,
                            run_id=run_id, app=app, model=e.model, attempt=attempt,
                            reason_code=e.reason_code, reason=str(e), raw_opener=e.raw_opener,
                            prompt_sha256=self.prompt_sha256, exc=store_exc)
                        # (2026-09-17) Remember this row so supervisor.py's shutdown sweep
                        # (_deadletter_stranded_opener_rejections) does not write a SECOND
                        # entry for it -- the failed store call above leaves this exact row
                        # stuck in the store's buffer, same args as just passed above.
                        self._deadlettered_rejection_keys.add(_rejection_identity_key(
                            run_id=run_id, app=app, model=e.model, attempt=attempt,
                            reason_code=e.reason_code, reason=str(e), raw_opener=e.raw_opener,
                            prompt_sha256=self.prompt_sha256))
                    # In-memory mirror of the row just above, independent of the store call's
                    # success -- see recent_rejections' docstring in __init__ for why this
                    # exists (the bug report's Recent opener rejections section reads this,
                    # not the store, exactly like recent_openers/recent_openers_snapshot).
                    # Unconditional now too, for the same reason as the store call just above.
                    self.recent_rejections.append({
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "app": app,
                        "model": e.model,
                        "attempt": attempt,
                        "reason_code": e.reason_code,
                        "reason": str(e),
                        "raw_opener": e.raw_opener,
                        "prompt_sha256": self.prompt_sha256,
                    })

                    if self.disabled:
                        # _exhaust() already ran above (the unpriceable-model guard) -- the
                        # service is globally disabled now, so there is nothing left to retry.
                        return None
                    if self.tracker.budget_reached():
                        # Re-checked here, not just before the first attempt: the retry rule
                        # never overrides the run budget, so a cap crossed by THIS attempt's
                        # spend must stop further retries immediately.
                        self._exhaust("run budget reached")
                        return None
                    if e.reason_code in (REASON_PROMPT_BLOCKED, REASON_RESPONSE_BLOCKED):
                        # Gemini identified this as a content-policy block, not malformed JSON
                        # or a stochastic bad draw.  Retrying another model with the same
                        # profile captures would simply resubmit the content Gemini withheld;
                        # do not evade that decision by rewriting, stripping, or isolating the
                        # profile.  The blocked attempt has already been billed and recorded
                        # above. Keep the service healthy for the next profile: the ranker has
                        # already made the like decision, so this one explicit outcome permits
                        # a commentless like.
                        blocked_stage = (
                            "the request" if e.reason_code == REASON_PROMPT_BLOCKED
                            else "the generated response"
                        )
                        self.last_skip_reason = (
                            f"Gemini's safety policy withheld {blocked_stage} for this profile; "
                            "it was not retried and no opener will be used for it. AUTO may "
                            "proceed with its existing like decision without a comment. Future "
                            "profiles remain eligible."
                        )
                        self.last_skip_allows_commentless_like = True
                        print(
                            "Opener: Gemini safety policy withheld content for this profile; "
                            "not retrying the same captured content. AUTO will proceed with its "
                            "existing like decision without a comment; future profiles remain "
                            "eligible."
                        )
                        return None
                    if attempt >= effective_max_attempts:
                        # Every attempt for this profile came back unusable. Per the owner's
                        # explicit rule, that many failures in a row is no longer one-off bad
                        # luck -- it means something is actually wrong -- so the whole run
                        # stops rather than ever falling back to a bare/commentless like.
                        self._exhaust(
                            f"{attempt} consecutive AI opener attempts for this profile were "
                            f"all rejected as unusable (most recent: {e}) -- retrying a bad "
                            "response is expected, but this many failures in a row means "
                            "something is actually wrong (a broken prompt, a degenerate model, "
                            "a schema bug), so opener generation is being disabled rather than "
                            "keep spending on retries that keep failing",
                        )
                        return None

                    # Retry: tell the model exactly what was wrong with its last attempt so a
                    # re-ask can actually fix it, rather than blindly repeating the same draw.
                    retry_hint = (
                        f"Your previous attempt was rejected and NOT sent: {e}. Write a new "
                        "opener that fixes this."
                    )
                    print(f"Opener: retrying this profile "
                          f"(attempt {attempt + 1}/{effective_max_attempts})...")
                    continue
                except OpenerError as e:
                    # Same class of per-profile failure, but raised without usage attached, so
                    # there is no billed amount to record -- just skip the opener this once.
                    # NOT retried, unlike OpenerParseError above: this is almost always
                    # deterministic (e.g. a corrupt photo that fails to decode -- see
                    # opener.py's _fit_images_to_budget), so a re-ask would resend the exact
                    # same bad payload and fail identically, making a retry pure wasted spend.
                    self._consecutive_bad_requests = 0
                    # Deliberately COUNTS toward the transient-failure latch rather than resetting
                    # it -- the opposite of OpenerParseError just above. Unlike OpenerParseError,
                    # reaching this branch does NOT prove the request reached (or was billed by)
                    # the provider: opener.py can raise OpenerError before any network call at all
                    # (e.g. its _fit_images_to_budget step failing to decode a corrupt photo -- see
                    # item B), so it proves nothing about whether the pipeline is healthy. A single
                    # corrupt screenshot is normal and must not escalate on its own -- but a
                    # SYSTEMICALLY corrupt capture pipeline (every `adb screencap` producing
                    # truncated PNGs, say) would surface as a long run of exactly this error, one
                    # per profile, and that is precisely the silent-forever degradation this latch
                    # exists to catch.
                    if self._register_transient_failure(e):
                        # Latched permanent by _register_transient_failure -- exhausted_reason
                        # (a global cause) now names this, so last_skip_reason (a per-call cause)
                        # is deliberately left alone rather than shadowing it with this profile's
                        # individual symptom.
                        pass
                    else:
                        self.last_skip_reason = f"opener call failed for this profile: {e}"
                        print(f"Opener: {e}; swiping without an opener for this profile.")
                    # Durable record even though this is not an OpenerParseError -- this branch
                    # was previously one of THREE failure kinds that left opener_rejections at
                    # zero rows forever (2026-09-06; see ops/OPENER-REDESIGN.md's dated
                    # addendum). No `model` is known here: as the comment above explains,
                    # opener.py can raise this before any provider request is ever built, so ""
                    # is recorded rather than a guess. Guarded exactly like the OpenerParseError
                    # branch's own store call above: a store outage must never take down opener
                    # generation.
                    try:
                        self.store.record_opener_rejection(
                            run_id, app, "", attempt, REASON_OPENER_ERROR, str(e), None,
                            prompt_sha256=self.prompt_sha256)
                    except Exception as store_exc:  # noqa: BLE001
                        print(f"Warning: failed to persist opener rejection record: {store_exc}")
                        _write_opener_rejection_deadletter(
                            self._deadletter_path, "opener_error", self.store,
                            run_id=run_id, app=app, model="", attempt=attempt,
                            reason_code=REASON_OPENER_ERROR, reason=str(e), raw_opener=None,
                            prompt_sha256=self.prompt_sha256, exc=store_exc)
                        # (2026-09-17) See the parse_error branch's own comment above --
                        # same reasoning, same shutdown-sweep consumer.
                        self._deadlettered_rejection_keys.add(_rejection_identity_key(
                            run_id=run_id, app=app, model="", attempt=attempt,
                            reason_code=REASON_OPENER_ERROR, reason=str(e), raw_opener=None,
                            prompt_sha256=self.prompt_sha256))
                    return None
                except Exception as e:  # noqa: BLE001
                    # HTTP-level / provider-level failures land here, and none of them are
                    # retried BY THIS METHOD: GeminiOpener.generate() already cascades a
                    # 429/404/5xx/transport failure across every configured model internally
                    # before it ever raises (see its class docstring) -- so by the time an
                    # exception reaches this branch, the whole model cascade already ran its
                    # course for this call, and looping again here would just repeat work
                    # generate() already did rather than genuinely try something new.
                    if isinstance(e, GeminiCapacityExhausted):
                        # Use the exception's own text rather than a generic line: GeminiOpener
                        # builds it per model and distinguishes "every model is out of DAILY
                        # quota, nothing works until the midnight Pacific reset" from "the
                        # cascade fell through with a transient per-minute cap in it, so a
                        # restart shortly may just work". This string becomes the hub's stop
                        # reason, and those two situations need opposite operator responses.
                        self._exhaust(str(e))
                        return None
                    # Permanent classification MUST run before the bad-request streak counter
                    # below: an invalid/revoked Gemini API key surfaces as an HTTP 400 (see
                    # _is_invalid_gemini_api_key), and it must latch immediately on the first
                    # occurrence like a 401 would, not spend 2 more profiles feeding the
                    # malformed-request streak and then get reported with the wrong diagnosis.
                    reason = _permanent_reason(e)
                    if reason is not None:
                        # Permanent (bad key/model): retrying it per-profile is pure waste, so
                        # stop attempting openers for the rest of the run instead.
                        self._exhaust(reason)
                        return None
                    if _is_bad_request(e):
                        # Only latch permanent after a STREAK of consecutive 400s -- see
                        # _BAD_REQUEST_LATCH_THRESHOLD's docstring for why a lone 400 must not
                        # kill openers for the whole run (it can be that profile's own bad
                        # capture, not a systemic request problem).
                        # Durable record for EVERY 400, latched or not -- previously one of
                        # three failure kinds that left opener_rejections at zero rows forever
                        # (2026-09-06; see ops/OPENER-REDESIGN.md's dated addendum). Placed
                        # before the latch decision below so it fires regardless of which of
                        # that decision's two `return None`s this call takes. No `model` is
                        # known: this is an HTTP-level rejection of the request, not a parsed
                        # provider response. Guarded like every other new call site here: a
                        # store outage must never take down opener generation.
                        try:
                            self.store.record_opener_rejection(
                                run_id, app, "", attempt, REASON_BAD_REQUEST, str(e), None,
                                prompt_sha256=self.prompt_sha256)
                        except Exception as store_exc:  # noqa: BLE001
                            print(f"Warning: failed to persist opener rejection record: {store_exc}")
                            _write_opener_rejection_deadletter(
                                self._deadletter_path, "bad_request", self.store,
                                run_id=run_id, app=app, model="", attempt=attempt,
                                reason_code=REASON_BAD_REQUEST, reason=str(e), raw_opener=None,
                                prompt_sha256=self.prompt_sha256, exc=store_exc)
                            # (2026-09-17) See the parse_error branch's own comment far
                            # above -- same reasoning, same shutdown-sweep consumer.
                            self._deadlettered_rejection_keys.add(_rejection_identity_key(
                                run_id=run_id, app=app, model="", attempt=attempt,
                                reason_code=REASON_BAD_REQUEST, reason=str(e), raw_opener=None,
                                prompt_sha256=self.prompt_sha256))
                        self._consecutive_bad_requests += 1
                        if self._consecutive_bad_requests >= _BAD_REQUEST_LATCH_THRESHOLD:
                            self._exhaust(
                                f"The opener provider rejected {self._consecutive_bad_requests} opener "
                                "requests in a row as malformed (bad param/schema) -- this looks "
                                "like a systemic request problem, not a one-off bad photo",
                            )
                            return None
                        # Below the latch: this is a per-call reason (see last_skip_reason's
                        # docstring), not yet a global exhaustion -- the service is still enabled
                        # and expected to serve the next profile normally.
                        self.last_skip_reason = (
                            f"opener request for this profile was rejected as malformed "
                            f"(HTTP 400): {e}"
                        )
                        print(f"Opener skipped (400 on this profile's request -- could be a bad "
                              f"capture; {self._consecutive_bad_requests}/"
                              f"{_BAD_REQUEST_LATCH_THRESHOLD} in a row before treating it as "
                              f"permanent, swiping without): {e}")
                        return None
                    self._consecutive_bad_requests = 0
                    # Transient network/timeout/rate-limit errors: skip this profile's opener
                    # but keep the service enabled so subsequent profiles can still get openers
                    # -- UNLESS this is now a streak (see _TRANSIENT_LATCH_THRESHOLD): an audit of
                    # this project demonstrated 50 consecutive bare RuntimeErrors sailing through
                    # this exact branch with the service left enabled and nothing on the hub
                    # showing anything was wrong, each one a live profile that got a bare like
                    # with no opener.
                    # Durable record for EVERY transient failure, latched or not -- previously
                    # one of three failure kinds that left opener_rejections at zero rows
                    # forever (2026-09-06; see ops/OPENER-REDESIGN.md's dated addendum). Placed
                    # before the latch decision below for the same reason as the HTTP-400
                    # branch above. No `model` is known: GeminiCapacityExhausted and the
                    # permanent-reason branch above already cover every case where the
                    # exception itself names one.
                    try:
                        self.store.record_opener_rejection(
                            run_id, app, "", attempt, REASON_TRANSIENT_ERROR,
                            f"{type(e).__name__}: {e}", None,
                            prompt_sha256=self.prompt_sha256)
                    except Exception as store_exc:  # noqa: BLE001
                        print(f"Warning: failed to persist opener rejection record: {store_exc}")
                        _write_opener_rejection_deadletter(
                            self._deadletter_path, "transient", self.store,
                            run_id=run_id, app=app, model="", attempt=attempt,
                            reason_code=REASON_TRANSIENT_ERROR,
                            reason=f"{type(e).__name__}: {e}", raw_opener=None,
                            prompt_sha256=self.prompt_sha256, exc=store_exc)
                        # (2026-09-17) See the parse_error branch's own comment far above --
                        # same reasoning, same shutdown-sweep consumer.
                        self._deadlettered_rejection_keys.add(_rejection_identity_key(
                            run_id=run_id, app=app, model="", attempt=attempt,
                            reason_code=REASON_TRANSIENT_ERROR,
                            reason=f"{type(e).__name__}: {e}", raw_opener=None,
                            prompt_sha256=self.prompt_sha256))
                    if self._register_transient_failure(e):
                        # Latched -- exhausted_reason now carries the global cause; leave
                        # last_skip_reason (a per-call cause) alone, same reasoning as the
                        # OpenerError branch above.
                        pass
                    else:
                        self.last_skip_reason = (
                            f"opener call failed for this profile (transient error): "
                            f"{type(e).__name__}: {e}"
                        )
                        print(f"Opener skipped (transient error, swiping without): "
                              f"{type(e).__name__}: {e}")
                    return None

                # Success -- whether on the first attempt or after one or more retries.
                self._consecutive_bad_requests = 0   # success -- request shape is fine
                self._consecutive_transient_failures = 0   # success -- the pipeline works end to end
                # A successful, end-to-end call proves the LAST failure (if any) is over -- clear
                # it so a stale per-call reason from a prior profile never lingers and gets
                # misread as describing this one (see last_skip_reason's docstring).
                self.last_skip_reason = None

                # ENTROPY GUARD (ops/OPENER-REDESIGN.md 3.6, see _apply_entropy_guard for the
                # full rationale). It may hand back a SECOND draw taken because this one opened
                # with words already sent this run. That extra call must not consume the
                # per-profile attempt budget whose exhaustion stops the whole run, and it does
                # not: these lines are lexically inside the `for attempt in ...` loop, but they
                # sit on the SUCCESS path, which returns unconditionally a few lines below, so no
                # extra draw can ever cause another `attempt` iteration. Keep the guard on a
                # returning path if it ever moves.
                #
                # It runs BEFORE any spend/opener recording below for one reason: exactly one
                # `openers` row per profile must exist, and it must hold the text that actually
                # gets sent. The draft the guard discards was still billed, so the guard records
                # THAT spend itself -- the money is tracked either way, while the opener row is
                # written once, below, for whichever draw survives.
                result, entropy_collision, entropy_regenerated = self._apply_entropy_guard(
                    run_id, profile, result,
                    items=items, should_stop=should_stop,
                    skip_models=frozenset(failed_models))

                try:
                    cost = self.tracker.record(result.model, result.usage)
                except KeyError:
                    # The API already ran (real credits spent) but its response echoed a model
                    # string with no budget.pricing entry, so spend can't be accounted for.
                    # Degrade the same way as budget-reached/out-of-credit rather than crash —
                    # continuing to spend with no way to track it would silently break the
                    # budget-enforcement contract the rest of this service is built around.
                    # cost=None (not 0.0): the real cost was nonzero, just unrecoverable, and
                    # a fabricated $0.00 would misreport actual spend in the stored record.
                    self._exhaust(f"no budget.pricing entry for model '{result.model}'; "
                                  "spend can no longer be tracked")
                    cost = None
                # getattr defaults here for the same defensive reason as the pre-existing
                # getattr on the old referenced_index (this block used to read only that
                # field): a simple/fake OpenerClient used by a test, or some future call site,
                # need not populate every field on the OpenerResult it constructs, and a bare
                # AttributeError from a call that otherwise fully succeeded would be a strange
                # way for this method to fail. Read HERE, above the store calls, rather than
                # after them, because `angle` and `item_description` are both part of the
                # persisted opener row below.
                #
                # `item_index` REPLACES `referenced_index` and is a different quantity, not a
                # rename: 1-based over the NUMBERED ITEMS the model was shown, where the old
                # field was 0-based over raw scroll frames (see OpenerPick's docstring and
                # ops/OPENER-REDESIGN.md 5.1/5.7). The default is ITEM_INDEX_ABSENT rather than
                # a bare 0 for the same reason: 0 is out of band under the new contract, and a
                # stand-in result that never set the field must not read as "she picked item 1".
                # Deliberately NO getattr fallback to "referenced_index": silently accepting an
                # old 0-based frame index here and treating it as an item number is precisely
                # the reinterpretation doc 5.3 exists to prevent, so a stale producer degrades
                # to ABSENT (loud, unusable) instead of to a plausible wrong item.
                item_index = getattr(result, "item_index", ITEM_INDEX_ABSENT)
                # WHICH LIST that number counts, straight off the result rather than inferred
                # from anything here. The getattr default is the UNTRANSLATABLE space, matching
                # OpenerResult's own default: a client that does not state its space gets a
                # pick nothing will convert into a tap, which is the safe direction. Reading it
                # from `result` (not from, say, whether `items` was passed) keeps one producer
                # of this fact -- generate(), which built the payload -- instead of two that
                # can drift.
                index_space = str(getattr(result, "index_space", INDEX_SPACE_MODEL_ITEMS)
                                  or INDEX_SPACE_MODEL_ITEMS)
                referenced = str(getattr(result, "referenced", "") or "")
                # The model's own words for what this opener is DOING -- telemetry only, never
                # read to make a decision (see OpenerPick.angle). "" when the model omitted it
                # or returned a non-string; opener.py has already stripped it.
                angle = str(getattr(result, "angle", "") or "")
                # The model's own short description of the ITEM it picked (doc 5.7). Read
                # unconditionally, regardless of caller, so every request stays the same shape.
                item_description = str(getattr(result, "item_description", "") or "")
                # Output of opener.py's deterministic redundancy MONITOR: content words this
                # opener restated from its own `referenced` note (doc 3.7). isinstance-checked
                # rather than trusted, so a fake/older result carrying a string or None here
                # degrades to [] instead of being iterated character by character into twelve
                # bogus markers. LOG ONLY, and that is not a soft preference: this monitor is a
                # documented LOWER BOUND on redundancy (a terse `referenced` defeats it
                # entirely), it has never been calibrated against real data, and five
                # consecutive rejections stop the whole run -- gating on it would make an
                # uncalibrated metric a run-killer. It never rejects anything here, ever.
                raw_markers = getattr(result, "redundancy_markers", None)
                redundancy_markers = ([str(m) for m in raw_markers]
                                      if isinstance(raw_markers, (list, tuple)) else [])
                staged_for_action = stage
                if redundancy_markers:
                    # opener.py prints when it computes these, including parsed drafts the
                    # entropy guard later rejects.  Staged Training/AUTO drafts must
                    # not get a second service-level message here: the draft monitor is already
                    # enough diagnostic evidence, and a staged draft can become a Dislike, Stop,
                    # or pre-send refusal.  Service-level redundancy output is reserved for the
                    # successful commit below, so each landed Like contributes exactly one such
                    # lifecycle record.
                    if not staged_for_action:
                        # Legacy/immediate callers retain their durable record at generation,
                        # but this method still cannot know whether their later device action
                        # will send the text. Keep the redundancy signal while making that
                        # boundary explicit instead of overstating delivery.
                        print(f"Opener: immediate-flow opener restates "
                              f"{len(redundancy_markers)} word(s) from its own `referenced` note "
                              f"({'; '.join(redundancy_markers)}). Delivery is not verified "
                              "here; logged only for offline calibration, never a rejection.")
                try:
                    self.store.record_spend(run_id, result.model, result.usage, cost)
                    # angle and item_description ride along as the 6th and 7th POSITIONAL
                    # arguments: both stores declare them as trailing `angle: str = ""` /
                    # `item_description: str = ""` parameters (see ranker/store.py and
                    # ranker/bigquery_store.py, each with its own column migration), and a
                    # positional call keeps working against the fakes in the test suite that
                    # accept *a. `referenced` is the defensively-read local above rather than
                    # result.referenced: a result missing that attribute should persist an empty
                    # note, not blow up mid-try and get reported as a failure to persist SPEND,
                    # which is a misleading thing to print about an AttributeError on
                    # `referenced` (the spend row above it was already written fine).
                    #
                    # 2026-09-05 (b): `prompt_sha256` is the ONE keyword argument here, and it
                    # is a keyword deliberately -- it is a trailing keyword-only parameter in
                    # both stores, so passing it positionally is not even possible, and this
                    # call's positional prefix stays exactly the seven arguments described
                    # above. The cost is paid by the test doubles: a fake declared `*a` alone
                    # raises TypeError here, so the fakes in tests/test_opener_service.py and
                    # tests/test_opener.py accept `**kw` (or name the parameter). That failure
                    # is silent in production shape -- this whole call sits in the non-fatal
                    # try/except below -- which is why the fakes were widened deliberately
                    # rather than left to surface as an unrelated-looking count assertion.
                    # Passed UNCONDITIONALLY here, unlike _record_staged_opener, which probes
                    # the sink for the parameter first. That asymmetry is deliberate: this is
                    # the legacy/immediate path, `self.store` on it is the configured Store
                    # (whose Protocol declares the parameter, as do both real backends), and
                    # the duck-typed-sink case the probe exists for is the STAGED path's. If a
                    # sink here ever did lack it, the TypeError would be swallowed by the same
                    # try/except and reported under the spend-record message -- so if that path
                    # ever gains real duck-typed sinks, copy the probe rather than the call.
                    if not staged_for_action:
                        self.store.record_opener(run_id, app, result.model, result.opener,
                                                 referenced, angle, item_description,
                                                 prompt_sha256=self.prompt_sha256)
                except Exception as e:  # noqa: BLE001
                    # Spend was already tracked in-memory by CostTracker (or deliberately
                    # marked unrecoverable above); store failure is non-fatal.
                    print(f"Warning: failed to persist opener spend record "
                          f"({_display_cost(cost)}): {e}")
                if self.tracker.budget_reached():
                    self._exhaust("run budget reached")
                # An un-staged legacy/immediate caller records its diagnostic opener here.
                # Staged AUTO and Training flows instead commit only after the worker has a
                # confirmed Like boundary.  A draft must not make an abandoned profile look
                # acted-on in a bug report.
                #
                # The three newer fields all exist to make a redesign VISIBLE in a bug report
                # rather than only in a console line nobody kept: `angle` is the model's own
                # account of what it was doing, `redundancy_markers` is the over-description
                # monitor's verdict on the text actually sent, and `entropy_collision` /
                # `entropy_regenerated` record whether this opener repeated an earlier opening
                # and whether a second draw was taken -- a REAL collision check on every call.
                # Extra keys are safe here: bugreport.py reads this dict key by key with .get
                # defaults, never by unpacking or exact comparison.
                #
                # The "index" KEY KEEPS ITS NAME AND CHANGES ITS MEANING, which is worth
                # knowing before comparing two bug reports across this commit: entries written
                # before 2026-08-12 hold a 0-based index into the raw scroll frames, entries
                # after hold the 1-based MODEL ITEM INDEX (doc 5.1/5.7, and OpenerPick's
                # docstring). The key was not renamed because bugreport.py reads it by that
                # name and a report is a human-read artefact, but `item_description` sitting
                # beside it is what makes the two eras distinguishable at a glance: it is
                # simply absent from every pre-change entry.
                #
                # "index_space" rides alongside it so a report never has to be dated to be
                # read: it names the list "index" counts, which is the one fact a bare small
                # integer cannot carry and whose absence is what let the two spaces be confused
                # in the first place.
                recent_entry = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "app": app,
                    "model": result.model,
                    # Training shares AUTO's provider budget and stages its opener through the
                    # same landed-action envelope; commit_opener refines this to ``training``
                    # when the landed decision source is manual.
                    "session_mode": "auto",
                    "index": item_index,
                    "index_space": index_space,
                    "referenced": referenced,
                    "angle": angle,
                    "item_description": item_description,
                    "redundancy_markers": redundancy_markers,
                    "entropy_collision": entropy_collision,
                    "entropy_regenerated": entropy_regenerated,
                    "opener": result.opener,
                    # The prompt-era digest this opener was generated under (see
                    # self.prompt_sha256, set from prompt_stamp() in __init__), so the always-on
                    # diagnostic view bugreport.py reads (recent_openers_snapshot) can be grouped
                    # by era exactly like the durable `openers` row already can -- added
                    # 2026-09-06, previously absent here even though record_opener has carried it
                    # since 2026-09-05 (b).
                    "prompt_sha256": self.prompt_sha256,
                }
                if not staged_for_action:
                    self.recent_openers.append(recent_entry)
                ngram = _leading_ngram(result.opener, _ENTROPY_NGRAM_WORDS)
                if ngram:
                    self._recent_opening_ngrams.append(ngram)
                # Positional, matching OpenerPick's field order: text, index (the MODEL ITEM
                # INDEX now -- see that dataclass's docstring), referenced, angle,
                # item_description. `index_space` is passed by KEYWORD rather than as a sixth
                # positional: it is the field that makes `index` interpretable at all, and a
                # bare trailing string in a five-argument positional call is exactly the kind
                # of thing a later edit drops or reorders without noticing.
                # prompt_sha256 by KEYWORD (see the dataclass field's own comment): the stamp is
                # taken HERE, at generation time, not read off the service when the draft is
                # later committed, so a staged draft is always attributed to the prompt it was
                # actually generated under.
                staged = (_StagedOpenerRecord(
                    run_id, app, result.model, result.opener, referenced, angle,
                    item_description, recent_entry,
                    prompt_sha256=self.prompt_sha256) if staged_for_action else None)
                return OpenerPick(result.opener, item_index, referenced, angle, item_description,
                                  index_space=index_space, _staged_record=staged)
            # Unreachable in practice: __init__ now rejects any max_attempts that isn't an
            # int >= 1 (see BUG 2), so range(1, effective_max_attempts + 1) always yields at
            # least one iteration, and
            # every iteration above returns -- from the should_stop check, a non-retried
            # exception, the success branch, or _exhaust() on the final retry attempt. Before
            # that guard existed, OpenerService(max_attempts=0) made this loop body never run at
            # all: 0 API calls, disabled stayed False, stop_requested stayed False, no reason was
            # ever recorded, and control fell straight through to this exact line on every call
            # -- reporting perfectly healthy while producing zero openers, forever. Kept as a
            # defensive fallback so this method's return type stays honest even if that invariant
            # is ever broken by a future edit.
            return None  # pragma: no cover

    def commit_opener(self, pick: OpenerPick, *, profile_id: str = "", decision: str = "like",
                      decision_source: str = "", decision_created_at: object | None = None,
                      pre_send_evidence: dict[str, object] | None = None,
                      profile_key: str = "") -> bool:
        """Persist one staged AUTO/Training opener after a landed Like, exactly once.

        This is intentionally a separate, explicit commit from generation: neither opening a
        profile nor generating a suggestion says that the account acted.  Pass, Stop, resync,
        and every pre-tap refusal simply drop the in-memory envelope with the card.

        `profile_key` (2026-09-06): the caller's own already-computed attribution key
        (ranker/profile_key.py) for the profile this Like landed on -- passed through
        unvalidated, including "", to the same store call `profile_id`/`decision` reach.  This
        method never derives it: the caller captured it BEFORE the device action that
        invalidates whatever the identity was read off of (see worker.py's
        `_current_profile_key`), so re-deriving it here would only ever see the wrong profile
        or nothing at all.
        """
        with self._lock:
            record = getattr(pick, "_staged_record", None)
            if record is None:
                return False
            try:
                _record_staged_opener(
                    self.store, record, pick, profile_id=profile_id, decision=decision,
                    decision_source=decision_source, decision_created_at=decision_created_at,
                    pre_send_evidence=pre_send_evidence, profile_key=profile_key)
            except Exception as exc:  # noqa: BLE001 -- a store outage must not erase a real Like
                print(f"Warning: failed to persist committed opener after landed like: {exc}")
                return False
            recent_entry = dict(record.recent_entry)
            recent_entry["session_mode"] = (
                "training" if decision_source == "manual" else "auto")
            self.recent_openers.append(recent_entry)
            markers = recent_entry.get("redundancy_markers")
            if isinstance(markers, list) and markers:
                # This is the first point at which a staged draft has both crossed durable
                # storage and been supplied with the worker's verified Like lineage.  It is
                # therefore the only staged-flow message allowed to describe the opener as
                # committed/landed rather than merely proposed for review.
                print(f"Opener: committed Like opener restates {len(markers)} word(s) from its "
                      f"own `referenced` note ({'; '.join(str(marker) for marker in markers)}). "
                      "Logged only for offline calibration, never a rejection.")
            pick._staged_record = None
            return True

    def discard_opener(self, pick: OpenerPick, *, profile_id: str = "",
                       decision: str = DECISION_NEVER_SENT,
                       decision_source: str = "", decision_created_at: object | None = None,
                       profile_key: str = "") -> bool:
        """Persist one staged AUTO/Training opener draft that will NEVER be committed, so the
        durable ``openers`` table stops being survivorship-biased toward only landed Likes.

        commit_opener's docstring already states the shape this mirrors: generation is only a
        draft, and a landed Like is the one event that has ever made a staged envelope durable.
        Everything else -- an explicit Dislike, a Stop, a targeting refusal, a driver exception --
        used to just drop the in-memory envelope with the card, which is exactly why the
        existing `openers` rows cannot answer "how often does the model write a bad opener":
        every draft that did NOT end in a Like was discarded before it was ever written down.
        This method is the other half of that same lifecycle -- same staging envelope, same
        `prompt_sha256` captured at generation (never re-derived here), same lock, same
        best-effort persistence discipline -- so a bad opener now leaves exactly as durable a
        trace as a good one, distinguished only by `decision`.

        Deliberately silent about WHAT happened beyond the caller-supplied `decision` string:
        this is generic telemetry plumbing, not a place to encode every driver-specific refusal
        reason. Callers (Training's explicit Dislike, AUTO's stop/refusal/exception paths) each
        know their own outcome and pass it through. The discard vocabulary those callers choose
        from is DECISION_NEVER_SENT (the default) and DECISION_SEND_UNVERIFIED, spelled at the
        top of this module; Training's explicit reject passes its own "dislike" instead. A
        discard is NEVER recorded as "like", whatever went wrong: that value means a VERIFIED
        landed Like and is what decides, downstream, which opener may own an observed outcome.

        Returns False, exactly like commit_opener, both when there was nothing staged to persist
        (pick was never staged, or was already committed/discarded -- `_staged_record` is None
        either way) and when the store write itself failed. Neither caller inspects the return
        value today -- there is no decision, send, or refusal left to make once a draft is being
        discarded -- but the boolean is kept for the same reason commit_opener keeps one: a
        future caller should not have to guess whether persistence happened.

        `profile_key` (2026-09-06): exactly the same attribution key commit_opener accepts, and
        for the same reason -- a discarded draft is still a real, billed opener about a real
        profile, and doc's whole point in adding this column at all is that BOTH populations
        (sent and discarded) must be attributable, not only the survivorship-biased Likes.
        """
        with self._lock:
            record = getattr(pick, "_staged_record", None)
            if record is None:
                return False
            try:
                _record_staged_opener(
                    self.store, record, pick, profile_id=profile_id, decision=decision,
                    decision_source=decision_source, decision_created_at=decision_created_at,
                    pre_send_evidence=None, profile_key=profile_key)
            except Exception as exc:  # noqa: BLE001 -- a store outage must not break a run
                # over telemetry for a draft that was never going to be sent anyway.
                print(f"Warning: failed to persist discarded opener draft: {exc}")
                return False
            pick._staged_record = None
            return True

    def recent_openers_snapshot(self) -> list[dict]:
        """A copy of the most recent committed opener records -- see recent_openers'
        docstring in __init__ for exactly what each entry records and why (the request item
        space and the model's own `referenced` claim diagnose an out-of-place opener).

        Returns list(self.recent_openers) under self._lock rather than handing back
        self.recent_openers itself: a bug-report reader running on another thread must never
        iterate a deque that a live worker thread is concurrently appending to (maybe_opener
        holds this same lock across its whole body, including the append), and a plain list is
        also a stable, JSON-serializable snapshot rather than a live view that keeps changing
        under the reader's feet."""
        with self._lock:
            return list(self.recent_openers)

    def recent_rejections_snapshot(self) -> list[dict]:
        """A copy of the most recent REJECTED opener attempts -- see recent_rejections'
        docstring in __init__ for exactly what each entry records. Same reasoning as
        recent_openers_snapshot above: returns a plain list copy under self._lock rather than
        the live deque, so a bug-report reader on another thread never iterates a deque a
        worker thread is concurrently appending to."""
        with self._lock:
            return list(self.recent_rejections)

    def is_rejection_row_deadlettered(self, row: dict) -> bool:
        """(2026-09-17) Whether `row` -- built the same way
        BigQueryStore.record_opener_rejection buffers it and pending_opener_rejections()
        snapshots it -- already has a per-call dead-letter entry from THIS service instance
        (see the four maybe_opener() `except Exception as store_exc:` sites above, and
        _rejection_identity_key's own docstring for exactly what "the same row" means here).

        The ONLY caller is supervisor.py's _deadletter_stranded_opener_rejections, which
        uses this to stop its shutdown sweep from writing a SECOND entry for a row whose
        per-call write already covered it -- the fix for the double-dead-lettering
        regression found the same day: a FAILED FLUSH LEAVES THE BATCH BUFFERED
        (ranker/bigquery_store.py's own _flush_table comment), so the row a shutdown
        pending_opener_rejections() snapshot sees is very often the EXACT SAME row a
        per-call exception handler already dead-lettered, not a new one -- and pre-fix,
        supervisor.py wrote a second entry for it every time, doubling the apparent loss.

        Locked, unlike a bare set membership test would need to be: a worker that is still
        wedged past its shutdown join timeout (see supervisor._join_worker_for_shutdown) can
        be concurrently INSIDE maybe_opener, still adding to the tracked set, while the
        shutdown path calls this from a different thread.
        """
        key = _rejection_identity_key(
            run_id=row.get("run_id", ""), app=row.get("app", ""), model=row.get("model", ""),
            attempt=row.get("attempt", 0), reason_code=row.get("reason_code", ""),
            reason=row.get("reason", ""), raw_opener=row.get("raw_opener"),
            prompt_sha256=row.get("prompt_sha256"))
        with self._lock:
            return key in self._deadlettered_rejection_keys

    def _exhaust(self, reason: str) -> None:
        """Flip the service permanently disabled and record WHY (exhausted_reason), so an
        operator staring at a stopped or degraded run sees the actual cause instead of a
        generic line. Every exhaustion path (the module docstring's list: budget reached, a
        latch tripped, a permanent provider error, GeminiCapacityExhausted, every retry
        attempt failing) reaches here and stops the whole run: there is no longer a
        configuration that keeps AUTOMATION going once the service has given up (see this
        class's module docstring for why the old budget.on_exhausted="swipe_without_opener"
        mode was removed outright).
        """
        with self._lock:
            if not self.disabled:
                print(f"Opener: {reason} -> stopping all workers")
            # First-writer-wins: once exhausted, later calls into _exhaust (e.g. a second
            # worker's own opener call failing right behind the first, now that the service
            # is disabled) must not clobber the reason an operator actually needs -- the
            # original cause, not whatever incidental error a later caller happened to hit.
            if self.exhausted_reason is None:
                self.exhausted_reason = reason
            self.disabled = True
            self.stop_requested = True
