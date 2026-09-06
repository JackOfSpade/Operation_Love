"""The stable, cross-time attribution key that lets an OUTCOME find its OPENER.

THE GAP THIS CLOSES
----------------------
`openers.profile_id` (see `worker.py`'s `lineage_profile_id`, minted as `uuid.uuid4().hex`) is
random per CARD -- generated fresh every time a profile is shown, purely so one committed
opener row can be bound to the ONE decision that sent it (see `record_opener`'s
`profile_id`/`decision`/`decision_created_at` trio, which is lineage WITHIN a single swipe,
not identity ACROSS time). It carries no information about WHO the card was. An outcome
learned days later -- she matched, she replied, she never responded -- has no way to find its
way back to the opener that earned it, because the id that "wrote" that profile is already
forgotten: nothing durable ties two different SIGHTINGS of the same person together.

`profile_key`, defined here, is that tie. It answers exactly one question -- "is this later
observation about the same captured profile as this stored opener" -- and nothing else.

THE INPUT: A CALIBRATED PIXEL FINGERPRINT, NOT A NEW MEASUREMENT
--------------------------------------------------------------------
`drivers.item_identity.ProfileIdentity.fingerprint` already exists for an unrelated reason (a
navigation safety gate: "is the screen still showing the profile this index was built from")
and is already, in that module's own words, "loggable and storable anywhere": a tuple of
`grid` grey-level block averages of Hinge's own sticky per-profile header strip -- the one
piece of the screen that carries her name and stays pixel-identical for the whole profile
(measured 0.000 apart across 143 refuting frames of one hand-scrolled read; see that module's
docstring for the full calibration). This helper does not re-derive, re-measure, or re-justify
any of that; it only HASHES the fingerprint `capture_profile_identity` already produced, once
per profile, regardless of how many times she is later re-shown or the index rebuilt.

WHY A HASH, NEVER THE RAW FINGERPRINT
-----------------------------------------
The fingerprint is a 1024-integer (at the shipped 64x16 grid) low-resolution rendering of a
real person's first name on flat app chrome. An equality JOIN needs only that two rows'
KEYS compare equal -- it never needs the pixels themselves -- and a cryptographic hash
preserves exactly that property (equal input -> equal digest, and nothing about the digest
is recoverable back to the input) while a bare fingerprint tuple sitting in a table whose
whole purpose is aggregate calibration would be a real rendering of a real name for no
attribution benefit at all. `openers.profile_key` and `opener_outcomes.profile_key` therefore
both store ONLY this hex digest. Nothing upstream of this function's return value is ever
persisted by either store.

WHY THE GRID RIDES INSIDE THE HASHED PAYLOAD
------------------------------------------------
`item_identity._IDENTITY_GRID` is a calibration constant, not a law of physics -- that
module's own comment records two rejected re-measurements at finer grids (128x32, 256x64).
A future recalibration that changes the grid would produce a longer or shorter fingerprint
tuple for what is, physically, the same strip. Folding `grid` into the hashed payload rather
than hashing the fingerprint alone means a grid change simply opens a new key namespace by
construction: a profile fingerprinted under the old grid and the same profile refingerprinted
under a new one hash to two DIFFERENT keys (no accidental collision, and no accidental
"same" either), rather than depending on whoever changes the grid to remember this module.

THE DECISION THIS MODULE MAKES, STATED RATHER THAN LEFT IMPLICIT: NO NAME IS MIXED IN
------------------------------------------------------------------------------------------
`ProfileIdentity` never carries an OCR'd name. `item_identity.py`'s own "WHAT THIS DOES NOT
DO" section is explicit that name-reading is a separate, best-effort, tesseract-dependent
codepath its safety GATE deliberately keeps out of its own decision, "because giving a safety
gate a component that can be absent at runtime would make the gate's strictness depend on the
host." This module makes the identical choice for the identical reason, generalized from "a
gate" to "a storage key": whether tesseract is on PATH differs by machine, so a name-bearing
key would make the SAME real profile hash to two different keys depending on which host
captured it -- which destroys the one property `profile_key` exists to provide (that two
observations of the same profile compare equal). It would also not buy real separating power
even where it IS available: the fingerprint already IS a rendering of that name at a fixed
font/size/position, so a normalized name string mostly restates bytes the hash already
covers rather than distinguishing two profiles the pixels already conflate (see the next
section for exactly when that conflation happens). For both reasons, `name` is deliberately
NOT a parameter of this function, and no caller should add one without re-opening this note.

WHAT THIS KEY IS: AN EQUALITY KEY, NOT A SIMILARITY MATCHER -- ITS ONE STATED LIMITATION
---------------------------------------------------------------------------------------------
Two consequences follow, and an owner reading `opener_outcomes` later must keep them
straight, because they are opposite failure directions:

  1. UNATTRIBUTED, NEVER MIS-ATTRIBUTED, ACROSS AN APP CHANGE. A header Hinge renders even
     slightly differently later -- a new app version's font hinting, a redrawn layout, a UI
     experiment -- fingerprints to different pixels and therefore a different key. The
     outcome then simply has no opener row to join to. That is the deliberately-chosen
     failure mode, inherited rather than invented: `item_identity.py` already made this exact
     trade for the identity GATE ("a header redrawn a few pixels lower is a REFUSAL ... a
     false stop and never a false go"), and this key rides on the same fingerprint rather
     than trying to be more lenient than the gate that produces its input.

  2. A MEASURED, NAMED MIS-ATTRIBUTION RISK ACROSS TWO DIFFERENT PEOPLE. Two different real
     profiles who happen to render an IDENTICAL header -- most plausibly two women sharing
     the exact same first name, same length, same glyphs, since the strip is centred text on
     flat chrome with nothing else biometric in it -- fingerprint to the SAME bytes and
     therefore the SAME key. This is not a weakness of SHA-256 (a cryptographic hash only ever
     collapses genuinely-equal inputs together); it is a property of the pixels this key is
     built from, and it is already a NAMED, MEASURED fact about this exact fingerprint: the
     identity gate's own calibration (`item_identity._IDENTITY_MATCH_MAX_DIST`'s comment)
     measures real distinct-profile pairs as close as 2.565 grey levels apart at this grid,
     i.e. two different people the gate itself came within a hair of calling "the same
     profile." A `profile_key` COLLISION across two different real people is therefore
     possible, and unlike (1) it loses nothing -- it silently MERGES two people's outcome
     history under one key. Do not build anything on `profile_key` UNIQUENESS across
     profiles; it is sufficient only for the one question this module opens with.

WHAT THIS IS NOT
-------------------
* Not a replacement for `profile_id`: that id is per-CARD lineage, unrelated and unchanged.
  `profile_key` is threaded ALONGSIDE it, never instead of it.
* Not a new capture, measurement, or calibration: it hashes whatever `ProfileIdentity` the
  caller already holds. It never touches a device, a frame, or any I/O.
* Not a guarantee of attribution: a caller with no known identity gets `None` back (see
  `profile_key_from_identity`) rather than a hash of a placeholder that would look like a
  real key. An outcome recorded against `None` (stored by the caller as `""`, matching this
  codebase's existing `profile_id=""` convention for "no identity available") simply cannot
  join to anything -- which is the correct, honest answer, not a bug to route around.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..drivers.item_identity import ProfileIdentity

# Bump ONLY if the canonical payload shape below changes (a field added/removed/renamed, or the
# encoding itself changed). A future reader diffing old vs. new keys can then tell "these were
# hashed under different rules" apart from "these are just two different fingerprints" without
# guessing from `created_at` -- the same reason `prompt_sha256` exists for the opener text itself
# (ranker/store.py, ranker/bigquery_store.py), generalized to this key's own encoding.
_KEY_VERSION = 1


def compute_profile_key(fingerprint: Sequence[int], grid: tuple[int, int]) -> str:
    """The hex SHA-256 of `fingerprint` and `grid`, canonically encoded. Pure; no I/O.

    `fingerprint` is `ProfileIdentity.fingerprint` -- grey levels, 0..255, row-major, already
    flattened to plain ints by that module for exactly this reason. `grid` is
    `ProfileIdentity.grid`, folded into the hashed payload so a future recalibration that
    changes the grid mints a new key namespace rather than silently colliding with (or
    silently diverging from) keys hashed under the old one -- see the module docstring's
    "WHY THE GRID RIDES INSIDE THE HASHED PAYLOAD" section for the full reasoning.

    Encoding is fixed and deliberately narrow: a JSON object with exactly three keys
    (`v`, `grid`, `fingerprint`), integers only, dict keys sorted and separators tightened so
    the same logical input always serializes to the same bytes regardless of Python's dict
    ordering. This is the same canonicalization shape `ranker.retractions.canonical_sha` uses
    for its own hashed payloads, applied here to a different value for a different reason
    (attribution key, not a correction fingerprint) -- kept as its own small function rather
    than imported from there so this module never has to reason about that module's opener-
    retraction-specific exclusions (e.g. `prompt_sha256` deliberately kept out of THAT hash)
    leaking into or constraining THIS one.

    Never raises for well-formed input; a caller passing something that cannot be coerced to
    `int` (the one thing this function does not defensively validate, matching
    `ProfileIdentity.fingerprint`'s own already-plain-int contract) gets whatever `int()` or
    `json.dumps` raises, which is a caller bug rather than a runtime condition to paper over.
    """
    payload = {
        "v": _KEY_VERSION,
        "grid": [int(grid[0]), int(grid[1])],
        "fingerprint": [int(value) for value in fingerprint],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def profile_key_from_identity(identity: "ProfileIdentity | None") -> str | None:
    """The attribution key for `identity`, or `None` when there is nothing to key.

    `identity` is normally `ItemIndex.identity` -- whatever `capture_profile_identity`
    (`drivers/item_identity.py`) produced for the profile a run is currently acting on.
    Returns `None`, never a placeholder hash, for every case that module's own `.known`
    property already calls "nothing to compare": `identity` itself is `None`, or its
    `fingerprint` is `None` (scroll-top-only capture, an app with no declared identity band,
    a split/uncorroborated read -- see that module's docstring for the full list of causes).
    A caller that receives `None` here should store `""` for `profile_key` at the call site,
    matching this codebase's existing `profile_id=""` convention for "no identity available"
    (see `record_opener`'s own parameter) rather than inventing a second empty-value spelling.

    Duck-typed deliberately (reads `.known`, `.fingerprint`, `.grid` off whatever is passed
    rather than an `isinstance` check): this module's only compile-time dependency on
    `drivers.item_identity` is the `TYPE_CHECKING`-guarded annotation above, so importing this
    helper never pulls that driver module (and everything it, in turn, needs) into a process
    that only stores or reads openers -- e.g. an offline calibration pass or the recording CLI
    this data layer exists to feed, neither of which touches a device.
    """
    if identity is None or not identity.known:
        return None
    return compute_profile_key(identity.fingerprint, identity.grid)
