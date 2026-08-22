# Still-photo discriminator — design of record (2026-08-21)

Status: **implemented fail-closed; awaiting the device bound campaign.** Numbered
targeting stays disabled until a verified bound artifact exists (section 5). Nothing
in this design flips `HINGE_POSITIVE_STILL_PHOTO_DISCRIMINATOR_READY` by hand; the
constant is replaced by an artifact-derived readiness that cannot be enabled by
editing code alone.

This document is the decision record for the missing positive still-photo
discriminator that blocks Hinge numbered-item targeting (see
`operation_love/targeting_policy.py`, `ops/RUNBOOK.md` section 2). It records what
was considered, what was refuted, and the exact contract that was implemented.
Detection/anti-bot findings stay in `ops/ANTI-BOT-RESEARCH.md`; this file is the
vision/targeting design.

## 1. The problem

`hinge_photos_only_v1` numbering was disabled fail-closed because every existing
signal is a rejector:

- `classify_crop` (PHOTO/WRITTEN/UNKNOWN) is a texture test; all 16 recoverable
  real video-card crops classify as PHOTO. Zero measured video margin.
- The 0.24 signature-drift ceiling rejects motion; a paused/stalled/ended video can
  have zero drift. No complement was ever measured.
- The 42x42 mute-control template (threshold 0.98, four positive sightings, no
  negative distribution) proves "is video" when it fires; its silence proves
  nothing because the control is not always rendered.

A real video came one reviewer approval from being hearted on 2026-08-21
(`ops/calibration/targeting_20260821T-heldout-d/hybrid_review/00004.png`, the only
confirmed 10.0.1 video frame on disk).

## 2. Candidate designs and verdicts

Three designs were produced independently, judged, then adversarially verified.
Full transcripts live in the session workflow journal; summary:

| design | core idea | judge | adversarial verify |
|---|---|---|---|
| `mute_reveal_window_v1` | three-way mute score band + measured attach window; absence becomes positive once the window passed | 7.2, recommended | **REFUTED** (five fatal objections, below) |
| `still_photo_dwell_v1` | N un-interacted screencaps over T seconds; target card rect byte-identical across all N | 6.4 | survives as a conjunct; cannot separate "still photo" from "video emitting no frames" on its own |
| `still_media_v2` | conjunction of all rejectors + dwell + reveal scroll, affirmative force = the measured accept rate | 5.6 | honest framing adopted; its reveal-scroll leg breaks `item_nav` entry-drift and was dropped |

Fatal objections that killed the mute-band-as-primary design (each verified
against the corpus):

1. **Ground-truth circularity.** Every "proven video" label on disk exists because
   the mute matcher fired. A bound measured on that corpus structurally excludes
   videos whose control never rendered — exactly the false-accept population.
2. **The "confidently-not-mute" band contains real videos.** Complete-ROI
   pre-match frames of proven videos score 0.267–0.305, under the proposed
   non-video ceiling.
3. **The attach window is degenerate** on the motivating corpus (every first match
   occurs before the card is fully in the viewport).
4. **Lazy-attach vs auto-hide is unidentifiable** from mid-scroll enumeration
   frames: continuous gesture activity is the one regime where an auto-hiding
   control would not hide.
5. **Timing mismatch**: numbering is decided at capture time inside
   `build_item_payload`; an action-frame hold cannot be a conjunct of that gate.

## 3. The design of record

Acceptance for numbering a photographic crop requires ALL of, computed at
capture/enumeration time (where numbering is decided) and re-proven at the action
frame:

- **C1 — existing rejectors, unchanged.** `classify_crop` == PHOTO, signature
  drift <= 0.24 across >= 2 frames, zero mute matches (>= 0.98), zero animation
  markers. Every current refusal string is preserved byte-for-byte
  (`hinge.py` prefix-matches `video_mute_v1: upper-left`; the exclusion vocabulary
  does not change).
- **C2 — dwell byte-exactness (primary positive observation).** With no input,
  N screencaps over a window W; the target card rect must be byte-identical
  (full-res RGB, zero tolerance) in every capture and page-level frameshift must
  measure 0px. N and W are hazard-randomized per the auto-mode rule; the shipped
  bound records the worst-case exact-run length observed on held-out videos and W
  must exceed it with margin.
  **Centering precondition (owner fact, 2026-08-21): Hinge autoplays a video
  only when its card sits at/near the center of the screen.** Zero motion is
  therefore evidence only if the card was held in the autoplay trigger zone for
  the entire burst: every dwell measurement first scrolls the card to center,
  records the achieved offset, and dwell_exact may only license numbering with
  centered proof. An off-center byte-exact card is UNSURE, never photo. Motion
  and mute-glyph evidence of "is video" remain valid at any position.
- **C3 — mute screen with a complete ROI on every dwell frame** (secondary,
  rejection-only). The screen must have RUN with a geometrically complete ROI and
  a successful decode on every dwell frame; any match, any incomplete ROI, or any
  decode failure refuses. No "absent proves still" claim is made — C3 exists so a
  control that renders during the dwell cannot be missed, not as positive proof.
- **C4 — a verified bound artifact exists** (section 5). Without it, C1–C3 are
  never even consulted: the payload numbers nothing, exactly as today.

Affirmative force is claimed from measurement, not mechanism: the accept rule is
licensed only by the held-out false-accept bound in the artifact, and the residual
it cannot exclude in principle (a video that emits no frame for W while its
control never renders) is exactly what that bound quantifies.

Implementation note (2026-08-21): during live enumeration the driver can only
dwell on cards with a complete sighting in the final indexed frame, so with a
bound installed most candidates still demote at the missing-dwell rung there. The
load-bearing dwell proof is taken in the calibration pre-heart window, digest
bound into `_StillPhotoProof` on the exact frames the reviewer approves. Offline
replay recomputes exactness from persisted frame bytes and deliberately ignores
any recorded verdict scalar.

### Gate split

The old single boolean simultaneously unblocked numbering, platform-registry Auto
availability, and config acceptance. These are now separate:

- **Numbering release**: artifact-derived (section 5). Enables numbered
  suggestions in OBSERVE and calibration capture only.
- **Auto availability**: stays statically blocked in the platform registry and
  config until a fresh production-OBSERVE release chain exists
  (`observe_release_evidence` / AI-reviewed equivalent). A weak or fraudulent
  bound artifact can never enable Auto by itself.

### Policy id

`hinge_photos_only_v1` is retired; the selection contract including C1–C4 is
`hinge_photos_only_v2`. Config validation pins the new id and rejects v1 with an
explicit superseded message. The archived 10.0.1 v1 mapping can never be
reinstalled (its config bytes `44491df5…` match no git revision of `config.yaml`;
verified 2026-08-21).

## 4. Measurement protocol (device, owner present)

Tool: `python -m tools.hinge_video_bound` (see `--help`). Runbook:

1. **Label channel (breaks the circularity).** Every card in the bound corpus is
   labeled by the OWNER at the phone, per card, via typed prompt: tap the card,
   observe whether it plays (motion/sound/scrubber), answer `video` / `photo` /
   `unsure`. `unsure` cards are excluded from the denominator but kept on disk.
   The mute matcher's output is recorded per frame but is NEVER the label. The
   artifact records `ground_truth_channel: owner_tap_to_play_v1` and
   `human_ground_truth: true`; the verifier refuses any artifact without them.
2. **Capture.** The harness persists EVERY frame (not just refusal dumps): the
   scroll-in frames, a hazard-randomized dwell burst per card, and per-frame
   monotonic timestamps. Labeling happens after the dwell burst so the tap cannot
   contaminate the dwell frames.
3. **Bound.** Per-card statistic on owner-labeled videos: the longest byte-exact
   dwell run and the C1–C3 accept verdict. Required for a shippable artifact:
   `>= 60` distinct owner-labeled video cards, `0` accepts (Rule-of-Three 95%
   upper bound 5%; 300 cards for 1%), and `<= 5%` false-refusal on `>= 60`
   owner-labeled still photos. Reported per CARD, never per frame pair.
4. **Freeze.** `bound.json` carries per-card verdicts, every frame sha256, device
   serial, Hinge build, frame geometry, config sha256, and its own digest; the
   config block `apps.hinge.still_photo_bound_evidence` binds the artifact by
   sha256 on the `observe_release_evidence` pattern.
5. The dedicated **60-second hold falsifier** runs first and is cheap: park one
   known video fully in view, no input, screencap ~0.5s for 60s. If it ever emits
   two consecutive byte-exact frames at card granularity, W grows or the design
   is abandoned honestly rather than patched.

Also required regardless of the discriminator: the calibration and held-out
splits must be recaptured (config bytes moved; the 2026-08-21 campaigns can never
re-pass review), and capture must not be attempted before the artifact exists
(a never-numbering target burns up to twelve real Passes; see RUNBOOK).

## 5. Readiness plumbing

`operation_love/targeting_policy.py` no longer exposes a hand-editable ready
boolean. Readiness is installed process-locally by config validation ONLY:

- `config.validate()` verifies `apps.hinge.still_photo_bound_evidence` (schema,
  sha256, on-disk artifact, ground-truth channel, thresholds of section 4) and
  installs a frozen summary via `install_verified_still_photo_bound(...)`.
- `hinge_targeting_unavailable_reason()` returns `None` only while a verified
  summary is installed; otherwise it returns the original blocker text.
- No config key -> no installation -> byte-identical behavior to the previous
  hardcoded `False` everywhere (verified by the test suite).
- The calibration checkpoint predicate `positive_still_photo_evidence_verified`
  is derived from the C1–C3 verdict on the digest-bound action frames, exactly as
  `target_frame_mute_control_screened_absent` already is. Emitting a checkpoint
  with an unearned predicate is structurally impossible, not just untested.

## 5a. Addendum 2026-08-21: owner-accepted circular AI-labeled channel

Later the same day the owner explicitly accepted the circular-risk trade-off and
directed an automated campaign: the tool drives the phone with the driver's
guarded humanized gestures, labels each card by AI vision (mute glyph or motion
means video; stable, clean, photographic means photo), and advances by a per-run
choice (never a standing default; the first run advances by Pass). This ships as
a SECOND channel, `ai_mute_glyph_circular_v1`, gated by the explicit acceptance
phrase `I_ACCEPT_CIRCULAR_AI_LABELED_STILL_PHOTO_BOUND` in both artifact and
config, with `human_ground_truth: false` REQUIRED (recording true is refused as
a lie). The owner-labeled channel remains implemented and preferred.

Recorded honestly and permanently: with AI labels the video-accept count is zero
BY CONSTRUCTION, because the label and the accept rule read the same pixels. The
circular bound therefore is NOT evidence against a static or stalled video whose
mute control never rendered; that residual risk is what the owner accepted. What
the campaign genuinely measures: the dwell window against real playing videos,
the photo false-refusal rate, deck video frequency at the live build, and a
labeled frame corpus. Section 4's owner-labeled protocol remains the only way to
measure a non-vacuous false-accept bound.

## 5b. Addendum 2026-08-21: the measured bound is not being collected

The owner, after the cost was measured against real deck data, decided the
held-out video false-accept rate is not worth its price and directed that
targeting ship on the deterministic test instead. This is a deliberate,
recorded acceptance, not an oversight.

The measured cost that drove it: campaign 3 observed videos at roughly one per
seven profiles, so 60 video cards means about 420 profiles and about 420 real
Passes, over many sittings. The owner judged that disproportionate.

What the owner accepts instead, in their words: a card moved to the center of
the screen should be playing if it is a video, and a playing video is
detectable. The deduction is sound given two owner-supplied facts, that Hinge
autoplays only at center, and that nothing else on a Hinge profile changes
pixels over time. The residual it cannot cover is a video that is not playing
at all while observed: stalled, buffering, unloaded, or an ended non-looping
clip. Such a card holds perfectly still and reads as a photo.

Consequently readiness gains a THIRD channel, an ACCEPTED ASSUMPTION
(`apps.hinge.still_photo_assumption_acceptance`, phrase
`I_ACCEPT_UNMEASURED_CENTERED_AUTOPLAY_ASSUMPTION`), mutually exclusive with a
measured bound. Every operator surface must state plainly, on every run, that
targeting is licensed by an unmeasured assumption and that no video
false-accept rate has been measured. AUTO remains separately blocked; an
assumption can never license it.

To shrink the accepted residual without any measurement, acceptance also
requires a deterministic RE-ATTACH PROBE: after a centered burst looks static,
the card is scrolled out of the center band and back, forcing Hinge to
re-attach media, and a second centered burst must also be byte-exact. A video
that failed to start on first centering gets a second trigger before anything
is called a photo.

Scope note: the owner reaffirmed that opener and like target PHOTOS ONLY, never
prompt cards, videos, or anything else, so the Plan B below stays unchosen.

The measured protocol in section 4 and the campaign tooling remain implemented
and are the only way to obtain a non-vacuous bound. Nothing stops a future
session from collecting one and swapping the assumption for a measurement.

## 5c. Addendum 2026-08-22: the acceptance ladder briefly made calibration structurally impossible

Recorded the day it was found, in the same spirit as 5a/5b: what shipped, what it
actually did, and what was decided.

**What happened.** The hardening commit that wired the v2 acceptance into the
codebase (`e5054f7f`, 2026-08-21) also threaded
`unnumber_without_evidence=unnumber_without_still_photo_evidence,
still_photo_dwell=None` into the calibration tool's two live enumeration loops.
With a `None` dwell, `still_photo_evidence_from_drift` substitutes an all-`None`
`StillPhotoDwell`, the C2 dwell rung refuses every selectable block, zero items
ever number, and `_target_scoped_prefix_reason` returns "photo-only payload is
unusable" on every one of the 60 bounded read scrolls — so every real profile
became a bounded pre-action skip. The first live campaign after the still-photo
licence was accepted skipped 2/2 real profiles this way (the skip reason was
recovered from the manifest's `reason_sha256`; at the time the plaintext detail
was hashed and discarded, itself fixed the same night). A replay of the
2026-08-14 alt3 frames through the real functions confirmed the regression
exactly: the pre-e5054f7f call shape numbers 6 items; the shipped shape numbers 0.
The loop's own comment described the correct design ("this loop has no
un-interacted dwell to offer; `_verified_still_photo_proof` takes the real one in
the pre-heart window") — the code contradicted it one line down.

**A second, structural blocker sat behind the first.** The ladder's
re-observation rung demanded read-scroll signature drift, which for the FIRST
card of a top-down read is unanswerable by construction: `_signature_drift`
returns `(None, ())` when no other frame's analysed band fully contains the
crop's page rows, and one read scroll moves the first card's rows out of the
band. Measured across seven archived real captures, item-1 drift was `None` in
four and near-zero in one. Separately, real still photos re-observed across read
scrolls routinely exceed the 0.24 static-photo ceiling (0.277–2.1 observed) —
that ceiling was measured on parked cards. Together these made depth-1 targets
(odd calibration ordinals, and production's first photo) permanently
unnumberable under the shipped wiring.

**What was decided (three parts, all fail-closed):**

1. Calibration capture-time numbering is CLASSIFIER-ONLY, as it was through the
   proven 2026-08-14 campaigns and as the loop comment always claimed. Numbering
   there selects which item to navigate to; no heart is spent on its strength.
   The measure/replay/entry-anchor rebuilds align to the same rule so they
   reproduce capture's numbering instead of refusing it.
2. The ladder's re-observation rung passes through when drift is UNMEASURABLE
   (`None`/no frames) and still refuses when drift is measurable and above the
   ceiling, or non-finite. The mandatory rungs below — policy licence, dwell
   byte-exactness, centering, complete mute screens, re-attach probe — carry the
   stability proof; they are a strictly stronger independent re-observation than
   read-scroll drift, and an unlicensed build still refuses at the policy rung
   immediately after. The accepted assumption's logic (5b) never depended on
   read-scroll drift.
3. `_verified_still_photo_proof` — the sole licence for touching a heart —
   measures signature drift over its OWN parked re-observations (the approved
   anchor frame, the no-input burst, and the post-re-attach burst, all required
   byte-identical over the rect) instead of inheriting the capture read's drift.
   Measured, never hardcoded to zero, so a future weakening of the
   byte-exactness requirement cannot silently pass the drift rung. Read-scroll
   drift remains recorded in manifests as capture-context diagnostics.

**What adversarial verification of the fix then established (same day, replaying
26 archived real profiles / 80 classifier-numbered items through the real
functions).** Three facts belong in this record:

1. Production was in the same vacuous-strict state, and the record above
   understated the change. At the pre-fix HEAD the production numbering gate
   accepted 0 of 26 real profiles — not one card in the replay corpus passed the
   ladder. The re-observation relaxation is therefore not a narrow adjustment;
   it is what makes production numbering non-vacuous at all. And every card
   production can now number has NO drift measurement (drift is None for 42.5%
   of real items and over the ceiling for another 31%), so the C1 drift leg of
   section 3 is satisfied by zero accepted cards: the dwell/centred/mute/
   re-attach ladder under the 5b assumption is the ENTIRE production gate in
   fact, not in theory. That is what the owner accepted, said plainly.

2. The residual is slightly wider than 5b's wording. Relative to 5b, the
   accepted set adds one named class: a card the read never re-observed, that
   WAS playing during the read scroll, showed no mute glyph or animation
   marker, and had stopped by both parked bursts. Read-scroll drift was the
   only rung that could have seen it; the mute-match and animation-marker
   exclusions screen it independently, so the class is narrow — but it is not
   empty, and it is now part of the accepted residual.

3. OPEN OWNER DECISION (recorded, deliberately not changed tonight): the 0.24
   ceiling — measured on parked cards — is still applied to read-scroll drift
   in production numbering, where real still photos measure 0.28–27.2 on the
   same estimator. Result: ~31% of real photographic cards are refused with the
   diagnosis "animated or video media is not targetable" (wrong for a photo,
   fail-closed), and because dwell evidence only exists for cards complete in
   the final parked frame, production numbering yields at most ONE item —
   always the last photo — on ~27% of profiles and nothing on the rest. The
   owner rule "the model picks WHICH item to like" currently degenerates to
   "the last photo or nothing." Fixing that needs one of: a read-scroll drift
   ceiling measured from real data, per-card centred dwell during the read
   (gesture cost), or an explicit acceptance of narrow numbering. Owner's
   call; nothing here changes it silently.

Two smaller notes for honesty: (i) the emitted calibration block still stamps
`item_selection_policy_id: hinge_photos_only_v2` while capture-time numbering is
classifier-only — the id gates config compatibility and the C1–C4 contract is
exercised where it matters (the pre-heart proof), but the block attests a
numbering policy the capture did not run; (ii) the ladder gained a
consistency rung the same day: a measurable drift reported over zero frames (or
frames with no drift) refuses as a corrupt measurement — no in-tree producer
emits that shape, but the first future producer that forgets the frames behind
its number must be caught, not silently passed.

**Blocker three, found by the diagnosis the first fix installed (2026-08-22, campaign
attempt 4).** With numbering and the licence both fixed, the pre-heart proof got one rung
further and refused at the re-attach probe: its exit stroke was hardcoded BACKWARD
(`_scroll_up_one`, content down), which rubber-bands at scroll top — exactly where a
profile's first photo parks, i.e. calibration's depth-1 target and production's item 1.
The measured clamp branch existed but had no second direction to try, so depth-1 targets
could never pass the proof, and production numbering was biased to deep-parked (last)
photos for the same reason. Fix: backward-first (the common deep-parked case), one
forward retry on a MEASURED clamp, refuse only when both directions leave the card
inside the trigger zone; every leg stays measured against the original anchor and the
unclamped path is draw-for-draw identical to before (proved at four seeds). One watch
item deliberately left in place: the proof still demands a byte-exact restore after the
return leg while the driver only promises a residual under half a scroll quantum — at
the top stop the clamp forces exactness, but a deep-parked target whose return
under-delivers will skip at that rung; if attempt 5+ shows such skips, the return leg
needs micro-step refinement, not a relaxed comparison.

**Blocker five, attempt 5 (same night).** With the probe fixed, the proof got to its
centring rung and honestly refused a card navigation had parked ~one scroll quantum
(216px) below scroll top: measured offset -0.229 vs the ±0.150 zone, while the same
profile's true scroll-top offset is -0.109 — centrable, just mis-parked. Navigation's
contract tolerates a ~109px residual; a top-parked card can have as little as ~74px of
zone margin, so depth-1 targets frequently park just outside the zone. The reviewer
SCROLL_UP/SCROLL_DOWN machinery could never help because a still-photo refusal skips
BEFORE any checkpoint. Fix: positional refusals are now separated from content refusals —
the parked offset is measured cheaply from the exact rows the target proof re-proved,
BEFORE the still-photo proof spends dwell/probe gestures; an off-zone park gets one
planned corrective scroll (sharing the bounded reviewer adjustment budget) and re-enters
the loop's tested re-navigate/re-prove path; only an in-zone park reaches the proof.
Content refusals (mute, motion, probe) keep their immediate-skip behaviour. The
budget-exhausted skip detail lists every offset tried so a hopeless geometry reads
differently from a converging correction.

**Blocker six, attempt 7: byte identity was never the driver's contract.** With
centring fixed, the proof reached its last rung and refused because
`probe.anchor != frame` — it demanded the re-attach probe put the page back
BYTE-FOR-BYTE. The probe never promised that: its return leg drives the MEASURED
net displacement back under half a read-scroll quantum (~109px), so byte identity
is luck. Depth-1 got it sometimes (the top stop clamps); depth-3 (even ordinals)
would essentially never. Fix: measure the residual instead of demanding zero. The
card rect is translated by the measured shift (`estimate_shift`'s stated
convention: content at row y of the earlier frame is at `y - delta_px` in the
later one), the second burst's byte-exactness runs at the translated rect where
it was actually captured, and — the part that makes this STRONGER than what it
replaces — the final C1 drift becomes a genuine TWO-POSITION re-observation of
the same card: a still photo re-rasterized ~100px away measures a small distance,
a video that advanced measures large and the 0.24 ceiling refuses it. Everything
downstream rebinds to the post-probe frame: the checkpoint, the mute screen
predicate, the still-photo predicate, the re-proved heart point, and the tap.
Refusing on byte identity had been trading a real measurement for an accident.

**Operational finding, attempt 6 (not a code defect).** The scroll-top gate
refused at 5.547 from the nearest filter-chips fingerprint — inside the
deliberate (3.0, 9.0) dead zone — and correctly declined to stroke an ambiguous
screen. Cell-level analysis showed the mismatch concentrated in the bottom two
rows of the band (columns under the chips' lower edge), i.e. the page had drifted
a few pixels off top while the app sat idle between runs, not a new chrome
variant. The fix is operational, not calibration: cold-start Hinge
(`am force-stop`) before a campaign rather than adding a fingerprint, because
registering a drifted-top fingerprint would widen the very dead zone that caught
it. Do NOT add a variant for this reading.

**Blocker seven, attempt 8: the review gate could not survive a clock tick.** The
campaign finally produced a real `automated_photo_heart` checkpoint — photo item 1,
all five predicates true, target independently confirmed as the photo card (not the
prompt card), centring offset -0.131 inside the zone, mute screen clean. The reviewer
APPROVED it. `_fresh_reviewed_target_point` then refused: it re-screencaps immediately
before the tap and demanded FULL-FRAME byte identity with the reviewed frame. Diffing
the reviewed PNG against the live framebuffer showed the only differing rows were
43-76 — Android's status-bar clock, which had ticked from 12:34 during the review. The
entire content band, card and heart included, was byte-identical.

That gate was therefore unsatisfiable by construction for any reviewer slower than 60
seconds, which is precisely what a careful reviewer inspecting a frame is. (The
2026-08-14 campaigns presumably won the race with a fast automated responder; nothing
in the design guaranteed it.) Fix: compare byte identity over the CONTENT BAND — the
scrolling region that by definition excludes the status bar and bottom nav, and that
always contains the reviewed heart — using the same `dwell_exact_over_rect` primitive
the dwell legs use. Everything after the comparison is unchanged: the reviewed-point
binding, the structural card/heart re-proof, and the profile-identity gate all still
re-run on the FRESH bytes, which is what makes narrowing the byte comparison a
correction of scope rather than a relaxation of the guarantee.

The general shape, worth naming: a guard whose scope is wider than its purpose does not
fail safe, it fails USELESS. Full-frame identity looked maximally strict and in practice
guaranteed that no careful review could ever be spent.

**Blocker eight, attempt 9: the identity gate kept the navigator's frame.** With the
byte comparison scoped correctly, the same pre-tap guard refused again — this time
`IdentityError: the identity band is 19.156 grey levels from the sticky header this
index was built from`. Measurement said otherwise: the live framebuffer was
byte-identical to the reviewed checkpoint frame across BOTH the identity band and the
content band (mean |diff| 0.000); only the clock rows differed. The 19.156 was measured
against a stale reference. `ItemTarget.identity` is bound to the NAVIGATION frame; the
re-attach probe then moves the page, and when its return leg leaves a residual the
checkpoint is rebound to the post-probe frame (blocker six) — but the identity reference
was not rebound with it. Hinge's identity band legitimately shows different content at
different scroll positions (profile-independent filter chips at top, the sticky
per-profile header once scrolled), so comparing across those two positions measures a
large distance on one unchanged profile.

Fix: the gate's reference is the REVIEWED frame — the one the checkpoint bound and the
tap will land on — because its question is "is the profile on screen now the profile
that was reviewed", and both frames sit at the same scroll position. "Same profile
across the probe" is separately established by the probe's measured displacement plus
the structural re-proof of exactly one card at the translated rows with exactly one
heart at the translated point. A genuinely different profile at tap time still refuses,
and that is pinned.

**Three blockers of one shape (six, seven, eight), all in the last ten feet.** Each was
a check bound to the wrong frame or the wrong scope: the proof demanded byte identity
the driver never promised; the freshness gate compared chrome that cannot affect a tap;
the identity gate compared across scroll positions where its own signal is designed to
differ. None was a perception failure — the perception chain had been correct since
blocker five. When a pipeline rebinds its subject mid-flight (here: the probe moving the
page), EVERY downstream check that holds a reference to the subject has to be rebound
with it, and the ones that are missed fail as confident, well-worded refusals rather
than as errors.

**Attempt 10: the first complete calibration profile, and blocker nine.** With the
identity gate corrected, the campaign captured a profile end to end for the first time:
photo item 1 hearted after a passing still-photo proof, the inline composer opened on
that exact photo (verified visually and by `verify_sheet_item`), and the profile Passed
without sending. The manifest banked 3 card-scroll frames, one composer item — a real
`own_intended` measurement pair — and a clean composer-clearing advance to a different
profile.

Profile 2 then aborted with `checkpoint refused: no still-photo (C1-C3) verdict is bound
to the exact frame whose heart would be approved`. Root cause: the caller rebound
`target_pre` to the proof's action frame only `if still_photo_proof.page_residual_px`.
The status-bar clock ticks during the probe, so `probe.anchor` routinely differs from the
pre-probe frame in the CLOCK ROWS ONLY; the shift estimator correctly measures zero (no
content moved), the residual is 0, and the rebind was skipped — while the proof had
already bound its verdict to `probe.anchor`. Every predicate read back against
`target_pre` then mismatched. Fix: gate the rebind on the BYTES changing, not on the
residual being non-zero; the residual only says how far to TRANSLATE, and zero is a valid
answer that still requires rebinding the frame the predicates are read back against.

**One root cause, three costumes.** The Android status-bar clock changing bytes without
changing content produced blocker seven (the freshness gate compared it), and blocker
nine (a rebind gated on content motion missed a chrome-only change). The general form is
worth more than the three fixes: **a frame identity is not a content identity.** Any check
that means "the screen has not changed underneath me" must say which REGION it means, and
any rebind that means "the frame I am bound to has changed" must test the bytes it is
bound to — not a proxy for why they might have changed.

**The lesson, stated once for both failures of the day:** a gate is only as real
as the evidence PATH that feeds it. Wiring the strict ladder into a loop that
can never possess dwell evidence did not make the loop safer — it made the loop
vacuous-strict (refuse everything), which reads identical to "working, deck
unsuitable" from the operator's chair and cost a live campaign. Every gate needs
at least one test that drives the REAL evidence constructor end-to-end under an
installed licence; the suite was green throughout because every capture-path
test monkeypatched all three hops (`build_item_index`, `build_item_payload`,
`_target_scoped_prefix_reason`) and the licence auto-reset fixture kept the
policy rung, never the dwell rung, as the answering rung in unit tests.

## 6. Plan B (documented, not chosen)

If held-out video accepts never reach zero, the honest fallback is to change the
SELECTION policy rather than weaken the bound: number only WRITTEN prompt cards
(offline margin currently 16/0 video-vs-written). That reverses the explicit
photos-only contract in `ops/OPENER-REDESIGN.md` and is the owner's call, not an
engineering default.
