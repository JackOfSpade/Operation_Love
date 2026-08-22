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

## 6. Plan B (documented, not chosen)

If held-out video accepts never reach zero, the honest fallback is to change the
SELECTION policy rather than weaken the bound: number only WRITTEN prompt cards
(offline margin currently 16/0 video-vs-written). That reverses the explicit
photos-only contract in `ops/OPENER-REDESIGN.md` and is the owner's call, not an
engineering default.
