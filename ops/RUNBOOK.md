# RUNBOOK — live bring-up (the human-in-the-loop steps)

Everything machine-independent is already built and unit-tested. This file
collects the steps that **require you, a real account, and your machine** — do
them in one sitting. OS-agnostic (macOS / Windows / Linux); the app auto-detects
the device (`python -m operation_love.runtime`).

---

## 1. One-time install

Use Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[ml,bq,hinge]"
cp .env.example .env                                    # add GEMINI_API_KEY
chmod 600 .env                                           # recommended on macOS/Linux
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). It MUST
be created in the SAME Google Cloud project whose free-tier quota you intend to
use — Gemini's free-tier rate limits are enforced per PROJECT, not per key,
so a key minted in a different project draws from a different (likely empty)
quota pool; view the project's active limits in AI Studio. Open the
repository-local `.env` and replace the placeholder with your real
`GEMINI_API_KEY`. The app loads this gitignored file at startup. Never put the
key in `config.yaml`, commit it, or paste it into logs/screenshots. Restart the
app after adding or rotating the key so the process receives the new value.

Openers are **Gemini-only** — the legacy Anthropic/Claude opener path has been
removed from the codebase entirely, not merely defaulted off. `opener.provider`
accepts nothing but `"gemini"`; `config.validate()` raises immediately at
startup for any other value. There is no fallback: a missing/invalid
`GEMINI_API_KEY`, or a configured model id Gemini doesn't recognize, aborts the
run rather than degrading to swiping without an opener.

At startup (before the slower store/model warmup), the supervisor calls
Gemini's ListModels endpoint to confirm every id in `opener.models` exists and
supports content generation, and that the key itself is valid — this turns a
typo'd model id or a bad/revoked key into an immediate, clear startup failure
instead of a confusing mid-run one. Set `opener.preflight: false` in
`config.yaml` to skip this network call (offline development only); model ids
then go unvalidated until the first real profile is processed.

Privacy: opener generation uploads the captured profile images and text to
Google. Google's [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing)
states that free-tier content may be used to improve Google products. Use a paid
tier if that free-tier data use is unsuitable.

GCP / BigQuery (storage of record):
- Put your **project_id** in `config.yaml` → `storage.bigquery.project_id`.
- Auth once: `gcloud auth application-default login` (tables auto-create on first run).
- Check the machine: `python -m operation_love.runtime` (should show your GPU/CPU + no missing components).

Hinge only — a physical Android phone over host-side ADB (**no emulator**: Play
Integrity flags automation on an emulated device, so the driver only talks to a
real phone — see `operation_love/drivers/hinge.py` and `apps.hinge` in
`config.yaml`). Full first-time device setup (eSIM, enabling ADB, the
no-on-device-automation-server guardrail, scrcpy, installing Hinge) is its own
runbook: **[ops/HINGE-PIXEL-RUNBOOK.md](HINGE-PIXEL-RUNBOOK.md)**. Once
`adb devices` lists your phone, put its serial in `config.yaml` →
`apps.hinge.serial` (or leave blank for the first device).

---

## 2. Live-verification hooks (the only TODOs in the code)

Both are config-overridable — **no code edits needed**, just fill `config.yaml`.

**Bumble (web) — removed.** Bumble discontinued its web app in August 2026, so
there is no live web target, `apps.bumble_web` config, or Bumble inspector tool.
The generic `operation_love.drivers.web` Playwright base remains reference-only
behind the optional `web` extra; ordinary setup and launchers intentionally install
neither it nor Chromium. Bumble itself is now a second Android target on the same
physical Pixel as Hinge below. Both modes are fail-closed because
`BUMBLE_SPEC.calibrated` and `BUMBLE_SPEC.observe_ready` are false, and the required
`upsell_dismiss` template is absent. Do not flip either readiness value until the
device checklist below has actually been completed.

**Bumble calibration checklist — BLOCKING, before `calibrated` is ever flipped
to `True`:**

⚠️ A previous version of this checklist said to "drag over SuperSwipe and
confirm nothing is purchased," as if a confirmation step protects every
mistake. It doesn't. Measured live on the device 2026-08-10 (read-only
screencap + uiautomator dump — see `BUMBLE_SPEC` in
`operation_love/drivers/android/bumble.py` for the full geometry):
Bumble's SuperSwipe has **two different outcomes** depending on the
account's SuperSwipe balance, and only one of them shows a confirmation
sheet at all.

- [ ] Real coordinates measured live on the device (not the placeholder
      guesses shipped in `BUMBLE_SPEC`/`config.yaml`'s `apps.bumble.coords`).
- [ ] `forbidden_zones` re-measured against the real SuperSwipe control, not
      left at its generously-oversized placeholder rect. (Do not "fix" the
      purchase-sheet hazard below by widening this rect instead of doing the
      two checks below — see the note next to `forbidden_zones` in
      `BUMBLE_SPEC` for why a screen-agnostic rectangle can't do that job.)
- [ ] **Non-zero balance: confirm the silent-spend path, live.** With the
      account holding at least one SuperSwipe, deliberately trigger a
      SuperSwipe on a disposable/burner profile and confirm it is spent with
      **no** confirmation prompt of any kind. If that's what happens (expected,
      per the 2026-08-10 measurement — the owner's account showed a balance of
      5 and no prompt), then the never-super-like guarantee for this path
      comes **entirely from the code** — `decide_gesture="card_swipe"` +
      `forbidden_zones` + `AndroidDriver._require_deck_confirmed`
      (`UnconfirmedScreenError`) — never from the app prompting first. There is
      nothing to dismiss and nothing to catch a mistake here; treat every one
      of those three mechanisms as load-bearing before trusting this path
      unattended.
- [ ] **Zero balance: confirm the purchase-sheet path, live.** Spend down to a
      zero SuperSwipe balance on the same disposable/burner profile, trigger
      another SuperSwipe, and confirm the purchase sheet appears (CTA "Get 30
      SuperSwipes for $39.99", measured at x 0.049-0.950, y 0.899-0.951).
      Verify the bot **never taps while that sheet is up** — no decide gesture
      should fire against it (`_require_deck_confirmed` should refuse, since
      the deck's like/pass glyphs won't be visible), and if
      `_handle_rose_upsell`/`_dismiss_via_zone` runs, confirm every dismiss tap
      lands inside `upsell_dismiss_zone` and never anywhere near the CTA.
- [ ] `upsell_dismiss` glyph template captured (required before `calibrated`
      can even be set — see `AndroidAppSpec.__post_init__`). Capture it against
      the sheet from the zero-balance check above (its heading or another
      sheet-identifying glyph — NOT the purchase CTA). `BUMBLE_SPEC` already
      declares `upsell_dismiss_zone` (the safe band to tap to dismiss it, with
      the measurement and margin arithmetic behind it); it stays inert until
      this template exists.
- [ ] **The swipe-direction assumption, verified live, on a disposable
      profile:** `AndroidDriver._swipe` (`operation_love/drivers/hinge.py`)
      only zone-checks a drag's touch-DOWN point, on the reasoning that a
      gesture is locked to whatever View captured `ACTION_DOWN` (standard
      Android dispatch) — a drag that merely travels over or ends on a
      control shouldn't press it. That reasoning is correct for ordinary
      Android views but has **never been verified against Bumble's actual
      UI**, and `AndroidDriver._scroll_to_top`'s undo-swipes routinely END
      with the finger sitting over Bumble's SuperSwipe location (the undo
      drag returns the card to the top, i.e. travels downward, ending low on
      the screen — exactly where the SuperSwipe zone lives). Before Bumble is
      ever run unattended: deliberately drag a card so the gesture passes
      over and ends on the SuperSwipe control on a disposable/burner Bumble
      profile, at BOTH a zero and a non-zero SuperSwipe balance (see the two
      checks above — a non-zero balance won't show a purchase screen even if
      this reproduces the bug, it will just silently consume a SuperSwipe),
      and confirm no SuperSwipe was spent and no purchase sheet appeared. If
      Bumble's button turns out to react on release (or via some other
      non-standard touch handling)
      rather than only on capture, `_swipe`'s touch-down-only check does not
      protect against it and the guard needs to change before this ships.

**Hinge (Android)** — one command does it (no `uiautomator2`/on-device inspector;
that's the exact automation footprint ops/HINGE-PIXEL-RUNBOOK.md §5 forbids):
```bash
python -m tools.hinge_inspect            # screencaps your phone; reports vision-hit vs fallback-coord
```
- It screencaps the current Hinge screen and runs the SAME template-matching the
  driver uses at runtime (`_load_template`/`_match_glyph` in
  `operation_love/drivers/hinge.py`) to locate the like-heart and pass-X glyphs,
  reporting a vision HIT or a FALLBACK (the fixed-fraction coord under
  `apps.hinge.coords`). A FALLBACK on a fully-loaded profile screen means the
  glyph templates need recapturing for your device, or `apps.hinge.coords` needs
  a live tweak.
- **Training decision boundary** is implemented and unit-tested: start the local Hub in
  `mode: training`. For each profile the worker generates a targeted opener, opens the exact
  item, types it, hides the keyboard, and publishes one verified snapshot. Choose **Like** in
  the Hub to send that exact typed opener, or **Dislike** to tap Hinge's visible pass **X**.
  The worker re-verifies the live profile/item/control surface after your choice; it records a
  manual label only after the selected device action is verified as landed. Do not make the
  decision manually on the phone while a Training checkpoint is open.

### Hinge targeted-opener calibration — BLOCKING before targeted text or targeted AUTO likes

The completed opener redesign intentionally ships **without** numeric targeting bounds. Do not
copy a value from an old calibration note, infer one from a nearby device, or choose a “safe
looking” number. On the actual phone that will run Hinge, add this mapping only after the
measurement protocol below is complete:

```yaml
apps:
  hinge:
    serial: <exact adb serial>
    targeting_calibration:
      schema_version: 3
      device: <the exact same adb serial>
      hinge_version_name: <exact versionName>
      frame_size_px: [<width>, <height>]
      composer_layout_id: hinge_inline_v1
      item_selection_policy_id: hinge_photos_only_v2
      identity_match_max_dist: <measured identity bound>
      inline_item_max_dist: <measured inline-item bound>
      calibrated_at: <date/time and measurement-run reference>
      identity_band: [<exact effective x0>, <exact effective y0>, <exact effective x1>, <exact effective y1>]
      content_band: [<exact effective y0>, <exact effective y1>]
```

Those are the **exact** `apps.<app>.targeting_calibration` keys: no missing or additional keys.
`schema_version` must be the integer `3`; legacy sheet calibrations are rejected.
`hinge_version_name`, `frame_size_px`, `composer_layout_id`, and `item_selection_policy_id` bind the evidence to the exact
live app/layout/display and selection contract. `hinge_photos_only_v1` is **superseded** and
config validation rejects it by name; the current contract is `hinge_photos_only_v2`
(see [ops/STILL-PHOTO-DISCRIMINATOR.md](STILL-PHOTO-DISCRIMINATOR.md)). The v2 discriminator
(un-interacted dwell byte-exactness plus complete-ROI mute screening on every dwell frame, on
top of the unchanged rejection signals: the `0.24` static-photo drift ceiling, the crop
classifier, and the mute matcher) is implemented fail-closed. It licenses numbering ONLY while
config validation has installed a still-photo readiness licence, and readiness cannot be enabled
by editing code. There are exactly three licence channels and they are not interchangeable:
a verified `apps.hinge.still_photo_bound_evidence` mapping from the owner-labeled video
false-accept bound campaign above; the same key carrying the explicitly accepted circular
AI-labeled channel; or `apps.hinge.still_photo_assumption_acceptance`, the owner's accepted but
UNMEASURED centered-autoplay assumption (decision 2026-08-21, ops/STILL-PHOTO-DISCRIMINATOR.md
section 5b). The bound key and the assumption key are mutually exclusive by design.

**The shipped config today carries the assumption acceptance**, so numbering readiness IS
installed and no still-photo work blocks a calibration campaign. Confirm rather than assume,
because this sentence is the kind that goes stale:

```bash
python -c "from operation_love import config, targeting_policy as tp; \
c = config.load('config.yaml'); config.validate(c); \
print('numbering readiness installed:', tp.hinge_targeting_unavailable_reason() is None)"
```

While NO licence is installed, every photographic-looking crop remains readable unnumbered
context and no targeted Training opener can be prepared. Training therefore stops before a
checkpoint instead of substituting a target. AUTO remains separately blocked behind the legacy
production release artifact under every licence channel, and an assumption can never license it.
Confirmed videos remain fully excluded, while WRITTEN and UNKNOWN crops remain readable
unnumbered context because neither aspect ratio nor a presumed item count is item-type evidence. Both distance fields must be finite positive numbers. `device` must exactly equal the nonempty
`apps.hinge.serial` ADB serial; it is a machine-checked binding, not free-form device evidence.
`calibrated_at` must be nonempty evidence text, not a placeholder. `identity_match_max_dist` must be strictly less than
the known 2.565 different-profile distance; this is a hard upper limit, not a recommended
setting. `inline_item_max_dist` must likewise be strictly less than the nearest known 14.91
foreign-card false-match distance; that too is only a hard upper limit, never a setting to copy.
`identity_band` is its four-number `[x0, y0, x1, y1]` rectangle and `content_band` is its
two-number vertical `[y0, y1]` span. Both must be finite, ordered normalized fractions and must
exactly equal Hinge's *effective* bands after any `apps.hinge` overrides. A calibration cannot be
reused after either crop changes; measure and record a new one.
The two values serve different tests and must be measured separately:

- `identity_match_max_dist` is the maximum distance for the sticky-header profile-identity
  comparison, used before navigation/tap and again on the opened inline composer.
- `inline_item_max_dist` is the absolute maximum distance for the selected inline card versus the exact
  stored numbered item, in addition to the item's closed-set nearest-item/separation test.
- `device` is the exact ADB serial of the physical device on which the bounds were measured. Keep
  its relevant display/app-build evidence with the private measurement ledger; `calibrated_at`
  identifies when and which measurement run produced that evidence.

#### Still-photo video false-accept bound campaign (do this before either protocol below)

Design of record: `ops/STILL-PHOTO-DISCRIMINATOR.md`. Section 4 is this procedure and section 5
is the readiness plumbing. Neither protocol below can complete until this campaign has produced
a verified `bound.json`, because capture cannot number an item without one.

`python -m tools.hinge_video_bound` never touches the screen. It runs read-only
`adb exec-out screencap` and asks questions on stdin; you do every scroll and every card tap by
hand. That is the point, not a convenience: the label channel has to be something the mute
matcher cannot produce, or the bound is circular (section 2, objection 1). The matcher still
runs and its per-frame scores are persisted beside each card, explicitly marked observational.
They are never a label.

1. **Run the falsifier first.** Park one KNOWN video fully in view, hands off the phone:

```bash
python -m tools.hinge_video_bound hold-test --config config.yaml \
  --out ops/calibration/videobound_hold_<UTC> --seconds 60 --interval 0.5
```

Read the verdict line, the changed-row-block table, and the never-changed row count. If any two
consecutive frames come back byte-exact over the content band, the dwell window has to grow past
that run with margin, or the dwell design is abandoned honestly. Do not patch around that result.

2. **Capture the corpus.** You scroll. For each card the harness waits for the screen to settle,
records a hazard-randomized dwell burst, and only then prints the label prompt, so your
tap-to-play can never land inside a dwell frame. Tap the card, watch whether it plays, then
answer `video`, `photo`, `unsure`, `skip`, or `done` to end the session.

```bash
python -m tools.hinge_video_bound capture --config config.yaml \
  --out ops/calibration/videobound_<UTC> --profiles 140
```

Section 4 requires at least 60 owner-labeled video cards and at least 60 owner-labeled still
photos. `unsure` and `skip` cards stay on disk and are excluded from both denominators, so
answer `unsure` freely rather than guessing. Never label from a thumbnail; tap it.

3. **Measure offline.** No device, no ADB, no screencap. It re-verifies every frame digest and
reports per CARD, never per frame pair.

```bash
python -m tools.hinge_video_bound measure ops/calibration/videobound_<UTC> --config config.yaml
```

It refuses, nonzero and without writing anything, on too few cards of either label, any
owner-labeled video that would have been accepted at the candidate dwell window, more than 5%
false refusals across the owner-labeled still photos, a window longer than the campaign actually
watched every card for, a frame whose sha256 does not match the manifest, or a label typed
before its burst closed. A refusal is a stop and a recapture. It is never a threshold to widen.

4. **Freeze the artifact and paste the block.**

```bash
python -m tools.hinge_video_bound emit ops/calibration/videobound_<UTC> --config config.yaml
```

This writes `bound.json` (per-card verdicts, every frame sha256, device serial, Hinge build,
frame geometry, config sha256, and its own `evidence_sha256`) and prints the exact
`apps.hinge.still_photo_bound_evidence` mapping. Paste it into `config.yaml` by hand; this tool
never edits config, so binding a bound to a running config stays a deliberate act.

5. **Re-run config validation**, which is the only thing that can install readiness:

```bash
python -c "from operation_love import config, targeting_policy as tp; \
c = config.load('config.yaml'); config.validate(c); \
print('numbering readiness installed:', tp.hinge_targeting_unavailable_reason() is None)"
```

`validate()` re-reads the artifact off disk, checks its sha256 against the pasted
`artifact_sha256`, requires `human_ground_truth: true` and the `owner_tap_to_play_v1` ground
truth channel, and re-checks the section 4 thresholds before installing anything. A mapping
whose numbers disagree with the artifact is fatal, never a warning.

**These numbers install NUMBERING readiness only.** They unblock numbered suggestions in Training
and in calibration capture, and nothing else. AUTO stays separately blocked behind its own
historical release artifact (the legacy-named `observe_release_evidence` or the AI-reviewed equivalent), so
a weak or fraudulent bound artifact can never enable AUTO by itself. The calibration and
held-out splits still have to be recaptured afterwards; a bound artifact is a prerequisite for
those protocols, not a substitute for them.

#### Held-out real-device measurement protocol

> **Runnable once a still-photo licence is installed — verify with the one-liner above.** This
> protocol was blocked for as long as nothing licensed numbering, because capture could not
> number any item and neither split could complete. The owner's accepted centered-autoplay
> assumption (2026-08-21) lifted that block; the shipped config carries it.
>
> A capture is still never free. It attaches to the phone and can spend up to twelve real Passes
> through bounded pre-action skips before the session aborts, on real profiles in the owner's
> real deck. Decide the advance action per run — it is an owner decision every time, never a
> standing default — and do not start a campaign you cannot sit through.

Use the fail-closed harness for the capture and measurement phases. It writes private frames
only below gitignored `ops/calibration/`, never taps a heart/types/sends, and refuses to print a
paste block unless both separately collected splits, every frame digest, the exact device/build
state, complete-profile item coverage, and all four supervised checks below validate:

```bash
python -m tools.hinge_calibrate capture --split calibration --profiles 4 \
  --record-operational-checks
python -m tools.hinge_calibrate capture --split heldout --profiles 3
python -m tools.hinge_calibrate measure \
  ops/calibration/targeting_<calibration-UTC> \
  ops/calibration/targeting_<heldout-UTC>
```

#### Recommended hybrid external-review campaign (no human ground truth)

After a Hinge update, this is the reusable default for automated recalibration. It is still
automated evidence, not human ground truth: a separate AI reviewer (Codex, Claude, or another
runner) inspects each private checkpoint frame before the next protected action. The capture writes an atomic PNG and a
self-hashed checkpoint JSON containing the exact frame hash, claimed state, and action plan. It
then waits on stdin for the reviewer to issue exactly `APPROVE <checkpoint-json-sha256>` or
`REFUSE <checkpoint-json-sha256>`. It additionally understands `RETRY`, `RESTART_PROFILE`,
`SCROLL_UP`, `SCROLL_DOWN`, and `ABORT` with that same hash. `SCROLL_*` is capped at three
planner-derived, guarded gestures and always forces vision to re-locate the target; no reviewer
can provide raw coordinates. EOF, a stale hash, malformed input, `REFUSE`, or `ABORT` cleanly
aborts that session. Before the first heart only, an unavailable/incomplete requested photo,
navigation refusal, or `RESTART_PROFILE` becomes a bounded **pre-action skip**: capture rewinds
to a confirmed ordinary deck top, emits a dedicated `skip_profile_without_heart` checkpoint,
and requires a new `APPROVE` before calling public `HingeDriver.dislike()`. It then proves a
composer-free, distinct new profile and retries the same ordinal. Skips are capped at three per
ordinal and twelve per session, recorded as hash-bound `skipped_attempts`, and never count as
calibration profiles. After a real heart, a post-tap refusal still aborts the session and the
profile never becomes evidence. Before it closes, capture may perform one **abort cleanup** only
when the current frame structurally proves Hinge's inline composer: it first re-proves the same
sticky identity and checks whether a stable, twice vision-located floating Pass is already
visible. That already-unfocused state taps Pass directly; it must **not** receive Android back,
which can leave Hinge. Otherwise it uses one guarded Android edge-back, then the same identity
and Pass proof. Neither route can tap `Send Like`. It prefers the ordinary exact photo verification; if
that comparison is the refusal itself, cleanup is explicitly recorded as an unverified-item
discard in manifest `abort_recoveries` (`calibration_evidence: false`). Any missing composer,
identity mismatch, moving/missing Pass, or recovery failure leaves the phone untouched from that
point and is recorded as `not_cleared`. This is escape handling, never a retry or a way to make a
bad heart valid. Modify code/config if needed and launch a fresh split directory from a confirmed
top; never substitute blind coordinates.

**Licence state, checked not assumed:** capture can only produce an `automated_photo_heart`
checkpoint while a still-photo licence is installed, and a reviewer can never override a missing
one — `APPROVE` is a safety layer on top of the policy, never a way around it. Run the readiness
one-liner above before a campaign; with the shipped assumption acceptance in place it prints
`True` and the commands below are the live procedure rather than a retained one. Every checkpoint
action plan must record `positive_still_photo_evidence_verified` and
`target_frame_mute_control_screened_absent`, both derived from the C1–C3 verdict on the
digest-bound action frames; the former hardcoded `photo_only_item_verified` intent label is gone,
and emitting an unearned predicate is structurally impossible rather than merely untested. The tool already takes another screenshot immediately before an approved
tap, requires byte equality with the approved frame, and repeats the card/heart, identity, and
mute-control checks at the identical point. The reviewer remains an independent safety layer:
if the frame visibly shows a mute/speaker-with-x control, or you cannot establish that it is a
still photo, issue `RESTART_PROFILE` rather than `APPROVE`.

```bash
python -m tools.hinge_calibrate capture --split calibration --profiles 4 --hybrid-review \
  --confirmation I_ACCEPT_EXTERNAL_REVIEWED_AUTOMATION_RISK \
  --reviewer-model <reviewer-model> --reviewer-process <reviewer-run-or-version> \
  --out ops/calibration/targeting_<campaign>-calibration
python -m tools.hinge_calibrate capture --split heldout --profiles 3 --hybrid-review \
  --confirmation I_ACCEPT_EXTERNAL_REVIEWED_AUTOMATION_RISK \
  --reviewer-model <reviewer-model> --reviewer-process <reviewer-run-or-version> \
  --out ops/calibration/targeting_<campaign>-heldout

python -m tools.hinge_calibration_review \
  ops/calibration/targeting_<campaign>-calibration \
  ops/calibration/targeting_<campaign>-heldout --config config.yaml \
  --out ops/calibration/targeting_<campaign>-independent-review.json

python -m tools.hinge_calibrate measure \
  ops/calibration/targeting_<campaign>-calibration \
  ops/calibration/targeting_<campaign>-heldout --config config.yaml \
  --accept-hybrid-reviewed-evidence \
  --confirmation I_ACCEPT_EXTERNAL_REVIEWED_AUTOMATION_RISK \
  --unattended-review ops/calibration/targeting_<campaign>-independent-review.json
```

The manifest records `external_ai_review` as the reviewer source plus every decision and
runner-agnostic reviewer ID/model/version/process,
checkpoint and frame hash, and `human_ground_truth: false`. A reviewer approval is independent review of a stated
frame/action plan, not independent human ground truth. Hybrid evidence stays distinct from both
supervised/manual capture and fully unattended diagnostic capture, and all three remain blocked
from AUTO until the historical release artifact validates the exact measured calibration.

#### Opt-in real sends: `--send-like`

Both automated campaigns default to a Pass-without-send after every calibrated heart — this tool
never sends a Priority Like unless separately opted into, on every other run. `--send-like` (plus
its own `--send-like-confirmation I_ACCEPT_REAL_PRIORITY_LIKE_SEND_RISK`, a phrase distinct from
`--confirmation`) is an owner-directed exception to that default, requested specifically because
this account runs unlimited HingeX likes: every profile's terminal action becomes a REAL,
PERMANENT Send Priority Like instead of Pass, using the exact same
tap/upsell-dismiss/landed-verification transport as an ordinary AUTO or legacy
manual-observation like (never the paid
Rose/upsell control). It requires **`--hybrid-review`** — a real Priority Like is permanent and
cannot be retracted, so every one of them must clear a reviewer checkpoint first, and `--unattended`
has no reviewer in the loop. Passing it alone, with `--unattended`, or with the wrong (or missing)
`--send-like-confirmation`, exits non-zero before any config or device action. That restriction also
keeps capture and measurement consistent: `measure --accept-automated-circular-evidence` (the
`--unattended` validator) authenticates only the Pass-without-send terminal action, so an unattended
send capture could never be measured even if it were allowed to run.

```bash
python -m tools.hinge_calibrate capture --split calibration --profiles 4 --hybrid-review \
  --confirmation I_ACCEPT_EXTERNAL_REVIEWED_AUTOMATION_RISK \
  --send-like --send-like-confirmation I_ACCEPT_REAL_PRIORITY_LIKE_SEND_RISK \
  --reviewer-model <reviewer-model> --reviewer-process <reviewer-run-or-version> \
  --out ops/calibration/targeting_<campaign>-calibration
```

The **abort-cleanup path never sends real Priority Likes, regardless of this flag**: a stranded
composer left open after a post-tap verification refusal is always cleared through the
calibration-only Pass route described above, never Send Like. `--send-like` only ever replaces the
ordinary end-of-profile Pass; it cannot substitute for, or appear inside, the recovery sequence
that clears an unsent composer.

Under `--hybrid-review`, the reviewer sees the real planned action before it happens: the
pre-action checkpoint's `action_plan.action` literally reads `automated_send_priority_like`, never
a disguised Pass. The manifest's `automation_acceptance` records `send_like_accepted` and
`terminal_advance_action` explicitly either way (`false` / `automated_pass` by default, `true` /
`automated_send_priority_like` only when accepted) — `hinge_calibration_review` and `measure` read
this acceptance from the manifest rather than inferring it from the trace shape, so a send trace
can never authenticate itself, and both refuse a session whose confirmation and acceptance
disagree, or whose ledger mixes Pass and Send terminal actions within one profile. Evidence
therefore never silently reads as Pass-only when real likes were actually sent, and a session can
never claim both outcomes for the same profile.

For automated and hybrid capture, the versioned strategy
`alternate_photo_1_3_by_profile_ordinal_v1` deliberately takes **one** photo heart before each
Pass: odd profile ordinals target photo model item 1 and even ordinals target item 3. This avoids
navigating after Hinge's persistent composer opens. Thus a four-profile calibration run records
`1, 3, 1, 3` and a three-profile held-out run records `1, 3, 1`. The manifest binds the strategy,
each profile's single `composer_items` value, and its action trace; review and measurement refuse
a multi-profile split that does not collectively cover both depths. Supervised/manual capture
continues to prompt for the owner's chosen photo item(s).

Automated captures use the separately named `photo_only_confirmed_prefix_v1` proof. It may stop
before Hinge's unseen lower profile tail only after the confirmed top, sticky identity, every
predecessor, and the chosen photo's complete crop/absolute heart ordinal are all proven. The
actual bottom-up navigation and post-tap composer comparison remain bound to that same target.
This is deliberately narrower than a closed-set profile payload: it is valid only for the one
alternating calibration photo recorded on that profile, never for production model input or a
runtime item list. The reviewer and measurement re-check the scope and refuse a target-scoped
manifest that claims the closed-set entry-anchor ledger.

After either automated Pass, Hinge may briefly show a Hinge+ promo instead of the next deck. The
automation never taps purchase UI. If the first post-Pass frame is not both a confirmed top and
the driver's ordinary Like+Pass deck, with no composer visible, it may issue exactly one guarded
Android edge-back and then requires that exact ordinary-deck proof. A second ambiguous frame
aborts; the trace records the recovery frame hashes and whether edge-back was used.

#### Fully unattended circular-risk diagnostic campaign (no human ground truth)

When the owner explicitly accepts the circular-risk trade-off, collect fresh evidence after a
Hinge build/layout/geometry change with this complete, repeatable campaign. This is a different
provenance mode from the supervised procedure above: it performs the ordinal-selected single photo heart and a
Pass-without-send through the guarded driver, writes `human_ground_truth: false`, and remains
blocked from AUTO exactly as before.

```bash
# Fresh directories every campaign; use the same unchanged config bytes for all four commands.
python -m tools.hinge_calibrate capture --split calibration --profiles 4 --unattended \
  --confirmation I_ACCEPT_UNATTENDED_CIRCULAR_CALIBRATION_RISK \
  --out ops/calibration/targeting_<campaign>-calibration
python -m tools.hinge_calibrate capture --split heldout --profiles 3 --unattended \
  --confirmation I_ACCEPT_UNATTENDED_CIRCULAR_CALIBRATION_RISK \
  --out ops/calibration/targeting_<campaign>-heldout

# Run in a second, offline process. It imports no capture, driver, vision, ADB, or touch code.
python -m tools.hinge_calibration_review \
  ops/calibration/targeting_<campaign>-calibration \
  ops/calibration/targeting_<campaign>-heldout --config config.yaml \
  --out ops/calibration/targeting_<campaign>-independent-review.json

# Freeze calibration bounds only after that exact self-hashed review succeeds.
python -m tools.hinge_calibrate measure \
  ops/calibration/targeting_<campaign>-calibration \
  ops/calibration/targeting_<campaign>-heldout --config config.yaml \
  --accept-automated-circular-evidence \
  --confirmation I_ACCEPT_UNATTENDED_CIRCULAR_CALIBRATION_RISK \
  --unattended-review ops/calibration/targeting_<campaign>-independent-review.json
```

The review validates the versioned unattended provenance schema, every listed PNG and
entry-anchor-ledger digest, declared framebuffer, exact capture manifest bytes, exact config
bytes, device/build binding, and the action-to-frame trace. It writes a deterministic,
self-hashed artifact with `human_ground_truth: false`; it is an independent *review process*,
not independent ground truth. It refuses a supervised capture, a mixed/cross-mode campaign,
tampering, a changed config, a different device/build/framebuffer, or a review that does not
cover exactly the sessions supplied to `measure`. Never alter a prior campaign to fit a new Hinge
release: make a new calibration + held-out pair and retain the old private evidence as superseded.

If a completed schema-v3 capture predates the automatic ledger but already has a complete,
hash-valid `manifest.json`, add that evidence without touching the phone:

```bash
python -m tools.hinge_calibrate verify-entry-anchor \
  ops/calibration/targeting_<capture-UTC>
```

This command refuses to infer frame roles from unmanifested PNGs. A session without a completed
manifest must be recaptured; numbered filenames alone cannot prove which screenshots were card
read frames, inline-composer states, or the terminal advance.

Every prompted heart tap and profile advance is performed by the operator's own hand. The tool
never asks for a per-item dismissal: hearting another requested item moves the persistent
composer, and only advancing to a new profile proves it cleared. Profile
ids must uniquely identify the real profile across both runs; `measure` also compares the two
splits' independently captured identity fingerprints and refuses an ambiguous possible reuse.
The `--record-operational-checks` prompts require traceable local run/debug references, not an
unreferenced assertion. The flag may be used on either capture as long as the supplied sessions
collectively carry all four completed checks.

Each completed capture writes `entry_anchor_ledger.json` before those prompts. This is the
traceable local reference for the entry-anchor check: it is a manifest-hash-bound, offline-only
replay of the saved card frames through the production bottom-up navigator. It requires a
confirmed top, complete photo-only numbering/translation, a measured entry/shift record for
each available photo item, and a cancellation dry run before a replay capture or scroll. It
does not hold an ADB transport, issue a phone gesture, produce a calibration bound, or authorize
AUTO. If it cannot replay the exact bytes, the capture remains unusable; recapture rather than
substituting an operator assertion.

1. On the intended physical device, collect a calibration set and a separately collected,
   held-out set of real profiles, captures, and opened inline composers. Keep profile identities,
   items, display scale, Hinge build, and capture/session references in the private measurement
   ledger; do not commit profile images or personal data.
2. Measure identity distances for same-profile composer/card pairs and deliberately different
   profile pairs. Freeze an identity bound from the calibration set only, strictly below 2.565;
   then apply it unchanged to the held-out set. It must produce **0 foreign-profile accepts**.
3. Independently measure inline selected-card distances for each intended numbered photo and every
   other numbered photo/profile available in the held-out material. Prompt composers are outside
   the product contract and must never be used as calibration targets. Freeze a separate absolute inline-item
   bound from the calibration set only, then apply it unchanged to held-out composer frames. It must
   produce **0 foreign-item or foreign-profile accepts**, including when the nearest stored item
   is otherwise attractive under the closed-set comparison.
4. Record both accepted and refused cases, the frozen bounds, device/build evidence, timestamps,
   item numbers, and every held-out result. A false refusal is a stop/re-measurement signal; a
   foreign accept invalidates the bound. Do not widen either bound to recover a refusal without a
   new calibration and fresh held-out evidence.

The release material in the next block is a **historical calibration/release record**, retained
to explain the installed AUTO release artifact. Observe mode and its external-controller API are
retired; do not use this material as a procedure for a new run. New manual collection uses the
Training workflow in section 3.

The historical release required staged supervised operational evidence. Before numeric calibration, confirm
the first three device checks. A preliminary passive-cycle/quota probe may be recorded at this
stage, but it is deliberately not a production OBSERVE release check: it does not construct the
Worker/hub/store or make a real profile request. After installing the measured candidate in
**OBSERVE only**, complete the fourth release-policy check below before enabling AUTO:

1. Run `python -m tools.hinge_calibrate observe-check --supervised --confirmation OBSERVE_ONLY`.
   Give its completed, self-hashed `observe_check.json` path to the capture tool's preliminary
   operational-check prompt. `measure` accepts that artifact so it can emit a **numeric candidate
   for OBSERVE**; it is intentionally not AUTO authorization.
2. Paste the candidate calibration. The following production-OBSERVE validation command and
   artifact name describe the retired release workflow; do **not** configure `mode: observe`.
   Then run
   `python -m tools.hinge_observe_release --debug-run … --run-id … --out …`. Hinge's debug
   folder is named with the Worker run ID and its first record binds that ID, so the verifier
   rejects a timestamp-named or unrelated `actions.jsonl` paired with another run's store rows.
   The verifier derives
   the pre-tap hub publication, post-tap item verification, and composer anchor control facts
   from the application's `actions.jsonl`; it accepts no hand-authored status/API booleans. The
   terminal LIKE must either include Hinge's observed `like_sending` transition or be a direct,
   frame-backed, non-truncated LIKE resolution after those ordered facts -- current Hinge can
   land directly on the next stable card before the optional heartbeat is emitted. It queries the
   configured active store by Worker run ID through its read-only
   aggregate release API (BigQuery when configured, SQLite otherwise) for a persisted manual
   **pass and like**, each as both a label and a decision, plus a successful Hinge opener row —
   never profile rows, embeddings, opener text, or images. A rejected opener or provider spend
   row cannot substitute for the successful opener. A naturally encountered refusal/paywall is additionally logged, but is
   not required and is never induced by the verifier. Paste its emitted
   `observe_release_evidence` mapping beside `targeting_calibration`. Config validation blocks
   Hinge AUTO until this exact artifact is present and bound to the calibration/device/build.

   If the owner had explicitly chosen the AI-driven hybrid workflow instead of a manual cycle,
   the **separate** non-manual release path below was used. It required
   `observe_evidence_source: external_ai_review` (or `automation`) and an exact
   `ai_reviewed_observe_controller` mapping with schema version, the same source, the acceptance
   token, and executor model/id/version/process. This makes the Worker persist that declared
   non-manual source; it does not make the original manual gate accept it. The controller then
   compiles a completed `hinge_ai_observe_action_provenance` JSON sidecar from the same Worker
   debug-run directory. It declares `human_ground_truth: false`, binds the run/actions hash and
   calibrated device/build/framebuffer, identifies the executor, and hashes the eight ordered
   Worker-owned bridge facts: reviewed PASS, hub pre-tap publication, composer anchor, post-tap
   target verification, reviewed open, physical reviewed Send attempt, reviewed LIKE decision,
   and verified LIKE landing. The open/send/landing records must bind the same photo item and
   nonempty opener length. A passive `like_sending` state is intentionally not required: current
   Hinge can resolve directly to a stable next card, and the verifier refuses to fabricate that
   transient. This bridge sequence cannot be satisfied by the manual Observe path or a prose
   attestation:

   **External-processing approval checkpoint:** before starting that real Worker, obtain the
   owner's explicit approval: “I approve sending Hinge profile data to Gemini and storing the
   run in the configured BigQuery/GCS for the AI-reviewed OBSERVE validation.” This is an
   operational disclosure/approval checkpoint, not a config token or schema field. Do not start
   the Worker without it; the run sends profile content to Gemini and persists its data through
   the configured BigQuery/GCS storage path.

   ```bash
   python -m tools.hinge_observe_ai_release provenance \
     --config config.yaml --debug-run data/hinge_debug/<worker-run-id> --run-id <worker-run-id> \
     --out ops/release/<worker-run-id>/provenance --source external_ai_review \
     --acceptance I_ACCEPT_AI_REVIEWED_OBSERVE_RELEASE_RISK \
     --executor-model <model> --executor-id <id> \
     --executor-version <version> --executor-process <process>
   ```

   Review it in a second process with a different executor identity/process, then verify it
   against the configured store's same-source persisted PASS+LIKE labels and decisions and a
   successful opener. These commands are offline-only; they do not drive the phone or send an
   opener themselves:

   ```bash
   python -m tools.hinge_observe_ai_release review \
     --debug-run data/hinge_debug/<worker-run-id> --run-id <worker-run-id> \
     --provenance ops/release/<worker-run-id>/provenance/hinge_ai_observe_action_provenance.json \
     --out ops/release/<worker-run-id>/independent-review \
     --reviewer-model <model> --reviewer-id <different-id> \
     --reviewer-version <version> --reviewer-process <different-process>

   python -m tools.hinge_observe_ai_release verify \
     --debug-run data/hinge_debug/<worker-run-id> --run-id <worker-run-id> \
     --provenance ops/release/<worker-run-id>/provenance/hinge_ai_observe_action_provenance.json \
     --review ops/release/<worker-run-id>/independent-review/hinge_ai_observe_independent_review.json \
     --out ops/release/<worker-run-id>/release \
     --acceptance I_ACCEPT_AI_REVIEWED_OBSERVE_RELEASE_RISK
   ```

   Paste only the emitted `ai_reviewed_observe_release_evidence` mapping. It includes the exact
   acceptance token, and AUTO accepts exactly one gate: this AI-reviewed artifact **or** the
   manual `observe_release_evidence`, never both. The modes are intentionally non-interchangeable:
   the manual validator remains the old manual-only policy, while the AI artifact never claims
   human ground truth and refuses manual persistence rows, incomplete runs, action/frame/hash
   mismatch, calibration/device/build mismatch, or a reviewer sharing the executor's identity
   and process.

- **Entry anchor / scroll ledger:** begin from a confirmed profile top, enumerate the profile,
  and inspect the capture's `entry_anchor_ledger.json` bottom-up entry anchor, frame/shift
  ledger, complete photo-only numbering, and cancellation-before-replay behaviour before
  authorizing a gesture. This is offline evidence only; it does not prove a live gesture was
  transported.
- **Item 1 inline composer identity:** heart item 1 without an intervening scroll. Hinge's
  inline composer was introduced in 9.134 and was measured on the historical 10.0.1 build. The
  live build is now 10.1.0; the 10.0.1 calibration remains immutable audit evidence, while the
  separately measured 10.1.0 schema-v3 calibration is installed for Training. Its matching
  still-photo assumption clears the separate numbering-policy gate. Do not relabel either
  calibration across builds; collect a new campaign after any later app/frame drift.
  The inline composer auto-focuses immediately, so collect an initial and a settled auto-focused
  reading rather than inventing an unfocused-to-focused transition. Confirm the selected card
  through the full calibration capture's item proof; the small operational recorder can prove
  composer topology and corroborated profile-header identity, not a card ordinal by itself.
  This covers the top state where the original identity strip contains profile-independent filter
  chips.
- **Gesture transport:** on a supervised disposable profile, manually heart another item and
  record the composer afterward, then advance the profile without sending and record the clear
  top. The frame record proves the visible composer/clear states; the traceable operator ledger
  is the evidence that the manual gestures and intended target transition actually occurred.

  Both this check and item-1 identity must point to the same completed
  `tools.hinge_operational_evidence` **v2** `manifest.json` (or its containing directory).
  `measure` re-hashes all six PNGs, checks the canonical manifest digest for unaccompanied
  corruption, and replays the top/composer/identity analyses from those bytes. It requires the
  exact auto-focused roles, topology results, phone serial/build/framebuffer, and effective
  identity-band binding. (The digest is not a signature against a filesystem owner who rewrites
  both a JSON record and its digest.) A v1 trace labelled
  `unfocused`/`focused`, a directory with arbitrary screenshots, or a written assertion is not
  evidence and must be recaptured with the current recorder.

  If a completed v3 capture was already paused in the old evidence prompts, do not hand-edit its
  manifest and do not rerun the four-profile capture. First record the fresh v2 six-frame trace,
  then, only after the capture has completed, run:

  ```bash
  python -m tools.hinge_calibrate attach-operational-evidence \
    ops/calibration/targeting_<existing> \
    ops/calibration/operational_evidence_<fresh> --config config.yaml
  ```

  This offline-only one-time migration re-authenticates the existing capture frames, validates
  the v2 evidence, creates the normal hash-bound `entry_anchor_ledger.json`, and atomically
  replaces only the entry/item-1/gesture reference fields. It refuses a session that already has
  an entry ledger rather than rewriting its audit trail. The entry-anchor check itself must name
  that session's ledger (the capture directory is accepted only as a shorthand for that fixed
  file); a run ID or prose assertion is rejected.
- **Historical post-calibration production OBSERVE validation (required release policy before AUTO):** run the actual
  production Worker/hub through a complete manual pass/heart/send cycle using the measured
  calibration. Verify hub pre-tap publication, post-tap item verification, a real profile
  provider/quota result, a persisted manual **pass and like** as both labels and decisions in the
  configured store, and a successful persisted Hinge opener. A rejection/spend row and naturally
  logged refusal/paywall event are diagnostic evidence only; neither can replace that successful
  opener. `python -m tools.hinge_calibrate observe-check` is only preliminary passive device +
  synthetic-quota evidence; it cannot substitute for this validation or authorize AUTO. The
  historical 10.0.1 calibration completed this procedure in manual run `d8547ff144b4`; that
  artifact remains evidence for that build only. It does not authorize the live 10.1.0 build.

For a missing, stale, or runtime-rejected calibration, `stop_kind=targeting_calibration` is
expected safety behaviour: the Hub renders **"targeting calibration must be renewed"**, **AUTO
stops before any opener request or targeted gesture**, and **Training stops before it can prepare
a targeted opener**. No action or label is recorded. Recapture and validate schema-v3 calibration
and still-photo evidence (or explicitly re-accept the assumption) for the live app build/device;
before enabling AUTO, also collect that calibration's production release evidence. This is not
permission to use a legacy anchored opener or a fixed first-item fallback. `stop_kind=targeting`
is different: it is reserved for a failure to attach an opener that already exists to its selected
item.

> These are the items deferred to "do live, at the end." Everything they plug
> into (capture, embed, store, ranker, openers, supervisor) already works and is
> tested.

---

## 3. Train your taste — Training mode (you decide, it learns)

```yaml
# config.yaml
enabled_apps: [hinge]      # the only platform the registry accepts today —
                            # Bumble isn't calibrated yet and can never run
                            # alongside Hinge anyway (one physical Android phone)
mode: training
```
```bash
python -m operation_love hub        # use the local Hub's Like / Dislike controls
python -m operation_love stats      # watch labels climb; "ranker ready" flips at ~min_labels
```
Make decisions on ~50–100 profiles (research sweet spot). For each Hinge profile, Training
generates and types an opener for its exact target item, then hides the keyboard so both Hinge
actions are visible in the snapshot. Choose **Like** in the Hub to send the typed opener or
**Dislike** to pass. Only a verified landed choice persists; the ranker retrains live and goes
from `defer` → ready mid-session. Training never calls the ranker to make the current choice.

---

## 4. Go autonomous — auto mode (it swipes for you)

**Current release state: AUTO is blocked pending fresh production-observe evidence.** The
legacy-named manual `observe_release_evidence` artifact for production run `d8547ff144b4`
remains installed beside the historical 10.0.1 calibration, but the live app is 10.1.0. The
separately measured 10.1.0 schema-v3 calibration and matching still-photo-assumption acceptance
already enable Training; they do not release AUTO. Perform a new production OBSERVE validation
for the 10.1.0 calibration and install its release evidence before AUTO. The still-photo licence
cannot substitute for release evidence.
It does not bypass opener generation, exact-item targeting, foreground ownership, paid-upsell
refusal, post-action verification, `halt_on_error`, or Stop. A failed/untargetable opener stops
the run; it never degrades to a bare like. The driver retains a private
`auto_opener_pre_send` screenshot after the opener is typed on the selected item and immediately
before the Send tap, and successful AUTO evidence is archived to private Cloud Storage with a
queryable BigQuery row. The per-gesture `uhid` transport has a round-trip-confirmed virtual-touch
lifecycle for every action. Editing config does not start a run.

```yaml
mode: auto
apps:
  hinge:
    touch_backend: uhid
limits: {}                                      # optional caps may be added deliberately
budget: { run_budget_usd: 5.00 }                # global opener cap
```
```bash
python -m operation_love
```
- The current config uses normal learned-model AUTO. It retains and durably archives pre-send
  opener evidence for every verified landed LIKE and continues until Stop, deck exhaustion,
  or a safety halt.
- Bumble (once calibrated): card drags only, with no per-swipe opener. It cannot be
  selected in either mode today.
- Hinge: likes with a Gemini-written, profile-specific opener. `opener.models`
  is a quality-descending cascade, tried strongest-first. A model that hits its
  per-DAY quota is skipped for the rest of that run (per-day quotas reset at
  midnight Pacific); a per-minute 429 is transient and is simply retried on the
  next profile without dropping the model. The bot never sends a commentless
  like: a rejected AI response is re-asked (with a correction hint) up to
  `opener.max_attempts` times (default 5) before giving up on that profile. If
  it's still bad after that, or every configured model's opener capacity is
  exhausted, the run stops entirely — the reason is shown in the hub — and it
  will NOT fall back to sending a bare like with no opener.
- Gemini quotas are applied per Google Cloud project, not per API key. Creating
  another key in the same project does not create another quota pool; view the
  project's active limits in Google AI Studio. A key must be created in the
  SAME project whose quota you intend to use (see step 1) or it draws from an
  unrelated pool.
- Every autonomous swipe is recorded for stats and optional daily limits, but is not fed
  back as a training label; learning remains grounded in your manual Training-mode decisions.

---

## 5. Always-on (optional) — run it off your laptop

Because state lives in BigQuery, the **same code runs on any always-on box**
(your AMD PC, a mini-PC, a cloud VM) with no migration. Run as a service:

```bash
# Linux (systemd) or just a screen/tmux session:
nohup python -m operation_love >> oplove.log 2>&1 &
```
- Hinge needs a physical Android phone reachable over ADB (USB, or wireless ADB
  on the same network) from wherever the process runs — no emulator, no special
  host virtualization support required. Bumble will need that same physical
  phone once it's calibrated; it can no longer run headless the way the old
  Bumble-web driver did, since it's now an Android target like Hinge.
- Ctrl-C / SIGTERM shuts down cleanly and flushes the store.

---

## Quick reference

## Training actions (localhost API)

The local Hub owns the only Training decision capability. The worker—not an API caller—opens the
targeted composer, types the opener, hides the keyboard, and performs the chosen Hinge action.
Treat each checkpoint as a one-card capability: use the current tokens, never coordinates or
arbitrary text, and do not submit another action until this card has a terminal result.

1. `GET /api/training/checkpoint?run_id=<run>&app=hinge` and retain the returned
   `profile_token` and `approval_token` exactly. A checkpoint includes the immutable snapshot
   and typed opener the operator is deciding on.
2. `POST /api/training/action` with JSON containing `command` (`like` or `dislike`), `run_id`,
   `app`, both tokens, and a unique `idempotency_token`.
3. Poll the checkpoint endpoint until the matching result is `completed`, `failed`, `rejected`,
   or `aborted`. A completed result means the physical Hinge action and its durable manual
   training record both succeeded. Obtain a fresh checkpoint after every terminal result.

A changed card, Stop, run mismatch, stale token, malformed command, or concurrent action is
rejected or aborted before device input. After Like or Dislike is selected, Hinge recaptures and
strictly re-verifies the live profile/item/composer controls before tapping. A failed, cancelled,
or stale action records no label.

| Want | Do |
|---|---|
| See progress | `python -m operation_love stats` |
| Learn from your decisions | `mode: training`, then use the Hub's Like / Dislike controls |
| Let it swipe | `mode: auto` |
| Cap volume (optional; uncapped by default) | `limits.max_per_run / max_per_day` |
| Cap spend | `budget.run_budget_usd` |
| Force a device | `OPLOVE_DEVICE=cpu|cuda|mps` |
| Run both apps at once | Not possible — Android shows one app in the foreground at a time, and Bumble/Hinge share the one physical phone; run one, then the other |
