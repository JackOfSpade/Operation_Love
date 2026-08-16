# RUNBOOK — live bring-up (the human-in-the-loop steps)

Everything machine-independent is already built and unit-tested. This file
collects the steps that **require you, a real account, and your machine** — do
them in one sitting. OS-agnostic (macOS / Windows / Linux); the app auto-detects
the device (`python -m operation_love.runtime`).

---

## 1. One-time install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[ml,bq,bumble,hinge,dev]"
python -m playwright install chromium                   # only for the dead Bumble-web reference driver/tests; not needed to run Hinge
cp .env.example .env                                    # add GEMINI_API_KEY
chmod 600 .env                                           # recommended on macOS/Linux
pytest -q                                               # sanity: all green
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

**Bumble (web) — no longer applicable.** Bumble discontinued its web app in
August 2026 (see `operation_love/platforms.py`), so there is no live target
left to run `tools/bumble_inspect.py` against; `apps.bumble_web` stays in
`config.yaml` and the tool stays in the repo only as reference for a possible
future web-based platform. Bumble itself is now a second Android target on
the same physical Pixel as Hinge below, but it isn't calibrated yet
(placeholder coordinates in `operation_love/drivers/android/bumble.py`,
`calibrated=False`) and there is no live-calibration runbook step for it yet
— do not flip that flag until it has actually been verified against the
device.

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
- **Observe tap detection** (`wait_for_decision()`) is implemented and unit-tested:
  the hub may show optional help before you choose. It is not a recommendation or
  decision: click the pass **X** to reject a profile, or, if you choose to like,
  click the suggested item's **heart** to open the inline comment / "Send Like"
  composer. Type the suggested opener manually and tap **Send Like** yourself. A like
  is persisted only after that final send advances the profile; a pass is
  persisted only after the card advances. The composer stays embedded under the selected item;
  hearting another item moves it, and advancing the profile clears it. Dry-run
  `mode: observe` and check it logs your manual decisions correctly.

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
      item_selection_policy_id: hinge_photos_only_v1
      identity_match_max_dist: <measured identity bound>
      inline_item_max_dist: <measured inline-item bound>
      calibrated_at: <date/time and measurement-run reference>
      identity_band: [<exact effective x0>, <exact effective y0>, <exact effective x1>, <exact effective y1>]
      content_band: [<exact effective y0>, <exact effective y1>]
```

Those are the **exact** `apps.<app>.targeting_calibration` keys: no missing or additional keys.
`schema_version` must be the integer `3`; legacy sheet calibrations are rejected.
`hinge_version_name`, `frame_size_px`, `composer_layout_id`, and `item_selection_policy_id` bind the evidence to the exact
live app/layout/display and selection contract. `hinge_photos_only_v1` numbers only crops
affirmatively classified as photos after their source sightings clear the card-local mute-control
screen; videos are excluded, while WRITTEN and UNKNOWN crops remain readable unnumbered context
because neither aspect ratio nor a presumed item count is item-type evidence. Both distance fields must be finite positive numbers. `device` must exactly equal the nonempty
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

#### Held-out real-device measurement protocol

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
from AUTO until the production OBSERVE release artifact validates the exact measured calibration.

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

The release requires staged supervised operational evidence. Before numeric calibration, confirm
the first three device checks. A preliminary passive-cycle/quota probe may be recorded at this
stage, but it is deliberately not a production OBSERVE release check: it does not construct the
Worker/hub/store or make a real profile request. After installing the measured candidate in
**OBSERVE only**, complete the fourth release-policy check below before enabling AUTO:

1. Run `python -m tools.hinge_calibrate observe-check --supervised --confirmation OBSERVE_ONLY`.
   Give its completed, self-hashed `observe_check.json` path to the capture tool's preliminary
   operational-check prompt. `measure` accepts that artifact so it can emit a **numeric candidate
   for OBSERVE**; it is intentionally not AUTO authorization.
2. Paste the candidate calibration, keep Hinge in `mode: observe`, and complete a real production
   OBSERVE validation with Hinge debug logging enabled. Then run
   `python -m tools.hinge_observe_release --debug-run … --run-id … --out …`. Hinge's debug
   folder is named with the Worker run ID and its first record binds that ID, so the verifier
   rejects a timestamp-named or unrelated `actions.jsonl` paired with another run's store rows.
   The verifier derives
   the pre-tap hub publication, post-tap item verification, composer anchor, and `like_sending`
   control facts from the application's `actions.jsonl`; it accepts no hand-authored status/API
   booleans. It queries the configured active store by Worker run ID through its read-only
   aggregate release API (BigQuery when configured, SQLite otherwise) for a persisted manual
   **pass and like**, each as both a label and a decision, plus a successful Hinge opener row —
   never profile rows, embeddings, opener text, or images. A rejected opener or provider spend
   row cannot substitute for the successful opener. A naturally encountered refusal/paywall is additionally logged, but is
   not required and is never induced by the verifier. Paste its emitted
   `observe_release_evidence` mapping beside `targeting_calibration`. Config validation blocks
   Hinge AUTO until this exact artifact is present and bound to the calibration/device/build.

   If the owner has explicitly chosen the AI-driven hybrid workflow instead of a manual cycle,
   use the **separate** non-manual release path. While Hinge remains in `mode: observe`, add
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
- **Item 1 inline composer identity:** heart item 1 without an intervening scroll. Hinge 9.134
  auto-focuses the composer immediately, so collect an initial and a settled auto-focused
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
- **Post-calibration production OBSERVE validation (required release policy before AUTO):** run the actual
  production Worker/hub through a complete manual pass/heart/send cycle using the measured
  calibration. Verify hub pre-tap publication, post-tap item verification, a real profile
  provider/quota result, a persisted manual **pass and like** as both labels and decisions in the
  configured store, and a successful persisted Hinge opener. A rejection/spend row and naturally
  logged refusal/paywall event are diagnostic evidence only; neither can replace that successful
  opener. `python -m tools.hinge_calibrate observe-check` is only preliminary passive device +
  synthetic-quota evidence; it cannot substitute for this validation or authorize AUTO.

Until all of this is recorded, absence of `targeting_calibration` is expected safety behaviour:
**AUTO stops before any targeted gesture**, and **OBSERVE withholds targeted opener text while
manual labels continue**. It is not permission to use a legacy anchored opener or a fixed first
item fallback.

> These are the items deferred to "do live, at the end." Everything they plug
> into (capture, embed, store, ranker, openers, supervisor) already works and is
> tested.

---

## 3. Seed your taste — observe mode (you decide, it learns)

```yaml
# config.yaml
enabled_apps: [hinge]      # the only platform the registry accepts today —
                            # Bumble isn't calibrated yet and can never run
                            # alongside Hinge anyway (one physical Android phone)
mode: observe
```
```bash
python -m operation_love            # make decisions manually on real profiles
python -m operation_love stats      # watch labels climb; "ranker ready" flips at ~min_labels
```
Make decisions on ~50–100 profiles (research sweet spot). For Hinge likes,
click the heart, manually type the opener shown in the hub, and tap **Send Like**;
only that final send/advance persists the like. The ranker retrains live and goes
from `defer` → ready mid-session.

---

## 4. Go autonomous — auto mode (it swipes for you)

```yaml
mode: auto
budget: { run_budget_usd: 5.00 }                # global opener cap
```
```bash
python -m operation_love
```
- Volume is deliberately uncapped by default (`limits: {}` in config.yaml) — a fixed
  swipe quota is itself a bot signature (identical hard stop, run after run). The
  human-timing model (`pacing:`, session micro-breaks) shapes when actions happen; the
  run ends when the profile queue runs out, a real stop condition occurs, or you stop it
  manually. If you genuinely want a temporary ceiling (e.g. a supervised first auto
  run), add it back under `limits: { max_per_run: 60, max_per_day: 100 }` or per-app
  under `apps.<app>.limits`.
- Bumble: swipes only (Bumble is the opener exception — no per-swipe message).
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
  back as a training label; learning remains grounded in your manual observe-mode decisions.

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

## Reviewed Observe actions (localhost API)

Ordinary `observe` remains manual.  A separately reviewed controller can request a Hinge
action only through the hub's checkpoint protocol; the hub queues it and the existing Hinge
Worker performs it at its own waiting boundary.  Nothing else may read the device or issue a
gesture.  This is deliberately stepwise: pass, open the exact suggested item, then send the
still-current suggested text.

The bridge is enabled only when Hinge Observe is explicitly configured with
`observe_evidence_source: external_ai_review` or `automation` and its reviewed-controller
metadata; ordinary manual Observe cannot be driven through this endpoint.  Treat each checkpoint
as a one-card capability: it exposes a `phase`, never use coordinates or cached tokens, and do
not submit the next action until the previous result is `completed`.  `targeted_like_open` is
accepted only after the Worker has durably recorded its pre-tap publication fact; only its
successful, independently verified completion moves the checkpoint to `sheet_open`, the sole
phase that accepts `send_current_suggestion`.  A Pass is accepted only before a reviewed-like
action begins.  Any failed/aborted reviewed action is terminal for that card: obtain a fresh
card rather than retrying around an uncertain on-screen state.

1. `GET /api/observe/checkpoint?run_id=<run>&app=hinge` and retain the returned
   `profile_token`, `suggestion_token`, and `item` exactly.
2. `POST /api/observe/action` with JSON containing `command`, `run_id`, `app`, both tokens, and
   a unique `idempotency_token`. `pass` has no `item`; `targeted_like_open` and
   `send_current_suggestion` require the checkpoint's exact `item`.
3. Poll the checkpoint endpoint until the matching result is `completed`, `failed`, `rejected`,
   or `aborted`; retry only a failed/aborted review with a fresh checkpoint and token.

Coordinates, arbitrary text, and generic tap/type commands are rejected. A changed card, stop,
run mismatch, stale token, wrong item, missing suggestion, or concurrent action is rejected or
aborted before input. Hinge reuses its current numbered-item index/anchor and re-verifies the
composer/item immediately before typing or sending. Completed reviewed decisions are stored as
`external_ai_review`; manual Observe decisions remain `manual`.

| Want | Do |
|---|---|
| See progress | `python -m operation_love stats` |
| Learn from your decisions | `mode: observe`, then make decisions manually |
| Let it swipe | `mode: auto` |
| Cap volume (optional; uncapped by default) | `limits.max_per_run / max_per_day` |
| Cap spend | `budget.run_budget_usd` |
| Force a device | `OPLOVE_DEVICE=cpu|cuda|mps` |
| Run both apps at once | Not possible — Android shows one app in the foreground at a time, and Bumble/Hinge share the one physical phone; run one, then the other |
