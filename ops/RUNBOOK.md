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
  click the pass **X** to reject a profile, or click its **heart** to open the
  comment / "Send Like" sheet. The hub then replaces its normal instruction with
  an opener suggestion; type it manually and tap **Send Like** yourself. A like
  is persisted only after that final send advances the profile; a pass is
  persisted only after the card advances. Dismissing the comment sheet does not
  record a decision — the same profile remains awaiting your choice. Dry-run
  `mode: observe` and check it logs your manual decisions correctly.

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

| Want | Do |
|---|---|
| See progress | `python -m operation_love stats` |
| Learn from your decisions | `mode: observe`, then make decisions manually |
| Let it swipe | `mode: auto` |
| Cap volume (optional; uncapped by default) | `limits.max_per_run / max_per_day` |
| Cap spend | `budget.run_budget_usd` |
| Force a device | `OPLOVE_DEVICE=cpu|cuda|mps` |
| Run both apps at once | Not possible — Android shows one app in the foreground at a time, and Bumble/Hinge share the one physical phone; run one, then the other |
