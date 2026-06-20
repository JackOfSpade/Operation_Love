# RUNBOOK — live bring-up (the human-in-the-loop steps)

Everything machine-independent is already built and unit-tested. This file
collects the steps that **require you, a real account, and your machine** — do
them in one sitting. OS-agnostic (macOS / Windows / Linux); the app auto-detects
the device (`python -m operation_love.runtime`).

---

## 1. One-time install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[ml,bumble,hinge,bq,dev]"
python -m playwright install chromium                   # Bumble browser
cp .env.example .env                                    # add ANTHROPIC_API_KEY
pytest -q                                               # sanity: all green
```

GCP / BigQuery (storage of record):
- Put your **project_id** in `config.yaml` → `storage.bigquery.project_id`.
- Auth once: `gcloud auth application-default login` (tables auto-create on first run).
- Check the machine: `python -m operation_love.runtime` (should show your GPU/CPU + no missing components).

Hinge only — Android emulator (no Android phone needed; it's a virtual phone):
- Install Android Studio → create an AVD with a **Google Play** image.
- Start the emulator; confirm `adb devices` lists it. Put its serial in
  `config.yaml` → `apps.hinge.serial` (or leave blank for the first device).
- Install Hinge (Play Store in the emulator, or `adb install hinge.apk`).
- Log into Hinge with **your phone number** (SMS to your real phone).

---

## 2. Live-verification hooks (the only TODOs in the code)

Both are config-overridable — **no code edits needed**, just fill `config.yaml`.

**Bumble (web)** — one command does it:
```bash
python -m tools.bumble_inspect          # opens Bumble headful; log in, reach a card, press ENTER
```
- It probes every **selector** (photo / bio / prompt / like / pass / empty) and
  prints OK/MISS for each. Any MISS → open DevTools, find the right CSS, drop it
  under `apps.bumble.selectors`, re-run.
- It then asks you to like/pass a few cards and prints what it **detected** —
  observe like/pass detection rides on the *same* like/pass selectors (a DOM
  click listener), so there's **no network reverse-engineering**. If detection
  prints the right LIKE/PASS, observe mode is wired.

**Hinge (Android)** — use the `uiautomator2`/`uiautodev` inspector:
- Confirm the **resource-ids** (like / pass / comment box / send / prompt / empty)
  under `apps.hinge.ids`.
- **Observe tap detection** (`wait_for_decision()`) is implemented and unit-tested:
  a like is detected when the comment / "Send Like" sheet opens (`send_like` /
  `comment_box`) and is then sent; a pass is detected when the prompt text changes
  to the next profile. Live-confirm those two ids are the ones the sheet exposes —
  if not, fix them under `apps.hinge.ids` (no code change). Then dry-run
  `mode: observe` and check it logs your manual like/pass correctly.

> These are the items deferred to "do live, at the end." Everything they plug
> into (capture, embed, store, ranker, openers, supervisor) already works and is
> tested.

---

## 3. Seed your taste — observe mode (you swipe, it learns)

```yaml
# config.yaml
enabled_apps: [bumble]     # or [bumble, hinge]
mode: observe
```
```bash
python -m operation_love            # swipe manually on real profiles
python -m operation_love stats      # watch labels climb; "ranker ready" flips at ~min_labels
```
Swipe ~50–100 profiles (research sweet spot). The ranker retrains live and goes
from `defer` → ready mid-session.

---

## 4. Go autonomous — auto mode (it swipes for you)

```yaml
mode: auto
limits: { max_per_run: 60, max_per_day: 100 }   # human-like caps
budget: { run_budget_usd: 5.00 }                # global opener cap (Claude)
```
```bash
python -m operation_love
```
- Bumble: swipes only (Bumble is the opener exception — no per-swipe message).
- Hinge: likes with a Claude-written, profile-specific opener.
- Every autonomous swipe is still a label, so it keeps improving.

---

## 5. Always-on (optional) — run it off your laptop

Because state lives in BigQuery, the **same code runs on any always-on box**
(your AMD PC, a mini-PC, a cloud VM) with no migration. Run as a service:

```bash
# Linux (systemd) or just a screen/tmux session:
nohup python -m operation_love >> oplove.log 2>&1 &
```
- Bumble runs headless anywhere. Hinge needs an emulator-capable host (KVM/WHPX),
  so the always-on box must be able to run the Android emulator.
- Ctrl-C / SIGTERM shuts down cleanly and flushes the store.

---

## Quick reference

| Want | Do |
|---|---|
| See progress | `python -m operation_love stats` |
| Learn from your swipes | `mode: observe`, then swipe |
| Let it swipe | `mode: auto` |
| Cap volume | `limits.max_per_run / max_per_day` |
| Cap spend | `budget.run_budget_usd` |
| Force a device | `OPLOVE_DEVICE=cpu|cuda|mps` |
| Run both apps at once | `enabled_apps: [bumble, hinge]` |
