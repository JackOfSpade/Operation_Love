# Operation Love v2

Personal dating-app assistant. A **local, private** preference ranker decides
who *you'd* swipe right on (learned from your own swipes), and **Gemini** writes
a natural, profile-specific opener. Targets **Hinge** (physical Android over
host-side ADB + vision template-match) — no AirDroid, no fixed pixel
coordinates — currently the only runnable platform. **Bumble** is a second
target on that same physical phone (its web app was discontinued in August
2026), but it isn't calibrated yet, so it can't run until that work is done.

> The original AHK 2.0 project is archived in [`legacy/`](./legacy) for reference
> to the *vision*, not the implementation. See the analysis that motivated this
> rewrite in the project history.

## Architecture

```
drivers/      app control: Hinge (host-side ADB + vision) — the only
              runnable platform; Bumble uses the same approach on the same
              phone but isn't calibrated yet
perception/   capture all photos + profile text -> Profile
vision/       local pyiqa quality filter + ArcFace/CLIP embeddings
ranker/       logistic-regression on YOUR swipe labels (BigQuery/SQLite)
opener/       Gemini writes the opener, enforced JSON output
costing.py    client-side spend tracking + per-run budget guard
supervisor    one worker per enabled app; owns shutdown + flush
worker.py     the per-app loop, training or auto (replaces main.ahk)
hub/          local control panel (state, server, page, launchers)
```

**Decision = local & private.** **Opener = Gemini** (the one cloud call, only
for likes). Opener generation sends the captured profile images and text to
Google. Google states that free-tier content may be used to improve its products;
see [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing). Use a paid
tier instead if that free-tier data use is unsuitable.

## Hardware & OS — cross-platform

Runs on **macOS, Windows, or Linux** and **auto-adjusts to each machine** — it
inspects the OS, accelerator, and installed components at startup
(`operation_love/runtime.py`) and adapts. The compute device is auto-detected
(`operation_love/device.py`): **MPS** on Apple Silicon, **CUDA** on NVIDIA,
**CPU** everywhere else (fine at tens–hundreds of profiles/day). If an optional
component isn't installed on a given machine, that feature is skipped with a
warning instead of crashing. Force a device with `OPLOVE_DEVICE=cpu|cuda|mps`.

Check what any machine will use:

```bash
python -m operation_love.runtime
# Operation Love — Darwin arm64 · Python 3.12.x · Apple GPU (MPS)
```

OS-specific notes:

| Piece | macOS | Windows | Linux |
|---|---|---|---|
| Local ML (torch, insightface, CLIP, pyiqa) | MPS | CUDA / CPU | CUDA / CPU |
| Bumble (same physical Android phone as Hinge — not yet calibrated) | ❌ | ❌ | ❌ |
| Hinge (physical Android phone, host-side ADB only) | ✅ (USB or wireless ADB) | ✅ | ✅ |
| Gemini opener | cloud — any OS | cloud | cloud |

## Cost control

`opener.models` is a QUALITY-DESCENDING cascade (strongest/newest first). Gemini
free-tier rate limits are enforced per Google Cloud project, not per API key, so
a model that returns a per-DAY quota 429 is dropped for the rest of THIS run
only (per-day quotas reset at midnight Pacific) and the next configured model is
tried; a per-minute 429 is transient and does not drop the model.

On an opener-capable app the bot never sends a commentless like. If the AI
returns a bad response (empty, breaks the style rules, ...), it's re-asked with
a correction hint, up to `opener.max_attempts` times (default 5). If it's still
bad after that, or opener capacity is exhausted (every configured model out of
free-tier quota, or the client-side `budget.run_budget_usd` cap is reached), the
run stops entirely — with the reason shown in the hub — instead of ever falling
back to a bare like with no opener.

Before a run starts, the supervisor calls Gemini's ListModels endpoint to
confirm every configured model id exists and supports content generation (and
that the key itself is valid) — this catches a typo'd model id or a bad/revoked
key immediately, before the slower store/model warmup runs, rather than as a
mid-run failure. Set `opener.preflight: false` to skip this network call (e.g.
offline development); model ids then go unvalidated until the first real call.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,bq,hinge]"
cp .env.example .env   # add GEMINI_API_KEY
chmod 600 .env         # recommended on macOS/Linux
```

Repository contributors can add the lint/test tools with
`pip install -e ".[dev]"`; production setup and generated launchers omit them.

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). It MUST
be created in the SAME Google Cloud project whose free-tier quota you intend to
use — free-tier limits are enforced per PROJECT, not per key, so a key minted in
a different project draws from a different (likely empty) quota pool. Set
`GEMINI_API_KEY` in the repository-local `.env`; the app loads that file at
startup and `.gitignore` excludes it. Never put the key in `config.yaml`, source
control, logs, or screenshots. Restart the app after adding or rotating the key.
Openers are Gemini-only — the legacy Anthropic/Claude opener path has been
removed entirely, not merely defaulted off. `opener.provider` accepts nothing
but `"gemini"`; any other value (including the old `anthropic`) fails
`config.validate()` at load time. A missing/invalid `GEMINI_API_KEY`, or a
configured model id Gemini's ListModels endpoint doesn't recognize, aborts the
run instead of silently falling back to swiping without openers.

## Concurrency & deployment

**The supervisor architecture is per-app** — it launches one worker per enabled
app (`enabled_apps: [hinge]`), all sharing the taste model, the BigQuery store,
and a single **global** opener budget; that design genuinely scales to more
than one app. What it does not do today is run two apps at once: Bumble and
Hinge are both Android targets sharing the one physical phone, Android
foregrounds a single app at a time, and the platform registry
(`operation_love/platforms.py`) refuses any selection that pairs two Android
platforms together. Right now Hinge is the only calibrated, runnable platform
— Bumble is a second Android target on the same phone but isn't calibrated
yet, and its old web path is dead (Bumble discontinued its web app in August
2026).

Run it with `python -m operation_love`. It's OS-agnostic (macOS/Windows/Linux,
GPU or CPU) and host-agnostic — since state lives in BigQuery, you can develop
on one machine and deploy the same code to an always-on box with no migration.

## Status

Done: scaffold, config, storage (BigQuery + SQLite), runtime auto-detect,
cost-control/global-budget, the Gemini opener, the **supervisor + per-app
worker** loop, and **the vision + personal-ranker core** — quality pre-filter,
ArcFace+CLIP embeddings, and a logistic-regression `PreferenceModel` that learns
your taste with cold-start gating (`defer` until enough labels). All pure logic
is covered by offline tests; model *inference* runs on a machine with the `ml`
extra installed.

**Two modes** (`mode: training | auto`):
- **training** — Hub-only human-in-the-loop learning. The worker assumes each
  profile is a provisional Like, generates an opener, reaches its exact target,
  types it, hides the keyboard, and pauses with the verified composer visible.
  Choose **Like** to send it or **Dislike** to pass the profile. The worker never
  consults the preference model for this choice; each verified human outcome is
  stored as a manual training label and retrains the ranker live. A failed,
  cancelled, or stale checkpoint records neither decision nor label.
- **auto** — normally, the bot swipes for you with the learned model. Decisions are recorded
  for stats and optional limits, but are not fed back as training labels. Hinge Auto is
  structurally fail-closed: an accepted still-photo assumption cannot license Auto, and a
  release artifact cannot bypass the targeting policy. The shipped configuration now carries a
  schema-v3 `targeting_calibration` renewed on the exact live Hinge 10.4.0 / 1080x2400 build on
  device `33111JEHN04475`, so Training is available. It is hybrid AI-reviewed circular-risk,
  photo-item-1-only evidence, explicitly not human ground truth. Its `observe_release_evidence` artifact for run
  `d8547ff144b4` remains historical 10.0.1 evidence and cannot release AUTO; AUTO still requires
  a fresh production release for 10.4.0. Once released,
  AUTO uses the learned ranker on every card,
  with per-gesture UHID, normal opener generation, exact-item targeting, Send verification, and
  all action safety gates. Configuration alone does not start a run.

**Hinge is code-complete** (host-side ADB + vision-located taps — no on-device
helper, no emulator; see ops/HINGE-PIXEL-RUNBOOK.md) and is the only currently
runnable platform. AUTO uses the learned ranker and is uncapped by default, continuing until
Stop, deck exhaustion, or a safety halt. Every LIKE retains a private pre-send snapshot after
the opener is typed and archives its durable evidence. A **stats** readout
(`python -m operation_love stats`) and human-like pacing remain available. Bumble is
ported to the same Android + vision approach on the same physical phone, but
still needs live calibration before it can run (see "Concurrency &
deployment" above). The generic Playwright driver base remains available through the
optional `web` extra as reference for a future browser-based platform; launchers do not
install it or Chromium now that no runnable target uses it.

Everything machine-independent is done and unit-tested (run `pytest` for the
current suite/test count; it runs across processes by default and takes about three minutes --
add `-n0` for a serial run when you want `--pdb`, `-s`, or a readable traceback). The only
remaining work needs your machine + a real account, and it's all batched in
**[ops/RUNBOOK.md](ops/RUNBOOK.md)**: install, connect the physical phone,
verify Training-mode behavior, and seed your taste.

The 2026-08-22 10.0.1 calibration and its release artifact, and the superseded 10.3.0 calibration,
remain historical evidence, not a license for the live 10.4.0 build. The shipped 10.4.0 schema-v3
hybrid AI-reviewed circular-risk photo-item-1-only calibration and matching still-photo-assumption
acceptance license Training; the calibration evidence is not human ground truth. AUTO additionally
needs fresh production release evidence. If a future build/frame mismatch is detected, it publishes
`stop_kind=targeting_calibration`: no opener is requested, no action is sent, and no label is
recorded. Recapture the schema-v3 calibration and measured still-photo evidence (or explicitly
re-accept the assumption) for that exact build/device. Coverage remains the separate limitation
described in
[ops/STILL-PHOTO-DISCRIMINATOR.md](ops/STILL-PHOTO-DISCRIMINATOR.md) section 5d.
