# Operation Love v2

Personal dating-app assistant. A **local, private** preference ranker decides
who *you'd* swipe right on (learned from your own swipes), and **Claude** writes
a natural, profile-specific opener. Targets **Bumble** (web) and **Hinge**
(physical Android over host-side ADB + vision template-match) — no AirDroid, no fixed pixel coordinates.

> The original AHK 2.0 project is archived in [`legacy/`](./legacy) for reference
> to the *vision*, not the implementation. See the analysis that motivated this
> rewrite in the project history.

## Architecture

```
drivers/      element-based control: Bumble (Playwright), Hinge (host-side ADB + vision)
perception/   capture all photos + profile text -> Profile
vision/       local pyiqa quality filter + ArcFace/CLIP embeddings   [Phase 2]
ranker/       logistic-regression on YOUR swipe labels (BigQuery/SQLite)  [Phase 3]
opener/       Claude writes the opener, enforced JSON output         [Phase 4]
costing.py    client-side spend tracking + per-run budget guard
supervisor    one worker per enabled app; owns shutdown + flush
worker.py     the per-app loop, observe or auto (replaces main.ahk)
hub/          local control panel (state, server, page, launchers)
```

**Decision = local & private** (photos never leave your machine).
**Opener = Claude** (the one cloud call; minimal data; only for likes).

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
| Bumble (Playwright web) | ✅ | ✅ | ✅ |
| Hinge (physical Android phone, host-side ADB only) | ✅ (USB or wireless ADB) | ✅ | ✅ |
| Claude opener | cloud — any OS | cloud | cloud |

## Cost control

Anthropic has no API to read your remaining credit, so spend is tracked
client-side from each response's token `usage` against `budget.run_budget_usd`
in `config.yaml`. On reaching the cap — or on the actual out-of-credit error —
the bot either stops or keeps swiping without openers (`budget.on_exhausted`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,bq,bumble,hinge,dev]"
cp .env.example .env   # add ANTHROPIC_API_KEY
pytest                 # cost-control tests run without a GPU or the SDK
```

## Concurrency & deployment

**One process runs all enabled apps at once.** A supervisor launches one worker
per app (`enabled_apps: [bumble, hinge]`), all sharing the taste model, the
BigQuery store, and a single **global** opener budget. Because the drivers are
element-based (CDP / ADB) rather than mouse-based, they don't fight over your
cursor, don't stop you using the machine, and several can run in parallel.

Run it with `python -m operation_love`. It's OS-agnostic (macOS/Windows/Linux,
GPU or CPU) and host-agnostic — since state lives in BigQuery, you can develop
on one machine and deploy the same code to an always-on box with no migration.

## Status

Done: scaffold, config, storage (BigQuery + SQLite), runtime auto-detect,
cost-control/global-budget, the Claude opener, the **supervisor + per-app
worker** loop, and **the vision + personal-ranker core** — quality pre-filter,
ArcFace+CLIP embeddings, and a logistic-regression `PreferenceModel` that learns
your taste with cold-start gating (`defer` until enough labels). All pure logic
is covered by offline tests; model *inference* runs on a machine with the `ml`
extra installed.

**Two modes** (`mode: observe | auto`):
- **observe** — shadow learning: you swipe manually on real profiles, the bot
  captures each, watches your like/pass, stores it as a label, and **retrains
  the ranker live** (transitions itself from `defer` → ready mid-session). No
  autonomous swiping. This is how you seed your taste — from real usage, not
  stock images.
- **auto** — the bot swipes for you with the learned model, and every swipe is
  still a label.

**Both apps are code-complete** (Bumble via Playwright, Hinge via host-side ADB +
vision-located taps — no on-device helper, no emulator; see ops/HINGE-PIXEL-RUNBOOK.md),
with autonomous-mode **rate limits** (`max_per_run` / `max_per_day`), a **stats**
readout (`python -m operation_love stats`), and human-like pacing.

Everything machine-independent is done and unit-tested (run `pytest` for the
current suite/test count). The only remaining work needs your machine + a
real account, and it's all batched in
**[ops/RUNBOOK.md](ops/RUNBOOK.md)**: install, log in, confirm the config-driven
DOM selectors / resource-ids and the observe-mode like/pass hooks, seed your
taste in observe mode, then flip to auto.
