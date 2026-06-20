# Operation Love v2

Personal dating-app assistant. A **local, private** preference ranker decides
who *you'd* swipe right on (learned from your own swipes), and **Claude** writes
a natural, profile-specific opener. Targets **Bumble** (web) and **Hinge**
(Android emulator) — no AirDroid, no fixed pixel coordinates.

> The original AHK 2.0 project is archived in [`legacy/`](./legacy) for reference
> to the *vision*, not the implementation. See the analysis that motivated this
> rewrite in the project history.

## Architecture

```
drivers/      element-based control: Bumble (Playwright), Hinge (Appium/uiautomator2)
perception/   capture all photos + profile text -> Profile
vision/       local pyiqa quality filter + ArcFace/CLIP embeddings   [Phase 2]
ranker/       logistic-regression on YOUR swipe labels (SQLite)      [Phase 3]
opener/       Claude writes the opener, enforced JSON output         [Phase 4]
costing.py    client-side spend tracking + per-run budget guard
orchestrator  the loop (replaces main.ahk)
```

**Decision = local & private** (photos never leave your machine).
**Opener = Claude** (the one cloud call; minimal data; only for likes).

## Hardware

Runs entirely on an **Apple Silicon MacBook (M-series)** — PyTorch uses the
`mps` GPU backend automatically (`operation_love/device.py`). NVIDIA/CUDA and
CPU are also supported. No AMD/ROCm needed. At tens–hundreds of profiles/day,
embedding is effectively instant.

## Cost control

Anthropic has no API to read your remaining credit, so spend is tracked
client-side from each response's token `usage` against `budget.run_budget_usd`
in `config.yaml`. On reaching the cap — or on the actual out-of-credit error —
the bot either stops or keeps swiping without openers (`budget.on_exhausted`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,bumble,hinge,dev]"
cp .env.example .env   # add ANTHROPIC_API_KEY
pytest                 # cost-control tests run without a GPU or the SDK
```

## Status

Phase 0 (scaffold + config + SQLite + cost control) is in. Drivers, vision,
ranker, and the Claude opener wiring land in subsequent phases — see the build
plan.
