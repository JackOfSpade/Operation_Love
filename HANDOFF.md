# HANDOFF — Operation Love v2

> Continuation doc for picking this up in a new environment (e.g. Antigravity +
> Claude extension). Read this top-to-bottom, then `ops/RUNBOOK.md` for the live
> steps. Everything machine-independent is built and tested; the only thing left
> is the **live bring-up** (needs a real machine + real accounts), and it has
> **not been started yet**.

---

## 1. What this is

A personal, **local, private** tool that automates swiping on **Bumble** and
**Hinge** to the owner's taste, and writes Hinge openers with Claude. It replaces
an old, janky, PC-bound AutoHotkey 2.0 project (archived under `legacy/`, kept
only as a vision reference — do **not** port its implementation).

**Ground rules (explicit from the owner):**
- Personal / non-commercial / single-user. Results never shared publicly.
- Fairness / privacy / legal concerns are explicitly **out of scope** — it's all
  local and private. Don't add consent/ethics scaffolding; just build the tool.
- Scope is **Bumble + Hinge only** (the owner's current subscriptions).

---

## 2. How to work in this repo (process)

- **Branch:** develop on `claude/hopeful-hopper-mzu9fm`. Never commit straight to
  `main`. Don't open a PR unless explicitly asked.
- **Auto-merge:** `.github/workflows/` has a CI-gated auto-merge kit. Pushing to
  the feature branch runs CI (`ci.yml`) and, on green, auto-merges to `main` and
  deletes the branch server-side. This is independent of the IDE — it just works
  off the push. (If you want to keep iterating on one branch, be aware each green
  push merges; re-create/checkout the branch as needed.)
- **Commit style:** clear, descriptive messages. Do **not** put any model
  identifier in commits, code, or pushed artifacts.
- **Tests are the contract:** every change keeps the suite green (see §6). Tests
  are written to run with plain `python3 tests/test_x.py` (no pytest required,
  though `pytest -q` also works).

---

## 3. Architecture (the mental model)

One process, started by a **supervisor**, runs one **worker thread per enabled
app** concurrently. Workers share a ranker, a storage layer, and a global Claude
budget. Each app has a **driver** that finds UI elements by id/text (never pixel
coordinates), so it's resolution- and OS-independent.

```
supervisor.py
  ├── loads config, validates, detects device/capabilities (runtime.py/device.py)
  ├── builds shared: Store (BigQuery|SQLite), PreferenceModel, OpenerService(+CostTracker)
  └── per enabled app → Worker(thread)
                          ├── driver (Bumble: Playwright/CDP | Hinge: uiautomator2/ADB)
                          ├── mode = "observe" (you swipe; it learns) or "auto" (it swipes)
                          ├── decider = RankerDecider(quality filter + embedder + model)
                          └── opener_service (Hinge only; Bumble is the opener exception)
```

**Two modes:**
- **observe (shadow learning):** YOU swipe manually on real profiles in the live
  app; the worker captures each card, watches your like/pass, embeds it, stores a
  label (`source="manual"`), and **retrains the ranker live**. This is how taste
  is seeded — from real usage, not stock images. Start here.
- **auto:** the trained ranker scores each profile and likes/dislikes itself,
  sending Claude openers where the app allows (Hinge). Every auto swipe also
  becomes a label, so it keeps improving.

**Key design decisions already made (don't re-litigate):**
- **Ranking = local personalized model**, not a generic "beauty score." ArcFace
  (facial structure) + CLIP (style/vibe) embeddings concatenated → scikit-learn
  LogisticRegression (`PreferenceModel`, with a pure-Python LR fallback). Cold-
  start gating: stays `defer` until `min_labels_to_engage` labels across both
  classes. Research sweet spot ≈ 50–100 labels.
- **Openers = Claude**, but provider-swappable (`opener/opener.py` is the only
  Anthropic-specific file). Uses structured outputs so it returns a bare opener,
  never "Here's your opener:" preamble (the original ChatGPT pain point).
- **Storage = BigQuery as system-of-record + in-memory hot cache** (owner has a
  GCP connector; "essentially free at our scale"; offloads local resources).
  SQLite backend also exists as a fallback.
- **Budget:** client-side spend tracking from Claude usage tokens (no credit-
  balance API). `run_budget_usd` caps spend per run across all workers; out-of-
  credit (400/403 billing) is detected and handled (`stop` or
  `swipe_without_opener`).
- **Human-like timing** (`human.py`): log-normal delay multipliers (Box-Muller,
  σ=0.22). `human_delay()` for app-facing pauses (between swipes, before/after
  typing an opener, between profile scrolls — can be shorter or longer).
  `human_cooldown()` for backoffs/minimum waits (never below the anchor).
  Internal watchdog/poll/nav timeouts stay fixed on purpose (not app-observable).
- **OS/device auto-detect:** MPS (Apple) → CUDA (NVIDIA) → CPU, override with
  `OPLOVE_DEVICE`. No assumptions about deployment OS.
- **Bumble opener exception:** Bumble (hetero) can't message at swipe time
  (women-first / post-match), so Bumble auto-mode swipes only. Hinge sends the
  opener with the like. This is intended, not a bug.

---

## 4. Repo map

| Path | Purpose |
|---|---|
| `operation_love/supervisor.py` | Orchestrates everything; one process, per-app worker threads, signal handling, flush/close. Entry: `run()`. |
| `operation_love/worker.py` | Per-app thread. `_observe_loop` (learn from your swipes) / `_auto_loop` (swipe for you). Uses `human_delay`/`human_cooldown`. |
| `operation_love/config.py` | Loads/validates `config.yaml` into typed dataclasses. `validate()` fails fast (unknown app, bad mode, bigquery needs project_id, opener model needs pricing). |
| `operation_love/human.py` | Log-normal human timing (`human_delay`, `human_cooldown`). |
| `operation_love/costing.py` | `ModelPricing`, `Usage`, `CostTracker` (thread-safe), `BudgetExceeded`, out-of-credit detection. |
| `operation_love/limits.py` | `RateLimiter` (max_per_run / max_per_day). |
| `operation_love/runtime.py`, `device.py` | Capability + device detection; the `python3 -m operation_love.runtime` banner. |
| `operation_love/drivers/base.py` | `DatingAppDriver` ABC (open_session, next_profile, like, dislike, out_of_profiles, + observe: current_profile, wait_for_decision). |
| `operation_love/drivers/bumble.py` | Bumble web via Playwright (sync, CDP, persistent login dir). Observe via DOM click listener. **Selectors are guesses — verify live.** |
| `operation_love/drivers/hinge.py` | Hinge Android (emulator) via uiautomator2/ADB. Opener sent with like. **resource-ids are guesses — verify live.** |
| `operation_love/perception/capture.py` | `Profile` dataclass (photos + bio + prompts + meta). |
| `operation_love/vision/embed.py` | `Embedder` (lazy insightface ArcFace + open_clip), `embed_profile` → None if no face. |
| `operation_love/vision/quality.py` | `QualityFilter` (pyiqa CLIP-IQA), drops blurry photos, fail-open. |
| `operation_love/ranker/model.py` | `PreferenceModel` (sklearn LR or pure-py fallback), cold-start gating. |
| `operation_love/ranker/decider.py` | `RankerDecider`: quality→embed→model→like/dislike/no_face/defer. `Decision` dataclass. |
| `operation_love/ranker/store.py` | `Store` Protocol + `SQLiteStore` (thread-safe). |
| `operation_love/ranker/bigquery_store.py` | `BigQueryStore` (batched streaming writes, of-record). |
| `operation_love/ranker/__init__.py` | `make_store(cfg)` factory. |
| `operation_love/opener/opener.py` | `AnthropicOpener` (Claude, structured JSON output). Only provider-specific file. |
| `operation_love/opener/service.py` | `OpenerService`: shared, thread-safe, global budget, `maybe_opener`, stop-on-exhaust. |
| `operation_love/stats.py` | `python3 -m operation_love stats` readout. |
| `operation_love/__main__.py` | CLI: `run` (default) / `stats`, `--config`. |
| `tools/bumble_inspect.py` | **Guided live bring-up helper** — probes selectors (OK/MISS) + validates observe detection + prints paste-ready config. Run on the real machine. |
| `ops/RUNBOOK.md` | The human-in-the-loop live steps (install → verify → seed → auto). |
| `config.yaml` | All runtime config. No hardcoded paths/coords. |
| `legacy/` | Archived original AHK 2.0 project — vision reference only. |

---

## 5. Status — done vs. pending

**DONE (built + unit-tested, 13 suites green):**
- Supervisor + concurrent per-app workers + global budget.
- Observe mode (shadow learning) and auto mode.
- Local ranker (ArcFace+CLIP → LogisticRegression), quality filter, cold-start gating.
- Claude opener with structured output + cost/budget tracking + out-of-credit handling.
- BigQuery store (of-record) + SQLite fallback; `make_store` factory.
- Human-like log-normal timing applied to all app-facing delays.
- OS/device auto-detect; capability degradation when extras missing.
- Bumble observe `wait_for_decision()` implemented via DOM click listener (no
  network reverse-engineering needed).
- `tools/bumble_inspect.py` guided inspector.
- CI + auto-merge-to-main GitHub Actions.

**PENDING — the live bring-up (NOT started). This is the whole remaining task.**
It needs a real machine + real Bumble/Hinge accounts, so it can't run in a
headless cloud container. Do it **Bumble first**, then Hinge. Full checklist in
`ops/RUNBOOK.md`. Summary:

1. **Install** on the target machine (`pip install -e ".[ml,bumble,hinge,bq,dev]"`,
   `playwright install chromium`).
2. **Bumble selectors + observe detection** — run `python3 -m tools.bumble_inspect`,
   log in, reach a card, press ENTER. Fix any `MISS` selector in
   `config.yaml → apps.bumble.selectors`; confirm LIKE/PASS detection prints.
3. **Hinge resource-ids + observe tap detection** — Android emulator (Google Play
   image) over ADB; confirm ids in `config.yaml → apps.hinge.ids`. Note:
   `HingeDriver.wait_for_decision()` is still a `NotImplementedError` stub — it
   needs the same treatment Bumble got (watch which control you tap). This is the
   one remaining code TODO.
4. **Seed taste** — `mode: observe`, swipe ~50–100 real profiles; ranker flips
   `defer → ready` live. Watch with `python3 -m operation_love stats`.
5. **Go auto** — `mode: auto`, set `limits` + `budget`, run.

---

## 6. Run & test commands

```bash
# tests (no pytest needed; or `pytest -q`)
for t in tests/test_*.py; do PYTHONPATH=. python3 "$t"; done

# device/capability banner
python3 -m operation_love.runtime

# the app
python3 -m operation_love           # run (mode from config.yaml)
python3 -m operation_love stats     # progress readout

# Bumble live inspector (on the real machine, after pip install -e ".[bumble]")
python3 -m tools.bumble_inspect
```

---

## 7. Environment gotchas (hit during setup)

- **macOS uses `python3`, not `python`** — `python -m venv` silently no-ops.
  Create the venv with `python3 -m venv .venv && source .venv/bin/activate`
  (after activating, `python` resolves inside the venv).
- **Run from the repo root**, on branch `claude/hopeful-hopper-mzu9fm` — the
  inspector/drivers only exist there. `pip install -e .` from `~` fails ("no
  pyproject.toml").
- **BigQuery backend requires `storage.bigquery.project_id`** — `validate()`
  rejects it empty. For a quick first seed without GCP, set
  `storage.backend: sqlite`. Auth for BQ: `gcloud auth application-default login`
  (tables auto-create on first run).
- **`ANTHROPIC_API_KEY`** (copy `.env.example` → `.env`) is only needed for
  openers (auto mode / Hinge). Observe-mode seeding does **not** need it.
- **`[ml]` extra is heavy** (torch, insightface, open_clip, pyiqa). The inspector
  alone only needs `[bumble]`. Observe seeding needs `[ml]` (for embeddings).
- Force a device with `OPLOVE_DEVICE=cpu|cuda|mps` if auto-detect misbehaves.

---

## 8. Things to verify / watch live

- Bumble selectors and Hinge resource-ids are best-effort guesses; the inspector
  will tell you which are wrong. All are config-overridable — no code edits.
- Bumble observe detection covers **mouse clicks** on the like/pass buttons. If
  you also swipe via keyboard shortcuts, confirm coverage (extend the listener if
  needed).
- Hinge `wait_for_decision()` still needs implementing (see §5.3).
- Confirm the Claude budget/credit-exhaustion path behaves as configured once
  real API calls happen (auto mode).

---

*Immediate next action for whoever picks this up:* run the Bumble inspector on
the real machine (§6), paste its OK/MISS + LIKE/PASS output, then fix selectors
and proceed to observe-mode seeding.
