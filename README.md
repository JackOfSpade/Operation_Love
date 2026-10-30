# Operation Love

Operation Love is a personal dating-app assistant with a local preference ranker,
configurable Gemini openers, and a browser-based control Hub. The maintained Python
package lives in `operation_love/`; configuration and device readiness checks determine
which apps and modes are available.

## Setup

Use Python 3.11 or newer. From the repository root on macOS or Linux:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[ml,bq,hinge]'
python -m operation_love.runtime
python -m operation_love hub
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. On macOS,
`Operation Love.command` also installs dependencies when needed and launches the Hub.
Follow the [setup runbook](ops/RUNBOOK.md) for credentials, storage, and configuration,
and the [Hinge device runbook](ops/HINGE-PIXEL-RUNBOOK.md) for Android setup and calibration.
Keep local credentials in the ignored `.env` file described by `.env.example`.

## Development

Install the test dependencies in the same environment:

```sh
python -m pip install -e '.[dev,ml,hinge]'
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python tools/backfill_prompt_eras.py --check
python -m unittest discover -s .github/tests -p 'test_merge_validated_pr.py'
```

Pytest uses at most two automatic workers to limit memory pressure. Use `-n0` for
serial debugging. Tests isolate device locks and use synthetic or mocked transports.
CI additionally uses `scripts/run_ci_test_files.py` and reconciles all shard manifests
to verify complete test coverage.

## Repository layout

- `operation_love/`: application runtime, Hub, drivers, vision, ranking, and storage.
- `tests/`: regression and integration tests.
- `tools/`: operator utilities and offline evidence analysis. Device-facing commands
  use the shared Android lock.
- `scripts/` and `.github/`: test sharding, workflows, and merge policy checks.
- `ops/`: setup instructions, operational runbooks, and retained release evidence.
- `legacy/`: archived Windows automation and research integrations. The archived BMI
  entry point reports unavailable inference; its analysis pipeline stops without a
  decision when required analysis is unavailable. Vendored research and OCR code remain
  separate from the maintained package.
