#!/bin/zsh
# Operation Love — set up if needed, then launch, in one double-click. Portable:
# resolves the project from this script's own location, so it works on any
# machine. Dependencies are (re)installed ONLY when they change; otherwise it
# launches straight away. The app runs from source, so there's no build step.
cd "${0:A:h}" || exit 1
notify() { osascript -e "display notification \"$1\" with title \"Operation Love\" sound name \"$2\"" >/dev/null 2>&1; }

PY=".venv/bin/python"
STAMP=".venv/.oplove-deps-stamp"
need_install=0
if [ ! -x "$PY" ]; then
  echo "> Creating project virtualenv (.venv)..."
  python3 -m venv .venv || { echo "x venv creation failed (is python3 installed?)"; notify "Setup failed: venv creation." "Basso"; exit 1; }
  need_install=1
elif [ ! -e "$STAMP" ] || [ pyproject.toml -nt "$STAMP" ]; then
  need_install=1                       # deps changed since the last install
fi

if [ "$need_install" -eq 1 ]; then
  echo "> Installing / refreshing dependencies (first run can take a few minutes)..."
  "$PY" -m pip install -e ".[ml,bq,bumble]" || { echo "x pip install failed."; notify "Setup failed at pip install." "Basso"; exit 1; }
  "$PY" -m playwright install chromium >/dev/null 2>&1
  "$PY" -m operation_love.runtime || { echo "x runtime check failed."; notify "Setup: runtime check failed." "Basso"; exit 1; }
  touch "$STAMP"
  notify "Operation Love is ready - launching." "Glass"
else
  echo "> Dependencies already up to date."
fi

echo "OK - launching the control hub (Ctrl-C to quit)..."
exec "$PY" -m operation_love hub
