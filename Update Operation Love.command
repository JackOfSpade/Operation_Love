#!/bin/zsh
# Update / set up Operation Love. Portable — resolves the project from this
# script's own location, so it works on any machine. Creates a project-local
# .venv on first run, then installs/refreshes dependencies. The app runs from
# source (editable install): code changes are live next launch, no build step.
cd "${0:A:h}" || exit 1
notify() { osascript -e "display notification \"$1\" with title \"Operation Love\" sound name \"$2\"" >/dev/null 2>&1; }

echo "==========================================="
echo "  Operation Love - Update / Setup"
echo "==========================================="

if [ ! -x ".venv/bin/python" ]; then
  echo "> Creating project virtualenv (.venv)..."
  python3 -m venv .venv || { echo "x venv creation failed (is python3 installed?)"; notify "Update failed: venv creation." "Basso"; exit 1; }
fi
PY=".venv/bin/python"

echo "> Installing / refreshing dependencies (first run can take a few minutes)..."
"$PY" -m pip install -e ".[ml,bq,bumble]" || { echo "x pip install failed."; notify "Update failed at pip install." "Basso"; exit 1; }

echo "> Ensuring the Bumble browser is installed..."
"$PY" -m playwright install chromium >/dev/null 2>&1

echo "> Verifying..."
"$PY" -m operation_love.runtime || { echo "x runtime check failed."; notify "Update: runtime check failed." "Basso"; exit 1; }

echo ""
echo "OK - Operation Love is ready."
notify "Operation Love is up to date!" "Glass"

# Auto-close this Terminal window shortly after exit (one-shot script).
if [ "$TERM_PROGRAM" = "Apple_Terminal" ]; then
  TTY_NAME=$(tty)
  ( sleep 1
    osascript -e 'tell application "Terminal"
  repeat with w in windows
    try
      if tty of selected tab of w is "'"$TTY_NAME"'" then
        close w
        exit repeat
      end if
    end try
  end repeat
end tell' >/dev/null 2>&1
  ) &
  disown 2>/dev/null
fi
exit 0
