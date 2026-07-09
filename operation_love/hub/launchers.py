"""OS-specific double-click launcher scripts, written into the project folder.

Kept separate from state/server so editing the launcher shell-script bodies
never touches request-handling or process-lifecycle code.
"""
from __future__ import annotations

# Portable launcher body — written INTO the project folder and committed, so it
# travels with the repo and works on any machine. It resolves the project from
# the script's OWN location (no absolute paths) and uses a project-local .venv.
# __EXTRAS__ is substituted by make_launchers.
#
# The single launcher: install deps only when they change, then launch. The app
# runs from source (editable install), so code changes are live with no rebuild;
# we re-run pip only when pyproject.toml is newer than the last install (stamped
# inside .venv) or there's no .venv yet. Then it launches the hub; when the
# browser hub tab closes, the hub exits and the launcher closes this Terminal tab.
_MAC_UPDATE_RUN = r'''#!/bin/zsh
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
  "$PY" -m pip install -e ".[__EXTRAS__]" || { echo "x pip install failed."; notify "Setup failed at pip install." "Basso"; exit 1; }
  "$PY" -m playwright install chromium >/dev/null 2>&1
  "$PY" -m operation_love.runtime || { echo "x runtime check failed."; notify "Setup: runtime check failed." "Basso"; exit 1; }
  touch "$STAMP"
  notify "Operation Love is ready - launching." "Glass"
else
  echo "> Dependencies already up to date."
fi

echo "OK - launching the control hub (Ctrl-C to quit)..."
TTY_NAME="$(tty)"
"$PY" -m operation_love hub
status=$?
if [ "$status" -eq 0 ] && [ -n "$TTY_NAME" ]; then
  /usr/bin/nohup /usr/bin/osascript \
    -e 'delay 0.2' \
    -e 'tell application "Terminal"' \
    -e 'repeat with w in windows' \
    -e 'repeat with t in tabs of w' \
    -e "if tty of t is \"$TTY_NAME\" then" \
    -e 'close t' \
    -e 'return' \
    -e 'end if' \
    -e 'end repeat' \
    -e 'end repeat' \
    -e 'end tell' >/dev/null 2>&1 &
fi
exit "$status"
'''

_LINUX_UPDATE_RUN = ('#!/bin/sh\ncd "$(dirname "$0")" || exit 1\n'
                     'PY=".venv/bin/python"\nSTAMP=".venv/.oplove-deps-stamp"\nNEED=0\n'
                     'if [ ! -x "$PY" ]; then python3 -m venv .venv || exit 1; NEED=1\n'
                     'elif [ ! -e "$STAMP" ] || [ pyproject.toml -nt "$STAMP" ]; then NEED=1; fi\n'
                     'if [ "$NEED" -eq 1 ]; then\n'
                     '  "$PY" -m pip install -e ".[__EXTRAS__]" || exit 1\n'
                     '  "$PY" -m playwright install chromium >/dev/null 2>&1\n'
                     '  "$PY" -m operation_love.runtime || exit 1\n'
                     '  touch "$STAMP"\nfi\n'
                     'exec "$PY" -m operation_love hub\n')

_WIN_UPDATE_RUN = ('@echo off\r\ncd /d "%~dp0"\r\n'
                   'set "PY=.venv\\Scripts\\python.exe"\r\n'
                   'set "STAMP=.venv\\.oplove-deps-stamp"\r\n'
                   'set NEED=0\r\n'
                   'if not exist "%PY%" ( python -m venv .venv || exit /b 1 & set NEED=1 ) else (\r\n'
                   '  powershell -NoProfile -Command "if(!(Test-Path \'%STAMP%\') -or (Get-Item \'pyproject.toml\').LastWriteTime -gt (Get-Item \'%STAMP%\').LastWriteTime){exit 1}else{exit 0}"\r\n'
                   '  if errorlevel 1 set NEED=1\r\n'
                   ')\r\n'
                   'if "%NEED%"=="1" (\r\n'
                   '  "%PY%" -m pip install -e ".[__EXTRAS__]" || exit /b 1\r\n'
                   '  "%PY%" -m playwright install chromium\r\n'
                   '  "%PY%" -m operation_love.runtime || exit /b 1\r\n'
                   '  echo ok> "%STAMP%"\r\n'
                   ')\r\n'
                   '"%PY%" -m operation_love hub\r\n')


def make_launchers(config_path: str = "config.yaml", extras: str = "ml,bq,bumble") -> None:
    """Write ONE portable double-click launcher INTO the project folder.

    It resolves the project from the script's own location (no absolute paths)
    and uses a project-local .venv, so the committed file works on any machine.
    The app runs from source, so there's no build step; dependencies are
    (re)installed only when they actually change (pyproject.toml newer than the
    last install, or no .venv yet). It then opens the hub.
    """
    import stat
    import sys
    from pathlib import Path

    proj = Path(config_path).resolve().parent
    plat = sys.platform

    def _write(name: str, body: str, executable: bool) -> None:
        p = proj / name
        p.write_text(body.replace("__EXTRAS__", extras))
        if executable:
            p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"Hub: wrote: {p.name}")

    if plat == "darwin":
        _write("Operation Love.command", _MAC_UPDATE_RUN, True)
    elif plat.startswith("win"):
        _write("Operation Love.bat", _WIN_UPDATE_RUN, False)
    else:  # linux / *bsd
        _write("operation-love.sh", _LINUX_UPDATE_RUN, True)

    print("      One launcher: installs deps only when they change, then opens the hub.")
    print("      Portable across machines (relative paths + project-local .venv).")
