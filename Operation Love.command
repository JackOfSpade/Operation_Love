#!/bin/zsh
# Operation Love — open the control hub. Portable: works wherever this folder is.
# Run "Update Operation Love" once first to create the .venv + install deps.
cd "${0:A:h}" || exit 1
PY=".venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" -m operation_love hub
