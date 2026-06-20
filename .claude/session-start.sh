#!/bin/bash
# SessionStart hook — reset the harness-assigned claude/* branch to origin/main.
set -uo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" != "true" ] && exit 0
cd "${CLAUDE_PROJECT_DIR:-$(pwd)}" || exit 0
[ -z "$(git remote 2>/dev/null)" ] && exit 0
git fetch origin main >/dev/null 2>&1 || exit 0
git reset --hard origin/main >/dev/null 2>&1 || true
exit 0
