#!/usr/bin/env bash
# Test battery for deny-destructive-git.sh.
#
# Fixtures live in a TSV FILE rather than on the command line on purpose: the guard scans the
# whole command string, so passing a destructive fixture as an argument would put it at command
# position in this runner's own invocation and the guard would (correctly) block the test run.
# Reading them from a file is the only way to exercise the guard without tripping it.
set -uo pipefail
cd "$(dirname "$0")"

pass=0; fail=0
while IFS=$'\t' read -r want cmd; do
  [ -z "${want:-}" ] && continue
  case "$want" in \#*) continue;; esac
  out="$(printf '%s' "$cmd" | jq -Rs '{tool_input:{command:.}}' | bash ./deny-destructive-git.sh)"
  if [ -n "$out" ]; then got=block; else got=allow; fi
  if [ "$got" = "$want" ]; then
    pass=$((pass+1))
  else
    fail=$((fail+1)); printf 'FAIL want=%-5s got=%-5s :: %s\n' "$want" "$got" "$cmd"
  fi
done < deny-destructive-git.cases.tsv

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
