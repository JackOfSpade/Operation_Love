#!/usr/bin/env bash
# Sourceable decision helpers for .github/workflows/auto-merge-to-main.yml.
#
# Factors the two decision PREDICATES that matter most — "is this SHA's CI
# green?" (fail-closed) and "is this ref already merged into that one?" — into
# functions the workflow `source`s directly, so it's straightforward to add
# unit tests against the same implementation the workflow runs (no shadow
# copy to drift).

# ci_conclusion_from_json <json> — given the JSON body of
# `gh api repos/<repo>/actions/workflows/<CI_WORKFLOW_FILE>/runs?head_sha=<sha>&per_page=1`
# (or "" / unparseable JSON, e.g. from a failed API call), print the conclusion
# of the most recent run:
#   - "success" / "failure" / "in_progress" / ... — a real run's conclusion.
#   - "none"    — valid JSON but no matching run yet (new commit; CI hasn't started/finished).
#   - "error"   — the API call failed, or returned empty/unparseable JSON.
# Requires `jq`. NOTE: jq treats a completely empty stdin as "no output, exit 0" (not an error), so an
# empty/missing `$1` is checked explicitly rather than relying on jq's own exit code for that case.
ci_conclusion_from_json() {
  local out
  if [ -n "$1" ] \
     && out="$(printf '%s' "$1" | jq -r '.workflow_runs[0].conclusion // "none"' 2>/dev/null)" \
     && [ -n "$out" ]; then
    printf '%s\n' "$out"
  else
    echo "error"
  fi
}

# is_ci_green <conclusion> — fail-closed: ONLY an exact "success" counts as green. Any other value
# (in-progress, failure, "none", "error") is NOT green, so a missing/ambiguous CI result blocks the
# merge instead of silently defaulting to allow.
is_ci_green() {
  [ "$1" = "success" ]
}

# is_ancestor_of <maybe-ancestor-ref> <descendant-ref> — true (exit 0) if the first ref's commit is
# reachable from the second, i.e. the first is already merged into the second. Used both for "already
# contained in main, just clean up" and for the re-confirm-before-delete ancestry check (a branch whose
# tip advanced after the merge decision was made must NOT look like an ancestor, so it must NOT be
# deleted).
is_ancestor_of() {
  git merge-base --is-ancestor "$1" "$2"
}
