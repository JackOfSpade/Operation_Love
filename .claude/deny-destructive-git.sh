#!/usr/bin/env bash
# PreToolUse(Bash) guard: refuse git commands that DESTROY UNCOMMITTED WORK.
#
# WHY THIS EXISTS. On 2026-08-08 a subagent ran a repo-wide `git stash` to capture a pristine
# file and wiped ~8 other agents' in-flight edits plus the user's own uncommitted
# ops/ANTI-BOT-RESEARCH.md changes. On 2026-09-06 it happened AGAIN, as
# `git checkout -- config.yaml`, destroying an uncommitted prompt rewrite -- and that time the
# agent's brief contained an explicit, capitalised "NEVER run git stash/checkout/reset".
#
# The lesson is that a PROMPT-LEVEL BAN IS NOT A CONTROL. It was present and it was ignored.
# Only the harness can actually stop a tool call, so the ban lives here instead.
#
# `git checkout -- <path>` is strictly worse than `git stash`: a stash is itself a snapshot and
# is recoverable, while checkout overwrites the working tree from the index and the previous
# bytes exist nowhere afterwards.
#
# Scans the WHOLE command string rather than its prefix, because the destructive call is usually
# chained (`pytest && git checkout -- x`). For the same reason this hook deliberately does NOT
# use the `if: "Bash(git *)"` filter, which only prefix-matches and would miss exactly that case.
#
# Over-blocking is the intended bias: a false positive costs one rephrase, a false negative costs
# unrecoverable work. Read-only git is fully allowed, including the recommended escape hatches:
#   git show HEAD:path   read a pristine file without touching the tree
#   git stash create     write a snapshot COMMIT OBJECT without touching the tree or stash list
set -uo pipefail

payload="$(cat)"
cmd="$(printf '%s' "$payload" | jq -r '.tool_input.command // ""' 2>/dev/null || true)"
[ -z "$cmd" ] && exit 0

# Collapse newlines/runs of spaces so multi-line and oddly spaced commands match the same way.
norm="$(printf '%s' "$cmd" | tr '\n\t' '  ' | tr -s ' ')"

# `git` must sit in COMMAND POSITION to count: start of string, after a separator (; & | ( `),
# or after a shell keyword. Without this anchor the guard also fires on a command that merely
# QUOTES the pattern -- writing a test, a doc, or a memory file ABOUT destructive git -- which
# happened immediately on first use. Anchoring keeps the case that actually matters, the chained
# invocation (`pytest && git checkout -- x`), while letting `echo "git checkout -- x"` through.
CMDPOS='(^|[;&|(`]|\bthen |\bdo |\belse |\{ )[[:space:]]*(env +)?([A-Za-z_][A-Za-z0-9_]*=[^ ]* +)*'

# has <extended-regex-after-the-anchor> -- true when it matches at command position.
has() { printf '%s' "$norm" | grep -Eq "${CMDPOS}$1"; }

deny() {
  jq -n --arg r "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $r
    }
  }'
  exit 0
}

ESCAPE_HATCH='Read a pristine file with `git show HEAD:<path>` (no worktree mutation), or snapshot everything with `SNAP=$(git stash create)` which writes a commit object WITHOUT touching the tree or the stash list. To restore one file from that snapshot later: `git show $SNAP:<path>`.'

# --- git stash: allow only the read-only / non-mutating subcommands ----------------------
if has 'git +stash\b'; then
  if ! printf '%s' "$norm" | grep -Eq 'git +stash +(create|list|show)\b'; then
    deny "BLOCKED: 'git stash' swallows every OTHER agent's uncommitted edits and the user's own, because git operations are worktree-global and cannot be scoped to one agent. This exact command destroyed ~8 agents' work on 2026-08-08. ${ESCAPE_HATCH}"
  fi
fi

# --- git checkout: block the working-tree-overwriting forms ------------------------------
# Blocks: `checkout -- <path>`, `checkout .`, `checkout -f/--force`.
# Allows: `checkout -b <new>`, `checkout <branch>` (git refuses that itself if it would clobber).
if has 'git +checkout +(.* )?(--( |$)|\.( |$)|-f( |$)|--force( |$))'; then
  deny "BLOCKED: 'git checkout --' / 'checkout .' overwrites the working tree from the index and DISCARDS uncommitted changes irreversibly -- unlike a stash, the previous bytes then exist nowhere. This exact command destroyed an uncommitted config.yaml rewrite on 2026-09-06. ${ESCAPE_HATCH}"
fi

# --- git restore: its DEFAULT is to discard worktree changes -----------------------------
# `--staged` alone only unstages and is safe, so it is permitted; anything touching the
# worktree is not. To unstage without this command, plain `git reset <path>` is allowed.
if has 'git +restore\b'; then
  if printf '%s' "$norm" | grep -Eq 'git +restore +(-S|--staged)\b' \
     && ! printf '%s' "$norm" | grep -Eq 'git +restore +.*(-W|--worktree)\b'; then
    : # `git restore --staged <path>` only unstages; the worktree is untouched.
  else
    deny "BLOCKED: 'git restore' discards uncommitted working-tree changes by default. Only 'git restore --staged' (unstage only) is permitted. ${ESCAPE_HATCH}"
  fi
fi

# --- git reset: only the modes that throw away the working tree --------------------------
# `--soft` and the default `--mixed` keep file contents and stay allowed.
if has 'git +reset\b.*(--hard|--merge|--keep)\b'; then
  deny "BLOCKED: 'git reset --hard/--merge/--keep' discards uncommitted working-tree changes. 'git reset' (mixed) and 'git reset --soft' keep file contents and are allowed. ${ESCAPE_HATCH}"
fi

# --- git clean: deletes untracked files, including newly created test/tool files ---------
if has 'git +clean\b.*( -[a-zA-Z]*[fdx]| --force)'; then
  deny "BLOCKED: 'git clean' deletes untracked files, which includes new files other agents just created. Use 'git clean -n' to preview instead, or delete specific paths explicitly."
fi

# --- git switch: the force/discard forms are checkout by another name --------------------
if has 'git +switch\b.*(--discard-changes|--force\b| -f( |$))'; then
  deny "BLOCKED: 'git switch --discard-changes/--force' discards uncommitted working-tree changes. ${ESCAPE_HATCH}"
fi

# --- INDIRECT INVOCATION ------------------------------------------------------------------
# The command-position anchor above is what lets prose and test fixtures mention these commands
# safely, but it also means a git call made THROUGH an interpreter is not in command position
# and would slip past: python -c "subprocess.run(['git','checkout','--','x'])", sh -c "...", eval.
# So when the command carries an interpreter/exec wrapper, fall back to scanning ANYWHERE for a
# destructive git pattern. Prose is unaffected because plain `echo`/`cat`/`grep` are not wrappers.
if printf '%s' "$norm" | grep -Eq '(subprocess|os\.system|os\.popen|shutil|Popen|check_call|check_output|run\(|system\(|(sh|bash|zsh) +-c|eval |xargs)'; then
  if printf '%s' "$norm" | grep -Eq "git['\"), ]+(checkout['\"), ]+(--|\.|-f)|restore|clean['\"), ]+-[a-zA-Z]*[fdx])" \
     || printf '%s' "$norm" | grep -Eq "git['\"), ]+reset['\"), ]+(--hard|--merge|--keep)" \
     || printf '%s' "$norm" | grep -Eq "git['\"), ]+stash([\"'), ]|$)(?!.*(create|list|show))" 2>/dev/null \
     || { printf '%s' "$norm" | grep -Eq "git['\"), ]+stash" \
          && ! printf '%s' "$norm" | grep -Eq "stash['\"), ]+(create|list|show)"; }; then
    deny "BLOCKED: a destructive git command reached through an interpreter or exec wrapper (subprocess/sh -c/eval/xargs) still destroys uncommitted work. Routing it indirectly does not make it safe. ${ESCAPE_HATCH}"
  fi
fi

exit 0
