#!/usr/bin/env bash
# Prints a short orientation at session start: git state, long jobs, next step.
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 0
b=$(git branch --show-current 2>/dev/null)
ahead=$(git rev-list --count main..HEAD 2>/dev/null || echo "?")
dirty=$(git status --porcelain 2>/dev/null | grep -vc ' 2\.')
echo "branch: $b (ahead of main: $ahead, changed files: $dirty)"
p=$(ps -axo args= 2>/dev/null | grep -E '^caffeinate .*\.py' | grep -oE '[A-Za-z_0-9]+\.py.*' | cut -c1-80 | sort -u | head -3)
[ -n "$p" ] && echo "running (caffeinate): $p"
if [ -f .claude/STATE.md ]; then
  echo "--- next (from .claude/STATE.md):"
  awk '/^## Next/{f=1; next} /^## /{f=0} f && NF' .claude/STATE.md | head -8
fi
exit 0
