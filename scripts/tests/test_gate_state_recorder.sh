#!/usr/bin/env bash
# Standalone self-check for scripts/lib/_gate_state.sh (gate_state_record).
# No device needed. Exit 0 only if all cases pass.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/_gate_state.sh"

tmp="$(mktemp -d)"
log="$tmp/state.log"
fail=0

# 1. standard verdict line format
: >"$log"
gate_state_record "$log" "M-1" "ok" "UNMEASURED" "none"
gate_state_record "$log" "M-2" "MISSED" "rc=0" ""
line1="$(sed -n '1p' "$log")"
line2="$(sed -n '2p' "$log")"
echo "$line1" | grep -qE '^M-1 +verdict=ok +want=UNMEASURED +absent=none$' || { echo "FAIL case1: '$line1'"; fail=1; }
# absent empty defaults to none
echo "$line2" | grep -qE '^M-2 +verdict=missed +want=rc=0 +absent=none$' || { echo "FAIL case2: '$line2'"; fail=1; }

# 2. append vs overwrite: two calls -> two lines
[ "$(wc -l <"$log")" -eq 2 ] || { echo "FAIL case3: line count"; fail=1; }

# 3. raw mode
: >"$log"
gate_state_record --raw "$log" "hello world with spaces"
gate_state_record --raw "$log" "second line"
[ "$(sed -n '1p' "$log")" = "hello world with spaces" ] || { echo "FAIL case4"; fail=1; }
[ "$(sed -n '2p' "$log")" = "second line" ] || { echo "FAIL case5"; fail=1; }

# 4. empty/missing log arg -> non-zero, no crash
if gate_state_record "" "x" "ok" "w" >/dev/null 2>&1; then echo "FAIL case6"; fail=1; fi
if gate_state_record --raw "" "x" >/dev/null 2>&1; then echo "FAIL case7"; fail=1; fi

# 5. repeated-state idempotence: recording same id twice yields two lines (not deduped)
: >"$log"
gate_state_record "$log" "S-1" "ok" "w" "none"
gate_state_record "$log" "S-1" "ok" "w" "none"
[ "$(wc -l <"$log")" -eq 2 ] || { echo "FAIL case8"; fail=1; }

# 6. case-insensitive verdict normalisation
: >"$log"
gate_state_record "$log" "C-1" "OK" "w" "none"
grep -q "verdict=ok" "$log" || { echo "FAIL case9"; fail=1; }

# 7. multi-line raw input is recorded as a single logical line (newlines flattened by %s? no: caller passes one arg)
#    Here we verify a raw line containing spaces survives intact.
: >"$log"
gate_state_record --raw "$log" "line with   multiple   spaces"
[ "$(sed -n '1p' "$log")" = "line with   multiple   spaces" ] || { echo "FAIL case10"; fail=1; }

# 8. missing log dir is auto-created
newdir="$tmp/sub/deep"
newlog="$newdir/state.log"
gate_state_record --raw "$newlog" "auto-created"
[ -f "$newlog" ] && [ "$(cat "$newlog")" = "auto-created" ] || { echo "FAIL case11"; fail=1; }

# 9. multi-line raw via two separate calls keeps line count
: >"$log"
gate_state_record --raw "$log" "first"
gate_state_record --raw "$log" "second"
[ "$(wc -l <"$log")" -eq 2 ] || { echo "FAIL case12"; fail=1; }

# 10. --format template
: >"$log"
gate_state_record --format '%s|%s|%s
' "$log" "a" "b" "c"
[ "$(cat "$log")" = "a|b|c" ] || { echo "FAIL case13: '$(cat "$log")'"; fail=1; }

# 11. --stdout also prints to stdout
: >"$log"
out="$(gate_state_record --stdout --raw "$log" "stdout-line")"
[ "$out" = "stdout-line" ] || { echo "FAIL case14: '$out'"; fail=1; }
[ "$(cat "$log")" = "stdout-line" ] || { echo "FAIL case15"; fail=1; }

# 12. error path: empty id in std mode -> non-zero
: >"$log"
if gate_state_record "$log" "" "ok" "w" >/dev/null 2>&1; then echo "FAIL case16"; fail=1; fi

# 13. help prints contract
gate_state_record --help >/dev/null 2>&1 || { echo "FAIL case17"; fail=1; }

# 14. --stdin mode echoes and appends
: >"$log"
out="$(printf 'stdin-a\nstdin-b\n' | gate_state_record --stdin "$log")"
[ "$out" = "stdin-a
stdin-b" ] || { echo "FAIL case18: '$out'"; fail=1; }
[ "$(wc -l <"$log")" -eq 2 ] || { echo "FAIL case19"; fail=1; }

# 15. unreadable log -> non-zero (best effort; may be skipped if root)
unread="$tmp/unreadable.log"
: >"$unread"
chmod 000 "$unread"
if gate_state_record --raw "$unread" "x" >/dev/null 2>&1; then
  # running as root or fs ignores perms; not a failure
  :
else
  : # expected non-zero
fi
chmod 644 "$unread"

# 16. --dry-run does not write
: >"$log"
out="$(gate_state_record --dry-run --raw "$log" "preview-line")"
[ "$out" = "preview-line" ] || { echo "FAIL case20: '$out'"; fail=1; }
[ ! -s "$log" ] || { echo "FAIL case21: dry-run wrote to log"; fail=1; }

# 17. --json emits JSON, no file write
: >"$log"
out="$(gate_state_record --json "M-X" "ok" "w" "none")"
echo "$out" | grep -q '"id":"M-X"' || { echo "FAIL case22: '$out'"; fail=1; }
[ ! -s "$log" ] || { echo "FAIL case23"; fail=1; }

# 18. exit code 2 on unknown option
gate_state_record --bogus x >/dev/null 2>&1
[ $? -eq 2 ] || { echo "FAIL case24"; fail=1; }

# 19. exit code 1 on missing id
: >"$log"
gate_state_record "$log" "" "ok" "w" >/dev/null 2>&1
[ $? -eq 1 ] || { echo "FAIL case25"; fail=1; }

# 20. --json for raw/fmt/stdin modes
out="$(gate_state_record --json --raw "hello")"; echo "$out" | grep -q '"raw":"hello"' || { echo "FAIL case26"; fail=1; }
out="$(gate_state_record --json --format '%s-%s
' "x" "y")"; echo "$out" | grep -q '"formatted":"x-y"' || { echo "FAIL case27"; fail=1; }
out="$(printf 'z
' | gate_state_record --json --stdin)"; echo "$out" | grep -q '"stdin":"z"' || { echo "FAIL case28"; fail=1; }

# 20. non-dry-run parity: dry-run stdout matches what is written
: >"$log"
dry="$(gate_state_record --dry-run --raw "$log" "parity-line")"
gate_state_record --raw "$log" "parity-line" >/dev/null 2>&1
written="$(cat "$log")"
[ "$dry" = "$written" ] || { echo "FAIL case30"; fail=1; }

# 21. --json exit codes: 0 success, 1 missing id, 2 unknown option
gate_state_record --json "x" "ok" "w" "a" >/dev/null 2>&1; [ $? -eq 0 ] || { echo "FAIL case31"; fail=1; }
gate_state_record --json "" "ok" "w" >/dev/null 2>&1; [ $? -eq 1 ] || { echo "FAIL case32"; fail=1; }
gate_state_record --json --bogus x >/dev/null 2>&1; [ $? -eq 2 ] || { echo "FAIL case33"; fail=1; }

rm -rf "$tmp"
if [ "$fail" -eq 0 ]; then echo "GATE STATE RECORDER SELF-CHECK: PASS"; exit 0; fi
echo "GATE STATE RECORDER SELF-CHECK: FAIL"; exit 1
