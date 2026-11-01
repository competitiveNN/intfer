#!/usr/bin/env bash
# Repeatable benchmark gate check for scripts/lib/_gate_snapshot.sh overhead.
#
# Usage: scripts/tests/bench_gate_snapshot.sh [N_FILES]
#   Default N_FILES=25. Exits 0 if overhead per file is under 20ms (adjustable
#   MAX_OVERHEAD_MS_PER_FILE). Prints timing for init, integrity, restore_all.
set -u
. "$PWD/scripts/lib/_gate_snapshot.sh"

N_FILES="${1:-25}"
MAX_OVERHEAD_MS_PER_FILE="${MAX_OVERHEAD_MS_PER_FILE:-20}"

TD="$(mktemp -d)"
for i in $(seq 1 "$N_FILES"); do
  head -c $((i * 1024)) /dev/urandom > "$TD/file_$i.txt"
done
cd "$TD" || exit 1
git init -q 2>/dev/null || true
git add -A 2>/dev/null || true
TARGETS=$(ls "$TD"/file_*.txt | tr '\n' ' ')

# init
start=$(date +%s%N)
gate_snap_init "$TARGETS" >/dev/null
end=$(date +%s%N)
init_ms=$(( (end - start) / 1000000 ))

# integrity
start=$(date +%s%N)
gate_snap_integrity "$TARGETS"
integrity_rc=$?
end=$(date +%s%N)
integrity_ms=$(( (end - start) / 1000000 ))

# modify all
for i in $(seq 1 "$N_FILES"); do
  echo "tampered" > "$TD/file_$i.txt"
done

# restore_all
start=$(date +%s%N)
gate_snap_restore_all "$TARGETS" >/dev/null
restore_rc=$?
end=$(date +%s%N)
restore_ms=$(( (end - start) / 1000000 ))

# cleanup
cd / || exit 1
rm -rf "$TD"
rm -rf "${GATE_SNAP_DIR:-}"

per_init=$(( init_ms / N_FILES ))
per_integrity=$(( integrity_ms / N_FILES ))
per_restore=$(( restore_ms / N_FILES ))
per_max=$per_init
[ "$per_integrity" -gt "$per_max" ] && per_max=$per_integrity
[ "$per_restore" -gt "$per_max" ] && per_max=$per_restore

echo "gate_snapshot benchmark (N=$N_FILES):"
echo "  init:        ${init_ms} ms (${per_init} ms/file)"
echo "  integrity:   ${integrity_ms} ms (${per_integrity} ms/file) rc=$integrity_rc"
echo "  restore_all: ${restore_ms} ms (${per_restore} ms/file) rc=$restore_rc"
echo "  max overhead per file: ${per_max} ms (threshold: ${MAX_OVERHEAD_MS_PER_FILE} ms)"

if [ "$per_max" -gt "$MAX_OVERHEAD_MS_PER_FILE" ]; then
  echo "FAIL: per-file overhead ${per_max} ms exceeds threshold ${MAX_OVERHEAD_MS_PER_FILE} ms"
  exit 1
fi

echo "PASS: gate_snapshot overhead within threshold"
exit 0
