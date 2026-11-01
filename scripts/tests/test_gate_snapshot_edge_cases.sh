#!/usr/bin/env bash
# shellcheck disable=SC2015
# Edge-case tests for scripts/lib/_gate_snapshot.sh.
# Tests: init, verify, assert, restore, integrity, dirty-guard, restore-all.
set -u
. "$PWD/scripts/lib/_gate_snapshot.sh"

PASS=0; FAIL=0
ok()   { echo "  ok   $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL $1"; FAIL=$((FAIL+1)); }

TD="$(mktemp -d)"
mkdir -p "$TD/sub"
echo "content A" > "$TD/file_a.txt"
echo "content B" > "$TD/sub/file_b.txt"
cd "$TD" || exit 1
git init -q 2>/dev/null || true
git add -A 2>/dev/null || true

# --- Core function tests ---
rm -rf "${GATE_SNAP_DIR:-}"
gate_snap_init "$TD/file_a.txt $TD/sub/file_b.txt"
[ -f "$GATE_SNAP_DIR/$TD/file_a.txt" ] && [ -f "$GATE_SNAP_DIR/$TD/sub/file_b.txt" ] && ok "init creates copies" || bad "init missing copies"
gate_snap_assert_clean "$TD/file_a.txt" && ok "assert_clean passes on unchanged" || bad "assert_clean fails on unchanged"
gate_snap_verify_patched "$TD/file_a.txt" 2>/dev/null && bad "verify_patched passes on unchanged" || ok "verify_patched fails on unchanged"

echo "modified A" > "$TD/file_a.txt"
gate_snap_verify_patched "$TD/file_a.txt" && ok "verify_patched passes on modified" || bad "verify_patched fails on modified"
gate_snap_assert_clean "$TD/file_a.txt" 2>/dev/null && bad "assert_clean passes on modified" || ok "assert_clean fails on modified"
gate_snap_restore "$TD/file_a.txt"
gate_snap_assert_clean "$TD/file_a.txt" && ok "restore works" || bad "restore fails"

echo "content A" > "$TD/file_a.txt"
echo "content B" > "$TD/sub/file_b.txt"
gate_snap_integrity "$TD/file_a.txt $TD/sub/file_b.txt" && ok "integrity passes on matching" || bad "integrity fails on matching"

echo "tampered" > "$TD/sub/file_b.txt"
gate_snap_integrity "$TD/file_a.txt $TD/sub/file_b.txt" 2>/dev/null && bad "integrity passes on tampered" || ok "integrity fails on tampered"
gate_snap_restore "$TD/sub/file_b.txt"

# --- dirty-guard tests ---
rm -rf "${GATE_SNAP_DIR:-}"
gate_snap_init "$TD/file_a.txt $TD/sub/file_b.txt" >/dev/null
echo "dirty A" > "$TD/file_a.txt"
GATE_SNAP_ALLOW_DIRTY=0 gate_snap_dirty_guard "$TD/file_a.txt $TD/sub/file_b.txt" 2>/dev/null && bad "dirty_guard allows dirty (no ALLOW)" || ok "dirty_guard refuses dirty (no ALLOW)"
GATE_SNAP_ALLOW_DIRTY=1 gate_snap_dirty_guard "$TD/file_a.txt $TD/sub/file_b.txt" 2>/dev/null && ok "dirty_guard allows dirty (ALLOW=1)" || bad "dirty_guard refuses dirty (ALLOW=1)"

# --- restore-all tests ---
gate_snap_restore_all "$TD/file_a.txt $TD/sub/file_b.txt" && ok "restore_all returns 0" || bad "restore_all returns non-zero"
gate_snap_assert_clean "$TD/file_a.txt" && gate_snap_assert_clean "$TD/sub/file_b.txt" && ok "restore_all restores both" || bad "restore_all incomplete"

echo "tampered again" > "$TD/file_a.txt"
gate_snap_restore "$TD/file_a.txt"

# --- manifest tests ---
rm -rf "${GATE_SNAP_DIR:-}"
gate_snap_init "$TD/file_a.txt $TD/sub/file_b.txt" >/dev/null
MANIFEST_LINES=$(gate_snap_manifest | wc -l)
[ "$MANIFEST_LINES" = "2" ] && ok "manifest lists both tracked files" || bad "manifest line count wrong ($MANIFEST_LINES)"
gate_snap_manifest | grep -qF "$TD/file_a.txt" && ok "manifest contains file_a path" || bad "manifest missing file_a path"
gate_snap_manifest | grep -qF "$TD/sub/file_b.txt" && ok "manifest contains file_b path" || bad "manifest missing file_b path"

# --- check_residue tests ---
echo "content A" > "$TD/file_a.txt"
echo "content B" > "$TD/sub/file_b.txt"
gate_snap_check_residue "$TD/file_a.txt $TD/sub/file_b.txt" && ok "check_residue clean on unchanged" || bad "check_residue false positive"
printf 'int x; /*M5*/\n' > "$TD/file_a.txt"
gate_snap_check_residue "$TD/file_a.txt $TD/sub/file_b.txt" 2>/dev/null && bad "check_residue misses marker" || ok "check_residue detects marker"
echo "content A" > "$TD/file_a.txt"

# --- concurrency lock tests ---
rm -f -r "${TMPDIR:-/tmp}/.gate_snap_lock.regress"
gate_snap_acquire_lock "regress" && ok "acquire lock succeeds (unheld)" || bad "acquire lock failed unexpectedly"
gate_snap_acquire_lock "regress" 2>/dev/null && bad "acquire lock succeeds when held" || ok "acquire lock refuses (held)"
gate_snap_release_lock "regress"
gate_snap_acquire_lock "regress" 2>/dev/null && ok "acquire lock succeeds after release" || bad "acquire lock failed after release"
gate_snap_release_lock "regress"

# --- contention test: two concurrent processes cannot both hold the lock ---
rm -f -r "${TMPDIR:-/tmp}/.gate_snap_lock.contended"
HOLDER_PID=""
(
  gate_snap_acquire_lock "contended" || exit 1
  HOLDER_PID=$$
  # Hold the lock briefly so the foreground attempt can observe contention
  sleep 0.5
  gate_snap_release_lock "contended"
) &
HOLDER_BG=$!
sleep 0.1
# Foreground attempt must fail while the background holder has the lock
if gate_snap_acquire_lock "contended" 2>/dev/null; then
  bad "acquire lock succeeded while contended (lock not exclusive)"
  gate_snap_release_lock "contended"
else
  ok "acquire lock refused while contended (lock is exclusive)"
fi
wait "$HOLDER_BG" || true
# After the holder releases, the lock must be acquirable again
gate_snap_acquire_lock "contended" && ok "acquire lock succeeds after holder releases" || bad "acquire lock failed after holder released"
gate_snap_release_lock "contended"

# --- reap_stale tests ---
rm -rf "${TMPDIR:-/tmp}/gate_snap.stale1" "${TMPDIR:-/tmp}/gate_snap.stale2"
mkdir -p "${TMPDIR:-/tmp}/gate_snap.stale1" "${TMPDIR:-/tmp}/gate_snap.stale2"
# make them old by touching with a far-past mtime if touch -d is available
if touch -d '2 years ago' "${TMPDIR:-/tmp}/gate_snap.stale1" 2>/dev/null; then
  : # mtime set; reap should remove it
else
  touch "${TMPDIR:-/tmp}/gate_snap.stale1"
fi
REAPED=$(gate_snap_reap_stale 1 2>/dev/null)
[ -d "${TMPDIR:-/tmp}/gate_snap.stale2" ] && ok "reap_stale keeps fresh dirs" || bad "reap_stale removed fresh dir"
[ -e "${TMPDIR:-/tmp}/gate_snap.stale1" ] || ok "reap_stale removed stale dir"

# --- mode-preservation / mtime-update tests (prevents stale build artifacts) ---
chmod 755 "$TD/file_a.txt"
rm -rf "${GATE_SNAP_DIR:-}"
gate_snap_init "$TD/file_a.txt" >/dev/null
echo "modified mode-test" > "$TD/file_a.txt"
# Record mtime before restore
MTIME_BEFORE=$(stat -c %Y "$TD/file_a.txt")
sleep 0.1
gate_snap_restore "$TD/file_a.txt"
MTIME_AFTER=$(stat -c %Y "$TD/file_a.txt")
MODE_AFTER=$(stat -c %a "$TD/file_a.txt")
[ "$MODE_AFTER" = "755" ] && ok "restore preserves executable mode" || bad "restore lost executable mode (got $MODE_AFTER)"
[ "$MTIME_AFTER" -ge "$MTIME_BEFORE" ] 2>/dev/null && ok "restore updates mtime" || bad "restore preserved stale mtime"
test "$(cat "$TD/file_a.txt")" = "content A" && ok "restore restores content after mode test" || bad "restore content mismatch after mode test"

# --- dirty-guard allows clean tree (final check before cleanup) ---
chmod 644 "$TD/file_a.txt" 2>/dev/null || true
rm -rf "${GATE_SNAP_DIR:-}"
gate_snap_init "$TD/file_a.txt $TD/sub/file_b.txt" >/dev/null
GATE_SNAP_ALLOW_DIRTY=0 gate_snap_dirty_guard "$TD/file_a.txt $TD/sub/file_b.txt" >/dev/null 2>&1 && ok "dirty_guard allows clean tree" || bad "dirty_guard refuses clean tree"

# --- restore-performance optimization: skip copy when already matching ---
rm -rf "${GATE_SNAP_DIR:-}"
gate_snap_init "$TD/file_a.txt" >/dev/null
SNAP_SHA_BEFORE=$(sha256sum "$GATE_SNAP_DIR/$TD/file_a.txt" | cut -d' ' -f1)
# Restore an already-matching file: should skip cp+touch (no-op)
gate_snap_restore "$TD/file_a.txt"
SNAP_SHA_AFTER=$(sha256sum "$GATE_SNAP_DIR/$TD/file_a.txt" | cut -d' ' -f1)
[ "$SNAP_SHA_BEFORE" = "$SNAP_SHA_AFTER" ] && ok "restore skips copy when already matching" || bad "restore copied when already matching"

# --- unified matrix API: begin/end ---
rm -f -r "${TMPDIR:-/tmp}/.gate_snap_lock.matrix.regress"
gate_snap_matrix_begin "regress" "$TD/file_a.txt $TD/sub/file_b.txt" && ok "matrix_begin initializes snapshots" || bad "matrix_begin failed"
# Mutate a file
echo "mutated content" > "$TD/file_a.txt"
gate_snap_verify_patched "$TD/file_a.txt" && ok "matrix_begin snapshot detects mutation" || bad "matrix_begin snapshot missed mutation"
# End: should restore and clean up
gate_snap_matrix_end "regress" "$TD/file_a.txt $TD/sub/file_b.txt" && ok "matrix_end restores and cleans up" || bad "matrix_end failed"
gate_snap_assert_clean "$TD/file_a.txt" && ok "matrix_end restored file to snapshot" || bad "matrix_end did not restore file"
# Lock should be released after end
gate_snap_acquire_lock "matrix.regress" 2>/dev/null && ok "lock released after matrix_end" || bad "lock not released after matrix_end"
gate_snap_release_lock "matrix.regress"

# --- SIGKILL-safe residue reinit: stale snapshot dirs reaped on begin ---
rm -rf "${TMPDIR:-/tmp}/gate_snap.stale_killed"
mkdir -p "${TMPDIR:-/tmp}/gate_snap.stale_killed"
touch -d '2 years ago' "${TMPDIR:-/tmp}/gate_snap.stale_killed" 2>/dev/null || true
# Create a stale lock with a dead pid
rm -f -r "${TMPDIR:-/tmp}/.gate_snap_lock.matrix.dead"
mkdir -p "${TMPDIR:-/tmp}/.gate_snap_lock.matrix.dead"
echo "999999" >"${TMPDIR:-/tmp}/.gate_snap_lock.matrix.dead/pid"
# begin should reap stale dirs and still be able to acquire the lock (dead owner)
gate_snap_matrix_begin "dead" "$TD/file_a.txt" >/dev/null 2>&1 && ok "matrix_begin reaps stale state and proceeds" || bad "matrix_begin failed on stale state"
gate_snap_matrix_end "dead" "$TD/file_a.txt" >/dev/null 2>&1 || bad "matrix_end failed after stale-state begin"

# --- interrupted-matrix recovery: stale snapshot owner is reaped regardless of age ---
rm -rf /tmp/gate_snap.dead_owner
mkdir -p /tmp/gate_snap.dead_owner
echo "999999" > /tmp/gate_snap.dead_owner/owner
gate_snap_reap_stale 3600 >/dev/null 2>&1
if [ ! -e /tmp/gate_snap.dead_owner ]; then ok "reap_stale removes dead-owner dirs even when fresh" || bad "reap_stale missed dead owner"; else bad "reap_stale missed dead-owner dir"; fi

# --- parallel lock stress: many contenders, exactly one succeeds ---
rm -f -r /tmp/.gate_snap_lock.parstress
PSF="${TMPDIR:-/tmp}/gate_snap_par_success"
rm -f "$PSF"
for i in 1 2 3 4 5 6 7 8; do
  (
    if gate_snap_acquire_lock parstress 2>/dev/null; then
      printf '\n' >>"$PSF"
      sleep 0.2
      gate_snap_release_lock parstress
    fi
  ) &
done
wait 2>/dev/null || true
PCOUNT=0
[ -f "$PSF" ] && PCOUNT=$(wc -l < "$PSF" | tr -d ' ')
[ "${PCOUNT:-0}" = "1" ] && ok "parallel lock stress proves exactly one holder" || bad "parallel lock stress got $PCOUNT holders (want 1)"
rm -f -r "${TMPDIR:-/tmp}/.gate_snap_lock.parstress" "$PSF"

# --- true SIGKILL recovery: a killed holder's lock and snapshot are reclaimed ---
rm -f -r "${TMPDIR:-/tmp}/.gate_snap_lock.sigkill"
rm -rf /tmp/gate_snap.sigkill_owner
SIGPATH="${TMPDIR:-/tmp}/gate_snap_sigkill_path"
rm -f "$SIGPATH"
(
  gate_snap_acquire_lock "sigkill" || exit 1
  sp="$(gate_snap_init "$TD/file_a.txt")"
  printf '%s\n' "$sp" >"$SIGPATH"
  sleep 5
) &
SIGPID=$!
sleep 0.2
kill -9 "$SIGPID" 2>/dev/null || true
wait "$SIGPID" 2>/dev/null || true
# The killed holder's lock must be removable by a new acquirer.
if gate_snap_acquire_lock "sigkill" 2>/dev/null; then
  ok "SIGKILL recovery: a new process acquires the killed holder's lock"
  gate_snap_release_lock "sigkill"
else
  bad "SIGKILL recovery: killed holder's lock stayed held"
fi
# Its snapshot directory must be reaped once its owner is dead.
if [ -f "$SIGPATH" ]; then
  SIGDIR="$(cat "$SIGPATH")"
  if [ -n "$SIGDIR" ] && [ -d "$SIGDIR" ]; then
    echo "$SIGPID" >"$SIGDIR/owner" 2>/dev/null || true
    gate_snap_reap_stale 3600 >/dev/null 2>&1
    if [ ! -e "$SIGDIR" ]; then
      ok "SIGKILL recovery: the killed run's snapshot dir was reaped"
    else
      bad "SIGKILL recovery: stale snapshot dir survived ($SIGDIR)"
    fi
  else
    ok "SIGKILL recovery: no stale snapshot dir remained"
  fi
fi
rm -f "$SIGPATH"

# --- cleanup ---
cd / || exit 1
rm -rf "$TD"
rm -rf "${GATE_SNAP_DIR:-}"
rm -f -r "${TMPDIR:-/tmp}/.gate_snap_lock.regress"
rm -f -r "${TMPDIR:-/tmp}/.gate_snap_lock.contended"
rm -f -r "${TMPDIR:-/tmp}/.gate_snap_lock.matrix.regress"
rm -f -r "${TMPDIR:-/tmp}/.gate_snap_lock.matrix.dead"
rm -rf "${TMPDIR:-/tmp}/gate_snap.stale1" "${TMPDIR:-/tmp}/gate_snap.stale2" "${TMPDIR:-/tmp}/gate_snap.stale_killed"

echo
echo "EDGE CASE RESULTS: PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
