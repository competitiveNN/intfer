#!/usr/bin/env bash
# SHARED: snapshot/restore discipline for mutation matrices that edit tracked
# source in place.
#
# Why: `git checkout --` destroys uncommitted work; a hash-vs-HEAD check
# mislabels an intentional repair as a state failure. Snapshotted bytes compared
# against the startup copy make the matrix run over an uncommitted working set
# safely, with an opt-in refusal for the default (careless) invocation.
#
# Env knobs:
#   GATE_SNAP_ALLOW_DIRTY=1   run over an uncommitted working set (opt-in)
#   INTFER_BUILD_DIR         passed through untouched
#   GATE_SNAP_REAP_STALE=1    reap stale snapshot dirs from killed runs on init
#
# API (all operate on the file named $1):
#   gate_snap_init "space separated target list"   create SNAPDIR + copies + manifest
#   gate_snap_dirty_guard "space separated target list"   refuse unless opt-in;
#       prints INPUT: UNCOMMITTED lines and returns 1 on refusal
#   gate_snap_verify_patched FILE   rc 0 iff FILE differs from its snapshot
#   gate_snap_assert_clean FILE      rc 0 iff FILE equals its snapshot
#   gate_snap_restore FILE           restore FILE from snapshot; rc 0 iff match
#   gate_snap_integrity LIST         rc 0 iff every target matches its snapshot
#   gate_snap_restore_all LIST        rc 0 iff every target restored
#   gate_snap_manifest                  print the init-time manifest (sha256 path)
#   gate_snap_acquire_lock NAME        mkdir lock; rc 0 iff acquired (concurrency)
#   gate_snap_release_lock NAME        release a previously acquired lock
#   gate_snap_reap_stale MAX_AGE_SEC   remove gate_snap.* dirs older than MAX_AGE_SEC
#   gate_snap_check_residue LIST       rc 0 iff no mutation markers remain in targets
#   gate_snap_matrix_begin NAME LIST   reap stale + acquire per-run lock + init snapshots
#   gate_snap_matrix_end NAME LIST     restore all + release lock + cleanup + residue check
#
# The snapshot dir lives in ${TMPDIR:-/tmp}; SNAPDIR is echoed on init and
# stored in GATE_SNAP_DIR.
GATE_SNAP_DIR=""
_GATE_SNAP_LIB="${BASH_SOURCE[0]}"

gate_snap_init() { # $1 = space-separated target list
  local f
  GATE_SNAP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gate_snap.XXXXXX")" || return 1
  mkdir -p "$GATE_SNAP_DIR" || return 1
  : >"$GATE_SNAP_DIR/manifest"
  printf '%s\n' "${BASHPID:-$$}" >"$GATE_SNAP_DIR/owner" 2>/dev/null || true
  for f in $1; do
    [ -e "$f" ] || continue
    mkdir -p "$GATE_SNAP_DIR/$(dirname "$f")"
    cp -a "$f" "$GATE_SNAP_DIR/$f"
    sha256sum "$f" | awk '{print $1, $2}' >>"$GATE_SNAP_DIR/manifest"
  done
  export GATE_SNAP_DIR
  printf '%s\n' "$GATE_SNAP_DIR"
}

gate_snap_dirty_guard() { # $1 = space-separated target list; returns 1 on refusal
  local dirty=0 f blob
  for f in $1; do
    if git diff --quiet -- "$f" 2>/dev/null; then :; else
      dirty=$((dirty+1))
    fi
  done
  if [ "$dirty" -gt 0 ] && [ "${GATE_SNAP_ALLOW_DIRTY:-0}" != "1" ]; then
    echo "REFUSAL: DIRTY-TREE scope=targets files=$dirty reason=in-place-mutation-snapshot-restored" >&2
    for f in $1; do
      git diff --quiet -- "$f" 2>/dev/null || echo "  dirty: $f" >&2
    done
    echo "  Set GATE_SNAP_ALLOW_DIRTY=1 to snapshot-restore over them." >&2
    return 1
  fi
  if [ "$dirty" -gt 0 ]; then
    for f in $1; do
      if ! git diff --quiet -- "$f" 2>/dev/null; then
        blob="$(git hash-object "$f" 2>/dev/null || sha256sum "$f" | cut -d' ' -f1)"
        echo "  INPUT: UNCOMMITTED file=$f blob=$blob" >&2
      fi
    done
  fi
  return 0
}

gate_snap_verify_patched() { # $1 = file
  local snap="${GATE_SNAP_DIR}/$1"
  [ -f "$snap" ] || return 0
  [ "$(sha256sum "$1" | cut -d' ' -f1)" != "$(sha256sum "$snap" | cut -d' ' -f1)" ]
}

gate_snap_assert_clean() { # $1 = file; rc 0 iff equals snapshot
  local snap="${GATE_SNAP_DIR}/$1"
  [ -f "$snap" ] || return 0
  [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$(sha256sum "$snap" | cut -d' ' -f1)" ]
}

gate_snap_restore() { # $1 = file
  local snap="${GATE_SNAP_DIR:-}/$1"
  [ -f "$snap" ] || return 0
  # Skip the copy if the file already matches the snapshot BOTH in content
  # (sha256) AND in mode. This fixes the case where a mutation changes only
  # the mode (e.g. chmod -x) while content stays identical -- the sha would
  # match, but the mode would not, so we must still restore.
  local cur_mode snap_mode cur_sha snap_sha
  cur_mode="$(stat -c '%a' "$1" 2>/dev/null || echo 0)"
  snap_mode="$(stat -c '%a' "$snap" 2>/dev/null || echo 0)"
  if [ "$cur_mode" = "$snap_mode" ]; then
    cur_sha="$(sha256sum "$1" | cut -d' ' -f1)"
    snap_sha="$(sha256sum "$snap" | cut -d' ' -f1)"
    if [ "$cur_sha" = "$snap_sha" ]; then
      return 0
    fi
  fi
  cp -p "$snap" "$1"
  touch "$1"
}

gate_snap_integrity() { # $1 = space-separated list; rc 0 iff all match
  local f ok=0
  for f in $1; do
    [ -f "${GATE_SNAP_DIR}/$f" ] || continue
    if [ "$(sha256sum "$f" | cut -d' ' -f1)" != "$(sha256sum "${GATE_SNAP_DIR}/$f" | cut -d' ' -f1)" ]; then
      echo "  FAIL target $f differs from its startup snapshot" >&2
      ok=1
    fi
  done
  return "$ok"
}

gate_snap_restore_all() { # $1 = space-separated list; rc 0 iff all restored
  local f ok=0
  for f in $1; do
    gate_snap_restore "$f" || ok=1
  done
  return "$ok"
}

gate_snap_manifest() { # prints the init-time manifest: sha256 path
  [ -f "${GATE_SNAP_DIR:-}/manifest" ] || return 0
  cat "${GATE_SNAP_DIR}/manifest"
}

gate_snap_acquire_lock() { # $1 = lock name; rc 0 iff acquired
  local lock="${TMPDIR:-/tmp}/.gate_snap_lock.$1"
  mkdir -p "$(dirname "$lock")" 2>/dev/null || true
  # Stale-lock self-heal: if the lock dir exists but its owner pid is dead,
  # remove it so a new run can proceed. Recycled-pid safety: if a live process
  # has inherited the pid, refuse (fail-closed, never dangerous).
  if [ -d "$lock" ]; then
    local owner
    owner="$(cat "$lock/pid" 2>/dev/null || echo unknown)"
    if [ -n "$owner" ] && [ "$owner" != "unknown" ] && kill -0 "$owner" 2>/dev/null; then
      return 1  # live owner holds the lock
    fi
    rm -rf "$lock" 2>/dev/null || true  # stale: remove and retry below
  fi
  if ! mkdir "$lock" 2>/dev/null; then
    return 1
  fi
  printf '%s\n' "${BASHPID:-$$}" >"$lock/pid" 2>/dev/null || true
  return 0
}

gate_snap_release_lock() { # $1 = lock name
  local lock="${TMPDIR:-/tmp}/.gate_snap_lock.$1"
  rm -rf "$lock" 2>/dev/null || true
}

gate_snap_reap_stale() { # $1 = max age in seconds; removes stale gate_snap.* dirs
  local max_age="${1:-3600}" now dir age removed=0 owner
  now="$(date +%s)"
  for dir in "${TMPDIR:-/tmp}"/gate_snap.*; do
    [ -d "$dir" ] || continue
    owner="$(cat "$dir/owner" 2>/dev/null || true)"
    remove=0
    if [ -n "$owner" ] && [ "$owner" != "unknown" ] && ! kill -0 "$owner" 2>/dev/null; then
      remove=1 # interrupted on killed run left a snapshot owned by a dead process
    else
      age=$(( now - $(stat -c %Y "$dir" 2>/dev/null || echo "$now") ))
      if [ "$age" -ge "$max_age" ]; then
        remove=1
      fi
    fi
    if [ "$remove" -eq 1 ]; then
      rm -rf "$dir" 2>/dev/null && removed=$((removed + 1))
    fi
  done
  [ "$removed" -gt 0 ] && echo "reaped $removed stale snapshot dir(s)" >&2
  return 0
}

gate_snap_check_residue() { # $1 = space-separated target list; rc 0 iff no markers
  local pat f
  for pat in '/*M1*/' '/*M2*/' '/*M4*/' '/*M5*/' '/*M6*/'; do
    for f in $1; do
      [ -f "$f" ] || continue
      grep -qF -- "$pat" "$f" 2>/dev/null && { echo "  residue marker $pat in $f" >&2; return 1; }
    done
  done
  return 0
}

# --- unified matrix API ------------------------------------------------------
# gate_snap_matrix_begin NAME LIST   reap stale + acquire per-run lock + init snapshots
# gate_snap_matrix_end NAME LIST     restore all + release lock + cleanup + residue check
#
# These wrap the boilerplate every mutation matrix repeats (reap stale dirs from
# killed runs, acquire a concurrency lock, snapshot targets, restore on exit,
# release lock, check residue). Using them is optional but recommended for new
# matrices so the discipline stays in one place and can't drift per-matrix.
gate_snap_matrix_begin() { # $1 = lock name, $2 = space-separated target list
  local name="$1" targets="$2"
  local lock="${TMPDIR:-/tmp}/.gate_snap_lock.matrix.${name}"
  # SIGKILL-safe residue reinit: reap stale snapshot dirs left by a killed run
  # before starting, so residue from a previous interrupted run can't be
  # mistaken for this run's mutation markers.
  if [ "${GATE_SNAP_REAP_STALE:-1}" = "1" ]; then
    gate_snap_reap_stale 3600 >/dev/null 2>&1 || true
  fi
  # Per-run lock manifest: record which run holds the lock so a stale lock can
  # be identified by owner pid.
  if ! gate_snap_acquire_lock "matrix.${name}"; then
    local owner
    owner="$(cat "${TMPDIR:-/tmp}/.gate_snap_lock.matrix.${name}/pid" 2>/dev/null || echo unknown)"
    echo "FAIL: matrix lock for '$name' is held by pid $owner; another run is active" >&2
    return 1
  fi
  echo "${BASHPID:-$$}" >"${TMPDIR:-/tmp}/.gate_snap_lock.matrix.${name}/manifest" 2>/dev/null || true
  gate_snap_init "$targets" >/dev/null || return 1
  export SNAPDIR="$GATE_SNAP_DIR"
  return 0
}

gate_snap_attest() { # $1 = stamp/log file; writes helper + manifest digests to stdout
  local stamp="${1:-${GATE_SNAP_DIR:-}/attestation}"
  local snap="${GATE_SNAP_DIR:-}"
  : > "$stamp" 2>/dev/null || return 0
  printf 'helper_sha256=%s\n' "$(sha256sum "${_GATE_SNAP_LIB:-scripts/lib/_gate_snapshot.sh}" 2>/dev/null | cut -d" " -f1)" >>"$stamp"
  if [ -n "$snap" ] && [ -f "$snap/manifest" ]; then
    printf 'manifest_sha256=%s\n' "$(sha256sum "$snap/manifest" | cut -d" " -f1)" >>"$stamp"
  fi
  printf '%s\n' "$stamp"
}

gate_snap_matrix_end() { # $1 = lock name, $2 = space-separated target list
  local name="$1" targets="$2"
  local ok=0
  # Restore all targets from snapshot.
  gate_snap_restore_all "$targets" >/dev/null || ok=1
  # Release the per-run lock.
  gate_snap_release_lock "matrix.${name}"
  rm -rf "${TMPDIR:-/tmp}/.gate_snap_lock.matrix.${name}" 2>/dev/null || true
  # Clean up the snapshot dir.
  rm -rf "${GATE_SNAP_DIR:-}" 2>/dev/null || true
  # Residue check: no mutation markers may remain after restore.
  if ! gate_snap_check_residue "$targets"; then
    echo "FAIL: mutation residue left in tracked sources after matrix run" >&2
    ok=1
  fi
  return "$ok"
}
