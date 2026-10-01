#!/usr/bin/env bash
# SHARED: enumerate the stamp population and the set of directories a matrix
# legitimately owns.
#
# WHY DERIVED, NEVER HAND-LISTED. A literal list of owned directories has to be
# edited every time a matrix is added, and the omission is SILENT: the new
# matrix's output directory reads as an unknown directory, the leak assertion
# fires, and the correct response -- to fix the list -- is indistinguishable
# from the wrong one. That happened: adding mutate_build_dir_consumers.sh made
# mutate_l0_cite.sh report a LEAK on a completely healthy tree.
#
# So the set is read from each mutate_*.sh's OWN path assignment (WORK=, LOG_DIR=,
# STAMP=). All three are needed because the matrices differ: mutate_run_all_checks
# stamps via WORK, mutate_device_precond via LOG_DIR, the rest via STAMP.
# Reading only two of them finds four of five and just moves the false alarm.
#
# WHY A SET AND NOT A COUNT. mutate_device_precond.sh writes TWO stamps into one
# directory, so a count-based census false-alarms on it, and a count-only fixture
# would cancel against a missing stamp. Compare membership.
#
# WHY THIS FILE EXISTS: the derivation had been written out three times by the
# time it was noticed -- the same "rule fixed at one site, re-derived at
# another" shape as patch_literal (audit #7/#8). New matrices source this.
#
# Sourced, not executed; sibling of _gate_enum.sh / _gate_patch.sh / _gate_pred.sh.
# Usage: . scripts/_gate_pop.sh   then: pop_dirs, known_dirs, pop_unknown_dirs
# Every caller defines BUILD_DIR (some export INTFER_BUILD_DIR, some only set
# the local), so accept either rather than forcing an export on each matrix --
# that hard requirement broke mutate_l0_cite.sh, whose M-H then reported the
# leak assertion as vacuous when the real cause was an unset variable.
pop_build_dir() {
  if [ -n "${INTFER_BUILD_DIR:-}" ]; then echo "$INTFER_BUILD_DIR"; return 0; fi
  if [ -n "${BUILD_DIR:-}" ]; then echo "$BUILD_DIR"; return 0; fi
  # Fail LOUDLY and non-zero. The previous form used ${VAR:?msg} inside an echo,
  # which printed the message and then went on to enumerate an EMPTY path --
  # caught by check_gate_pop.sh case (e): it returned rc=0 over zero
  # directories, i.e. a vacuous enumeration wearing a successful exit code.
  echo "ERROR: neither INTFER_BUILD_DIR nor BUILD_DIR is set -- cannot enumerate" >&2
  return 1
}

pop_dirs() {
  local bd
  bd="$(pop_build_dir)" || return 1
  find "$bd" -name '*last_pass.stamp' 2>/dev/null \
    | xargs -r -n1 dirname | xargs -r -n1 basename | LC_ALL=C sort -u
}

known_dirs() {
  local sd; sd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  grep -hoE '(WORK|LOG_DIR|STAMP)="[^"]*"' "$sd"/mutate_*.sh 2>/dev/null \
    | sed -e 's/^[A-Z_]*="//' -e 's/"$//' \
    | sed -E -e 's@^\$\{[^}]*\}/@@' -e 's@^\$(BUILD_DIR|ROOT|PWD)/build/@@' -e 's@^\$(BUILD_DIR|ROOT|PWD)/@@' \
    | grep -vE '^[\$/{]' \
    | awk -F/ 'NF>0 {print $1}' \
    | LC_ALL=C sort -u | tr '\n' ' '
}

# The members of the population that no matrix owns -- i.e. a fixture stamp.
pop_unknown_dirs() {
  local k
  k="$(known_dirs)" || return 1
  pop_dirs | while read -r d; do
    [ -n "$d" ] || continue
    case " $k " in *" $d "*) ;; *) echo "$d" ;; esac
  done
}

# Assert the population AS FOUND, before any state mutates anything. Exists
# separately from the after-run comparison because POP_BEFORE vs POP_AFTER alone
# cannot see a leak that was already present when the matrix started: it sits
# inside the baseline. Measured -- planting a fixture made a self-comparing
# assertion exit 0 (audit #19).
pop_assert_no_leak() { # $1 = label for the message
  local unknown
  unknown="$(pop_unknown_dirs)"
  if [ -n "$unknown" ]; then
    echo "  FAIL  [$1] stamp(s) in director(ies) no matrix owns:"
    printf '%s\n' "$unknown" | sed 's/^/          /'
    echo "        a checker is leaving fixture stamps in \$BUILD_DIR, so"
    echo "        run_all_checks.sh step 3/3 would compare the wrong set."
    return 1
  fi
  echo "  ok    [$1] $(pop_dirs | grep -c .) stamp dir(s), all owned by a known matrix"
  return 0
}