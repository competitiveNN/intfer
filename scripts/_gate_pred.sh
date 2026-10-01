#!/usr/bin/env bash
# SHARED: predicate comparison with the POLARITY NAMED AT THE CALL SITE.
#
# WHY THIS EXISTS. An inverted predicate was found THREE times in two sessions,
# every time scoring correct code as broken:
#   * mutate_gate_enum.sh: verify_patched() failed when the marker WAS present.
#   * mutate_gate_enum.sh: residue_scan returns 0 when CLEAN, and the call
#     branched on success.
#   * mutate_gate_enum.sh: state M-I asserted the guard's own FAIL line, while
#     consumer_state scores "expect must be ABSENT" -- and for that state the
#     guard firing IS the success.
# All three cost a debugging round on code that was working. All three are the
# same bug: the writer had to remember an unstated convention about which side
# of the comparison the marker is on, and nothing at the call site recorded it.
#
# The fix is not "be careful". It is to make the convention part of the CALL:
#
#   pred_expect_absent <label> <text> <pattern>   # the pattern must NOT appear
#   pred_expect_present <label> <text> <pattern>  # the pattern MUST appear
#
# Both print the polarity in their verdict, so a reader of the log -- and a
# mutation state that mutates only this rendering -- can see which side was
# meant. `run_state` in the matrices echoes the same label.
#
# Sourced, not executed; sibling of _gate_enum.sh and _gate_patch.sh.
pred_fail=0

pred_expect_absent() { # $1 label, $2 haystack, $3 ERE
  if printf '%s\n' "$2" | grep -qE -- "$3"; then
    printf '  FAIL  [%s] expected ABSENT, but found: %s\n' "$1" "$3"
    pred_fail=1
    return 1
  fi
  printf '  ok    [%s] expected absent, and it is absent\n' "$1"
  return 0
}

pred_expect_present() { # $1 label, $2 haystack, $3 ERE
  if printf '%s\n' "$2" | grep -qE -- "$3"; then
    printf '  ok    [%s] expected present, and it is present\n' "$1"
    return 0
  fi
  printf '  FAIL  [%s] expected PRESENT, but not found: %s\n' "$1" "$3"
  pred_fail=1
  return 1
}

pred_reset() { pred_fail=0; }
pred_bad() { echo "$pred_fail"; }