#!/usr/bin/env bash
# SHARED mutation helpers: patch_literal, verify_patched, residue scan.
#
# WHY THIS EXISTS. patch_literal() was ALREADY copied verbatim into three
# matrices (mutate_fold_matrix.sh, mutate_device_precond.sh,
# mutate_run_all_checks.sh) and a fourth copy was about to be written for
# mutate_gate_enum.sh -- the audit #7/#8 "rule fixed at one site, re-derived at
# another" shape, one level down. The three existing copies are left alone
# deliberately: they are exercised by passing matrices and converting them
# carries risk that outweighs the dedup. This file exists so the NEXT matrix
# does not add a fourth.
#
# WHY LITERAL PATCHING, NOT sed. Two concrete failures, both recorded:
#   * a pattern CONTAINING `||` only matched because the sed delimiter was
#     escaped to `\|` -- the audit #17b delimiter collision, sitting live in a
#     state that passes today and would silently stop matching after a small
#     edit, turning into a verdict about the CHECKER when the fault is the state.
#   * sed replaces the FIRST match and reports nothing about how many there
#     were, and one anchor was a PREFIX of another, so a later state still
#     matched a line an earlier state had already mutated.
# So: exact text, and the anchor must occur EXACTLY ONCE or the patch is refused.
#
# Usage:  . scripts/lib/_gate_patch.sh
#         patch_literal <file> <exact-old> <exact-new>
#         verify_patched <file> <marker>        # marker absent -> syntax error
#         residue_scan <root> <marker>...       # fail if any marker survives
patch_literal() {
  # *** EXACTLY THREE ARGUMENTS: file, exact-old, exact-new. *** Asserted, not
  # assumed. This helper was called with FIVE by a matrix author who passed one
  # single-quoted string per LINE of a two-line case, expecting them to
  # concatenate. They do not: each quoted string is a separate ARGUMENT, so the
  # helper replaced the first line with the second, wrote no mutation marker,
  # and returned SUCCESS -- five states reporting "PATCH VERIFY FAIL" with no
  # error anywhere. Found only with `bash -x`.
  # A single-quoted bash string MAY span lines: put the newline INSIDE one
  # argument rather than adding arguments.
  if [ "$#" -ne 3 ]; then
    echo "patch_literal: expected 3 args (file, exact-old, exact-new), got $#" >&2
    echo "  Patching a MULTI-LINE block? Put the newline INSIDE ONE single-quoted" >&2
    echo "  argument -- do NOT pass one quoted string per line." >&2
    return 2
  fi
  python3 - "$1" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
n = src.count(old)
if n != 1:
    sys.stderr.write("patch_literal: anchor found %d time(s) in %s, want exactly 1: %r\n"
                     % (n, path, old[:90]))
    sys.exit(1)
open(path, "w").write(src.replace(old, new))
PY
}

# Confirm a patch LANDED and left the file valid.
# Returns 0 when the marker IS present (the mutation applied). The first
# version of this helper had the test INVERTED -- it failed when the marker was
# found, i.e. it reported "the mutation did not apply" for every mutation that
# applied -- which scored 5 of 6 states MISSED for a reason that had nothing to
# do with them, and (because those states returned early) skipped their restore,
# so a mutated helper leaked into the residue scan of the next run.
#
# A patch that leaves the file syntactically broken is a HARNESS fault, not
# evidence about the checker (audit #17b verify_patched).
verify_patched() { # $1 = file, $2 = a marker string unique to the mutation
  if ! grep -q -- "$2" "$1"; then
    echo "PATCH VERIFY FAIL: '$2' NOT present in $1 -- the mutation did not apply"
    return 1
  fi
  if command -v bash >/dev/null 2>&1; then
    bash -n "$1" 2>/dev/null || {
      echo "PATCH VERIFY FAIL: $1 is syntactically broken after patching."
      echo "  This is a STATE fault (a fragment was left dangling), not a verdict"
      echo "  about the checker. Fix the anchor so it spans the whole unit."
      return 1
    }
  fi
  return 0
}

residue_scan() { # $1 = root, rest = markers that must NOT survive
  local root="$1"; shift
  local found=0 m
  for m in "$@"; do
    if grep -rql -- "$m" "$root" 2>/dev/null; then
      echo "RESIDUE: marker '$m' still present under $root"
      grep -rl -- "$m" "$root" 2>/dev/null | sed 's/^/  /'
      found=1
    fi
  done
  return $found
}