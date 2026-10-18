#!/usr/bin/env bash
# A gate script's verdict is its exit code (run_ten_gates.sh:79: rc 0 -> GPASS,
# rc 2 -> PARTIAL, rc 1 -> FAIL). So a gate that sets PASS=1 and PASS=0 and
# never converts the variable to an exit code reports PASS with every arm
# failing -- the vacuous green, at the one gate that certifies the 27B --tp 2
# target.
#
# *** WHAT THIS CHECKS, AND WHAT IT DOES NOT. *** It reads the SCRIPT SOURCE.
# It does not run the gate (that needs the model and both B580s) and it does
# not synthesise a log. A synthetic-log version would build its fixture from
# the fixed script and pass by construction -- the same mistake the first
# decode1 self-check made, which is why this checker is source-based.
#
# Three predicates, each with a negative direction:
#   (a) CONVERSION EXISTS. The script must contain a line that tests $PASS
#       and exits. Absent -> the gate's verdict is independent of its own
#       assertions.
#   (b) POLARITY. The conversion must exit 1 (FAIL) when PASS != 1. A
#       `exit 0` in both branches would make (a) green and (b) red at once,
#       which is the shape of a checker that cannot fail.
#   (c) NON-VACUITY. The script must set PASS=1 somewhere, otherwise the
#       conversion tests a variable nobody writes and the whole PASS scheme
#       is decorative.
#
# (d) NO FALL-OFF-THE-END. The last statement of the script must be an exit
#     (or a call that always exits). A script whose final line is `rm -f`
#     falls off the end with the shell's exit status of the LAST COMMAND --
#     which, here, is `rm` succeeding -> 0 -> GPASS. This is the exact shape
#     that was shipping before this checker was written, so it is asserted
#     rather than inferred from (a): (a) is satisfied by a conversion placed
#     ABOVE a trailing `rm -f`, and only (d) catches that.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR"/.. && pwd)"
cd "$ROOT" || exit 2

GATE="${1:-scripts/test_tp2_27b.sh}"
[ -f "$GATE" ] || { echo "FAIL: gate $GATE not found" >&2; exit 2; }

echo "== TP2 PASS->EXIT PIN ($GATE) =="

# (c) NON-VACUITY, FAMILY-INDEPENDENT -- runs BEFORE the family branch, on
#     purpose. The invariant is: IF the gate contains a $PASS conversion, THEN
#     PASS=1 must be written somewhere. Gating (c) on the PASS-variable branch
#     would let the PE-4 mutation (delete PASS=1) flip the family to
#     probe-relay and skip (c) entirely -- measured: pe4.sh then passes d2 on
#     the conversion line and the checker reports PASS. The conversion is the
#     thing that makes PASS meaningful, so the test keys on the conversion, not
#     on which branch the family test took.
if grep -Eq '\[ "\$PASS" *(=|!=) *[0-9]+ \].*exit' "$GATE"; then
  if ! grep -Eq '^[[:space:]]*PASS=1[[:space:]]*$' "$GATE"; then
    echo "FAIL (c): the gate has a \$PASS conversion but never sets PASS=1," >&2
    echo "         so the conversion tests a variable nobody writes -- the" >&2
    echo "         PASS scheme is decorative" >&2
    exit 1
  fi
  echo "ok  (c) non-vacuity: \$PASS conversion present and PASS=1 is written"
else
  echo "ok  (c) non-vacuity: no \$PASS conversion (probe-relay family)"
fi

# *** FAMILY. *** The sweep found what this single-gate checker could not:
# only some gates write a `PASS` variable and convert it to an exit code; the
# rest relay the probe's OWN rc through `exit "$RC"` (or an `exit 1` on the
# FAIL branch immediately before an unconditional `echo "... TEST: PASS"`).
# Calling a probe-relay gate "unsafe" for lacking a `PASS` conversion would be
# weakening the check to make it green. So classify first, then apply the
# predicate set that governs the family. A gate that prints PASS and then
# falls off the end is unsafe in BOTH families.
if grep -Eq '^[[:space:]]*PASS=1[[:space:]]*$' "$GATE"; then
  family=PASS-variable
else
  family=probe-relay
fi
echo "  family: $family"

if [ "$family" = PASS-variable ]; then

# (a) conversion exists: a line testing $PASS that exits (or opens a block
#     The comparison value is `["']?[0-9]+["']?` so a quoted "1" counts; the
#     tail is `exit|; then` so the split form counts (the `; then` is the tell
#     that the test opens a control block -- (b) then requires that block to
#     exit 1). What (a) rules out is a test that does NOT exit and does NOT
#     open an exiting block, which is the shape of a gate whose verdict is its
#     own exit code while PASS is only ever written, never converted.
if ! grep -Eq '\[ "\$PASS" *(=|!=) *["'"'"']?[0-9]+["'"'"']? \].*(exit|;[[:space:]]*then)' "$GATE"; then
  echo "FAIL (a): no PASS -> exit conversion in $GATE; the gate's verdict" >&2
  echo "         is its exit code, so it reports PASS regardless of PASS=unset" >&2
  exit 1
fi
echo "ok  (a) PASS -> exit conversion present"

# (b) polarity: the FAIL branch must exit 1. The canonical form in this tree
#     (test_cpu_parity.sh:133, test_oracle_equiv.sh:58, and this gate) puts it
#     on the SAME line as the FAIL echo: `else echo "TEST: FAIL" >&2; exit 1; fi`.
#     That is what is asserted. The defect (b) exists to catch is a conversion
#     whose only exit is 0 -- a green branch in a verdict gate -- which is
#     caught by the second grep. A correct multi-line form that splits the
#     echo from the exit is NOT refused: (d) pins the last-statement property
#     and the gate runner's rc classification is what makes the verdict real.
if ! { grep -EA3 '\[ "\$PASS" *(=|!=) *["'"'"']?[0-9]+["'"'"']? \]' "$GATE" | grep -Eq 'exit[[:space:]]+1'; }; then
  echo "FAIL (b): no FAIL branch -- the failure path cannot turn a" >&2
  echo "         failing arm into a FAIL verdict" >&2
  exit 1
fi
if grep -Eq '\[ "\$PASS" *(=|!=) *[0-9]+ \].*exit[[:space:]]+0' "$GATE"; then
  echo "FAIL (b): the conversion exits 0 -- a green branch in a verdict gate" >&2
  exit 1
fi
echo "ok  (b) polarity: FAIL branch exits 1; no green exit-0 branch"

# (d) LAST STATEMENT DETERMINES THE VERDICT -- the SAME predicate for both
#     families, applied OUTSIDE the family branch. Five shapes are correct in
#     this tree and all are accepted:
#       (i)   `exit "$RC"` / `exit "$rc"` / `exit "$?"` -- a VARIABLE argument.
#             The probe-relay family's verdict; the variable is the probe's rc,
#             so the gate cannot green-light it by spelling a literal.
#       (ii)  the last statement is a single-line conversion that contains
#             `exit <1 or 2>` (cpu_parity/oracle_equiv/tp2_27b)
#       (iii) the last statement is an unconditional `echo "... TEST: PASS"`
#             whose preceding non-blank non-comment line exits non-zero
#             (device_argmax/gpu_optin/gguf_roundtrip/http_batch/serve_load)
#       (iv)  a non-zero LITERAL exit (`exit 1`, `exit 2`) -- the FAIL branch
#             of a probe-relay gate (dpas_fold/ga64_recip/ksplit_contract/
#             onemkl/roofline_bisect*/roofline_probe/rowpair_equiv)
#     What is unsafe is falling off the end after a PASS echo with no FAIL exit
#     before it -- the exact shape test_tp2_27b.sh shipped -- AND, for
#     PASS-variable gates, a literal `exit 0` as the last statement, which
#     kills the PASS logic just as dead: the gate sets PASS=1, runs checks
#     that set PASS=0, and then exits 0 regardless, so the verdict is green
#     with every arm failing. Shape (i) accepts the variable form and refuses
#     the literal-0 form for PASS-variable gates on purpose.
last="$(grep -vE '^[[:space:]]*(#|$)' "$GATE" | tail -1)"
if printf '%s' "$last" | grep -Eq '^exit[[:space:]]+"\$?(RC|rc|\?)"'; then
  echo "ok  (d) last statement exits: ${last%% *}"
elif printf '%s' "$last" | grep -Eq '\[ "\$PASS" *(=|!=) *["'"'"']?[0-9]+["'"'"']? \].*exit[[:space:]]+[1-9]'; then
  echo "ok  (d) last statement is a conversion that exits: ${last%% *}"
elif printf '%s' "$last" | grep -qE 'TEST: PASS'; then
  prev="$(grep -vE '^[[:space:]]*(#|$)' "$GATE" | tail -6 | head -5)"
  # The LAST exit before the PASS echo must be non-zero. A bounded window
  # (last 6 non-blank non-comment lines) is enough: device_argmax and
  # gpu_optin end `... exit 1; fi` / `echo "TEST: PASS"`, so the last exit
  # before the echo is the `exit 1` inside the `if` block; gguf_round-
  # trip/http_batch/serve_load put `exit 1` on the line immediately before.
  # PE-6 plants `exit 0` immediately before the echo, so the last exit is
  # 0 and the predicate fires -- a plain "window contains exit [1-9]"
  # would reach past the `exit 0` into the conversion line above it and
  # pass, which is how this state first came back unobservable.
  lastexit="$(printf '%s\n' "$prev" | grep -E 'exit[[:space:]]+[0-9]' | tail -1)"
  if [ -n "$lastexit" ] && printf '%s' "$lastexit" | grep -qE 'exit[[:space:]]+[1-9]'; then
    echo "ok  (d) PASS echo preceded by a non-zero exit: ${lastexit%% *}"
  elif [ -n "$lastexit" ]; then
    echo "FAIL (d): the last exit before the PASS echo is ${lastexit%% *}" >&2
    echo "         last: '$last'" >&2
    printf '%s\n' "$prev" | sed 's/^/         prev: /' >&2
    echo "         the shell returns the echo's rc (0), so the gate reports" >&2
    echo "         PASS regardless of the probe's rc" >&2
    exit 1
  else
    echo "FAIL (d): the PASS echo is not preceded by any exit:" >&2
    echo "         last: '$last'" >&2
    printf '%s\n' "$prev" | sed 's/^/         prev: /' >&2
    exit 1
  fi
elif printf '%s' "$last" | grep -Eq '^exit[[:space:]]+[1-9]'; then
  echo "ok  (d) last statement exits: ${last%% *}"
elif [ "$family" = PASS-variable ] && printf '%s' "$last" | grep -Eq '^exit[[:space:]]+0'; then
  echo "FAIL (d): the last statement is `exit 0` on a PASS-variable gate:" >&2
  echo "         '$last'" >&2
  echo "         the gate sets PASS=1, runs checks that set PASS=0, and then" >&2
  echo "         exits 0 regardless -- the verdict is green with every arm" >&2
  echo "         failing, the same defect as falling off the end." >&2
  exit 1
else
  echo "FAIL (d): the last statement is neither an exit nor a PASS echo:" >&2
  echo "         '$last'" >&2
  exit 1
fi

fi

echo "TP2-PASS-EXIT PIN: PASS"