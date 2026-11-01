#!/usr/bin/env bash
# Mutation matrix for scripts/checks/check_dpas_shape_audit.sh.
#
# WHAT THE CHECKER CLAIMS: no tensor and no TP=2 shard in the model reaches a
# shape the three ESIMD DPAS kernels cannot lower (N%16, K%32, (bits,group)
# combo), and the predicate the engine consults (src/engine/engine.cpp:201,
# mirrored one-for-one in scripts/lib/_dpas_pred.sh) is the SAME predicate this
# audit re-derives. A green verdict is only worth the strength of both halves.
#
# WHY THESE STATES ARE WHAT THEY ARE. AGENTS.md rule 11: "If the artifact under
# test is correct by construction, weakening the CHECKER is unobservable --
# mutate the ARTIFACT and require the checker to REJECT it." The artifact here
# is the model's own tensor table (the .intifer header), so DS-1 and DS-3 plant
# a model with the two shapes that can plausibly be shipped: one tensor whose
# K does not divide the DPAS op, and a model with no quantized tensor at all
# (the anti-vacuity case, where "no violations" would otherwise be read as
# evidence about nothing).
#
# DS-2 and DS-4 mutate the CHECKER, and each is the exact defect it exists to
# prevent rather than a random break: DS-2 is the re-derived copy of the
# predicate drifting from scripts/lib/_dpas_pred.sh (AGENTS.md rule 4 -- the
# "remembered constant that decays into a guess"), and DS-4 is the inverted
# comparison, the bug class this repo's own AC-4 matrix was written for.
#
# EVERY STATE IS NON-DESTRUCTIVE: DS-1 and DS-3 run against synthetic models
# planted under build/, DS-2 and DS-4 patch a COPY of the checker under build/
# and run that. The tracked tree is never edited (AGENTS.md rule 16) and the
# residue assertion at the end re-proves it.
#
# States:
#   DS-1  a tensor's K is not a multiple of 32          [must REJECT]
#   DS-2  the re-derived predicate uses TN=8 / OP=16   [must REJECT]
#   DS-3  the model has no INT2/4/8 tensor             [must REJECT]
#   DS-4  the checker's own violation test is inverted [must REJECT]
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/dpas_shape_matrix"
# *** WIPE LOCK, OUTSIDE THE TREE IT WIPES (check_matrix_sandbox_locks.sh). ***
# The sandbox below is a FIXED path wiped on every invocation, so two concurrent
# runs share one tree: the second's wipe empties the first's mid-state and the
# first then grades a corpus the other has just rebuilt (MEASURED 2026-10-21 on
# mutate_durable_reads.sh). Taken BEFORE the first wipe and released on
# EXIT/INT/TERM, so a run that dies mid-state does not brick the next.
LOCK="$BUILD/.mutate_dpas_shape_audit.lock"
mkdir -p "$BUILD" 2>/dev/null || true
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FAIL: another mutate_dpas_shape_audit.sh is running (lock: $LOCK)" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 130' INT
trap 'rmdir "$LOCK" 2>/dev/null || true; exit 143' TERM

CHECK="scripts/checks/check_dpas_shape_audit.sh"
# The real model, so DS-2 can run against it. check_dpas_shape_audit.sh falls
# back to this path when INTFER_27B_MODEL is unset, and the matrix must not
# depend on that fallback being the same string the checker hard-codes.
REAL_MODEL="${INTFER_27B_MODEL:-/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer}"

rm -rf "$SB"; mkdir -p "$SB" || exit 2
PRISTINE="$SB/pristine.chk"
cp "$CHECK" "$PRISTINE"
restore() { cp "$PRISTINE" "$CHECK"; }

patch() { # $1 file, $2 old, $3 new  -- EXACT literal, never sed (AGENTS rule 10)
  python3 - "$1" "$2" "$3" <<'PYX'
import sys
p, o, n = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
c = s.count(o)
if c != 1:
    print("anchor %r matched %d times, want exactly 1" % (o[:48], c), file=sys.stderr)
    sys.exit(1)
open(p, "w").write(s.replace(o, n, 1))
PYX
}

total=0; caught=0; RESULTS=()
run_state() { # $1 id, $2 expect-regex
  local id="$1" expect="$2" out rc
  total=$((total + 1))
  out="$(bash "$CHECK" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  restore
  if [ "$rc" -eq 0 ]; then
    RESULTS+=("$id MISSED '(still PASS)'")
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then
    RESULTS+=("$id MISSED '(wrong reason; rc='$rc')'")
  else
    caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
  fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  DS-1  a tensor's K is not a multiple of 32          [must REJECT]"
  echo "  DS-2  the re-derived predicate uses TN=8 / OP=16   [must REJECT]"
  echo "  DS-3  the model has no INT2/4/8 tensor             [must REJECT]"
  echo "  DS-4  the checker's own violation test is inverted  [must REJECT]"
  exit 0
fi

# --- synthetic models, planted as DATA (the scheme vocabulary has quotes) -----
python3 - "$SB" <<'PYX'
import json, os, struct, sys
sb = sys.argv[1]

def write(name, tensors):
    d = {"model_id": "synthetic", "tensors": tensors}
    s = json.dumps(d).encode()
    open(os.path.join(sb, name), "wb").write(b"INTFER" + struct.pack("<Q", len(s)) + s)

# DS-1: the shapes the 27B actually ships, with ONE K that does not divide the
# DPAS op (OP=32). Every other tensor is a valid 27B shape, so the state attacks
# exactly one defect class.
write("k_violation.intifer", [
    {"name": "q_proj", "N": 12288, "K": 5120,
     "qcfg": {"scheme": "INT4", "group_size": 64}},
    {"name": "bad_k", "N": 12288, "K": 5121,
     "qcfg": {"scheme": "INT4", "group_size": 64}},
    {"name": "gate", "N": 17408, "K": 5120,
     "qcfg": {"scheme": "INT2", "group_size": 16}},
])

# DS-3: no quantized tensor at all. "No violations found" here is a statement
# about an empty answer, which is the vacuous-green shape.
write("fp16_only.intifer", [
    {"name": "w_a", "N": 5120, "K": 6144,
     "qcfg": {"scheme": "FP16", "group_size": 0}},
    {"name": "w_b", "N": 6144, "K": 5120,
     "qcfg": {"scheme": "FP16", "group_size": 0}},
])
PYX
[ $? -eq 0 ] || { echo "DPSA MUTATION MATRIX: FAIL (could not plant synthetic models)"; exit 1; }

echo "== baseline =="
if bash "$CHECK" >"$SB/base.log" 2>&1; then echo "  ok baseline rc=0"; else
  echo "  FAIL baseline" >&2; tail -5 "$SB/base.log" >&2; exit 1; fi
echo "== states =="

# DS-1: the model's OWN tensor table carries a K the kernel cannot lower. The
# checker must reject the model, not the checker.
INTFER_27B_MODEL="$SB/k_violation.intifer" run_state DS-1 \
  'violate a DPAS constraint'

# DS-2: the re-derived copy of the predicate in the checker's enumeration block
# drifts off the one scripts/lib/_dpas_pred.sh defines. The self-check is untouched
# (it sources _dpas_pred.sh directly), so this isolates the DRIFT detector --
# the AGENTS.md rule-4 failure mode in one line.
#
# The mutation is a STRICTER combo set, not a looser one: the checker must read
# the model as NOT-eligible while _dpas_pred.sh still says it IS. A looser
# predicate (TN=8/OP=16, a dropped combo) leaves every 27B shape eligible in
# BOTH copies, so the drift detector has nothing to compare -- which is itself
# the finding, and why this state is written the way it is.
#
# This state runs against the REAL 27B model, not the synthetic one: the
# synthetic model's only INT2 tensor is gate/up at group 16, which the
# stricter combo set rejects, so the SHAPE verdict fires first and the drift
# detector is never reached. The real model has the same gate/up tensors, so
# the drift detector has cases to disagree about and the shape verdict stays
# green (the real model has no shape violations).
patch "$CHECK" 'def want_eligible(scheme, gs, Nl, K):
    bits = {"INT2": 2, "INT4": 4, "INT8": 8}.get(scheme, 16)
    if bits == 16:
        return 1
    G = gs if gs else K
    combo = (bits, G) in {(4, 64), (8, 32), (2, 32), (2, 16)}' \
                 'def want_eligible(scheme, gs, Nl, K):
    bits = {"INT2": 2, "INT4": 4, "INT8": 8}.get(scheme, 16)
    if bits == 16:
        return 1
    G = gs if gs else K
    combo = (bits, G) in {(4, 64), (8, 32)}'
INTFER_27B_MODEL="$REAL_MODEL" run_state DS-2 're-derived predicate disagrees with'
# restore so the tree is clean for the states that follow
cp "$PRISTINE" "$CHECK"

# DS-3: no INT2/4/8 tensor. The anti-vacuity floor is the ONLY thing standing
# between this and a green verdict that certified an empty harvest.
INTFER_27B_MODEL="$SB/fp16_only.intifer" run_state DS-3 \
  'no INT2/INT4/INT8 tensor'

# DS-4: the checker's own violation predicate is INVERTED. Every one of the 242
# eligible tensors must now read as a violation, so a checker that still prints
# PASS has lost the ability to fail -- the always-red twin, run red on purpose.
patch "$CHECK" '        ok_shard = combo and (Nl % TN == 0) and (K % OP == 0)' \
                 '        ok_shard = not (combo and (Nl % TN == 0) and (K % OP == 0))'
run_state DS-4 'violate a DPAS constraint'

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done
if ! cmp -s "$CHECK" "$PRISTINE"; then
  echo "  FAIL residue: $CHECK differs from its pre-matrix state" >&2; exit 1
fi
echo "  ok residue: $CHECK byte-identical to its pre-matrix state"

echo
if [ "$caught" -eq "$total" ]; then
  echo "DPAS-SHAPE MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  DS-1: a tensor whose K does not divide the DPAS op is caught"
  echo "  DS-2: the re-derived predicate drifting from _dpas_pred.sh is caught"
  echo "  DS-3: a model with no quantized tensor is caught, not certified green"
  echo "  DS-4: an inverted violation test is caught -- the checker can still fail"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "DPAS-SHAPE MUTATION MATRIX: FAIL ($caught/$total caught)" >&2
echo "RESULT: FAIL ($caught/$total caught)" >&2
exit 1
