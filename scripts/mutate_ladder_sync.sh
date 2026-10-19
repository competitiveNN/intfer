#!/usr/bin/env bash
# Mutation matrix for scripts/check_ladder_sync.sh.
#
# WHY NOT THE USUAL IN-PLACE CONVENTION. Every other mutate_*.sh in scripts/
# patches the tracked source in place and restores it from a pristine copy, which
# is fine because it snapshots and its own residue check proves byte-identity.
# THIS one cannot safely do that: the working tree currently carries ANOTHER
# session's uncommitted quantisation work (src/quant_spec.cpp is one of its
# files -- see the memory note concurrent_sessions_quant_work_20261019), and
# AGENTS.md's standing rule forbids a destructive probe over uncommitted edits.
# The checker already exposes the seam (INTFER_LADDER_REAL), so the states are
# built into a scratch dir and the checker is pointed at them. Nothing tracked is
# ever written, and the residue proof compares sha256s of the tracked files.
#
# LS-1 FP16 rung reappears on LINEAR_ATTN_QKV/_Z/_OUT. This is the ORIGINAL 19 GB
# defect in its exact shape: FP16 as a candidate is what let the budget search hand
# 56.8% of the model's bytes back un-quantized.
# LS-2 g32 and g16 swap order on those roles. The XMX BK=32 dequant reads ONE
# scale per group, so g32-first is a scale-traffic policy, not a formatting choice.
# LS-3 the INT2 rungs are dropped. RCO then has no INT2 to choose at all, and
# nothing else in the tree would notice -- this is the "weights must stay INT2"
# mandate silently becoming unenforceable.
# LS-4 a divergence the STATIC predicates cannot see: EMBED's INT4 activation
# scheme flips to FP16. Roles 11/12/16 still satisfy every static check, so only
# the compiled diff can reject it -- this state exists to prove the diff is live.
# LS-5 a benign comment edit. Must stay GREEN. Without it the matrix would pass
# against a checker that fails everything, which is the vacuity this repo has a
# memory tag for.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 2

BUILD="${INTFER_BUILD_DIR:-$ROOT/build}"
SB="$BUILD/ladder_sync_matrix"
CHECK="scripts/check_ladder_sync.sh"
REAL="src/quant_spec.cpp"

mkdir -p "$BUILD" || exit 2
rm -rf "$SB"; mkdir -p "$SB"

# Residue authority: anything tracked that changes under this matrix is a failure.
TRACKED=("$REAL" "src/quant/quant_spec.cpp" "$CHECK" "include/intfer/quant_spec.h")
declare -A PRE
for f in "${TRACKED[@]}"; do PRE["$f"]="$(sha256sum "$f" | cut -d" " -f1)"; done

total=0; caught=0; RESULTS=()
run_state() { # $1 id, $2 expect-regex, $3 want: reject | keep-green
  local id="$1" expect="$2" want="$3" out rc
  total=$((total + 1))
  out="$(INTFER_LADDER_REAL="$SB/$id.cpp" bash "$CHECK" 2>&1)"; rc=$?
  printf '%s\n' "$out" >"$SB/$id.log"
  if [ "$want" = keep-green ]; then
    if [ "$rc" -eq 0 ]; then caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''")
    else RESULTS+=("$id MISSED '(FALSE NEGATIVE threat -- rc='$rc')"); fi
    return
  fi
  if [ "$rc" -eq 0 ]; then RESULTS+=("$id MISSED '(still PASS)'")
  elif ! printf '%s' "$out" | grep -qE -- "$expect"; then RESULTS+=("$id MISSED '(wrong reason; rc='$rc')")
  else caught=$((caught + 1)); RESULTS+=("$id CAUGHT ''"); fi
}

if [ "${1:-}" = "--list" ]; then
  echo "states:"
  echo "  LS-1  FP16 rung on LINEAR_ATTN_QKV/_Z/_OUT   [must REJECT]"
  echo "  LS-2  g32/g16 order swapped                   [must REJECT]"
  echo "  LS-3  INT2 rungs dropped                      [must REJECT]"
  echo "  LS-4  EMBED act divergence (diff-only)        [must REJECT]"
  echo "  LS-5  benign comment edit                      [must stay GREEN]"
  exit 0
fi

echo "== baseline =="
if bash "$CHECK" >"$SB/base.log" 2>&1; then echo "  ok baseline rc=0 (18/18 roles agree)"; else
  echo "  FAIL baseline" >&2; tail -8 "$SB/base.log" >&2; exit 1; fi
echo "== states =="

# build every mutant from a copy of the REAL ladder; the program on stdin must
# bind `n` to the mutated text. The program is captured by THIS function's stdin
# and passed as argv, because `python3 - <<EOF` already owns stdin -- reading it
# again inside the child yields "" and the state silently becomes a no-op (which
# is exactly how the first draft of this matrix "passed" its keep-green state).
# The no-op guard is real: a mutation that does not change the source would make
# the state vacuous in both directions.
mk() { cp "$REAL" "$SB/$1.cpp"; }
edit() { # $1 = target file, stdin = python that binds `n`
  local target="$1" prog
  prog="$(cat)"
  python3 - "$target" "$prog" <<'PYX'
import sys
p, prog = sys.argv[1], sys.argv[2]
s = open(p).read()
g = {"s": s}
exec(prog, g)
n = g["n"]
assert isinstance(n, str) and n != s, "mutation was a no-op: %r" % (n[:60],)
open(p, "w").write(n)
PYX
}

# LS-1 the original 19 GB defect: an FP16 rung on the three byte-carrying roles.
mk LS-1
edit "$SB/LS-1.cpp" <<'PYE'
o = """      v.push_back(mk(QuantScheme::INT8, 32, a8));
      return v;
    }
    case TensorRole::LINEAR_ATTN_A:"""
mu = """      v.push_back(mk(QuantScheme::INT8, 32, a8));
      v.push_back(mk(QuantScheme::FP16, 0, ActScheme::FP16, false));
      return v;
    }
    case TensorRole::LINEAR_ATTN_A:"""
n = s.replace(o, mu, 1)
PYE

# LS-2 the XMX scale-traffic policy inverted (g32 no longer first).
mk LS-2
edit "$SB/LS-2.cpp" <<'PYE'
o = "      v.push_back(mk(QuantScheme::INT2, 32, a8));  // BK-aligned: 1 group/tile\n      v.push_back(mk(QuantScheme::INT2, 16, a8));  // accurate rung, +0.5 bpw\n"
mu = "      v.push_back(mk(QuantScheme::INT2, 16, a8));\n      v.push_back(mk(QuantScheme::INT2, 32, a8));\n"
n = s.replace(o, mu, 1)
PYE

# LS-3 INT2 disappears from those roles entirely.
mk LS-3
edit "$SB/LS-3.cpp" <<'PYE'
o = "      v.push_back(mk(QuantScheme::INT2, 32, a8));  // BK-aligned: 1 group/tile\n      v.push_back(mk(QuantScheme::INT2, 16, a8));  // accurate rung, +0.5 bpw\n"
n = s.replace(o, "", 1)
PYE

# LS-4 a divergence only the compiled diff can see: EMBED/LM_HEAD's INT4 activation.
mk LS-4
edit "$SB/LS-4.cpp" <<'PYE'
o = "      v.push_back(mk(QuantScheme::INT4, 64, a8));\n      v.push_back(mk(QuantScheme::INT8, 32, a8));\n      v.push_back(mk(QuantScheme::FP16, 0, ActScheme::FP16, false));"
i = s.index(o)
mu = "      v.push_back(mk(QuantScheme::INT4, 64, ActScheme::FP16));\n      v.push_back(mk(QuantScheme::INT8, 32, a8));\n      v.push_back(mk(QuantScheme::FP16, 0, ActScheme::FP16, false));"
n = s[:i] + mu + s[i+len(o):]
PYE

# LS-5 benign: a comment only. The candidate tables are byte-identical, so the
# checker must PASS -- otherwise it cannot tell a real divergence from prose.
mk LS-5
edit "$SB/LS-5.cpp" <<'PYE'
o = "// Floor >= INT4 (sensitive). Offer INT4/INT8/FP16."
mu = "// Floor >= INT4 (sensitive). Offer INT4/INT8/FP16 (prose only)."
n = s.replace(o, mu, 1)
PYE

run_state LS-1 'carries an FP16 rung' reject
run_state LS-2 'does not offer INT2 g32 FIRST' reject
run_state LS-3 'does not offer INT2 g32 FIRST' reject
run_state LS-4 'two ladders DIVERGE' reject
run_state LS-5 '' keep-green

for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done

bad=0
for f in "${TRACKED[@]}"; do
  now="$(sha256sum "$f" | cut -d' ' -f1)"
  [ "$now" = "${PRE[$f]}" ] || { echo "  FAIL residue: $f changed under the matrix" >&2; bad=$((bad+1)); }
done
[ "$bad" -eq 0 ] && echo "  ok residue: ${#TRACKED[@]} tracked file(s) byte-identical to pre-matrix state"

echo
if [ "$caught" -eq "$total" ] && [ "$bad" -eq 0 ]; then
  echo "LADDER-SYNC MUTATION MATRIX: PASS ($caught/$total states)"
  echo "  LS-1: the FP16-rung regression (the 19 GB defect) is caught"
  echo "  LS-2: the g32-before-g16 XMX scale-traffic policy is caught"
  echo "  LS-3: losing the INT2 rungs is caught -- the INT2 mandate stays enforceable"
  echo "  LS-4: a static-predicate-invisible divergence is caught by the compiled diff"
  echo "  LS-5: a benign prose edit stays GREEN -- the checker is not just always-red"
  echo "RESULT: PASS ($caught/$total states caught)"
  exit 0
fi
echo "LADDER-SYNC MUTATION MATRIX: FAIL ($caught/$total caught, $bad residue)" >&2
echo "RESULT: FAIL ($caught/$total caught, $bad residue)" >&2
exit 1
