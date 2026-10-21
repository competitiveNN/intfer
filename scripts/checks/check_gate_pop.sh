#!/usr/bin/env bash
# Behavioural self-check for scripts/lib/_gate_pop.sh.
#
# WHY. It is a brand-new shared helper, and the class it belongs to -- "rule
# fixed at one site, re-derived at another", the patch_literal shape -- requires
# a falsifiability proof AT ITS OWN HOME. The only evidence so far was indirect:
# three converted matrices happening to pass.
#
# It is gated on the two defects this session ACTUALLY HIT while converting
# those matrices, because a helper that has already produced both of them is
# exactly the helper that needs a test:
#   (a) PREFIX STRIPPING assumed `${INTFER_BUILD_DIR:-$PWD/build}` while the
#       matrices write `$ROOT/build`, so two legitimate directories
#       (device_precond_matrix, gate_inventory_matrix) were reported as
#       FOREIGN on a clean tree -- an ALWAYS-RED leak check, manufactured by
#       the refactor itself.
#   (b) it required INTFER_BUILD_DIR to be EXPORTED, while one matrix sets only
#       the local BUILD_DIR, so its M-H died with "must be set by the caller"
#       and declared its own leak assertion VACUOUS.
#
# AUDIT #21 APPLIES: mutate the INPUT SHAPE, not the helper's leniency, and
# require rejection. A helper cannot be made to fail by relaxing it -- that is
# the unobservable mutation.
#
# Usage: scripts/checks/check_gate_pop.sh
# Exit: 0 = all cases behaved, 1 = otherwise.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/intfer_pop.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
trap 'rm -rf "$WORK"; exit 130' INT
trap 'rm -rf "$WORK"; exit 143' TERM

# A synthetic scripts/ holding the HELPER plus matrix-shaped files whose path
# assignments we control. Pointing the helper at a synthetic tree is what makes
# the prefix cases testable at all.
SYN="$WORK/scripts"; BD="$WORK/build"
mkdir -p "$SYN" "$BD"
cp "$SCRIPT_DIR/../lib/_gate_pop.sh" "$SYN/"
# load_helper: source the helper with INTFER_BUILD_DIR set (the normal case)
load_helper() { ( set +u; . "$SYN/_gate_pop.sh"; ) >/dev/null 2>&1; }

bad=0
echo "== GATE POP SELF-CHECK (_gate_pop.sh) =="
# Counted so the coverage assertion below can confirm every case was
# REACHED, rather than trusting that the script ran to the end.
# One counter PER CASE, deliberately named n_a..n_e rather than a shared
# n_cases. A shared counter cannot be targeted by a single-line mutation
# anchor (all five increments are identical text), so the matrix could not
# remove one case without matching all five.
n_a=0; n_b=0; n_c=0; n_d=0; n_e=0; n_f=0

# ---- (a) every REAL prefix form a matrix actually uses must be recognised ---
echo "-- (a) prefix forms: every form a real matrix writes is OWNED --"
# The heredocs below carry the literal prefix forms ON PURPOSE -- they are the
# fixtures under test, not paths this script resolves. Tagged so the
# build-dir consumer checker does not read them as an AUDIT #8 violation.
cat > "$SYN/mutate_m1.sh" <<'EOF'   # MUTANT payload
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/device_precond_matrix"
EOF
cat > "$SYN/mutate_m2.sh" <<'EOF'   # MUTANT payload
STAMP="$BUILD_DIR/fold_matrix"
EOF
cat > "$SYN/mutate_m3.sh" <<'EOF'   # MUTANT payload
WORK="$BUILD_DIR/run_all_mutate"
EOF
cat > "$SYN/mutate_m4.sh" <<'EOF'   # MUTANT payload
STAMP="$PWD/build/l0_cite_mutate"
EOF
mkdir -p "$BD"/{device_precond_matrix,fold_matrix,run_all_mutate,l0_cite_mutate}
for d in device_precond_matrix fold_matrix run_all_mutate l0_cite_mutate; do
  echo "sha=deadbeef" > "$BD/$d/last_pass.stamp"
done
got="$(INTFER_BUILD_DIR="$BD" bash -c ". '$SYN/_gate_pop.sh'; known_dirs")"
miss=""
for d in device_precond_matrix fold_matrix run_all_mutate l0_cite_mutate; do
  case " $got " in *" $d "*) ;; *) miss="$miss $d";; esac
done
if [ -n "$miss" ]; then
  echo "  FAIL  these REAL matrix directories are not recognised as owned:$miss"
  echo "        that is the PWD-vs-ROOT build-prefix defect, and it makes the"
  echo "        leak check always-red on a clean tree."
  bad=$((bad+1))
else
  echo "  ok    all four prefix forms recognised: $got"
  n_a=$((n_a+1))
fi

# ---- (b) the converse: a clean synthetic tree must be QUIET ----------------
echo "-- (b) converse: a clean tree is NOT reported as leaking --"
out="$(INTFER_BUILD_DIR="$BD" bash -c ". '$SYN/_gate_pop.sh'; pop_assert_no_leak clean")"; rc=$?
if [ "$rc" -eq 0 ]; then
  echo "  ok    ok -- the assertion stays SILENT on a clean tree"
  n_b=$((n_b+1))
else
  echo "  FAIL  rc=$rc; a clean tree was reported as a leak"; printf '%s\n' "$out" | sed 's/^/        /'
  bad=$((bad+1))
fi

# ---- (c) and it must go RED on a genuinely foreign directory ---------------
echo "-- (c) a foreign directory IS reported (the check can fail) --"
mkdir -p "$BD/fixture_leak"; echo x > "$BD/fixture_leak/last_pass.stamp"
out="$(INTFER_BUILD_DIR="$BD" bash -c ". '$SYN/_gate_pop.sh'; pop_assert_no_leak planted")"; rc=$?
rm -rf "$BD/fixture_leak"
if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q 'fixture_leak'; then
  echo "  ok    rejected, and it named fixture_leak"
  n_c=$((n_c+1))
else
  echo "  FAIL  rc=$rc (want non-zero); the foreign dir was not named"; printf '%s\n' "$out" | sed 's/^/        /'
  bad=$((bad+1))
fi

# ---- (d) INTFER_BUILD_DIR exported vs only a LOCAL BUILD_DIR ---------------
# The (b) defect: the helper demanded the exported variable and died when a
# caller set only the local. Both shapes must work.
echo "-- (d) variable shape: exported INTFER_BUILD_DIR OR local BUILD_DIR --"
r1="$(INTFER_BUILD_DIR="$BD" bash -c ". '$SYN/_gate_pop.sh'; pop_dirs | wc -l")"
r2="$(BUILD_DIR="$BD" bash -c ". '$SYN/_gate_pop.sh'; pop_dirs | wc -l")"
if [ "$r1" = "$r2" ] && [ "$r1" -gt 0 ]; then
  echo "  ok    both shapes enumerate the same $r1 dir(s)"
  n_d=$((n_d+1))
else
  echo "  FAIL  exported=$r1 local=$r2 -- the helper still requires one shape"
  bad=$((bad+1))
fi

# ---- (e) and with NEITHER set it must fail LOUDLY, not silently enumerate -
echo "-- (e) with neither variable set it must fail closed --"
out="$(env -u INTFER_BUILD_DIR -u BUILD_DIR bash -c ". '$SYN/_gate_pop.sh'; pop_dirs" 2>&1)"; rc=$?
if [ "$rc" -ne 0 ]; then
  echo "  ok    refused (rc=$rc) instead of reporting an empty population as ok"
  n_e=$((n_e+1))
else
  echo "  FAIL  returned rc=0 with no build dir -- a vacuous enumeration"; printf '%s\n' "$out" | head -3 | sed 's/^/        /'
  bad=$((bad+1))
fi

# ---- (f) A COMMENT IS NOT A DECLARATION ------------------------------------
# MEASURED 2026-10-04. A new matrix explained in its comment block that STAMP
# declarations must start with a known base var, and quoted a sibling's real
# declaration as an EXAMPLE. known_dirs() read that prose as a declaration, so
# the new matrix's own directory was reported as an unowned leak -- a false alarm
# on a healthy tree, which is the exact failure _gate_pop.sh's header says the
# derivation exists to prevent. Both directions are required: the comment must be
# IGNORED, and the real declaration in the same file must still be harvested.
echo "-- (f) a comment quoting a declaration is ignored; the real one is not --"
cat > "$SYN/mutate_f.sh" <<'EOF'   # MUTANT payload
# e.g. mutate_l0_cite.sh: STAMP="$BUILD_DIR/quoted_example_mutate"
LOG_DIR="${INTFER_BUILD_DIR:-$ROOT/build}/real_decl_mutate"
EOF
k="$(INTFER_BUILD_DIR="$BD" bash -c ". '$SYN/_gate_pop.sh'; known_dirs")"
rm -f "$SYN/mutate_f.sh"
if printf '%s' "$k" | tr ' ' '\n' | grep -qx 'real_decl_mutate' \
   && ! printf '%s' "$k" | tr ' ' '\n' | grep -qx 'quoted_example_mutate'; then
  echo "  ok    real declaration harvested, the commented one ignored"
  n_f=$((n_f+1))
else
  echo "  FAIL  got: $k"
  echo "        want real_decl_mutate present and quoted_example_mutate absent --"
  echo "        reading a comment as a declaration is a false leak on a clean tree"
  bad=$((bad+1))
fi

# A DROPPED CASE MUST FAIL. Each case bumps `bad` only when its assertion
# FAILS; merely deleting a case removes an ok line and leaves bad=0, so the
# checker still exited 0 with a measurement silently gone. Caught by
# mutate_gate_pop.sh -- all five states scored MISSED until this existed.
n_ok=0
for v in "$n_a" "$n_b" "$n_c" "$n_d" "$n_e" "$n_f"; do n_ok=$((n_ok+v)); done
if [ "$n_ok" -lt 6 ]; then
  echo "  FAIL  [coverage] only $n_ok of 6 cases reached -- a case was"
  echo "        deleted rather than failed. A removed measurement is not a pass."
  bad=$((bad+1))
fi

echo
if [ "$bad" -eq 0 ]; then
  echo "GATE POP SELF-CHECK: PASS (prefix forms, converse, and both variable shapes)"
  echo "RESULT: _gate_pop.sh recognises every real matrix path form and fails closed"
  exit 0
fi
echo "GATE POP SELF-CHECK: FAIL ($bad case(s))"
echo "RESULT: _gate_pop.sh does NOT behave as its three callers assume"
exit 1