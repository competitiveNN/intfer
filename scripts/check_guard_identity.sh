#!/usr/bin/env bash
# A guard keyed on a resource NAME cannot see duplicates when the name itself
# carries the per-instance discriminator.
#
# THE DEFECT, MEASURED on scripts/bench_when_quiet.sh. It had an artifact
# collision guard -- refuse to append to an existing $RUNLOG -- which was
# correct, staged in both directions, and mutation-proven (state B-R). It also
# could not see a CONCURRENT run, because the artifact name is
#
#     when_quiet_<second>_<shortHEAD>_$$.log
#
# which carries `$$`. Two LIVE runs therefore get DIFFERENT names, neither sees
# the other, and the guard is structurally incapable of detecting the thing it
# was named for.
#
# WHY THAT MATTERED RATHER THAN BEING COSMETIC. These are 27B `--tp 2` batches. A
# second one loads the same CPU and the same GPU, so BOTH runs' arms cross the
# load threshold and BOTH artifacts withhold every figure -- while both artifacts
# stay perfectly CLEAN. The failure is invisible in the very artifacts the guard
# was written to protect, and two runs cannot each obtain the quiet window they
# are both waiting for. A pid-owning lock is the only thing that can see it,
# because a lock directory does NOT carry `$$`.
#
# THE RULE, derived from source rather than from the one file I happened to fix
# (audit #7/#8: a hand-remembered list is a citation, not a closure):
#
#     a script that (1) refuses when its resource path ALREADY EXISTS, and
#     (2) builds that path from a per-instance discriminator ($$ / a pid / a
#     timestamp / a uuid), MUST also hold a shared lock -- a lock whose name
#     does NOT carry the discriminator, whose owner is recorded, and whose
#     liveness is proved with `kill -0` before it is taken over.
#
#     THE CONVERSE, and it is what keeps the rule from crying wolf: a script
#     that refuses on an existing path whose name carries NO discriminator is
#     self-protecting, because the name IS the key. Two runs then collide BY
#     CONSTRUCTION and the refusal is the whole mechanism. No lock required.
#     Requiring one there would force a pointless lock on every idempotent
#     re-run, which is how a check like this gets disabled.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${INTFER_SCRIPTS_DIR:-$SCRIPT_DIR}"

# classify is done in python, not grep, and the reason is worth recording: the
# FIRST version harvested the discriminator from ANYWHERE in the file and matched a
# lock only in the `${X:-y}` form. Run against the real tree it reported 14 of 19
# sites as violations, and all 14 were defects in the RULE, not the tree:
#   * check_bench_load_gate.sh writes `echo $$ > "$GD/live/pid"` -- the `$$` is a pid
#     VALUE and the path is FIXED. The rule has to look inside the GUARDED VARIABLE'S
#     OWN ASSIGNMENT, not anywhere in the file.
#   * run_all_checks.sh declares `LOCK="$LOCK_DIR/.run_all_checks.lock"`, a plain
#     assignment. The rule required the `${X:-y}` form and so reported a script that
#     HAS a correct pid-owning lock, with kill -0 and an atomic mkdir, as unlocked.
#   * check_*/mutate_* quote the unguarded shape as fixture and patch text.
# A check that cries wolf is the mirror defect and trains everyone to ignore the
# line, so the rule is calibrated to the tree and then re-run -- which is the
# audit #23 lesson (enumerate the CLAIM, not the files the first fix touched).
#
# *** VERDICT IN A GLOBAL, CAPTURED, NOT LOST IN $(...) *** -- the recorded lesson
# that a helper assigning to a global inside command substitution dies in the
# subshell and the parent reads the stale value. So the python prints two lines and
# the shell ASSIGNS from the captured string.
classify() {
  local f="$1" out rc
  out="$(python3 "$TMP/classify.py" "$f")" || out="RC 2"
  REASON="$(printf '%s\n' "$out" | sed -n 's/^REASON //p')"
  rc="$(printf '%s\n' "$out" | sed -n 's/^RC //p' | head -1)"
  # RETURN the code, never print it. An earlier version PRINTED it, which silently
  # made every caller-s `[ $? -eq 1 ]` read 0 -- so the self-check rejected correct
  # rule text while the real tree looked clean. A helper that reports its verdict
  # any way other than the one its callers read is worse than one that does not
  # report at all.
  [ -n "$rc" ] || rc=2
  return "$rc"
}

# *** BEHAVIOURAL SELF-CHECK FIRST, and it GATES. *** Executable fixtures driven
# through this exact function -- never a grep of this script's own source, which is
# the self-defeating form recorded in mutate_fold_matrix.sh's own self-check.
TMP="$(mktemp -d "${TMPDIR:-/dev/shm}/guard_identity.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
trap 'rm -rf "$TMP"; exit 130' INT
trap 'rm -rf "$TMP"; exit 143' TERM
self_bad=0

cat >"$TMP/classify.py" <<'PYEOF'
#!/usr/bin/env python3
"""classify <file>: print 'RC <n>' and 'REASON <text>'. 0 ok, 1 violation, 2 n/a."""
import re, sys

DISC = ("$$", "$(date", "uuid", "UUIDGEN", "${EPOCH", "BASHPID", "$(hostname")

def main() -> int:
    path = sys.argv[1]
    src = open(path, errors="replace").read()
    base = path.rsplit("/", 1)[-1]
    # Verifiers may PLANT the defect: they quote the unguarded shape as fixture or
    # patch text. Excluded by PREFIX, same rationale and same named cost as
    # check_dri_probe.sh: a harness that plants a defect must be able to name it.
    if base.startswith("check_") or base.startswith("mutate_"):
        print("RC 2"); return 0
    src = re.sub(r"(?m)^[ \t]*#.*$", "", src)          # a comment is not a mechanism
    if not re.search(r"refus|already exists|FILE_", src, re.I):
        print("RC 2"); return 0
    # Guarded variables: the ones tested with [ -e "$V" ] / [ -f "$V" ].
    guards = set(re.findall(r"\[[ \t]*-[ef][ \t]*\"\$\{?([A-Za-z_]\w*)\}?\"", src))
    if not guards:
        print("RC 2"); return 0
    # A guard is only SUBJECT to this rule when the guarded name itself is
    # per-instance. Look inside that variable's own assignment -- not the file.
    discriminated = []
    for v in sorted(guards):
        m = re.search(r"(?m)^[ \t]*" + re.escape(v) + r"=(.*)$", src)
        if not m:
            continue
        val = m.group(1)
        hit = [d for d in DISC if d in val]
        if hit:
            discriminated.append((v, hit[0]))
    if not discriminated:
        print("RC 0"); return 0                            # fixed name: self-protecting
    v, d = discriminated[0]
    # A shared lock: declared in ANY form, and its OWN name must not carry the
    # discriminator or it serialises nothing.
    locks = re.findall(r"(?m)^[ \t]*([A-Za-z_]*LOCK[A-Za-z_]*)=(.*)$", src)
    good = [n for n, val in locks if not any(x in val for x in DISC)]
    if not locks:
        print("REASON refuses an existing '%s'-discriminated path (\"$%s\") but takes NO" % (d, v))
        print("REASON shared lock, so two LIVE runs get different names and neither sees the")
        print("REASON other -- concurrent runs load the same CPU/GPU and BOTH lose every figure")
        print("RC 1"); return 0
    if not good:
        print("REASON the only lock ('%s') carries the discriminator itself, so it is" % locks[0][0])
        print("REASON per-instance and serialises nothing")
        print("RC 1"); return 0
    if not re.search(r"kill -0", src):
        print("REASON the lock ('%s') records an owner but never proves it with" % good[0])
        print("REASON kill -0, so a lock left by a killed run blocks forever")
        print("RC 1"); return 0
    if not re.search(r'mkdir "\$' + re.escape(good[0]) + r'"', src):
        print("REASON no `mkdir \"$%s\"` acquisition, so the take is not atomic and two" % good[0])
        print("REASON simultaneous runs can both believe they hold it")
        print("RC 1"); return 0
    print("RC 0"); return 0

sys.exit(main())
PYEOF


# (a) THE DEFECT: refuses an existing $$-named artifact, no lock.
cat >"$TMP/no_lock.sh" <<'EOF'
#!/usr/bin/env bash
RUNLOG="$DIR/when_quiet_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
if [ -e "$RUNLOG" ]; then
  echo "FAIL: $RUNLOG already exists -- refusing to append." >&2
  exit 2
fi
EOF
# (b) THE FIX: the same, plus a pid-owning lock whose name is NOT per-instance.
cat >"$TMP/with_lock.sh" <<'EOF'
#!/usr/bin/env bash
WQ_LOCK="${INTFER_WQ_LOCK:-$RUNLOG_DIR/.bench_when_quiet.lock}"
RUNLOG="$RUNLOG_DIR/when_quiet_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
if ! mkdir "$WQ_LOCK" 2>/dev/null; then
  wq_owner="$(cat "$WQ_LOCK/pid" 2>/dev/null || true)"
  if [ -n "$wq_owner" ] && ! kill -0 "$wq_owner" 2>/dev/null; then
    rm -rf "$WQ_LOCK"; mkdir "$WQ_LOCK" 2>/dev/null
  else
    echo "FAIL: another run holds $WQ_LOCK -- refusing." >&2
    exit 2
  fi
fi
echo $$ > "$WQ_LOCK/pid"
if [ -e "$RUNLOG" ]; then
  echo "FAIL: $RUNLOG already exists -- refusing to append." >&2
  exit 2
fi
EOF
# (c) THE CONVERSE: refuses an existing FIXED-name path. Self-protecting: two runs
#     collide by construction, so the refusal IS the mechanism and no lock is owed.
cat >"$TMP/fixed_name.sh" <<'EOF'
#!/usr/bin/env bash
OUT="$DIR/result_block.sh"
if [ -e "$OUT" ]; then
  echo "FAIL: $OUT already exists -- refusing." >&2
  exit 2
fi
EOF
# (d) NOT A SUBJECT: no refusal at all.
cat >"$TMP/no_refusal.sh" <<'EOF'
#!/usr/bin/env bash
RUNLOG="$DIR/run_$$.log"
echo hi > "$RUNLOG"
EOF
# (e) A lock whose OWN name carries the discriminator is per-instance, so it
#     serialises nothing -- the trap-door version of "has a lock".
cat >"$TMP/lock_is_per_instance.sh" <<'EOF'
#!/usr/bin/env bash
LOCK="${MYLOCK:-$DIR/.lock_$$}"
RUNLOG="$DIR/run_$$.log"
mkdir "$LOCK" 2>/dev/null || { echo "FAIL: held by another run." >&2; exit 2; }
echo $$ > "$LOCK/pid"
kill -0 "$(cat "$LOCK/pid")" 2>/dev/null
if [ -e "$RUNLOG" ]; then
  echo "FAIL: $RUNLOG already exists -- refusing." >&2
  exit 2
fi
EOF

classify "$TMP/no_lock.sh";               a=$?
classify "$TMP/with_lock.sh";             b=$?
classify "$TMP/fixed_name.sh";            c=$?
classify "$TMP/no_refusal.sh";            d=$?
classify "$TMP/lock_is_per_instance.sh";  e=$?
[ "$a" -eq 1 ] || { echo "SELF-CHECK FAIL: a \$\$-named refusal with no lock was not REJECTED (rc=$a)"; self_bad=$((self_bad+1)); }
[ "$b" -eq 0 ] || { echo "SELF-CHECK FAIL: the pid-owning lock shape was not ACCEPTED (rc=$b)"; self_bad=$((self_bad+1)); }
[ "$c" -eq 0 ] || { echo "SELF-CHECK FAIL: a FIXED-name refusal was not ACCEPTED (rc=$c) -- the rule would cry wolf"; self_bad=$((self_bad+1)); }
[ "$d" -eq 2 ] || { echo "SELF-CHECK FAIL: a script with no refusal is not exempt (rc=$d)"; self_bad=$((self_bad+1)); }
[ "$e" -eq 1 ] || { echo "SELF-CHECK FAIL: a per-instance LOCK name was not REJECTED (rc=$e)"; self_bad=$((self_bad+1)); }

echo "== GUARD IDENTITY CHECK (a name-keyed guard cannot see duplicates it names) =="
if [ "$self_bad" -eq 0 ]; then
  echo "  ok    self-check: \$\$-named refusal with no lock REJECTED; pid-owning lock"
  echo "        ACCEPTED; FIXED-name refusal ACCEPTED (the converse); no-refusal"
  echo "        exempt; a per-instance LOCK name REJECTED"
else
  echo
  echo "GUARD IDENTITY: FAIL (self-check did not behave; the rule is not trustworthy)"
  exit 1
fi

shopt -s nullglob
SCRIPTS=("$SCRIPTS_DIR"/*.sh)
shopt -u nullglob
self_is() { [ "$1" = "$0" ]; }

viol=0; sites=0
for f in "${SCRIPTS[@]}"; do
  self_is "$f" && continue
  classify "$f"; r=$?
  [ "$r" -eq 2 ] && continue
  sites=$((sites + 1))
  if [ "$r" -eq 0 ]; then
    printf '  ok    %-34s guarded name is self-protecting, or holds a shared lock\n' "$(basename "$f")"
  else
    printf '  FAIL  %-34s %s\n' "$(basename "$f")" "$REASON"
    viol=$((viol + 1))
  fi
done

# *** POPULATION GUARD. *** The SUBJECT must still exist -- not a violation. A
# healthy tree must be allowed to have none (the audit #20 shape, inverted).
if [ "$sites" -eq 0 ]; then
  echo "FAIL: no script in scripts/ refuses an existing resource path, so a green" >&2
  echo "      verdict would certify a population it never harvested." >&2
  exit 1
fi

echo "  ..    $sites script(s) refuse an existing resource path"
echo
if [ "$viol" -eq 0 ]; then
  echo "GUARD IDENTITY: PASS ($sites refusal site(s); every discriminated name holds a shared lock)"
  echo "RESULT: a guard keyed on a \$\$- or timestamp-bearing name is paired with a lock"
  echo "        whose name does NOT carry the discriminator, so a concurrent run is seen"
  exit 0
fi
echo "GUARD IDENTITY: FAIL ($viol of $sites refusal site(s) cannot see a concurrent run)"
echo "RESULT: a per-instance artifact name is guarded without a shared lock"
exit 1
