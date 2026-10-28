#!/usr/bin/env bash
# Derive, from SOURCE, which mutation matrices wipe a FIXED sandbox, and assert each
# of them holds a concurrency lock OUTSIDE the tree it wipes.
#
# WHY THIS EXISTS (the defect, and it is measured, not hypothetical). A matrix whose
# sandbox is a fixed path under $BUILD does `rm -rf "$SB"` at startup. Two
# invocations of that matrix therefore share ONE sandbox: the second's rm -rf empties
# the first's temp root mid-state, and the first then grades its state against a
# corpus the other instance has just restored or re-copied.
#
# MEASURED 2026-10-21 on mutate_durable_reads.sh, running two instances side by side,
# both symptoms:
#   - one instance died at its BASELINE with "FAIL baseline rc=1" and the checker's
#     own refusal "7 SUBJECT_READS marker(s) against 8 reader(s)" -- a reader whose
#     file was mid-copy when the other instance touched the tree;
#   - and a state that plants a defect and then sees a HEALTHY corpus is the
#     signature of a restore from another instance, not of a weak predicate.
#
# *** THE POPULATION PREDICATE, and it is SEMANTIC, not a name grep. *** Three
# conditions are NECESSARY and all three are required:
#   (1) FIXED      the path does not derive from mktemp. A mktemp path is unique per
#                  invocation, so nothing is shared and there is nothing to lock.
#   (2) WIPED      the path is rm -rf'd. A stamp file that is overwritten is not the
#                  shared-sandbox collision class.
#   (3) A WORKING DIRECTORY, wiped ACROSS STATES -- the path holds the matrix's
#                  mutable working state, so a second invocation's wipe lands while a
#                  first invocation is mid-state.
#
# Condition (3) is where two hand-written classifiers of this file DISAGREED, and
# both were wrong, in OPPOSITE directions. The disagreement is recorded because it is
# the reason this predicate is derived from semantics and not from a variable name:
#   - A name-based pass (SB/WORK/LOG_DIR/SANDBOX/...) MISSED mutate_l0_cite.sh, whose
#     sandbox is named MUT and is wiped+rebuilt per state at line 98. Hand-verified:
#     two invocations collide exactly as the class describes.
#   - A value-based pass counted mutate_engine_cert.sh's MUT, which is a FILE removed
#     with `rm -f` (not a directory), and mutate_gpu_recover.sh's PROBE, which is
#     created and destroyed INSIDE one function (asfound_leak) and never spans a
#     state. Hand-verified: neither is the collision class.
# So the predicate below requires the wiped path to be a DIRECTORY that a STATE
# rebuilds or occupies, and excludes single-artefact files.
#
# WHAT IS CHECKED, all from comment-stripped EXECUTABLE code:
#   1. population  every matrix meeting (1)(2)(3) is DERIVED, so a new matrix joins
#                  the population on its own and no hand-maintained list can go stale.
#   2. lock_exists the matrix declares LOCK and acquires it with mkdir.
#   3. lock_outside the lock path is NOT inside (nor equal to) any fixed path the
#                  matrix wipes. A lock inside the wiped tree is deleted by the wipe
#                  it is supposed to survive -- the exact defect in
#                  mutate_run_all_checks.sh, whose LOCK="$WORK/.lock" is removed by
#                  its own `rm -rf "$WORK"`, and whose lock is taken AFTER that wipe.
#   4. lock_before the mkdir acquiring the lock precedes the first wipe.
#   5. exit_release the lock is released on EXIT (rmdir inside a trap naming EXIT),
#                  so a run that dies mid-state does not brick the next one.
#
# ANTI-VACUITY: the derived population must be non-empty (>= 3 members) or the
# checker refuses rather than reading "nothing to check" as PASS.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

echo "== SANDBOX-LOCK COVERAGE OF THE MUTATION MATRICES =="

MATCHERS="$(ls "$ROOT"/scripts/matrices/mutate_*.sh 2>/dev/null)"
[ -n "$MATCHERS" ] || { echo "  FAIL no mutation matrix found under $ROOT/scripts" >&2; exit 1; }

# The derivation is python3, embedded exactly as check_durable_reads.sh embeds its
# resolver: the population is derived from code, not asserted by hand.
derive() { python3 - "$ROOT" <<'PYQ'
import glob, os, re, sys
root = sys.argv[1]

def strip_comments(s):
    return "\n".join(l for l in s.split("\n") if not re.match(r'^\s*#', l))

def strip_heredocs(s):
    # heredoc bodies are SYNTHETIC CORPUS the matrix builds, not its own behaviour
    return re.sub(r"<<-?'?[A-Za-z_][A-Za-z_0-9]*'?.*?\n[ \t]*[A-Za-z_][A-Za-z_0-9]*[ \t]*\n", "", s, flags=re.S)

def val_map(s):
    v = {}
    for m in re.finditer(r'^[ \t]*([A-Za-z_][A-Za-z_0-9]*)=("[^"\n]*"|\'[^\'\n]*\')[ \t]*$', s, re.M):
        v[m.group(1)] = m.group(2).strip("\"'")
    return v

def expand(x, v, depth=12):
    out = x
    for _ in range(depth):
        b = out
        for _i in range(4):
            out = re.sub(r'\$\{[A-Za-z_][A-Za-z_0-9]*:-([^{}]*)\}', r'\1', out)
            out = re.sub(r'\$\{([A-Za-z_][A-Za-z_0-9]*)\}', lambda m: v.get(m.group(1), m.group(0)), out)
        for k, vv in v.items():
            out = out.replace("${%s}" % k, vv).replace("$" + k, vv)
        if out == b:
            break
    return out

# Condition (3): the wiped path must be a DIRECTORY that spans states. A path is
# accepted when it is assigned from a variable whose name denotes a working tree
# (SB/WORK/LOG_DIR/SANDBOX/.../MUT) OR when the wipe is immediately followed by a
# rebuild into it (`rm -rf "$X"; cp -r ... "$X"`), which is the per-state rebuild
# shape measured in mutate_l0_cite.sh:98. A wipe of a FILE (rm -f) and a wipe of a
# directory created and destroyed inside one function are both excluded.
WORKDIR = ("SB", "WORK", "LOG_DIR", "SANDBOX", "TROOT", "MUT", "MUTROOT", "TMPD",
           "CRROOT", "PRISTINE_DIR", "SANDBOX_TREE")

for p in sorted(glob.glob(os.path.join(root, "scripts/matrices/mutate_*.sh"))):
    raw = open(p).read()
    s = strip_comments(strip_heredocs(raw))
    v = val_map(s)
    v.setdefault("ROOT", root)
    v.setdefault("SCRIPT_DIR", os.path.join(root, "scripts/matrices"))
    name = os.path.basename(p)

    # --- the lock, if any ---
    lockname = lockpath = lockraw = None
    lockline = acquire = 0
    for i, line in enumerate(s.split("\n"), 1):
        m = re.match(r'^[ \t]*([A-Za-z_][A-Za-z_0-9]*)=("?)([^"\n]*?)\2?(\.lock)"?[ \t]*$', line)
        if m and lockname is None:
            lockname = m.group(1)
            # RAW is kept alongside the expanded value, because the two disagree in
            # the case that matters: `LOCK="BUILD/x.lock"` (a literal variable
            # NAME) expands to a plausible absolute path while the SHELL would
            # create a directory called BUILD under the CWD. The raw text is what
            # gets judged for that shape.
            lockraw = m.group(3)
            # keep the ".lock" SUFFIX: the lock is the DIRECTORY named by the
            # assignment, and the inside/outside test compares it against the
            # wiped tree. Dropping the suffix compared the wrong two paths.
            lockpath = expand(m.group(3) + m.group(4), v)
            lockline = i
            continue
        if lockname and acquire == 0 and re.search(r'mkdir\s+(-p\s+)?"?\$%s"?' % re.escape(lockname), line):
            acquire = i
    if lockname and lockpath:
        # RELEASE: computed BEFORE any early-exit, because a predicate that is
        # evaluated after a `continue` is a predicate that cannot fire for the row
        # it was meant to grade -- and it must be computed on the RAW path, not the
        # expanded one, for the same reason the BAD check below uses the raw text.
        released = False
        if re.search(r'\brm(dir)?\b[^"]*"\$%s"' % re.escape(lockname), s):
            released = True
        has_exit_trap = any(l.startswith("trap ") and "EXIT" in l for l in s.split("\n"))
        release = bool(released and has_exit_trap)
        # *** THE RAW TEXT IS JUDGED, NOT THE EXPANDED VALUE. *** MEASURED DEFECT in
        # this checker's own first run: a propagation script wrote the lock as
        # LOCK="BUILD/name.lock" -- the variable NAME as literal text, so the lock
        # would be created in a directory called BUILD under the CWD instead of
        # under the build root. Two things went wrong, and the second is the
        # dangerous one:
        #   (a) the EXPANDED value still resolved to an absolute path (the resolver
        #       expands BUILD whether or not the shell would), so a check on the
        #       expanded path PASSED a tree whose lock was a no-op; and
        #   (b) worse, the planted matrix then fell OUT of the derived population
        #       (22 -> 21) and the checker still printed PASS -- the matrix stopped
        #       being checked and NOTHING said so. That is the vacuous-green shape
        #       this repo refuses, in the checker that exists to prevent it.
        # So the RAW assignment is judged: a lock path that does not begin with a
        # variable reference or an absolute path is refused outright.
        raw = lockraw
        if not (raw.startswith("$") or raw.startswith("/") or raw.startswith('"')):
            print("\x1f".join([name, "BAD", str(lockline), raw, str(acquire),
                  "yes" if release else "no", ""]))
            continue
        lockpath = expand(lockpath, v)
    # RELEASE: the lock is removed AND an EXIT trap exists. The removal may sit
    # in a FUNCTION the trap calls (mutate_run_all_checks.sh removes it inside
    # cleanup(), and its trap line reads only `trap cleanup EXIT`), so the removal
    # is searched across the file and the EXIT trap is required separately.
    release = False
    removes = False
    for line in s.split("\n"):
        if lockname and re.search(r'\brm(dir)?\b[^"]*"\$%s"' % re.escape(lockname), line):
            removes = True
        if lockpath and re.search(r'\brm(dir)?\b[^"]*%s' % re.escape(lockpath), line):
            removes = True
    has_exit_trap = any(l.startswith("trap ") and "EXIT" in l for l in s.split("\n"))
    release = bool(removes and has_exit_trap)

    # --- the fixed paths this matrix wipes ---
    lines = s.split("\n")
    wiped = []
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r'\brm\s+(-[a-zA-Z]+\s+)*-?[a-zA-Z]*[rf][a-zA-Z]*\s+(?:"\$(\w+)"|\$\{?(\w+)\}?)', line):
            var = m.group(2) or m.group(3)
            if var is None:
                continue
            if re.search(r'rm\s+(-[a-zA-Z]+\s+)*-?[a-zA-Z]*f[a-zA-Z]*\s+"\$%s"' % re.escape(var), line) \
               and not re.search(r'rm\s+(-[a-zA-Z]+\s+)*-?[a-zA-Z]*r[a-zA-Z]*\s+"\$%s"' % re.escape(var), line):
                continue      # rm -f only: a file, not a wiped directory
            rawv = v.get(var)
            if rawv is None:
                continue
            ex = expand(rawv, v)
            if "mktemp" in ex or not ex.startswith("/"):
                continue      # immune (unique per invocation) or not a path
            # condition (3): a working directory. Either the name denotes one, or the
            # wipe is IMMEDIATELY followed by a rebuild into the same path.
            rebuild = False
            if i < len(lines):
                nxt = lines[i]
                if re.search(r'(cp|mkdir)\b[^|]*"\$%s"' % re.escape(var), nxt):
                    rebuild = True
            if var not in WORKDIR and not rebuild:
                continue
            wiped.append((i, var, ex))
    if not wiped:
        continue
    print("\x1f".join([name, str(len(wiped)), str(lockline), lockpath or "",
                       str(acquire), "yes" if release else "no",
                       ",".join("%d:%s:%s" % (w[0], w[1], w[2]) for w in wiped)]))
PYQ
}

ROWS="$(derive)"
[ -n "$ROWS" ] || { echo "  FAIL the derivation produced no rows -- a population of nothing" >&2
                   echo "        must not read as PASS" >&2; exit 1; }

population=0; failed=0
while IFS=$'\x1f' read -r name nfix lockline lockpath acquire release wiped; do
  [ -n "${name:-}" ] || continue
  # A "BAD" row is a matrix whose lock path is not an absolute path -- the shape a
  # propagation script produces when it writes the variable NAME as literal text
  # (LOCK="BUILD/x.lock"). It is a FAILURE, never a silent skip: a row dropped here
  # is a matrix whose lock was never checked, which is the vacuous-green shape.
  if [ "$nfix" = "BAD" ]; then
    failed=$((failed + 1))
    printf '  FAIL  %-40s LOCK-PATH-NOT-ABSOLUTE(%s)\n' "$name" "$lockpath" >&2
    printf '        the lock path must be an absolute path; a literal variable name\n' >&2
    printf '        here would create the lock under the CWD, not the build root\n' >&2
    continue
  fi
  case "$nfix" in ''|*[!0-9]*) continue;; esac
  [ "$nfix" -gt 0 ] || continue
  population=$((population + 1))
  bad=""
  [ "$lockline" -gt 0 ] || bad="$bad LOCK-NOT-DECLARED"
  [ "${acquire:-0}" -gt 0 ] || bad="$bad LOCK-NOT-ACQUIRED"
  [ "$release" = "yes" ] || bad="$bad LOCK-NOT-RELEASED"
  IFS=',' read -ra _w <<<"$wiped"
  for _e in "${_w[@]}"; do
    _d="${_e#*:}"; _d="${_d#*:}"
    case "$lockpath" in "$_d"|"$_d"/*) bad="$bad LOCK-INSIDE-WIPED-TREE($_d)";; esac
  done
  first="$(printf '%s\n' "$wiped" | tr ',' '\n' | cut -d: -f1 | sort -n | head -1)"
  case "$first" in ''|*[!0-9]*) first="";; esac
  if [ -n "$first" ] && [ "${acquire:-0}" -gt "$first" ]; then bad="$bad LOCK-AFTER-WIPE"; fi
  if [ -n "$bad" ]; then
    failed=$((failed + 1))
    printf '  FAIL  %-40s %s\n' "$name" "$bad" >&2
    printf '        lock=%s (acquired at line %s, released=%s)\n' "${lockpath:-<none>}" "${acquire:-0}" "$release" >&2
  else
    printf '  ok    %-40s lock=%s released=%s\n' "$name" "$lockpath" "$release"
  fi
done <<<"$ROWS"

echo
if [ "$population" -lt 3 ]; then
  echo "  FAIL derived only $population matrix(es) with a fixed-wipe sandbox; a checker" >&2
  echo "        that examined nothing must not read as PASS" >&2
  exit 1
fi
echo "-- derived population: $population matrix(es) with a fixed-wipe sandbox"
echo "   (matrices whose sandbox derives from mktemp are immune: unique per invocation)"
if [ "$failed" -gt 0 ]; then
  echo "RESULT: FAIL ($failed of $population matrix(es) unpinned)" >&2
  exit 1
fi
echo "RESULT: PASS ($population matrix(es), each locked outside the tree it wipes)"
exit 0
