#!/usr/bin/env python3
"""Derive two facts about scripts/run_all_checks.sh's dispatch() and STEPS lists.

Kept in its OWN file rather than a heredoc inside the checker, for one measured
reason: embedded in a bash heredoc the backslashes in the regexes were mangled and
the parser died with "missing ), unterminated subpattern" -- which surfaced as the
checker silently finding NOTHING to compare, i.e. reporting two empty sets as
agreeing. That is the vacuous-green shape, arrived at through quoting.

Prints, one per line, and exits 0 always so the caller decides the verdict:

  ARM-MISMATCH <label> <runs>
      a case arm whose body executes a path segment other than its own label.
      THIS IS THE SHIPPED BUG: an arm whose path became
          bash "$SCRIPT_DIR/a.sh b.sh"
      is valid shell, passes `bash -n`, and fails rc=127 at runtime.

  LABEL <name>
  STEP  <name>
      the derived populations, so the caller can enforce non-emptiness.

Why the whole path segment is compared and not just the first ".sh" in it: the
mangled path's FIRST .sh still equals the label, so a reader that stops there sees
agreement. That was the second version of this parser and DS-1 passed it.
"""
import re
import sys

WS = r"[ \t]*"
ARM_RE = re.compile(r"^" + WS + r"([A-Za-z0-9_]+\.sh)" + WS + r"\)" + WS + r"$")
# The pattern needs a literal double-quote inside a character class, so the
# regex is SINGLE-quoted. An earlier version used r"..." and the embedded `"`
# terminated the python string -- which is the second quoting failure in this
# file's life, and the third overall in this session where a regex did not survive
# being written down. Each one surfaced as the scan finding NOTHING rather than as
# a syntax error, which is the shape worth remembering.
BODY_RE = re.compile(r'\$SCRIPT_DIR/([^"]*)"')
END_RE = re.compile(r"^" + WS + r";;" + WS + r"$")

STEPS_ASSIGN = re.compile(r"^" + WS + r'STEPS="([^"]*)"')
STEPS_APPEND = re.compile(r"\$STEPS ([A-Za-z0-9_. ]+)\"")
NAME_RE = re.compile(r"^[A-Za-z0-9_]+\.sh$")


def main(path):
    lines = open(path, errors="replace").read().split("\n")

    labels, steps = set(), set()
    cur, buf = None, []

    def flush(cur, buf):
        if not cur:
            return
        for line in buf:
            m = BODY_RE.search(line)
            if m and m.group(1) != cur:
                print("ARM-MISMATCH %s %s" % (cur, m.group(1)))

    for line in lines:
        m = ARM_RE.match(line)
        if m:
            flush(cur, buf)
            cur, buf = m.group(1), []
            continue
        m = STEPS_ASSIGN.match(line)
        if m:
            steps.update(t for t in m.group(1).split() if NAME_RE.match(t))
        m = STEPS_APPEND.search(line)
        if m:
            steps.update(t for t in m.group(1).split() if NAME_RE.match(t))
        if cur:
            buf.append(line)
            if END_RE.match(line):
                flush(cur, buf)
                cur, buf = None, []
    flush(cur, buf)

    # Labels come from the same single pass but are emitted here so the caller can
    # assert the population is non-empty rather than trusting an empty set to agree
    # with another empty set.
    for line in lines:
        m = ARM_RE.match(line)
        if m:
            labels.add(m.group(1))
    for n in sorted(labels):
        print("LABEL %s" % n)
    for n in sorted(steps):
        print("STEP %s" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "scripts/run_all_checks.sh"))
