#!/usr/bin/env python3
"""Find glob-bearing command invocations that sit inside a `shopt -s nullglob`
region. Kept in its own file rather than a heredoc: AUDIT #32 is about a glob
collapsing because of how it was WRITTEN, and a heredoc is exactly the quoting
context that has gone wrong repeatedly in this repo (mangled backslashes, a
premature `PYEOF`, a `'0'` literal terminating a shell string). The scanner for
the glob-collapse class should not itself depend on fragile quoting.

Prints, one per line, and always exits 0 so the caller decides the verdict:

  POPULATION <n>
      how many glob-bearing command invocations were examined inside a
      nullglob region. The caller floors this: a population of zero means the
      scan is broken, not that the repo is clean.

  SITE <file>:<line> [cmd] <text>
      a DEGENERATE site -- a command whose only argument is a glob, run while
      nullglob is on. A no-match therefore leaves the command with NO arguments,
      and `ls` (like cat/rm/head/tar/cp/mv) lists the current directory instead
      of erroring.
"""
import os
import re
import sys

# Commands with a meaningful ZERO-ARGUMENT form. These are the ones where a
# vanished glob changes the meaning of the command rather than making it fail.
CMDS = ("ls", "cat", "rm", "head", "tar", "wc", "sort", "cp", "mv", "cut", "tee")

GLOB = re.compile(r"\*|\?|\[")
NULLGLOB_ON = re.compile(r"shopt -s nullglob")
INVOCATION = re.compile(r"^[\s(]*\(?\s*(%s)\b(.*)$" % "|".join(CMDS))
ARRAY_ASSIGN = re.compile(r"^[\s(]*[A-Za-z_][A-Za-z0-9_]*=\(")


def scan(base="scripts"):
    sites = []
    population = 0
    for name in sorted(os.listdir(base)):
        if not name.endswith(".sh"):
            continue
        full = os.path.join(base, name)
        try:
            lines = open(full, errors="replace").read().split("\n")
        except OSError:
            continue
        enabled = False
        for n, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            enabled_here = False
            if NULLGLOB_ON.search(stripped):
                enabled = True
                enabled_here = True
            if not enabled:
                continue
            if "shopt -u nullglob" in stripped:
                enabled = False
                continue
            # A command can follow `shopt -s nullglob` ON THE SAME LINE, inside
            # a subshell: `( shopt -s nullglob; cp "$DIR"/*.py "$DEST" )`. A
            # line-start-only regex misses it entirely -- and it measured
            # POPULATION 0 on a tree that HAS such a line, which is the vacuous
            # harvest shape again, one level up. So when the line itself enabled
            # nullglob, every `;`-separated segment is examined too.
            segs = []
            if enabled_here:
                segs = [seg.strip() for seg in line.split(";")]
            else:
                segs = [line]
            for seg in segs:
                m = INVOCATION.match(seg)
                if not m:
                    continue
                cmd, rest = m.group(1), m.group(2)
                rest = rest.split(" #")[0]      # drop a trailing comment
                if not GLOB.search(rest):
                    continue
                population += 1
                # ALLOWED FORMS -- each makes a no-match mean "empty", which is
                # the answer, rather than "run the command with nothing".
                if "compgen -G" in rest:
                    population -= 1
                    continue
                if ARRAY_ASSIGN.match(rest):
                    population -= 1
                    continue      # arr=(glob)
                if "|" in rest:
                    population -= 1
                    continue      # piped: empty stream, not a listing
                sites.append((name, n, cmd, stripped[:78]))
                population -= 1
    return population, sites


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "scripts"
    pop, sites = scan(base)
    print("POPULATION %d" % pop)
    for name, n, cmd, text in sites:
        print("SITE %s:%d [%s] %s" % (name, n, cmd, text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
