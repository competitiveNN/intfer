#!/usr/bin/env python3
"""Attribute every `sycl::free(...act)` to its ENCLOSING Engine function, and
report which of them free the buffer INTFER_FUSE_ACT caches into.

Kept in its own file for the same reason the nullglob scanner is: the rules here
are about nesting and aliasing, not about a pattern, and a heredoc has been the
wrong tool for that in this repo more than once.

Emits, one per line:

  FREE <function> <line>
      a free of a buffer that is NOT the cached one (informational).

  CACHED-FREE <function> <line> <has_clear:0|1>
      a free of ws_[d].act -- the buffer the activation cache lives in. The
      third field says whether a cache clear appears within the preceding few
      lines.

The attribution is the whole point. There are three frees matching
`sycl::free(w?.act)`:

    54   Engine::reset              ws_[d].act   <- the CACHED buffer
    662  Engine::ensure_workspace   w.act        <- the CACHED buffer (alias `w`)
    698  Engine::ensure_workspace48 w.act        <- ws48_[d].act, a DIFFERENT one

A line-local grep cannot tell 662 from 698: identical text, identical local
alias, different owner. Two earlier versions of the caller failed on exactly
that -- one counted only `ws_[d].act` and so MISSED the ensure_workspace
invalidation, the other counted `w.act` too and then failed a healthy tree over a
buffer the cache never touches.
"""
import re
import sys

# NOTE the optional ~: the destructor is `~Engine()`, and a pattern
# without it attributes its frees to NO function -- which silently classified a
# cached-buffer free as unrelated, so removing its invalidation was invisible.
FN = re.compile(r"^\s*[\w:<>*&\s]*Engine::~?(\w+)\s*\(")
FREE = re.compile(r"sycl::free\(\s*(ws_\[d\]|w)\.act")
CLEAR = "act_cache_ptr_ = nullptr"

# The only function whose local `w` alias is the CACHED workspace. The DPAS one
# (ensure_workspace48) and reset both free ws48_ as well, so the alias alone is
# ambiguous and the FREE TARGET is what decides.
WS_OWNERS = ("ensure_workspace",)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "src/engine/engine.cpp"
    src = open(path, errors="replace").read().split("\n")
    cur, curline = None, 0
    for i, l in enumerate(src):
        m = FN.match(l)
        if m:
            cur, curline = m.group(1), i
        if l.strip().startswith("//") or not FREE.search(l):
            continue
        # Classify by the FREE TARGET, not by the enclosing function. A
        # per-function test cannot work: Engine::reset frees BOTH ws_ and ws48_,
        # so its body mentions ws48_ and every ws_ free inside it got excluded.
        target = FREE.search(l).group(1)
        # group 1 excludes the trailing `.act`, so compare against `ws_[d]`
        cached = (target == "ws_[d]") or (cur == "ensure_workspace")
        if cached:
            window = "\n".join(src[max(curline, i - 4):i + 1])
            print("CACHED-FREE %s %d %d" % (cur, i + 1, 1 if CLEAR in window else 0))
        else:
            print("FREE %s %d" % (cur, i + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
