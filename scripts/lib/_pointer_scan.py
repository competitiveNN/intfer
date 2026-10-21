#!/usr/bin/env python3
"""Classify every HISTORY.md title citation in AGENTS.md against HISTORY.md.

Kept in its OWN file rather than a heredoc inside the checker, for the reason
_dispatch_scan.py's own header records: embedded in a bash heredoc the
backslashes in these regexes were mangled and the parser died, which surfaced
as the scan finding NOTHING -- two empty sets agreeing, the vacuous-green shape
reached through quoting. Every regex here is written once, in python, and the
bash side only does set arithmetic on the lines this file prints.

WHY THIS FILE EXISTS. AGENTS.md was split on 2026-10-04 and its long sections
were MOVED to HISTORY.md, leaving `Moved to HISTORY.md ("<title>")` pointers.
HISTORY.md was later truncated (509 lines destroyed), and the repo recorded
that honestly twice over: HISTORY.md carries a banner naming the eleven dated
entries whose bodies are gone, and each AGENTS.md pointer to one of them is
annotated in place. A pointer-integrity checker that asserts the naive
"the title must exist as a heading" refuses TEN of the eleven correct titles
and would have to be weakened to pass -- the exact "weaken the gate to make it
green" defect. So a title resolves by any of FOUR rules, and a title that
resolves by NONE of them is the defect:

  OK-HEADING  the title is a real heading in HISTORY.md
  OK-TEXT     the title is present verbatim in HISTORY.md as text (a banner
              phrase, a bold-text record, a table cell), not necessarily a
              heading -- the body need not be gone for a non-heading title
  OK-BANNER   the title is named in HISTORY.md's destroyed-entries banner, so
              its absence is RECORDED PROVENANCE, not an accident
  DANGLING    the title resolves none of the above, but its AGENTS.md site
              carries an explicit [DANGLING annotation within +/- 3 lines
              (proximity asserted, not assumed: a DANGLING note 500 lines away
              from the pointer it excuses is prose, not a pointer annotation)
  UNRESOLVED  none of the four -- an unaccounted pointer

ALSO EMITTED, and it is the banner's own staleness half: a banner entry that
AGENTS.md no longer cites is a record about a pointer that no longer exists.

Prints, one per line, and exits 0 always so the caller decides the verdict:

  OK-HEADING <title>
  OK-TEXT <title>
  OK-BANNER <title>
  DANGLING <title>
  UNRESOLVED <title>
  BANNER-UNCITED <title>
  CITES <n>
"""
import re
import sys

CITE_RE = re.compile(r'HISTORY\.md\s*\("([^"]+)"\)', re.S)
HEADING_RE = re.compile(r'^#{1,4} (.+)$', re.M)
BANNER_ANCHOR = '**The 11 dated entries whose HISTORY.md detail is gone.**'


def flat(s):
    return re.sub(r'\s+', ' ', s).strip()


def main(agents_path, hist_path):
    agents = open(agents_path, errors='replace').read()
    hist = open(hist_path, errors='replace').read()

    # ---- the populations in HISTORY.md ----
    headings = set(flat(h) for h in HEADING_RE.findall(hist))
    hist_text = flat(hist)
    m = re.search(re.escape(BANNER_ANCHOR) + r'(.*?)'
                  r'\*\*What is not lost', hist, re.S)
    if m is None:
        sys.stderr.write('the destroyed-entries banner is absent from HISTORY.md; '
                         'a harvest of zero banner entries must not read as '
                         '"nothing is dangling"\n')
        sys.exit(3)
    banner = set(flat(t) for t in re.findall(r'`([^`]+)`', m.group(1)))
    if not banner:
        sys.stderr.write('the banner carries no entries\n')
        sys.exit(3)

    # ---- every citation in AGENTS.md, with its line number ----
    # the proximity assertion on [DANGLING needs the citation's line, and a
    # char offset is not a line start, so count newlines rather than index a map
    a_lines = agents.split('\n')

    cites = []
    for m2 in CITE_RE.finditer(agents):
        title = flat(m2.group(1))
        if not title:
            continue
        cites.append((title, m2.start()))

    seen = set()
    for title, off in cites:
        if title in seen:
            continue
        seen.add(title)
        # ORDER IS SPECIFICITY, and it is load-bearing: a banner entry ALSO
        # appears verbatim in HISTORY.md (the banner is inside it), so testing
        # OK-TEXT first would file every destroyed entry as "present as text"
        # and the banner's own provenance would never be consulted.
        if title in headings:
            print('OK-HEADING %s' % title)
        elif title in banner:
            print('OK-BANNER %s' % title)
        elif title in hist_text:
            print('OK-TEXT %s' % title)
        else:
            # proximity: the annotation must sit next to the pointer it excuses
            ln_no = agents.count('\n', 0, off) + 1
            near = '\n'.join(a_lines[max(0, ln_no - 4): ln_no + 3])
            if re.search(r'\[DANGLING', near):
                print('DANGLING %s' % title)
            else:
                print('UNRESOLVED %s' % title)

    cited = set(t for t, _ in cites)
    for t in sorted(banner - cited):
        print('BANNER-UNCITED %s' % t)

    print('CITES %d' % len(cites))
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('usage: _pointer_scan.py AGENTS.md HISTORY.md\n')
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
