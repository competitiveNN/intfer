#!/usr/bin/env python3
"""Derive the citation facts of the engine survey corpus (the pipe-format reports).

Kept in its own file rather than a heredoc inside the checker, for the same
measured reason `_dispatch_scan.py` gives: backslashes in regexes get mangled by
the bash heredoc, and the failure mode is the parser dying silently so the
harvest returns NOTHING -- a corpus with no citations read as a corpus with no
defects, which is the vacuous-green shape this repo keeps refusing.

PRINTS FACTS, one per line, and exits 0 always, so the caller decides the verdict:

  POP <n>
      how many reports the population derivation found. ZERO is printable on
      purpose: the caller refuses it, so "looked at nothing" can never read as a
      PASS.

  CLONE <report> <line> <clone-abs-path>
      a clone declaration (`clone at <path>` or `Clone: <path>`), and the line it
      is on. Resolution is BLOCK-SCOPED: a citation is resolved against the
      most recent declaration at or above it, so a file that surveys several
      upstreams (engines_sources.txt) resolves each record against its own clone.

  CITE <report> <line> <token> <loclist> <status> <resolved-or-'-'>
  MENTION <report> <line> <token> <status> <resolved-or-count>
      every harvested token, with its classification.

  DEFECT <class> <report> <line> <token-or-detail>
      one line per token whose class is a DEFECT, so a caller (or a mutation
      matrix) can refuse for a NAMED class rather than for "something failed".

  MENTION_AMBIG <report> <line> <token> <path>...
      a bare mention whose basename exists at more than one path, with those
      paths. This is the "is it disambiguated?" fact a reader needs; the count
      alone is not the fact.

  SAMEBASE <report> <line> <token> <resolved> <competing-path>...
      the same-basename audit: a resolved citation whose BASENAME also exists at
      some other path. This is informational, not a defect -- it is the reason a
      path is required at all.

  SUMMARY <class> <count>
  CONTROL_BAD <description>

THE RULE (derived from the judge directive, and stated in each report header):

  a citation is location-bearing AND path-qualified.

  * location-bearing means the token carries `:<numbers>`; a bare `README.md:6`
    is a citation, and a `README.md` with no location is a MENTION -- a separate
    class, so the two never collapse.
  * path-qualified means the token carries a path: either a `/` (a directory
    component), an `intfer/`/`upstream/` tree qualifier, or the ` (ROOT)`
    disambiguator the engines_sources rule already uses. A bare BASENAME
    (`engine.cpp:337`) is REFUSED: it resolves only by guessing, and it is how a
    correct citation once got declared FABRICATED against the wrong file.

  Resolution is TREE-AWARE and headed by the report itself:

      intfer/<p>       the repo root
      upstream/<p>     the clone named in the report (block-scoped)
      <p>              repo root first, then the clone; existing in BOTH is an
                       ambiguity, not a resolution
      /abs/path        resolves exactly at that path, with no fallback, and is
                       refused as a HOUSE-STYLE violation: an absolute path is
                       machine-specific and cannot be read by another checkout

  Sentinels mark a mention that CANNOT resolve, so a real path can never be
  laundered into one:

      external/<p>     outside both trees on purpose (a system header)
      absent/<p>       the author asserts the file does not exist

  Either way the path must NOT resolve: a sentinel on a file that exists is
  claiming an absence that is false, which is the defect in a new dress.
"""

import os
import re
import sys

# ---- the population: reports that declare the pipe-record contract -----------
# Deliberately NOT the style statement: deriving the population from the very
# marker the checker also requires would let a report drop out of the corpus by
# deleting its own header, and a smaller corpus reads as a quieter one.
MARKER = re.compile(r'^#\s*Format:\s*NAME\s*\|', re.M)

# ---- the header statement of the house path style ----------------------------
# The statement must NAME the four qualifiers, not merely exist: a marker
# carrying no rule is a decorative marker, which is the shape this repo keeps
# refusing.
STYLE_RE = re.compile(r'^#\s*Path style:', re.M)
STYLE_TAG = '# Path style:'
STYLE_QUALIFIERS = ('intfer/', 'upstream/', 'external/', 'absent/')

# ---- the tree names, the sentinels, and the file extensions -------------------
PREFIX = ('intfer', 'upstream', 'external', 'absent')
# The extension set is EXPLICIT, not "anything after a dot".  Three measured
# reasons:
#   * `0.05/0.95` is a ratio, not a file;
#   * `e.g` is prose, not a file;
#   * `docs/models/qwen3.8-flash-next/EXPERIMENTS.md` must be ONE token -- an
#     extension rule that accepts a digit extension splits it at `.8` and then
#     cites `flash-next/EXPERIMENTS.md`, which resolves in neither tree.
# The alternation is ordered LONGEST-FIRST and that is load-bearing: with `cu`
# before `cuh` the real citation `paged_prefill.cuh:322-360` was cut at `.cu`
# and harvested as a NOT_FOUND mention of a file that does not exist.  That was
# measured on the corpus, not imagined.
EXT = (r'(?:hxx|hpp|hh|cpp|cxx|cc|cuh|cu|h|go|md|txt|sh|py|rs|yaml|yml|json'
       r'|toml|cmake|inl|ipp|mm)')
RUN = r'[A-Za-z0-9_+./-]+'
# A path token: an optional tree qualifier or sentinel, then an absolute or a
# relative path, ending in a KNOWN extension.  The run is greedy and the
# extension is backtracked, so the whole path is taken as one token.
#
# The NEGATIVE LOOKAHEAD after the extension is load-bearing, and it was added
# after a measured failure: the extension backtracking could match a PREFIX of
# a longer path.  A clone directory named `/tmp/tmp.hXXXXXX/clone` was
# harvested as the token `/tmp/tmp.h` -- a path that does not exist -- because
# the run gave back characters until `.h` matched the one-letter extension and
# the rest of the name was left behind.  The extension must be the END of the
# path token, so the character after it may not be a path character.
PATH_CORE = (r'(?:(?:' + '|'.join(PREFIX) + r')/)?'
             r'(?:/[A-Za-z0-9_+./-]+|[A-Za-z0-9_+./-]+)\.' + EXT
             + r'(?![A-Za-z0-9_+./-])')
TOKEN_RE = re.compile(
    r'(?<![\w./-])'
    r'(?P<tok>' + PATH_CORE + r')'
    r'(?P<root> \(ROOT\))?'
    r'(?P<loc>(?::\d+(?:[-,]\d+)*)?)'
)
URL_RE = re.compile(r'\b(?:https?|ftp|git)://\S+')
# A token followed by a colon and NOT a number: the location list is malformed.
MALFORMED_RE = re.compile(
    r'(?<![\w./-])(?P<tok>' + PATH_CORE +
    r')(?P<root> \(ROOT\))?:(?P<bad>[^\s0-9])'
)
# A clone declaration: `clone at <abs path>` (the survey-header spelling) or
# `Clone: <abs path>` (the per-survey-block spelling). The path terminates at
# whitespace, a comma, a paren, a COLON or a semicolon -- the colon matters:
# "re-derived against the upstream clone at /path/Strata: nothing was copied"
# is PROSE mentioning a clone, and without the colon in the terminator class it
# was captured as a declaration with a trailing colon, which then broke every
# citation after that line by resolving against a path that does not exist.
CLONE_RE = re.compile(r'(?:clone at|Clone:)\s*(/[^\s,):;]+)')

SKIP_DIRS = {'.git', 'build', 'node_modules', '__pycache__', '.venv', 'venv'}

# Classes that are DEFECTS (the caller must fail on any of these).
FAIL_CLASSES = (
    'BARE_CITATION', 'NO_CLONE', 'NOT_FOUND', 'AMBIGUOUS_TREE',
    'OUT_OF_RANGE', 'MALFORMED_LOCATION', 'UNRESOLVED_MENTION',
    'SENTINEL_MISUSE', 'ABSOLUTE_CITATION', 'ABSOLUTE_MENTION',
    'STYLE_STATEMENT_MISSING',
)
# Classes that are REPORTED but are not defects.  A bare MENTION is not a defect
# on its own -- it is CLASSED by whether its basename is ambiguous across the
# trees, which is the fact a reader needs in order to know whether the prose
# disambiguated it.
INFO_CLASSES = ('MENTION_OK', 'ABSENT_OK', 'EXTERNAL_OK', 'UNRESOLVED_MENTION',
                'MENTION_AMBIGUOUS', 'MENTION_UNIQUE', 'MENTION_UNKNOWN')


def nlines(path):
    """The number of lines in a file, counting a final unterminated line."""
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return None
    n = data.count(b'\n')
    if data and not data.endswith(b'\n'):
        n += 1
    return n


def index_basenames(roots):
    """basename -> [paths], over the given roots, skipping build trees."""
    idx = {}
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for dp, dn, fn in os.walk(r):
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
            for f in fn:
                idx.setdefault(f, []).append(os.path.join(dp, f))
    return idx


def split_token(tok, root):
    """A token -> (kind, rel) where kind is one of
    'intfer' | 'upstream' | 'external' | 'absent' | 'abs' | 'plain'."""
    for p in PREFIX:
        if tok.startswith(p + '/'):
            return p, tok[len(p) + 1:]
    if tok.startswith('/'):
        return 'abs', tok
    if root:
        # `engines_sources.txt (ROOT):187` -- the disambiguator the engines_sources
        # rule already writes, resolved against the repo root.
        return 'intfer', tok
    return 'plain', tok


class Corpus(object):
    """The report corpus, its clone declarations, and the harvested tokens."""

    def __init__(self, corpus_dir, repo):
        self.corpus_dir = corpus_dir
        self.repo = repo
        self.files = []
        self.clones = []          # (report, line, path)
        self.cites = []           # dicts
        self.mentions = []
        self.facts = []
        self.style_missing = []   # (report, line, detail)
        self.samebase = []
        self.mention_ambig = []   # (report, line, token, [paths])
        self.report_clones = {}   # report -> [clone paths it declares]
        self._idx_cache = {}

        # ---- the population, DERIVED ----------------------------------------
        if os.path.isdir(corpus_dir):
            for name in sorted(os.listdir(corpus_dir)):
                if not name.endswith('.txt'):
                    continue
                p = os.path.join(corpus_dir, name)
                if not os.path.isfile(p):
                    continue
                try:
                    txt = open(p, encoding='utf-8', errors='replace').read()
                except OSError:
                    continue
                if not (MARKER.search(txt) or 'clone at ' in txt or 'Clone: ' in txt):
                    continue
                self.files.append(p)
        for p in self.files:
            self._scan(p)

    # ------------------------------------------------------------------ scan
    def _scan(self, path):
        rel = os.path.basename(path)
        txt = open(path, encoding='utf-8', errors='replace').read()
        lines = txt.split('\n')

        # block-scoped clone declarations
        decl = []   # (line, clone)
        for i, ln in enumerate(lines, 1):
            m = CLONE_RE.search(ln)
            if m:
                decl.append((i, m.group(1)))
                self.clones.append((rel, i, m.group(1)))
                self.report_clones.setdefault(rel, [])
                if m.group(1) not in self.report_clones[rel]:
                    self.report_clones[rel].append(m.group(1))
                self.facts.append(('CLONE', rel, i, m.group(1)))

        # the header style statement
        self._style_check(rel, lines)

        decls = {i: c for (i, c) in decl}
        decl_lines = [i for (i, _c) in decl]

        # the URL spans, so a token inside a URL is never harvested
        url_spans = [m.span() for m in URL_RE.finditer(txt)]
        in_url = lambda off: any(a <= off < b for (a, b) in url_spans)

        line_of = lambda off: txt.count('\n', 0, off) + 1
        clone_at = lambda ln: (lambda ls: max([l for l in ls if l <= ln], default=None))(decl_lines)

        for m in TOKEN_RE.finditer(txt):
            if in_url(m.start()):
                continue
            tok = m.group('tok')
            loc = m.group('loc') or ''
            ln = line_of(m.start())
            cline = clone_at(ln)
            clone = decls.get(cline) if cline else None
            if loc:
                self.cites.append(dict(report=rel, line=ln, tok=tok, root=m.group('root') or '',
                                       loc=loc, clone=clone, has_clone_decl=bool(clone),
                                       resolved=None, status=None))
            else:
                self.mentions.append(dict(report=rel, line=ln, tok=tok,
                                          clone=clone, has_clone_decl=bool(clone),
                                          resolved=None, status=None))

        # a malformed location list: a colon followed by something that is not a
        # number. URLs are already excluded, and `external:`/`absent:` are
        # sentinel prefixes, not tokens, so they cannot self-match.
        for m in MALFORMED_RE.finditer(txt):
            if in_url(m.start()):
                continue
            tok = m.group('tok')
            if tok.split('/', 1)[0] in ('external', 'absent'):
                # ONLY the SENTINEL prefixes are skipped here.  The earlier form
                # skipped every tree qualifier, and that included `upstream/` and
                # `intfer/` -- so a malformed location on a QUALIFIED citation
                # was never detected at all.  Measured: the mutation matrix's
                # EC-11 state planted `upstream/src/engine/engine.cpp:abc` and
                # the checker passed it.  The sentinels are skipped because a
                # colon after `external/foo.hpp` is prose ("...: the system
                # header"), not a malformed location.
                continue
            ln = line_of(m.start())
            self.cites.append(dict(report=rel, line=ln, tok=tok, root='', loc=':' + m.group('bad'),
                                   clone=None, has_clone_decl=False, resolved=None,
                                   status='MALFORMED_LOCATION'))

    def _style_check(self, rel, lines):
        """Does the report header carry the house path style, and does it NAME
        the four qualifiers?  A missing statement is a FAIL class because the
        rule is machine-checkable, and a prose rule the checker cannot see is a
        prose rule nobody enforces."""
        hit = None
        for i, ln in enumerate(lines):
            if STYLE_RE.match(ln):
                hit = i
                break
        if hit is None:
            self.style_missing.append((rel, 1, 'no "%s" statement in the header' % STYLE_TAG))
            return
        # the statement may wrap; take the whole contiguous comment block
        i = hit
        while i < len(lines) and lines[i].lstrip().startswith('#'):
            if i > hit + 40:
                break
            i += 1
        stmt = '\n'.join(lines[hit:i])
        missing = [q for q in STYLE_QUALIFIERS if q not in stmt]
        if missing:
            self.style_missing.append(
                (rel, hit + 1, 'the style statement does not name %s' % ', '.join(missing)))

    # -------------------------------------------------------------- resolution
    def _candidates(self, kind, rel, has_root, clone):
        """The candidate files for a token, in resolution order."""
        if kind == 'intfer':
            return [os.path.join(self.repo, rel)]
        if kind == 'upstream':
            return [os.path.join(clone, rel)] if clone else []
        if kind == 'abs':
            return [rel]
        out = [os.path.join(self.repo, rel)]
        if clone:
            out.append(os.path.join(clone, rel))
        return out

    def resolve(self, tok, root, clone, has_clone_decl, is_citation):
        """-> (status, resolved_path)."""
        kind, rel = split_token(tok, root)
        if kind == 'plain' and '/' not in tok and is_citation:
            # A location-bearing citation with no path.  It resolves only by
            # GUESSING which file of that name is meant, which is how a correct
            # citation once got declared FABRICATED against the wrong file.
            # REFUSED INSIDE THE RESOLVER so the positive control exercises the
            # same code path the corpus does.
            return 'BARE_CITATION', None
        if kind in ('external', 'absent'):
            cands = self._candidates('plain', rel, False, clone)
            hit = [c for c in cands if os.path.isfile(c)]
            if hit:
                return 'SENTINEL_MISUSE', None
            return ('EXTERNAL_OK' if kind == 'external' else 'ABSENT_OK'), None
        if kind == 'abs':
            # An absolute path still RESOLVES (so the reader sees which file it
            # named), but it is refused: it is machine-specific and cannot be
            # read by another checkout.
            hit = [c for c in self._candidates('abs', rel, bool(root), clone)
                   if os.path.isfile(c)]
            if not hit:
                return 'NOT_FOUND', None
            return ('ABSOLUTE_CITATION' if is_citation else 'ABSOLUTE_MENTION'), hit[0]
        if kind == 'upstream' and not has_clone_decl:
            return 'NO_CLONE', None
        cands = self._candidates(kind, rel, bool(root), clone)
        hit = [c for c in cands if os.path.isfile(c)]
        if not hit:
            return 'NOT_FOUND', None
        if len(hit) > 1:
            return 'AMBIGUOUS_TREE', None
        return 'OK', hit[0]

    def classify_citation(self, tok, root, clone, has_clone_decl, loc):
        """A citation -> (status, resolved_or_None), resolution AND range check.
        One entry point for both the corpus and the positive control, so a
        control that passes while the corpus fails is not constructible."""
        st, res = self.resolve(tok, root, clone, has_clone_decl, True)
        if st not in ('OK', 'ABSOLUTE_CITATION'):
            return st, None
        if not loc or not re.fullmatch(r':\d+(?:[-,]\d+)*', loc):
            return 'MALFORMED_LOCATION', res
        n = nlines(res)
        bad = [x for x in map(int, re.findall(r'\d+', loc))
               if x < 1 or (n is not None and x > n)]
        if bad:
            return 'OUT_OF_RANGE', res
        return st, res

    def trees(self):
        """The repo plus every clone the corpus names."""
        out = [self.repo]
        seen = set()
        for _r, _l, c in self.clones:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def report_trees(self, report):
        """The repo plus the clones THIS report declares.

        The scope is per report on purpose.  A global index makes a bare
        mention point at a file in a DIFFERENT upstream's tree -- measured: the
        token `llama.cpp` in the jitllm report was "uniquely" resolved to
        pulsarlm's `pulsarlm-core/src/model/llama.cpp`, which is a different
        project entirely and reads as a citation the report never made.  The
        question a reader asks of a bare mention is "is this basename ambiguous
        between the trees THIS report talks about?", so that is the scope.
        """
        out = [self.repo]
        for c in self.report_clones.get(report, []):
            if c not in out:
                out.append(c)
        return out

    def index_for(self, report):
        """basename -> [paths] over the report's own trees (cached)."""
        roots = tuple(self.report_trees(report))
        if roots not in self._idx_cache:
            self._idx_cache[roots] = index_basenames(roots)
        return self._idx_cache[roots]

    def audit(self):
        """Classify every harvested token, then run the same-basename audit."""
        for c in self.cites:
            if c['status']:
                continue                      # already classified (malformed)
            st, res = self.classify_citation(c['tok'], c['root'], c['clone'],
                                              c['has_clone_decl'], c['loc'])
            c['status'] = st
            c['resolved'] = res

        for m in self.mentions:
            tok = m['tok']
            if not ('/' in tok or tok.split('/', 1)[0] in PREFIX):
                # A bare mention is not a defect, but it IS a class: the reader
                # needs to know whether the basename is ambiguous across the
                # trees this report talks about (so the prose MUST disambiguate
                # it) or unique.
                comp = self.index_for(m['report']).get(os.path.basename(tok), [])
                if len(comp) > 1:
                    m['status'] = 'MENTION_AMBIGUOUS'
                    m['resolved'] = str(len(comp))
                    self.mention_ambig.append((m['report'], m['line'], tok, comp))
                elif len(comp) == 1:
                    m['status'] = 'MENTION_UNIQUE'
                    m['resolved'] = comp[0]
                else:
                    m['status'] = 'MENTION_UNKNOWN'
                    m['resolved'] = '-'
                continue
            st, res = self.resolve(tok, '', m['clone'], m['has_clone_decl'], False)
            if st == 'NOT_FOUND':
                # A path-qualified mention names a file that is not there.  The
                # judge directive says path-qualified mentions are
                # EXISTENCE-CHECKED, so this is a defect -- but it keeps its own
                # class so a citation failure and a mention failure are not
                # reported as the same thing.
                st = 'UNRESOLVED_MENTION'
            m['status'] = st
            m['resolved'] = res if st in ('OK', 'ABSOLUTE_MENTION') else None

        # ---- the same-basename audit -----------------------------------------
        for c in self.cites:
            if not c['resolved'] or c['status'] not in ('OK', 'ABSOLUTE_CITATION'):
                continue
            base = os.path.basename(c['resolved'])
            comp = [p for p in self.index_for(c['report']).get(base, [])
                    if p != c['resolved']]
            for p in comp:
                self.samebase.append((c['report'], c['line'], c['tok'], c['resolved'], p))

    # ------------------------------------------------------------------ output
    def emit(self):
        for f in self.facts:
            print(' '.join(str(x) for x in f))
        for (rel, line, detail) in self.style_missing:
            print('DEFECT STYLE_STATEMENT_MISSING %s %d %s' % (rel, line, detail))
        for c in self.cites:
            print('CITE %s %d %s %s %s %s' % (c['report'], c['line'], c['tok'],
                                              c['loc'] or '-', c['status'] or '?',
                                              c['resolved'] or '-'))
            if c['status'] in FAIL_CLASSES:
                print('DEFECT %s %s %d %s' % (c['status'], c['report'], c['line'], c['tok']))
        for m in self.mentions:
            print('MENTION %s %d %s %s %s' % (m['report'], m['line'], m['tok'],
                                              m['status'] or '?', m['resolved'] or '-'))
            if m['status'] in FAIL_CLASSES:
                print('DEFECT %s %s %d %s' % (m['status'], m['report'], m['line'], m['tok']))
        for s in self.mention_ambig:
            print('MENTION_AMBIG %s %d %s %s' % (s[0], s[1], s[2], ' '.join(s[3])))
        for s in self.samebase:
            print('SAMEBASE %s %d %s %s %s' % (s[0], s[1], s[2], s[3], s[4]))
        counts = {}
        for c in self.cites:
            counts[c['status'] or '?'] = counts.get(c['status'] or '?', 0) + 1
        for m in self.mentions:
            counts[m['status'] or '?'] = counts.get(m['status'] or '?', 0) + 1
        counts['SAMEBASE'] = len(self.samebase)
        counts['POP'] = len(self.files)
        counts['STYLE_MISSING'] = len(self.style_missing)
        for k in sorted(counts):
            print('SUMMARY %s %d' % (k, counts[k]))


# ---- the POSITIVE CONTROL, run through the SAME resolver ----------------------
def run_control(corpus):
    """Classify synthetic strings so a blind resolver cannot pass.

    Every assertion goes through `corpus.resolve` or `corpus.classify_citation`,
    i.e. through the code the corpus itself uses.  A control that reimplemented
    the rule would certify the control, not the resolver.
    """
    ctl = corpus
    hist = os.path.join(ctl.repo, 'HISTORY.md')
    if not os.path.isfile(hist):
        return ['the control needs HISTORY.md at the repo root and it is not there']
    n_hist = nlines(hist) or 0
    bad = []
    if split_token('HISTORY.md', '')[0] != 'plain':
        bad.append('the control cannot run: a bare basename is no longer classed plain')
    # (1) a bare basename citation must be REFUSED, not resolved
    st, res = ctl.resolve('HISTORY.md', '', None, False, True)
    if st != 'BARE_CITATION':
        bad.append('a bare BASENAME citation was not refused as BARE_CITATION (got %s)' % st)
    # (2) a qualified citation must resolve to the repo root
    st, res = ctl.resolve('intfer/HISTORY.md', '', None, False, True)
    if st != 'OK' or res != hist:
        bad.append('intfer/HISTORY.md did not resolve to the repo root (got %s)' % st)
    # (3) a location PAST EOF must be OUT_OF_RANGE, through the same entry point
    if n_hist:
        st, res = ctl.classify_citation('intfer/HISTORY.md', '', None, False,
                                        ':%d' % (n_hist + 4242))
        if st != 'OUT_OF_RANGE':
            bad.append('a location past EOF did not read OUT_OF_RANGE (got %s)' % st)
        st, res = ctl.classify_citation('intfer/HISTORY.md', '', None, False,
                                        ':%d' % n_hist)
        if st != 'OK':
            bad.append('a location AT EOF did not read OK (got %s)' % st)
    # (4) a token that resolves nowhere must be NOT_FOUND
    st, res = ctl.resolve('intfer/no/such/report.txt', '', None, False, True)
    if st != 'NOT_FOUND':
        bad.append('a token that resolves nowhere did not read NOT_FOUND (got %s)' % st)
    # (5) a sentinel on a file that EXISTS must be refused -- and the path must
    #     be written WITHOUT a tree qualifier, because split_token strips the
    #     first pipe segment as the sentinel prefix
    st, res = ctl.resolve('absent/HISTORY.md', '', None, False, False)
    if st != 'SENTINEL_MISUSE':
        bad.append('the absent sentinel on a file that EXISTS was not refused (got %s)' % st)
    # (6) an absent sentinel on a file that really is absent must pass
    st, res = ctl.resolve('absent/intfer/no/such/report.txt', '', None, False, False)
    if st != 'ABSENT_OK':
        bad.append('the absent sentinel on a genuinely missing file did not pass (got %s)' % st)
    # (7) the external sentinel must pass
    st, res = ctl.resolve('external/sycl/ext/oneapi/graph.hpp', '', None, False, False)
    if st != 'EXTERNAL_OK':
        bad.append('the external sentinel did not pass (got %s)' % st)
    # (8) an absolute path must be refused as a house-style violation for BOTH a
    #     citation and a mention
    st, res = ctl.resolve(hist, '', None, False, True)
    if st != 'ABSOLUTE_CITATION':
        bad.append('an absolute path citation was not refused as ABSOLUTE_CITATION (got %s)' % st)
    st, res = ctl.resolve(hist, '', None, False, False)
    if st != 'ABSOLUTE_MENTION':
        bad.append('an absolute path mention was not refused as ABSOLUTE_MENTION (got %s)' % st)
    return bad


def main(argv):
    corpus_dir = 'docs/engines'
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    mode = 'audit'
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('--corpus', '-c'):
            i += 1
            corpus_dir = argv[i]
        elif a in ('--repo', '-r'):
            i += 1
            repo = os.path.abspath(argv[i])
        elif a in ('--mode', '-m'):
            i += 1
            mode = argv[i]
        elif a in ('--facts', '-f'):
            mode = 'facts'
        i += 1
    if not os.path.isdir(corpus_dir):
        # ZERO is PRINTS ARE facts. Printing nothing is what makes a missing
        # corpus read as a quiet one, so the shape is printed and the caller
        # refuses it.
        print('SUMMARY POP 0')
        return 0
    corpus = Corpus(corpus_dir, repo)
    corpus.audit()
    if mode != 'facts':
        for b in run_control(corpus):
            print('CONTROL_BAD %s' % b)
    corpus.emit()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
