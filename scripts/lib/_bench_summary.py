#!/usr/bin/env python3
"""Summarise one batch of bench_reports_batch.sh records. See that harness for the
protocol; this file exists so the summary is TESTABLE without a GPU.

*** WHY IT IS ITS OWN FILE. *** The rule this repo keeps re-learning (AUDITs
#3/#12/#19: a rule fixed at one site and re-derived at another) applies to test
seams too: an assumption the harness cannot redirect is one it cannot test.
The summary was a heredoc inside the harness, so the only way to drive it
was to run a 27B benchmark -- which is why the failed-arm defect below survived
from the E2 batch until the run itself died. It is now a plain file with argv,
`scripts/checks/check_bench_load_gate.sh` section (7) drives it against fixtures in both
directions, and `scripts/matrices/mutate_bench_load_gate.sh` states B-J..B-N break it.

*** EXIT CODE IS PART OF THE CONTRACT. *** 0 = summary printed, every record
carried data. 3 = at least one record produced NO measurement, so the table's
dashes are ABSENCE OF DATA and the batch is not a result. The harness does not
rely on this alone -- it counts failures in the shell too and exits 2 -- but a
standalone invocation must not report success over a dead batch, and the harness
MUST propagate a non-zero rc from here (AUDIT #28: it used to discard it).

Buckets are grouped by whether they may be believed: nk/nw/nr are exact counters,
kernel is device-side event time, and submit/waitblock/copy/readback/usm_write are
HOST wall clock that tracks the box.
"""
import re, sys, statistics

log, arms, ref = sys.argv[1], sys.argv[2].split(), sys.argv[3]

# *** THE RECORD IS PARSED BY FIELD NAME, NOT BY SPLITTING THE WHOLE LINE. ***
# It used to be `dict(kv.split("=", 1) for kv in line.split())`, which raises
# ValueError on the FIRST token that does not contain '=' -- and the bucket line is
# embedded in the record, so any '='-less token in it (a quote in the text, a
# positional field the engine ever adds) crashed the summariser with a traceback,
# printed NO summary, and left the harness exiting 0 (AUDIT #28, MEASURED). One
# batch, no table, exit 0. So: keys before BUCKETS are parsed as tokens, and
# BUCKETS is taken as the REMAINDER OF THE LINE. A token with no '=' is not a
# crash, it is a malformed record, and a malformed record is a FAILED record.
REQUIRED = ("kernel", "nk", "nw", "nr")
P = {"kernel": r"\bkernel=([\d.]+)ms",
     "submit": r"\bsubmit=([\d.]+)ms",
     "waitblock": r"\bwaitblock=([\d.]+)ms",
     "copy": r"\bcopy=([\d.]+)ms",
     "readback": r"\breadback=([\d.]+)ms",
     "usm_write": r"\busm_write=([\d.]+)ms",
     "nk": r"\bnk=(\d+)", "nw": r"\bnw=(\d+)", "nr": r"\bnr=(\d+)"}


def field(line, name):
    """Value of `name=` from a bucket line, or None."""
    m = re.search(P[name], line or "")
    return m.group(1) if m else None


recs = []
for line in open(log, errors="replace"):
    line = line.rstrip("\r\n")
    if not line.startswith("ARM="):
        continue
    d = {}
    # The bucket value is taken by CHARACTER SPAN, not by splitting the line:
    #   * `BUCKETS=""` must yield the empty string. A regex that takes the text
    #     up to the LAST quote captures the closing quote as data instead, every
    #     field then looks "incomplete", and a genuinely dead record is misnamed
    #     (measured while writing this).
    #   * keys on BOTH sides of the span are parsed. The writer emits FAILED=
    #     AFTER the span, and a parse that only looked at the head silently lost
    #     it -- so a dead record looked like an untagged one (measured).
    #   * a truncated write with no closing quote still yields its text.
    i = line.find('BUCKETS="')
    if i >= 0:
        vs = i + len('BUCKETS="')
        # The value ends at the NEXT quote -- not at end-of-line, because the
        # writer emits FAILED= AFTER the span, and not at the LAST quote, because
        # that would swallow FAILED= into the bucket text. A quote inside the
        # bucket line therefore truncates it, which shows up as a NAMED
        # incomplete record rather than as a crash or a silent dash.
        q = line.find('"', vs)
        if q >= 0:
            d["BUCKETS"] = line[vs:q].strip()
            rest = line[:i] + line[q + 1:]
        else:                      # truncated write: keep what arrived
            d["BUCKETS"] = line[vs:].strip()
            rest = line[:i]
    else:
        rest = line
    for kv in rest.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k] = v
        else:
            # Not a crash. Recorded so the refusal below can NAME the defect
            # instead of aborting on it.
            d.setdefault("MALFORMED", kv)
    recs.append(d)

# *** PAIR ROUNDS BY IDENTITY, NOT BY POSITION. *** The log is round-major, so
# the i-th data row of arm A and the i-th of arm B were the same round -- WHILE
# BOTH ARMS HAD EVERY ROUND. An arm that produced no measurement in round 1 would
# shift every later row down one, and the comparison would silently become
# "round 2 vs round 1". With the FAILED rows now present that is reachable, and a
# mis-paired delta is worse than no delta. Counters-only mode has no ROUND= key,
# so the per-arm ordinal is the round identity there.
per_arm = {}
for r in recs:
    per_arm.setdefault(r.get("ARM"), []).append(r)
for arm, rs in per_arm.items():
    for i, r in enumerate(rs):
        r["RND"] = r.get("ROUND") or str(i + 1)

# *** WHAT COUNTS AS A MEASUREMENT. *** Keyed on the DATA, not on the log's own
# tag, for two reasons: it also diagnoses records written by the PREVIOUS
# harness (which recorded `BUCKETS=""` as an ordinary row), and it cannot be
# satisfied by a tag that a bug failed to write. *** AND IT IS NOT JUST
# "non-empty": a record whose buckets were only PARTIALLY recovered -- a truncated
# write, a quote in the text -- used to print '-' in the exact-counter columns and
# be counted as a good round (AUDIT #28, MEASURED). Absent-of-data is now named
# per field, so the dash table is unreachable for an arm that has data.
failed = {}
for r in recs:
    sig = r.get("FAILED") or ""
    if not sig and r.get("MALFORMED"):
        sig = "malformed_record_token:%s" % r["MALFORMED"]
    if not sig:
        b = (r.get("BUCKETS") or "").strip()
        if not b:
            sig = "no_bucket_data_in_record"
        else:
            miss = [k for k in REQUIRED if field(b, k) is None]
            if miss:
                sig = "incomplete_bucket_line_missing:" + "+".join(miss)
    # Per-RECORD, not per-arm: a half-dead arm must keep contributing its
    # SURVIVING rounds, which is what the round-identity pairing exists for.
    r["OK"] = not sig
    if sig:
        failed.setdefault(r.get("ARM"), []).append("r%s: %s" % (r.get("RND"), sig))


def series(arm, key):
    """{round -> value}. Keyed by round identity so a missing round cannot
    shift the pairing (see the RND note above). Per-round, not median: the spread
    is the quantity that decides whether a delta is a result at all."""
    out = {}
    for r in recs:
        if r.get("ARM") != arm or not r.get("OK"):
            continue
        v = field(r.get("BUCKETS"), key)
        if v is not None:
            out[r["RND"]] = float(v)
    return [out[k] for k in sorted(out)]


def med(v):
    return statistics.median(v) if v else None


def num(v):
    return ("%.1f" % v) if v is not None else "-"


def cnt(v):
    # nk/nw/nr are exact integer counters; printing them as "63488.0" implies a
    # measurement precision they do not have, which is the same overclaim as the
    # 6-decimal millisecond table this replaced.
    return ("%d" % v) if v is not None else "-"


def spr(v):
    return (max(v) - min(v)) if len(v) > 1 else 0.0


def host(v):
    """median (spread). AUDIT #26: a central value is never printed without the
    spread that bounds it -- the retracted '-53% submit saving' was computed from
    exactly these numbers, so a bare median here is the same footgun."""
    return "%s(%s)" % (num(med(v)), num(spr(v))) if v else "-"


for a in arms:
    rs = [r for r in recs if r.get("ARM") == a]
    if not rs:
        print("%-9s (no records at all -- the batch never ran this arm)" % a)
        continue
    tps = []
    for r in rs:
        try:
            tps.append(float(r["TPS"]))
        except (ValueError, KeyError):
            pass
    if tps:
        held = len(rs) - len(tps)
        # AUDIT #23: say how many rounds were withheld. A median over the
        # surviving rounds, printed without this, reads exactly like a median
        # over all of them -- and "never the best one" is this repo's rule for a
        # reason. A partial withholding must be visible in the table.
        extra = ""
        if held:
            extra = "   [%d/%d round(s) WITHHELD: box contended]" % (held, len(rs))
        print("%-9s tok/s %s  median=%6.2f%s" %
              (a, "  ".join("%6.2f" % t for t in tps), statistics.median(tps), extra))
    else:
        nfail = len(failed.get(a, []))
        if nfail:
            print("%-9s !! %d of %d round(s) produced NO MEASUREMENT -- %s"
                  % (a, nfail, len(rs), "; ".join(failed[a])))
        else:
            print("%-9s (wall-time withheld: box contended, %d/%d rounds)" % (a, len(rs), len(rs)))
    print("          exact  : nk=%s nw=%s nr=%s" %
          tuple(cnt(med(series(a, k))) for k in ("nk", "nw", "nr")))
    k = series(a, "kernel")
    if k:
        pct = (100.0 * spr(k) / med(k)) if med(k) else 0.0
        print("          kernel : %s ms   median=%s  spread=%s (%.1f%%)" %
              (" ".join("%.1f" % x for x in k), num(med(k)), num(spr(k)), pct))
    print("          host   : %s   median(spread), NOT load-insensitive" %
          " ".join("%s=%s" % (b, host(series(a, b))) for b in
                   ("submit", "waitblock", "copy", "readback", "usm_write")))

# ---- the delta table: does the effect clear the run-to-run spread?
rounds = max([len(series(a, "kernel")) for a in arms] or [0])
print()
if rounds == 0:
    print("== kernel delta vs %s : NO arm produced a measurement ==" % ref)
    print("   There is nothing to compare. Read the refusal below, not a verdict.")
elif rounds < 2:
    print("== kernel delta vs %s : SINGLE ROUND per arm, no spread estimate ==" % ref)
    print("   A one-shot delta cannot be separated from run-to-run noise. Report the")
    print("   value as a single shot, never as a measured cost or saving.")
else:
    print("== kernel delta vs %s, PER ROUND (median delta inside the spread is NOT a result) ==" % ref)
    print("   noise floor per row = max(spread of the two arms being compared), i.e.")
    print("   the same arm+flags re-measured. Spread is the ONLY available noise")
    print("   estimate, so it bounds what can be claimed, it does not prove noise.")
    print("   *** AND THE SPREAD IS A PROPERTY OF THIS RUN, NOT OF THE ARM. ***")
    print("   Measured 2026-10-02, same model/shape/rounds: BASE spread 1875.5 ms in")
    print("   the E1 run and 64.5 ms in the E3 run -- an order of magnitude apart.")
    print("   So 'CLEARS the spread' means 'clears the noise actually observed here'.")
    print("   Quote it with the run's spreads, never as the arm's noise.")
for a in arms:
    if a == ref:
        continue
    ak = series(a, "kernel")
    if failed.get(a):
        print("   %-9s !! %d round(s) produced no measurement -- so its delta below is"
              " computed over the SURVIVING rounds only, and the batch is NOT a result"
              % (a, len(failed[a])))
    rpa = dict(zip([r["RND"] for r in recs if r.get("ARM") == a and r.get("OK")], ak))
    rpr = dict(zip([r["RND"] for r in recs if r.get("ARM") == ref and r.get("OK")],
                   series(ref, "kernel")))
    # Paired on ROUND IDENTITY, not position -- a missing round must not shift
    # the comparison onto a different round.
    common = sorted(set(rpa) & set(rpr), key=lambda s: (len(s), s))
    if not common:
        print("   %-9s (no rounds in common with %s)" % (a, ref))
        continue
    ds = [rpa[k] - rpr[k] for k in common]
    md = statistics.median(ds)
    noise = max(spr([rpa[k] for k in common]), spr([rpr[k] for k in common]))
    # *** THE VERDICT IS GATED ON THE ROUNDS IN COMMON, NOT ON THE BATCH. ***
    # It used to be gated on the global round count -- the MAX over arms -- so an
    # arm holding ONE surviving round in common with a 3-round reference was
    # scored against a spread of 0.0 (spr of a 1-element list), and any non-zero
    # delta printed "CLEARS the spread ... every round agrees in sign: yes"
    # (MEASURED 2026-10-02: "d=+98.0 (rounds 2) ... CLEARS the spread: report
    # +98.0 ms (+0.60%); every round agrees in sign: yes"). That is precisely the
    # false claim AUDIT #26 retracted for the E1 PREQ summary, arriving through a
    # different door -- and with a vacuous sign check, since one round trivially
    # agrees with itself. A delta needs >=2 paired rounds to have a noise estimate
    # at all; below that the only honest verdict is that none exists.
    if len(common) < 2:
        v = ("NO SPREAD ESTIMATE: only %d round in common -- a single paired delta "
             "cannot clear an unmeasured noise floor. Report it as a single shot, "
             "never as a cost or a saving." % len(common))
    elif abs(md) > noise:
        agree = all([(d > 0) == (md > 0) for d in ds])
        v = ("CLEARS the spread (floor=%.1f ms): report %+.1f ms (%+.2f%%); every"
             " round agrees in sign: %s"
             % (noise, md, 100.0 * md / statistics.median([rpr[k] for k in common]),
                "yes" if agree else "NO -- a single"
                " outlier round is carrying it, so treat it as unresolved"))
    else:
        v = ("INSIDE the spread (|%.1f| <= %.1f): NOT a result. Signs across rounds: %s"
             % (abs(md), noise, "".join("+" if d > 0 else "-" for d in ds)))
    print("   %-9s d=%s (rounds %s)  median=%+.1f ms  -> %s" %
          (a, " ".join("%+.1f" % d for d in ds), ",".join(common), md, v))
if recs:
    print("\n(n=%d runs, load1 at each run is in %s)" % (len(recs), log))
if failed:
    print("")
    print("!!!! %d RECORD(S) PRODUCED NO MEASUREMENT. THIS BATCH IS NOT A RESULT. !!!!"
          % sum(len(v) for v in failed.values()))
    for arm in sorted(failed):
        for s in failed[arm]:
            print("!!!!   %s %s" % (arm, s))
    print("!!!! The dashes above are ABSENCE OF DATA, not a neutral result. A batch")
    print("!!!! whose arms died (device lost, OOM, timeout) produces a table of")
    print("!!!! dashes that looks like 'no effect' -- that is the failure mode this")
    print("!!!! check exists to refuse. See scripts/gpu_recover.sh diag / verify.")
    sys.exit(3)