#!/usr/bin/env python3
"""Summarise one batch of bench_reports_batch.sh records. See that harness for the
protocol; this file exists so the summary is TESTABLE without a GPU.

*** WHY IT IS ITS OWN FILE. *** The rule this repo keeps re-learning (AUDITs
#3/#12/#19: a rule fixed at one site and re-derived at another) applies to test
seams too: an assumption the harness cannot redirect is one the harness cannot
test. The summary was a heredoc inside the harness, so the only way to drive it
was to run a 27B benchmark -- which is why the failed-arm defect below survived
from the E2 batch until the run itself died. It is now a plain file with argv,
`scripts/check_bench_load_gate.sh` section (7) drives it against fixtures in both
directions, and `scripts/mutate_bench_load_gate.sh` state B-J breaks the guard.

*** EXIT CODE IS PART OF THE CONTRACT. *** 0 = summary printed, every record
carried data. 3 = at least one record produced NO measurement, so the table's
dashes are ABSENCE OF DATA and the batch is not a result. The harness does not
rely on this alone (it counts failures in the shell too, and exits 2), but a
standalone invocation must not report success over a dead batch.

Buckets are grouped by whether they may be believed: nk/nw/nr are exact counters,
kernel is device-side event time, and submit/waitblock/copy/readback/usm_write are
HOST wall clock that tracks the box.
"""
import re, sys, statistics
log, arms, ref = sys.argv[1], sys.argv[2].split(), sys.argv[3]
recs = []
for line in open(log, errors="replace"):
    if not line.startswith("ARM="):
        continue
    d = dict(kv.split("=", 1) for kv in line.split())
    # BUCKETS carries the whole bucket line, which CONTAINS SPACES. Whitespace
    # splitting truncated it at the first space, so BUCKETS became "@decode]" and
    # every nk/nw/nr/kernel column in the summary printed "-" for every batch --
    # i.e. the device-side columns, the entire point of counters-only mode, were
    # structurally incapable of populating. Quoted at the writer, extracted by
    # regex here.
    m = re.search(r'BUCKETS="([^"]*)"', line)
    if m:
        d["BUCKETS"] = m.group(1)
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

failed = {}
for r in recs:
    # A record with NO bucket data is a failed run, whatever the log says. Keying
    # this on the absence of data rather than on the FAILED= tag matters twice
    # over: it also diagnoses logs written by the PREVIOUS version of this harness
    # (which recorded `BUCKETS=""` as an ordinary row), so a batch that was
    # already lost cannot be summarised into a clean table of dashes.
    if r.get("FAILED") or not (r.get("BUCKETS") or "").strip():
        sig = r.get("FAILED") or "no_bucket_data_in_record"
        failed.setdefault(r.get("ARM"), []).append("r%s: %s" % (r.get("RND"), sig))

P = {"kernel": r"\bkernel=([\d.]+)ms",
     "submit": r"\bsubmit=([\d.]+)ms",
     "waitblock": r"\bwaitblock=([\d.]+)ms",
     "copy": r"\bcopy=([\d.]+)ms",
     "readback": r"\breadback=([\d.]+)ms",
     "usm_write": r"\busm_write=([\d.]+)ms",
     "nk": r"\bnk=(\d+)", "nw": r"\bnw=(\d+)", "nr": r"\bnr=(\d+)"}


def series(arm, key):
    """{round -> value}. Keyed by round identity so a missing round cannot
    shift the pairing (see the RND note above). Per-round, not median: the spread
    is the quantity that decides whether a delta is a result at all."""
    out = {}
    for r in recs:
        if r.get("ARM") != arm:
            continue
        m = re.search(P[key], r.get("BUCKETS", "") or "")
        if m:
            out[r["RND"]] = float(m.group(1))
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


for a in arms:
    rs = [r for r in recs if r.get("ARM") == a]
    if not rs:
        print("%-9s (no data)" % a)
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
    print("          host   : %s   (NOT load-insensitive)" %
          " ".join("%s=%s" % (b, num(med(series(a, b)))) for b in
                   ("submit", "waitblock", "copy", "readback", "usm_write")))

# ---- the delta table: does the effect clear the run-to-run spread?
rk = series(ref, "kernel")
rounds = max([len(series(a, "kernel")) for a in arms] or [0])
print()
if rounds < 2:
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
        print("   %-9s !! %d round(s) produced no measurement -- its delta above is"
              " over the SURVIVING rounds only, and the batch is NOT a result"
              % (a, len(failed[a])))
    rpa = dict(zip([r["RND"] for r in recs if r.get("ARM") == a
                    and re.search(P["kernel"], r.get("BUCKETS", "") or "")],
                   series(a, "kernel")))
    rpr = dict(zip([r["RND"] for r in recs if r.get("ARM") == ref
                    and re.search(P["kernel"], r.get("BUCKETS", "") or "")],
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
    if rounds < 2:
        v = "single shot -- not a result"
    elif abs(md) > noise:
        agree = all([(d > 0) == (md > 0) for d in ds])
        v = ("CLEARS the spread: report %+.1f ms (%+.2f%%); every round agrees in sign: %s"
             % (md, 100.0 * md / statistics.median([rpr[k] for k in common]),
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
