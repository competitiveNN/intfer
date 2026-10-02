# Optimization batch reports

One file per batch of optimization experiments handed to the user for
benchmarking. The user benchmarks **from** these reports, so a batch delivered
without one is an unfinished batch.

Naming: `reports/<YYYY-MM-DD>-<batch-slug>.md`.

## `reports/` holds ONLY batches that have NOT been benchmarked yet

Once a batch has been benchmarked and a verdict is recorded, **move it to
`reports/archive/`** (`git mv`, so history follows). `reports/` is a work queue:
the user's pending-test list is read off it, and a decided batch sitting there
makes the queue lie about what is left to do.

Two consequences, both deliberate:

- **`gate_inventory.sh` globs `reports/*.md` non-recursively** to reconcile the
  doc tallies. Archived reports therefore drop out of that scan automatically,
  which is correct: a decided batch must not keep driving the live tally.
- **`check_doc_counts.sh` / `mutate_doc_counts.sh` point at one specific report
  by path** (the E1/E2/E3 verdicts, which moved to `archive/`). Any script that
  cites a report path must be updated in the same commit that moves the file —
  a dangling citation is a green checker reading a file that is not there.

**Archiving, not deleting.** The repo's standing rule is that a true record must
not be erased: rewriting history to match today's state destroys the only
evidence that the mistake or the dead end happened. `archive/` keeps it.

Every experiment in a report must state:

1. flag + kill switch + default state
2. what it changes and the mechanism (why it should help)
3. the exact benchmark command, plus the quiet-box rule (`load1 < 2`)
4. the load-**insensitive** numbers the agent measured (kernel/submit/
   waitblock/copy/usm_write/readback ms, `nk`/`nw`/`nr` counters) — these are
   device-side counters, **not** wall time
5. expected direction and the win/neutral/loss decision rule
6. which correctness gates already passed, and which stream-identity
   invariant holds for the A/B arms

Never fill in a tok/s or wall-time number the agent did not measure; leave a
blank `RESULT:` line for the user to fill from a quiet-box run. Keep reports
alive: re-scope, merge or kill an experiment by **editing the report**, so the
pending-test list never goes stale.
