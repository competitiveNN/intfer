# Optimization batch reports

One file per batch of optimization experiments handed to the user for
benchmarking. The user benchmarks **from** these reports, so a batch delivered
without one is an unfinished batch.

Naming: `reports/<YYYY-MM-DD>-<batch-slug>.md`.

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
