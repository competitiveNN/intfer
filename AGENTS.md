# intfer — port of ninfer to 2× Intel Arc Pro B580

Live gate tally:

    gate tally: 18 PASS, 0 PARTIAL, 0 SKIPPED CITATION(`scripts/gate_inventory.sh` derives 18 from `scripts/gates/test_*.sh`)

Re-derive box/gate state with `scripts/gate_inventory.sh` (never read it off this file).
All historical records, design rationale, perf directives, and tooling catalog are in `HISTORY.md` and `docs/audit-log-tooling.md`, `docs/*.md`, `PLAN.md`.

Standing rule: commit (or stash) uncommitted work before running a destructive probe; never `reset --hard` over uncommitted work. Do not commit or push unless the operator explicitly asks.
