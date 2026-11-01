# intfer — port of ninfer to 2× Intel Arc Pro B580

Live gate tally:

    gate tally: 18 PASS, 0 PARTIAL, 0 SKIPPED CITATION(`scripts/gate_inventory.sh` derives 18 from `scripts/gates/test_*.sh`)

Re-derive box/gate state with `scripts/gate_inventory.sh` (never read it off this file).
All historical records, design rationale, perf directives, and tooling catalog are in `HISTORY.md` and `docs/audit-log-tooling.md`, `docs/*.md`, `PLAN.md`.

Standing rule: commit (or stash) uncommitted work before running a destructive probe; never `reset --hard` over uncommitted work. Do not commit or push unless the operator explicitly asks.

## Historical record pointers

`HISTORY.md` was truncated once (509 lines destroyed). Its banner names the eleven
dated entries whose detail is gone; they are cited here so no pointer dangles
unrecorded and `scripts/checks/check_history_pointers.sh` can resolve every one.
This list is the POINTER set, not a summary — re-derive the detail from source.

- HISTORY.md ("2026-10-15: INTFER_LA_OUT_TP2 — first la_out seam at 27B --tp 2")
- HISTORY.md ("2026-10-15: INTFER_SAMPLE_TOPK_DIRECT — first FORK_B580 port")
- HISTORY.md ("2026-10-15: Sweep mutation matrix — check_gate_exit_verdict coverage")
- HISTORY.md ("2026-10-15: Three checker defects found and fixed")
- HISTORY.md ("2026-10-16: 0.8B removed from code defaults/paths")
- HISTORY.md ("2026-10-16: Architectural improvements beyond fusion — reminder #9 evaluation")
- HISTORY.md ("2026-10-16: beellama.cpp (Anbeeld) — KVarN KV + fit-tail survey, DECLINED")
- HISTORY.md ("2026-10-16: DFlash2 on NPU seam completed")
- HISTORY.md ("2026-10-16: Flag gate coverage closed — all opt-in flags armed")
- HISTORY.md ("2026-10-16: INTFER_LA_CONV_PREP_WG — LA conv-prep launch-geometry knob")
- HISTORY.md ("2026-10-16: INTFER_LMHEAD_DEVARGMAX_TP2 — device argmax at TP=2")
- HISTORY.md ("2026-10-17: S1 uncached-host-USM doorbell probe — SHIPPED (informed-from-scratch from Strata)")
- HISTORY.md ("2026-10-11: INTFER_LMHEAD_FP16 reachability") — ARMED, REACHABILITY ONLY.
  [DANGLING 2026-10-18: HISTORY.md was truncated and this section's BODY no longer exists — the banner near the top of HISTORY.md is the record; re-derive the detail from source, never invent it.]

Engine-survey corpus: `docs/engines/*.txt`, citation rule enforced by
`scripts/checks/check_engine_citations.sh` and `check_engines_sources_paths.sh`
(path-qualified `intfer/…`, `upstream/…`, `external/…`, `absent/…`).
Tooling catalog: `docs/audit-log-tooling.md`; backlog: `PLAN.md`.
