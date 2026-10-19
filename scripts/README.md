# scripts/ — where the tooling lives, and where to put new things

136 executable files (127 `.sh`, 8 `.py`, 1 `.c`), two data directories, and one
retirement directory. **The flat root is load-bearing, not untidy** — see
[The root stays flat](#the-root-stays-flat) before you move anything into a
subdirectory.

If you are about to add a checker or a mutation matrix, **read
`docs/audit-log-tooling.md` first** (project instruction). It carries the ~40
tooling failure modes — vacuous gates, inverted predicates, stale artifacts,
a checker certifying itself — that everything here exists to prevent. Most of
the comments in these files are records of a defect that was *measured*, not
predicted; they are the cheapest way to avoid repeating it.

---

## Orientation: what kind of thing is this?

The directory is organised by **filename prefix**, and that prefix is a *contract*,
not decoration. A file's prefix tells you which derived population it belongs to,
and each of those populations is read off the filesystem by a checker.

| prefix | what it is | count | the population is defined by |
|---|---|---|---|
| `test_` | correctness gates — run the model or a probe, print `TEST: PASS/FAIL` | 18 | `ls scripts/test_*.sh` |
| `check_` | checkers — grade the tree without running it (text, source, hashes) | 34 | `ls scripts/check_*.sh` |
| `mutate_` | mutation matrices — plant a defect on a temp copy and require the pair to REFUSE | 44 | `ls scripts/mutate_*.sh` |
| `_` | shared helpers — *sourced* by the families above, never run directly | 12 | nothing expands them; each is named by its callers |
| *(none)* | orchestrators, entry points, operator harnesses, model build drivers | 28 | hand-written lists (`STEPS`, `dispatch()`) |

Two directories share the root: `archive/` (retired scripts, see
`archive/README.md`) and `fixtures/` (synthetic device fixtures used by the
device-precondition checkers).

### Reading order

1. `gate_inventory.sh` — what the 18-gate set currently resolves to, and whether
   the doc tallies agree. Run this before trusting any count in a report.
2. `run_all_checks.sh` — the single entry point that runs the checker band, the
   gate band and the matrix band, and writes the certificate.
3. `run_ten_gates.sh` — the correctness gates only.
4. `docs/audit-log-tooling.md` — why the tooling is shaped this way.

---

## The root stays flat

**Five separate mechanisms pin `scripts/<name>` to the root. Each one fails the
build if a file moves, and none of them is a stylistic preference.**

1. **Population globs.** `test_*.sh`, `check_*.sh` and `mutate_*.sh` are the
   *definitions* of their families, not a naming style:
   `check_checker_matrix_coverage.sh:31-32`,
   `check_gate_exit_verdict.sh:36-37`, `run_ten_gates.sh`'s `GATE_GLOB`,
   `check_pending_operator.sh:38`, `mutate_e2_ctx.sh:3`,
   `check_scripts_live.sh:35`. A subdirectory removes a file from its family
   *without deleting it* — which is exactly what `archive/` is for.

2. **`check_scripts_live.sh` enumerates at `-maxdepth 1`** (`:83`). Its own
   comment (`:80-81`) says the depth limit is load-bearing. Move a file down and
   it leaves the population of the only checker that certifies a script is
   *reachable* — silently. That is the vacuous-green direction: the check keeps
   printing PASS while covering less.

3. **`check_durable_claims.sh` `p_target_exists` / `p_pair_hook`** (`:43`) require
   every `REGEN(...)` marker and every sha pair to name a script that **exists in
   `scripts/`**. ~150 live `REGEN(bash scripts/<name>.sh)` markers in `AGENTS.md`
   and `HISTORY.md` would all break.

4. **`HISTORY.md` carries ~108 `scripts/...` citations, and rewriting a true
   historical record to point at today's path is the laundering failure
   `AGENTS.md` explicitly refuses.** Moving files would either rewrite history or
   leave citations that cannot be re-derived.

5. **`run_all_checks.sh` `dispatch()` + `STEPS`** are two hand-written lists
   reconciled by `check_dispatch_steps_consistency.sh`. A move needs a path map
   in the dispatcher *and* the same change in its 1281-line mutation matrix
   (`mutate_run_all_checks.sh`), which anchors on the literal case labels.

> **So: do not create `scripts/gates/`, `scripts/checkers/`, `scripts/matrices/`
> or `scripts/helpers/`.** The organisation you want is already expressed by the
> prefixes, and this file is the map. If a genuine grouping need arises, fix the
> five mechanisms above in the same commit as the move — partial moves are how
> vacuous gates ship.

---

## The `_`-prefixed helpers

Twelve files, `sourced` by the families and never executed directly:
`_gate_build.sh _gate_enum.sh _gate_loadlog.sh _gate_patch.sh _gate_pop.sh
_gate_precond.sh _gate_pred.sh _dpas_pred.sh` plus `_act_cache_scan.py _bench_summary.py
_dispatch_scan.py _nullglob_scan.py`.

They are reached through a **composed path**, not a literal:
`SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` then
`. "$SCRIPT_DIR/_gate_pred.sh"`. Several checkers *generate* a stub gate in a
heredoc and rewrite the placeholder with `sed -i "s|X|$SCRIPT_DIR|g"`
(`check_device_precond.sh:269`) — so the helpers' directory is assumed by
generated code, not only by hand-written lines. Keep them flat and keep the
`_` prefix.

`scripts/_gate_enum.sh`'s `enum_require_nonempty` and `_gate_pred.sh`'s
`pred_*` are the two guards every checker reaches for; reuse them rather than
hand-rolling a population count.

---

## Entry points worth knowing

| script | use it for |
|---|---|
| `run_all_checks.sh` | everything: checker band → gate band → matrix band → certificate |
| `run_ten_gates.sh` | the correctness gates alone |
| `gate_inventory.sh` | the authoritative gate tally and doc/tally reconciliation |
| `knob_sweep.sh` | operator-only 27B `--tp 2` product sweep over the launch-geometry knobs |
| `model_byte_roofline.sh` | achieved-vs-ceiling calculator over a recorded kernel figure |
| `gpu_recover.sh` | box-state remedy ladder — `diag` → `rebind` → `verify` (see below) |
| `with_oneapi.sh <cmd>` | run anything that needs `setvars.sh`; a bare shell enumerates **no** GPU |
| `g.sh` | fixed-string grep for ad-hoc terminal use (grep's rc is a trap — see its header) |
| `guard.sh` | dirty-tree guard used by the matrices |
| `build.sh` / `build_quant_real.sh` | CMake build / real-weight quantize build |
| `run.sh` / `run_bin.sh` / `serve_27b.sh` | convenience wrappers for the CLI |

`INTFER_QUICK=1` on `run_all_checks.sh` runs the checker band only. Never run the
GPU- or model-backed steps from an agent session unless the operator scheduled it.

---

## Box state: never report it without `setvars.sh`

`with_oneapi.sh` wraps a command with the oneAPI environment. A session that has
not sourced `/opt/intel/oneapi/setvars.sh` reports **zero** GPUs, and that
reading is wrong — `SYCL: sycl-ls not available` is a statement about `PATH`.
This is pinned by `scripts/check_env_not_box_state.sh`.

If the second B580 stops enumerating, the remedy ladder is read-only first:
`scripts/gpu_recover.sh diag` → `sudo scripts/gpu_recover.sh rebind` →
`verify` (`0` = 2 devices, `1` = measured 1, `2` = unmeasurable). Do not lead
with the BAR/small-BAR theory. The one-rung-deeper kit is retired in
`scripts/archive/` and `archive/README.md` says so.

---

## Where a NEW script goes

| you want to add | it lives at | and you must |
|---|---|---|
| a correctness gate that runs the model or a probe | `scripts/test_<name>.sh` | be picked up automatically by `run_ten_gates.sh`'s `GATE_GLOB`; prefer modifying an existing gate over adding one |
| a checker that grades the tree without running it | `scripts/check_<name>.sh` | add a `dispatch()` arm **and** a `STEPS` entry in `run_all_checks.sh` (arm 0b and 0c refuse otherwise), and a matching `scripts/mutate_<name>.sh` |
| a mutation matrix | `scripts/mutate_<name>.sh` | ditto; every checker must have a dispatched matrix |
| a helper sourced by gates/checkers | `scripts/_<name>.sh` | sourced via `$SCRIPT_DIR/`, never executed |
| a model build / quantize driver | flat root for now (e.g. `build_quant_real.sh`) | a directory for these is the obvious first candidate *if* the five pinning mechanisms are ever unwired |
| an operator A/B or benchmark | flat root (`bench_*.sh`) | the wall-time queue is **closed**; decide on `nk`/`nw`/`nr` and the host buckets, never on a tok/s |
| a diagnostic that only ever ran once | **do not add it** | record the finding in `docs/audit-log-tooling.md`; if you must keep code, put it in `scripts/archive/` with the subject that ended |

**Before you add anything:** `check_scripts_live.sh` requires every script to be
*reachable* — either named verbatim in a live source, or expanded by a runner's
glob. A script that nothing runs and no entry point names is the defect that
checker now refuses. `check_checker_matrix_coverage.sh` requires every checker to
have a dispatched matrix. `check_hw_constants.sh` requires every hardware figure
in the docs it scans to be carried. Derive the population, never hand-keep a list.

---

## One hazard specific to this file

`check_scripts_live.sh` treats **any live source that names a file** as a call
site, and it deliberately excludes only `build/ .git/ docs/ reports/ node_modules/`
and `scripts/archive/` (`:129-131`). Its header says a comment naming a retired
script is the rule's one known *miss*, and that it names no real script itself
because "a checker that cites the files it judges is a checker that can mark them
reachable by saying so."

`scripts/README.md` is **not** in that exclusion list, so anything this file names
becomes a reference that could keep an unreachable script alive. This file
therefore names only entry points that are *already* reachable through
`AGENTS.md` or the gate globs, and describes the rest by pattern and prefix. When
you edit it, keep it that way — and if you find yourself needing to enumerate,
add this path to the checker's prose exclusion in the same commit, consistent
with the checker's own rule that a doc naming a file is not evidence that
anything calls it.
