# scripts/ — where the tooling lives, and where to put new things

Two address spaces share this directory: a **flat root** of entry points and a
**prefixed hierarchy** under `gates/`, `checks/`, `matrices/`, `lib/`, `model/`,
`l0/`, `build/`, `run/`, `bench/`, `tools/` plus the retirement directory
`archive/` and the `fixtures/` data dir.

```
scripts/
  build.sh                 flat entry points (9 .sh/.py/.c) + this README
  gate_inventory.sh
  gpu_recover.sh
  g.sh
  guard.sh
  knob_sweep.sh
  run_all_checks.sh
  run_ten_gates.sh
  with_oneapi.sh
  README.md

  gates/          test_*   correctness gates (18 .sh + test_quantized_model.py)
  checks/         check_*  tree graders (38 .sh)
  matrices/       mutate_* defect matrices (48 .sh)
  lib/            _*       sourced helpers (13 .sh/.py), never executed directly
  model/            *.py/*.sh tokenization / oracle / roofline calculator (4)
  l0/               *.c/*.sh Level-Zero enumeration (2)
  build/            *.sh   build / quantize drivers (4)
  run/              *.sh   CLI convenience wrappers (4)
  bench/            *.sh   operator bench/serve wrappers (3)
  tools/            *.sh   xmx_profile.sh (1)
  archive/        retired scripts (18), see archive/README.md
  fixtures/       synthetic device fixtures (data, not scripts)
```

If you are about to add a checker or a mutation matrix, **read
`docs/audit-log-tooling.md` first** (project instruction). It carries the ~40
tooling failure modes — vacuous gates, inverted predicates, stale artifacts,
a checker certifying itself — that everything here exists to prevent. Most of
the comments in these files are records of a defect that was *measured*, not
predicted; they are the cheapest way to avoid repeating it.

---

## The prefix is a contract

A file's prefix is the *definition* of its population, not decoration. Each
population is read off the filesystem by a checker, so a misplaced file drops
out of its family *without being deleted* — the vacuous-green direction.

| prefix | lives at | what it is | population is defined by |
|---|---|---|---|
| `test_` | `scripts/gates/test_*.sh` | correctness gates — run the model or a probe, print `TEST: PASS/FAIL` | `ls scripts/gates/test_*.sh` (`run_ten_gates.sh` `GATE_GLOB`) |
| `check_` | `scripts/checks/check_*.sh` | checkers — grade the tree without running it (text, source, hashes) | `ls scripts/checks/check_*.sh` |
| `mutate_` | `scripts/matrices/mutate_*.sh` | mutation matrices — plant a defect on a temp copy, require REFUSE | `ls scripts/matrices/mutate_*.sh` |
| `_` | `scripts/lib/_*` | shared helpers — sourced by the families above, never run directly | named by each caller's `$SCRIPT_DIR`-composed path |

The four families are the ones `check_scripts_live.sh` certifies as
**reachable** (it enumerates `scripts/` at `-maxdepth 2`, pruning
`archive/`). `archive/` is explicitly excluded: anything there is retired by
definition, see `archive/README.md`.

---

## How paths resolve

A script in a subdirectory computes the project root as

```
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
```

(one more `..` than a flat-root script, which uses `scripts/` itself).

Helpers are reached through a **composed path**, never a literal:

* from a subdirectory script: `. "$SCRIPT_DIR/../lib/_gate_pred.sh"`
* from a flat-root script: `. "$SCRIPT_DIR/lib/_gate_pred.sh"`

The stub-generator in `check_device_precond.sh` rewrites its placeholder with
`sed -i "s|X|$SCRIPT_DIR|g"` and sources `X/../lib/_gate_precond.sh`, so the
helpers' directory is assumed by generated code, not only hand-written lines.
`scripts/lib/_gate_enum.sh`'s `enum_require_nonempty` and
`scripts/lib/_gate_pred.sh`'s `pred_*` are the two guards every checker reaches
for — reuse them rather than hand-rolling a population count.

---

## Entry points worth knowing

| script | use it for |
|---|---|
| `run_all_checks.sh` | everything: checker band → gate band → matrix band → certificate |
| `run_ten_gates.sh` | the correctness gates alone |
| `gate_inventory.sh` | authoritative gate tally + doc/tally reconciliation |
| `knob_sweep.sh` | operator-only 27B `--tp 2` product sweep over launch-geometry knobs |
| `model_byte_roofline.sh` | achieved-vs-ceiling calculator over a recorded kernel figure |
| `gpu_recover.sh` | box-state remedy ladder — `diag` → `rebind` → `verify` |
| `with_oneapi.sh <cmd>` | run anything that needs `setvars.sh` (a bare shell enumerates no GPU) |
| `g.sh` | fixed-string grep for ad-hoc terminal use |
| `guard.sh` | dirty-tree guard used by the matrices |
| `build.sh` / `build_quant_real.sh` | CMake build / real-weight quantize build |
| `run.sh` / `run_bin.sh` / `serve_27b.sh` | convenience wrappers for the CLI |
| `xmx_profile.sh` (tools/) | join xmxmon capture with internal buckets |

`INTFER_QUICK=1` on `run_all_checks.sh` runs the checker band only. Never run the
GPU- or model-backed steps from an agent session unless the operator scheduled it.

---

## Where a NEW script goes

| you want to add | it lives at | and you must |
|---|---|---|
| a correctness gate | `scripts/gates/test_<name>.sh` | picked up automatically by `run_ten_gates.sh`'s `GATE_GLOB` |
| a checker | `scripts/checks/check_<name>.sh` | add a `dispatch()` arm **and** a `STEPS` entry in `run_all_checks.sh` (arm 0b/0c refuse otherwise), plus a matching `scripts/matrices/mutate_<name>.sh` |
| a mutation matrix | `scripts/matrices/mutate_<name>.sh` | ditto; every checker must have a dispatched matrix |
| a helper | `scripts/lib/_<name>.sh` | sourced via `$SCRIPT_DIR/../lib/` from subdirs, `$SCRIPT_DIR/lib/` from root |
| a build/quantize driver | flat root (`build_*.sh`) | |
| a CLI wrapper / operator harness | flat root (`run_*.sh` / `bench_*.sh`) | |
| a diagnostic that ran once | **do not add it** | record in `docs/audit-log-tooling.md`; keep code in `scripts/archive/` if required |

**Before you add anything:** `check_scripts_live.sh` requires every script to be
*reachable* — named verbatim in a live source, or expanded by a runner's glob.
A script nothing runs and no entry point names is the defect this checker now
refuses. `check_checker_matrix_coverage.sh` requires every checker to have a
dispatched matrix. Derive the population, never hand-keep a list.

---

## Box state: never report it without `setvars.sh`

`with_oneapi.sh` wraps a command with the oneAPI environment. A session that has
not sourced `/opt/intel/oneapi/setvars.sh` reports **zero** GPUs, and that
reading is wrong — `SYCL: sycl-ls not available` is a statement about `PATH`.
This is pinned by `scripts/checks/check_env_not_box_state.sh`.

If the second B580 stops enumerating, the remedy ladder is read-only first:
`scripts/gpu_recover.sh diag` → `sudo scripts/gpu_recover.sh rebind` →
`verify` (`0` = 2 devices, `1` = measured 1, `2` = unmeasurable). Do not lead
with the BAR/small-BAR theory. The deeper kit is retired in
`scripts/archive/` and `archive/README.md` says so.

---

## One hazard specific to this file

`check_scripts_live.sh` treats **any live source that names a file** as a call
site, and it deliberately excludes only `build/ .git/ docs/ reports/
node_modules/` and `scripts/archive/`. Its header says a comment naming a retired
script is the rule's one known *miss*, and that it names no real script itself
because "a checker that cites the files it judges is a checker that can mark them
reachable by saying so."

`scripts/README.md` is **not** in that exclusion list, so anything this file
names becomes a reference that could keep an unreachable script alive. This file
therefore names only entry points that are already reachable through `AGENTS.md`
or the prefix globs, and describes the rest by pattern and prefix. When you edit
it, keep it that way — and if you find yourself needing to enumerate, add this
path to the checker's prose exclusion in the same commit, consistent with the
checker's own rule that a doc naming a file is not evidence that anything calls it.
