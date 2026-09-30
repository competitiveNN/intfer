# Pending benchmark backlog — 27B on 2x Arc Pro B580

Date: 2026-09-30. Author: maki (agent). Nothing here has a wall-time number
from me; every `RESULT:` line is deliberately blank and is yours to fill.

Scope note (AGENTS.md): the optimization target is the **27B** model
(`Qwen3.8-27B-int2.intifer`, 19 GB, INT2/G16+G32 + INT4/G64 + INT8/G32).
The 0.8B numbers elsewhere in AGENTS.md are legacy reference only. On 27B,
**TP=1 OOMs** (lm_head INT8 tile alloc fails, ~19 GB on one ~12 GB B580), so
every 27B command below is `--tp 2`.

Environment gotchas that will otherwise look like engine bugs:

```bash
export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}
# otherwise SYCL cannot enumerate ANY GPU: "No devices matching filter 'Arc'"
# (verify with sycl-ls; xpu-smi still shows the GPUs — do not debug the engine)
```

Quiet-box rule: **load1 < 2 at the start of every round** (`awk '{print $1}'
/proc/loadavg`). Run the arms **interleaved**, at least 3 rounds, and report
every round — not the best one. Wall-time deltas on a loaded box have
produced several retracted claims in this project.

```bash
# wait for a quiet box before round 1 (and re-check before each round)
until awk '{exit !($1 < 2)}' /proc/loadavg; do sleep 30; done; echo "load1=$(awk '{print $1}' /proc/loadavg)"
```

## PRECONDITION — both B580s must enumerate, or none of this can run

Every command below is `--tp 2`, so **do not start a round until both GPUs are
visible to the compute runtime**. Check this first, every session:

```bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}
sycl-ls | grep -c '\[level_zero:gpu\]'     # must print 2
xpu-smi --list-gpus | grep -c 'Arc'        # must print 2
```

If that prints **1**, stop: this is the box state recorded in AGENTS.md
AUDIT #9/#11, not an engine regression. With one GPU the engine correctly
clamps `--tp 2` down to TP=1, 27B does not fit (~19 GB vs ~12 GB), and the run
dies during prefill. You will now also see a new, deliberate diagnostic that
did not exist before commit `bc851e5`:

```
Engine: WARNING: TP=2 requested but only 1 Arc GPU(s) discovered — continuing with TP=1. ...
```

That warning plus a later `rc=139` / `FATAL: signal 6` with
`cpu_page_fault_memory_manager.cpp` in the abort is the **known 1-GPU
signature**, not a new bug. Do not debug it and do not report it as a result —
wait for both GPUs. (The clamp itself is correct; only its silence was a defect,
and that is what the warning fixes.)

Do not verify GPU presence with `lspci` — it is not installed on this box, so
it answers "0" regardless of the hardware. Do not trust `ls -la /dev/dri`
either; use `find /dev/dri -maxdepth 1`. Both traps produced false readings
during AUDIT #10/#11.

---

## E1 — `INTFER_DPAS_PREQ=1` (device GA64 activation-quant pre-pass)

**Default state: OFF** (and `INTFER_DPAS48=1` is also OFF by default).
`INTFER_DPAS_PREQ=1` **alone is inert by design** — it is nested inside the
DPAS48 path, and `scripts/test_gpu_optin.sh` asserts exactly that. Never test
it on its own.

### What it changes
DPAS48 quantizes activations to INT8/GA64. Before this flag, the whole
GA64 quant ran on the host (measure: `copy` bucket, GA64 act-quant ~6 ms/token
at 27B scale), which is what made DPAS A net wall-time loss. The flag moves
only the **quantize + store half** into a standalone SYCL pre-pass kernel,
dep-chained into the DPAS launch so it adds **no host wait**. One pre-pass per
device, never one for both (each rank owns its own `sycl::context`).

Bit-exactness is by construction: the per-64-slice `(scale, inv)` pair is
still computed **on the host** (single shared site, `ga64_recip()`) and
shipped to the device verbatim. Do not "optimize" this by computing the
reciprocal on the device — the device `1.0f/s` lowers to `rcp.approx` and
measurably disagrees with the host on ~10% of elements (AGENTS.md AUDIT #2/#3).

### What I already measured (counters, not wall time)
0.8B TP=1 decode, n=24 CLI stream gate:

| counter | DPAS48 | DPAS48+PREQ | meaning |
|---|---|---|---|
| `nk` (kernel launches) | 1116 | 1488 | +372 = 4 tok x 93 GEMM calls — the pre-pass really fires |
| `nw` (host waits) | 744 | 744 | unchanged: no extra host round-trip |

### Expected direction
It removes part of the host `copy` bucket but **keeps** the O(M*K) host
max-abs reduction (the price of shipping the exact reciprocal bits), so
expect roughly **half** of the originally promised host act-quant saving, and
read the `copy` bucket as including that reduction. The device-side kernel
saving alone was already measured at -15% on 27B.

### The A/B baseline is DPAS48, not base
This matters more than the flag itself. DPAS48 **on its own** was measured to
lose: ~30% decode and ~11% prefill at M=361, and it loses on *every* bucket at
M=1 (kernel +48%, submit +73% at a *lower* launch count, copy 2x, readbacks 2x).
So a "win" for PREQ over base can still be a loss overall. Test three arms:

- `base` (both flags off)
- `DPAS48`
- `DPAS48 + PREQ`

### Stream-identity invariant (read before believing any tok/s)
The correct invariant is **DPAS48+PREQ == DPAS48**, bit-identical. `base` vs
`DPAS48` is *expected* to differ (DPAS48 quantizes activations, base does not)
and is not a bug. If DPAS48+PREQ and DPAS48 print different token ids, stop —
that is a real regression in the pre-pass.

### Gates already passed (my side, load-insensitive)
- `scripts/test_ga64_recip.sh` — 480,645 slices / 30.7M elements bit-identical; 192,000 tie slices verified to be 1-ulp-discriminating
- `scripts/test_cpu_parity.sh`, `scripts/test_http_batch.sh`
- `scripts/test_gpu_optin.sh` — TP=1 and TP=2, plus a wide ~97-token prefill arm (M=97) that covers the 2*M*nGA row stride

### Commands (27B, TP=2, decode)
```bash
BIN="${INTFER_BUILD_DIR:-build}/examples/intfer_cli/intfer_cli"
M=/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer
TOK="--tokenizer-py scripts/bpe_tokenize.py --detok-py scripts/bpe_detokenize.py"
export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}

for r in 1 2 3; do
  echo "== round $r load1=$(awk '{print $1}' /proc/loadavg) =="
  for arm in "BASE" "DPAS48" "PREQ"; do
    case $arm in
      BASE)  unset INTFER_DPAS48 INTFER_DPAS_PREQ ;;
      DPAS48) export INTFER_DPAS48=1; unset INTFER_DPAS_PREQ ;;
      PREQ)  export INTFER_DPAS48=1 INTFER_DPAS_PREQ=1 ;;
    esac
    $BIN --gpu --tp 2 -m $M -p "The capital of France is" -n 64 --no-stream $TOK 2>&1 \
      | grep -aE "tokens in|decode:"
  done
done
```
Prefill leg (one 361-token chunk = whole prompt in a single pass):
```bash
P361="$(python3 -c "print('The history of artificial intelligence and machine learning. '*40)")"
INTFER_PREFILL_CHUNK=1 INTFER_PREFILL_CHUNK_M=361 \
  $BIN --gpu --tp 2 -m $M -p "$P361" -n 5 --no-stream $TOK 2>&1 | grep -aE "tokens in"
```
(`INTFER_DPAS48`/`INTFER_DPAS_PREQ` still come from the loop's env.)

### Decision rule
- **WIN** — PREQ beats DPAS48 by >3% decode in 3+ rounds at load1 < 2, and
  DPAS48+PREQ beats base overall.
- **NEUTRAL** — within ±3% of DPAS48.
- **KILL** — PREQ does not close the DPAS48 deficit; both stay default-OFF.
  That is the expected outcome, and it is why nothing about PREQ is on by default.

RESULT (27B TP=2 decode, 3 rounds, load1 at each round):

RESULT (27B TP=2 prefill chunk361):

---

## E2 — `INTFER_INT8_KV=1` (+ `INTFER_KV_Q8=1`, `INTFER_KV_HAD=1`)

**Default state: OFF.** Each layer is a separate flag; test them as the
stack they were designed as.

### What it changes
Groupwise-INT8 KV cache, shrinking the KV bytes the engine must read on every
decode step. `INTFER_KV_Q8=1` adds the append-only per-token G64 Q8 mirror;
`INTFER_KV_HAD=1` selects half-away-from-zero rounding on the append (the
rounding the host oracle uses).

### Why it is worth measuring
Every perf lever tried so far attacks **launch/submit overhead**
(fusion, waves, L4/L5, DPAS). Decode itself is bandwidth-bound on weight +
KV reads, and KV is the part of that traffic the current INT2/G16 weights do
not already shrink. This is also the flag that buys headroom for long contexts
and for concurrency on a ~12 GB/card TP=2 split.

### Expected direction
- decode: improves, and the effect should **grow with context length** (test
  long ctx, not just a 5-token prompt — at short ctx the KV share is tiny and
  the result will be noise)
- prefill: neutral to mildly better (less write traffic, no math change)
- VRAM: strictly lower for the cache

### Gates already passed (my side, load-insensitive)
- `scripts/test_gpu_optin.sh` 10/10 — base / KV / Q8 / Q8+HAD / SSM_GPU, x TP=1 and TP=2, streams identical
- `scripts/test_cpu_parity.sh`, `scripts/test_http_batch.sh`, `scripts/test_serve_load.sh`

### Stream-identity invariant
**base == KV == Q8 == Q8+HAD**, byte-identical. Unlike E1, all four arms are
meant to be identical. Verify the ids first; only then believe a timing.

The E2 snippets reuse `$BIN`, `$M` and `$TOK` from E1's block — define them
first if you run E2 on its own:

```bash
BIN="${INTFER_BUILD_DIR:-build}/examples/intfer_cli/intfer_cli"
M=/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer
TOK="--tokenizer-py scripts/bpe_tokenize.py --detok-py scripts/bpe_detokenize.py"
```

```bash
for mode in "INTFER_INT8_KV=1" "INTFER_INT8_KV=1 INTFER_KV_Q8=1" \
            "INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1"; do
  echo "--- $mode ---"
  env $mode $BIN --gpu --tp 2 -m $M -p "The capital of France is" -n 8 \
    --no-stream --ids $TOK 2>&1 | grep -a "decode step"
done
```

### Commands (27B, TP=2, long context decode)
```bash
LONG="$(python3 -c "print('The history of artificial intelligence and machine learning. '*400)")"
for r in 1 2 3; do
  echo "== round $r load1=$(awk '{print $1}' /proc/loadavg) =="
  for arm in "BASE" "KV" "KVQ8"; do
    case $arm in
      BASE) unset INTFER_INT8_KV INTFER_KV_Q8 INTFER_KV_HAD ;;
      KV)   export INTFER_INT8_KV=1; unset INTFER_KV_Q8 INTFER_KV_HAD ;;
      KVQ8) export INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1 ;;
    esac
    $BIN --gpu --tp 2 -m $M -p "$LONG" -n 64 --no-stream $TOK 2>&1 \
      | grep -aE "tokens in|decode:"
  done
done
```
If a long prompt blows the KV cap, report that as a capacity finding rather
than dropping the arm — the cap behaviour is part of what you are testing.

### Decision rule
- **WIN** — KV (and/or KVQ8) beats base by >3% decode at long ctx, 3+ rounds,
  streams identical.
- **NEUTRAL at short ctx** is an expected, acceptable result: report it, do
  not retest on a loaded box hoping for more.
- **KILL** — no decode gain at long ctx, or any stream divergence.

RESULT (27B TP=2 long-ctx decode, 3 rounds, load1 at each round):

---

## E3 — `INTFER_DPAS48_FOLD=1`

Implemented and gated; awaiting quiet-box numbers. Full report (with the
measured device-side counters, the mutation proof, and the row-stride bug the
probe caught during development) is in
**`reports/2026-09-30-dpas48-fold.md`** — read that file for the command.
Run its pre-flight before spending a quiet window on it:
`./scripts/test_dpas_fold.sh` (builds its own target first, so it cannot
report PASS for a stale binary).

**Its non-vacuity proof is itself a runnable script, not prose.**
`./scripts/mutate_fold_matrix.sh` re-derives the S0-S4 matrix (each state applied
alone from a clean header, per-state logs, tree restored and re-verified), and
asserts the *character* of every state rather than dumping output. Run it if you
touch the fold layout. It refuses to start on a dirty tree, because it edits
tracked files. Current result: 5/5 as characterized, where
S1 (wrong default) is invisible to the device readback and S4 (wrong reader
index) is invisible to the stride contract — so neither gate alone suffices.

## E4 — PROPOSED (not implemented): cooperative 2D-tiled DPAS block

**Status: design only. There is no flag and no code yet — do not go looking for
`INTFER_DPAS48_TILED` or similar; it does not exist.** Pre-registered here so
the pending list stays honest about what the project still owes a
measurement.

**What is left.** E3 removed the *arithmetic* replication: the per-(row,op)
activation fold no longer runs once per 16-N tile. It did not touch the *load*
replication. In `quant_gemm_dpas48` every WI still does its own
`vb.copy_from(B + ((size_t)t * nOp + o) * op_bytes)` — one B operand load per
WI per op, so the same B op-tile is fetched `nT` times per row-group from L1/L2
rather than once per block.

**The shape being proposed.** One WI per 16-N tile is the problem; the fix is a
genuinely 2D work-item block (a strip of tiles per WI, or a cooperative
subgroup that loads a B op-tile once into registers and reuses it across
several tiles), so B traffic per output element drops from `nT` to ~1.

**Carry one lesson from AUDIT #12 into this kernel specifically.** Reordering
the work means rewriting the *index* expressions, and a stride/contract check
is structurally blind to indexing: mutation S4 drops the `+ nOp` that selects
the Sfull half of the fold row, and the stride contract still reports 80/80
agreement — only a device readback catches it (60/64 outputs). E3 now keeps
every read behind `fold_index_sfull`/`fold_index_s0` in
`include/intfer/gemm/quant_gemm.hpp` so the expression has one home, unit-gated
over 360 (m,stride,nOp,o) tuples. Do the same for any new E4 indexing: one
named helper, one gate, and do not rely on the stride contract to notice.

This is
the same class of change as `quant_gemm_dpas4_row` (`INTFER_DPAS_ROW`, already
built and measured ~460x SLOWER — so a naive "one WI per row" is already known
to be wrong; the geometry has to actually cooperate, not just re-partition).

**Why it is on the prefill-4000 list.** Prefill 4000 tok/s needs XMX doing real
work, and the EU pipes are the whole prefill story today. DPAS is the only
XMX path available, and it is gated on exactly this kernel being competitive at
wide N and M>1.

**Honest blocker, stated up front.** A 2D/cooperative DPAS kernel changes the
accumulation ORDER of the FP32 `acc_tile`, so unlike E3 it is **not** a
bit-exact hoist. It needs a tolerance-based gate, not `==`.

**That contract is now WRITTEN DOWN: `reports/2026-09-30-e4-dpas-2d-contract.md`.**
Key decision: gate **backward error** (`max |seq-blocked| / max |t_o|`), not
result-relative error, because the kernel's `2*c - 15*S` zero-point correction
makes the terms cancel and the result ~1000x smaller than the largest term — a
result-relative bound would be dominated by cancellation. Tolerances are
measured, not guessed, and enforced today by a section of
`examples/dpas_fold_probe` (gated via `scripts/test_dpas_fold.sh`):
`backward_err <= 4.0e-5` (measured worst 1.38e-5 at K=28672), stored-FP16
movement `<= 16` ULP (measured 8), changed outputs `<= 1%` (measured 0.32%).
Stream identity is **reported, not asserted** — asserting it would be tie-luck
(AUDIT #3b). Read the contract file before writing any E4 code; it also lists
the five conditions that would invalidate the tolerances.

RESULT (27B TP=2 prefill chunk361, vs DPAS48 and DPAS48+FOLD baselines):

---

## Do NOT spend a quiet window on these (already measured, verdict final)

Listed so nobody re-runs them; details and numbers in AGENTS.md.

| flag / idea | verdict |
|---|---|
| `INTFER_DPAS48=1` alone | default-OFF at all shapes (M=1/128/361): loses every bucket at M=1 |
| `INTFER_L4_FUSE=1` (scalar) | variance wash; certified infra, no tok/s claim either way |
| `INTFER_L5_FRONT=1` | dead end as built: +144 launches/token, waitblock 5x |
| `INTFER_SSM_GPU=1` | -9% decode |
| device act-quant fused **into** the DPAS kernel | dead end (O(K) replication), do not retry |
| SYCL command graphs / L0 immediate submit | same floor as plain UR submit |
| prefix cache (`INTFER_NO_PREFIX_CACHE=1` kill switch) | quiet-box A/B already done 2026-09-26, verdict in |
| OpenVINO | evaluated, not adopted |

### The queue is blocked on box state, not on missing experiments
E1/E2/E3 are all `--tp 2`. As of this writing the box enumerates **one** GPU to
the compute runtime, so **none of them can run** — see the PRECONDITION at the
top. That is a known box state (AGENTS.md AUDIT #9/#11), not an engine bug, and
not something to re-investigate: when both GPUs enumerate, the queue is ready.
Do not use a loaded or 1-GPU window and do not report a number from one.

**The 2-GPU-dependent gate arms now SKIP instead of FAILing** — and, since audit
#13, they skip **per arm** rather than per gate (`scripts/_gate_precond.sh`).
This matters for reading a gate run. On a 1-GPU box `test_device_argmax.sh` and
`test_gpu_optin.sh` still run and still assert every arm that works on one GPU,
skip only their `--tp 2` arms, and exit **2 = PARTIAL**; `test_tp2_27b.sh` has
no 1-GPU arm at all and exits **2 = SKIPPED**. None exit 0 (which would claim
coverage nobody got) and none exit 1 (indistinguishable from a real regression,
and it trains everyone to ignore red gates). So a full run on this box reads
**8 PASS, 2 PARTIAL, 1 SKIPPED**.

Two things audit #13 corrected, both worth knowing before you touch these
scripts:

- A whole-gate `gate_require_gpus 2` was tried on `test_device_argmax.sh` and
  was **wrong**. That gate is mostly 1-GPU; skipping all of it also dropped the
  serve fence-lock test, which guards a crash that no stream gate can see. Skip
  the arms, not the gate.
- `test_gpu_optin.sh` was **never guarded at all** and was reporting a **false
  PASS**: its assertions are stream-identity, so re-running "tp 2" as tp 1 on a
  1-GPU box satisfied every one of them. A false pass is worse than the false
  fail the precondition was introduced to remove.

### OUTSTANDING — blocked on the second GPU (do NOT report as done)

These are **unverified here**, not "good enough". Nothing in this repo may be
described as covered by them until they have actually run.

| item | what is unverified | how to close it |
|---|---|---|
| `test_device_argmax.sh` | only its 5 `--tp 2` arms (TP=2 base/flag identity, TP2 spec PARTS=0, TP2 spec streams). The TP=1 arms — including the serve fence lock — **do** run here | `sycl-ls` must print 2 `[level_zero:gpu]` lines, then `bash scripts/test_device_argmax.sh` (expects rc=0 instead of rc=2) |
| `test_gpu_optin.sh` | only its whole `tp 2` pass (~14 arms). The `tp 1` pass **does** run here | same condition, then `bash scripts/test_gpu_optin.sh` (expects rc=0 instead of rc=2) |
| `test_tp2_27b.sh` | everything: 27B TP=2 greedy decode vs oracle, and the ~19GB-vs-~12GB VRAM headroom claim | same condition, then `bash scripts/test_tp2_27b.sh` |
| E1 / E2 / E3 A/Bs | every tok/s number in those sections | needs both GPUs *and* a quiet window (load1 < 2) |
| E4 | unimplemented by design; no flag exists | pre-registered only, see above |

If a future session cannot get a second GPU, the correct statement is
"8 PASS, 2 PARTIAL, 1 SKIPPED, E1-E4 unmeasured" — **not** a full pass, and
never a tok/s figure from a 1-GPU or loaded window.

### Self-check for the fold matrix (run it after any refactor of that code)

`bash scripts/mutate_fold_matrix.sh --anchors-only` is a <1s, no-build,
no-mutation pre-flight that fails if any M1/M2/M4/M5/M6 mutation target has
moved, duplicated or vanished, or if a read site has stopped routing through
`fold_index_sfull`/`fold_index_s0`. The full run stamps
`build/fold_matrix/last_pass.stamp` with the HEAD sha it was proven against, so
"the matrix passed" and "the matrix passed on THIS code" stay distinguishable.


## Also not yet implemented (no flag to test — do not go looking)

- rewritten **2D-tiled / cooperative** DPAS kernel (B operands loaded once per
  block, not once per WI). Registered as **E4** above with its design and its
  numerics blocker; it is no longer lumped in here because it now has a written
  shape. The K-fold arithmetic half of that work SHIPPED as E3
  (`INTFER_DPAS48_FOLD`, gated).
- batched multi-launch L0 command lists amortizing one sync over many kernels
  (single immediate append+sync was measured at the same floor as UR submit)

---

### Recording results

Paste results back into this file's `RESULT:` lines (commit it) or just reply
with them; either way they get folded into AGENTS.md. If an experiment comes
out neutral or loses, it stays default-OFF and gets a "closed" row in the table
above — that is the expected outcome for most of this list, and reporting it is
what stops the next session from re-running it.
