# Batch: fused multi-GEMM decode launch + decode1 work-group knob

> ## ==> REVERTED. THIS BATCH BROKE 27B DECODE AND HAS BEEN ROLLED BACK. <==
>
> Both experiments were implemented, then **measured to break 27B `--tp 2`
> decode** (decode `kernel` 1284 ms → 231 114 ms, output `id=-1`), and the code,
> the gate arms, the mutation matrix and the `run_all_checks.sh` wiring have all
> been removed. `src/gemm/quant_gemm.cpp` and `src/engine/engine.cpp` are restored
> to the last known-good state.
>
> **The finding that matters is kept** — M=1 decode is EU-parallelism-bound, and
> a bit-exact multi-GEMM fusion is the right *direction*. What is wrong is the
> *shape*: sharing the M=1 INT2 per-output arithmetic through a device function
> breaks it, with identical arithmetic, and no flag was set.
>
> Full account, including the bisect and the generalizable lesson, is in
> `AGENTS.md` under "REVERTED 2026-10-04: FUSED MULTI-GEMM DECODE". Read that
> before re-attempting. **Do not start splitting K across work-items on a
> structural argument** — it touches the same kernel body.
>
> Every `RESULT:` line below stays blank, permanently. There is nothing to fill:
> the arms no longer exist.

so **decide on the load-insensitive counters**.

---

## The measurement that motivated this batch

27B `--tp 2`, decode M=1, counters only, `load1` 0.96–1.58:

```
kernel 44 %   waitblock 22 %   submit 17 %   usm_write 9 %   readback 8 %
```

Decode reads **9.16 GiB of weights per device per token** in **257.7 ms** of
kernel = **35.5 GiB/s ≈ 21 % of the B580's ~179 GB/s**.

The obvious reading — "the GEMV kernel is not saturating bandwidth" — is
**wrong**, and the batch exists because that was established by measurement
rather than left as a guess.

**Why the kernel is parallelism-bound, not bandwidth-bound.** At M=1 the
single-GEMM decode kernel's total work-item count is **exactly `N`** (one output
element per work-item), and that is **invariant to the work-group shape**. At
27B TP=2 a layer GEMM has N≈2048 per rank over **512 Xe cores ≈ 4 threads/EU**,
and `kThr=16` cannot even fill one SIMD.

`INTFER_DECODE1_MR=1` (MR=4 outputs per work-item, chosen to raise
memory-level parallelism *and* share one activation load across 4 outputs, and
chosen bit-exact) came out **2.3× slower**:

| arm | kernel | waitblock |
|---|---|---|
| off | 2055.1 ms | 1037.7 ms |
| on | **4698.7 ms** | 2362.5 ms |

`nk`/`nw`/`nr` identical (7936/7936/2048) — device work, not plumbing. MR did
not add parallelism, it **divided an already-starved budget by four**. Output
tiling at M=1 is **anti-parallel**; do not re-attempt it.

That leaves exactly two ways to raise the work-item budget:

* **(a) split K across work-items** — raises parallelism *and* coalesces the
  weight loads, but **reassociates the FP32 sum**, so it needs a tolerance-based
  numerics contract agreed **before** any code. **Not this batch.**
* **(b) fuse INDEPENDENT GEMMs into one `nd_range`** — bit-exact, because every
  output keeps its own work-item and its own K order. **This batch.**

---

## E5 — `INTFER_FUSED_DECODE1=1` (fused multi-GEMM decode launch)

### Flag / kill switch / default

| | |
|---|---|
| Flag | `INTFER_FUSED_DECODE1=1` |
| Kill switch | `INTFER_NO_FUSED_DECODE1=1` |
| Default | **OFF** — unset is byte-identical to the previous commit |
| Interaction | rides the existing wave seam, so `INTFER_NO_WAVE=1` still restores fully sequential submission |

### What it changes, and why it should help

A wave holds `nl` **independent** GEMMs — q/k/v is 3, gate/up is 2 — and each
currently launches its own `N`-wide kernel back-to-back. One launch carrying
`sum(N_i)` work-items instead of `N_i` occupies **2–3× the EUs for identical
arithmetic**, which is exactly the resource that is starved.

Side effect: `nk` and `nw` drop by `(nl-1)` per device, so fewer launches too.

### Why it is bit-exact (structurally, not by inspection)

* The per-output INT2 arithmetic was **extracted verbatim** into one file-local
  routine `decode1_out_int2_tab`, and **both** the single-GEMM kernel and the
  fused kernel call it. The K order, the scale table, the tail expression and the
  byte indexing therefore cannot land in one caller and not the other.
* The extraction **deliberately preserves a pre-existing asymmetry**: the main
  chunk applies the scale via a register table (`av * wtab[code]`) while the
  scalar tail uses `av * scale * canon_value(...)`. Those associate differently —
  `(av*scale)*cv` vs `av*(scale*cv)` — and that ~1 ulp difference is the
  oracle-gated consequence of the table hoist. Tidying it would be a numerics
  change disguised as a refactor.
* It takes the **global** code pointer plus `ng`, not a pre-offset row pointer.
  Byte offsets are computed on the global index `(ng*K + kg)/4`, so when `K` is
  not a multiple of 4 the rows are not byte-aligned and `row_base/4` is
  fractional; pre-offsetting would have silently changed which bytes are read.
* The slot → (set, output) mapping is a **host-computed prefix sum**. The device
  consumes it and never re-derives it.

### Where it declines (bit-identical fall-through to the sequential loop)

`M != 1`; `nl < 2`; more than 8 sets; **any non-INT2 scheme** (the shared helper
is the INT2 routine, so another scheme would be a different arithmetic path and
must decline rather than be approximated); a null buffer; `group_size <= 0`; and
any set whose `num_groups`, `num_groups_w` or `out_offset` is not
self-consistent. Those last three are **verified, not trusted** — `out_offset`
is the device's only source of the mapping, so a wrong one is a silently wrong
answer rather than a crash. DPAS48 already declines upstream via the existing
wave fallback.

### Expected direction

Fewer launches (`nk`/`nw` down by `(nl-1)` per device), and a **higher fraction of
EUs occupied** during each decode wave. `kernel` should fall if occupancy is
really the binding constraint; if it does not fall, **occupancy was not the
constraint** and the 21 % bandwidth figure has a different cause — that is a
real and useful negative, so record it rather than discarding the run.

**Win:** `nk`/`nw` down, `kernel` down. **Neutral:** `nk` down but `kernel`
flat (occupancy was not binding). **Loss:** `kernel` up — report it, it would
mean the fused launch serialised something it should have overlapped.

### Correctness already done

* `scripts/test_gpu_optin.sh` — arms at **TP=1 and TP=2**: flag-on equals base at
  n=24; the **kill switch** also equals base; `INTFER_DECODE1_WG` at 32 and 8;
  FUSED+WG=32 together; and the **wide (~97-row) arm** where the kernel must
  decline. The wide arm is the only one that can catch a broken decline — the
  n=24 arms never exercise M>1.
* `scripts/mutate_fused_decode.sh` — 4 artifact states (G-A shared-routine
  bypass, G-B device-side slot re-derivation, G-C eligibility weakened, G-D the
  `M != 1` decline removed) plus **G-E, the converse, which runs first**:
  healthy streams identical *and* the fused path proven **reachable** by
  launch-counter movement. Without the reachability half a kernel that always
  declines would satisfy every other state without ever running.
* **Anchor pre-flight, and the verdict is COUNTED, not inherited.** Every anchor
  is located up front by *substring occurrence count in python* — the same
  counting method `patch_literal` uses, so the guard and the thing it guards
  agree about what "exactly once" means (a multi-line anchor is one line to
  `grep -c` and many occurrences to python). It refuses **drifted** (0
  occurrences) *and* **ambiguous** (2 occurrences) anchors; the second matters
  because it is the state where a patch would rewrite whichever comes first,
  which is a different mutation than the state describes.
  `scripts/mutate_fused_decode.sh --anchors-only` runs it with no build, no
  model and no mutation.

  *Verified in both directions without a model run* (measured, not asserted):
  healthy tree → `ok (4/4)`, rc=0; G-C reindented → `FAIL (1 of 4 stale)`, rc=1,
  refuses to start; G-D's anchor duplicated → `FAIL (1 of 4 stale)`, rc=1;
  restored → `ok (4/4)`, rc=0.

  The first version printed `ANCHOR STALE` and then exited **0** — python's
  non-zero status discarded by a script that is not `set -e`. A loud message
  followed by a green summary is worse than no pre-flight, so the failure is now
  accumulated in `ANCHOR_BAD` and both the summary line and the exit code derive
  from that count.

RESULT (TP=1): ______________________
RESULT (TP=2): ______________________

---

## E6 — `INTFER_DECODE1_WG=<n>` (M=1 decode work-group size)

### Flag / kill switch / default

| | |
|---|---|
| Flag | `INTFER_DECODE1_WG=8\|16\|32\|...` (clamped to [1,256], rounded down to a power of two) |
| Default | **16** — identical to the historical `kThr`, so unset is unchanged |
| Scope | the **M=1 decode kernels only**. `kThr` itself is deliberately untouched, because it also sets the M>1 tiled kernels' 2D geometry and prefill has its own measurements |

### What it changes, and why it should help

A **pure work-group shape change**: one work-item per output, same K loop, same
order. Hence bit-exact by construction.

### Its ceiling — stated so it is not oversold

Total work-items is `ceil(N/nWG)*nWG ≈ N` and is **essentially invariant to
`nWG`** (2048 items at N=2048 whether nWG is 8, 16 or 32). So this **cannot** fix
the ~4-threads/EU starvation — that needs more work-*items* (E5) or a K-split (a).
What it can move is second-order:

* **lane utilisation** — `kThr=16` may under-fill a 32-wide SIMD;
* **work-group count**, i.e. how many EUs the scheduler can spread across:
  **256 groups at nWG=8, 128 at 16, 64 at 32.**

Expect a second-order effect, **not** a bandwidth or occupancy fix.

### Expected direction

**Win:** `kernel` down modestly at one of 8/32. **Neutral:** flat — the expected
outcome, and it confirms nWG is not the constraint. **Loss:** one value up and
the other down, i.e. 16 is a genuine local optimum.

### Correctness already done

Same `test_gpu_optin.sh` arms (bit-exact at 32 and 8, plus inert in the wide
`M>1` arm, plus composed with FUSED).

RESULT (nWG=32): ______________________
RESULT (nWG=8):  ______________________

---

## Reproducing

```bash
# quiet box is NOT required -- these are load-insensitive counters
export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}
sycl-ls | grep -c '\[level_zero:gpu\]'      # must print 2

M=/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer
for arm in "BASE:" "FUSED:INTFER_FUSED_DECODE1=1" "WG32:INTFER_DECODE1_WG=32" "WG8:INTFER_DECODE1_WG=8"; do
  n=${arm%%:*}; e=${arm#*:}
  echo "== $n"
  env $e build/examples/intfer_cli/intfer_cli -m "$M" --gpu --tp 2 -n 16 \
      -p "The capital of France is" 2>&1 | grep -E 'GPU profile @decode'
done
```

Read the counters, not tok/s. Report the **per-round** `kernel` values with their
spread if you take more than one round, and state the counting basis.

---

## Status and provenance (read before trusting any of the above)

**Verified without running the model, re-checked against the COMMITTED bytes:**
`intfer_gemm`, `intfer_engine` and `intfer_cli` compile clean; `bash -n` clean on
`test_gpu_optin.sh`, `mutate_fused_decode.sh`, `run_all_checks.sh`; the wide arms
appear **exactly once** each and **inside** the `for tp` loop, with their base
(`WB`) assigned before both of its uses; **zero** occurrences of an undefined
base variable in that file -- in the working tree *and* in `HEAD`; all four
mutation anchors located exactly once, with the pre-flight refusing both a
drifted (0) and an ambiguous (2) anchor, per the table above; and the coverage
sections green. Re-derive that last one rather than take it from here:

```bash
INTFER_RUN_ALL_COVERAGE_ONLY=1 bash scripts/run_all_checks.sh
```

It prints its own census and asserts every `check_*`/`mutate_*` is both
**dispatched** (0b) and **scheduled** (0c). **No count is quoted in this file on
purpose** -- a hand-counted total in prose is a citation, not a count, and it goes
stale the moment a script is added.

**NOT verified, and it needs the model:** the four mutation states and the
stream-identity arms themselves. Those are the certification of bit-exactness and
they have not been executed. `RESULT:` lines stay blank until they are.

**Commits for this batch are DERIVED, not transcribed here.** Listing the shas
in prose was the previous version of this paragraph and it was wrong for the same
reason the old census was wrong: a hand-maintained list in a document is a copy
that goes stale, and a reader has no way to check it without re-deriving it anyway.
Derive it instead -- the range starts at the commit that records the measured
negative result which motivated the batch:

```bash
git log --oneline 40219bf^..HEAD
```

No sha is named here, and that is deliberate: the commit that rewrites this
sentence is itself inside the range above, so any sha it quoted would be stale
the moment it landed. A paragraph that forbids transcription must not commit it
in its own closing line. Re-run the command rather than trusting the sentence.

---

## Explicitly not attempted

* **K-split across work-items** (option a). Raises parallelism *and* coalescing,
  but reassociates the FP32 sum, so it needs a tolerance-based numerics contract
  agreed **before** code — the recorded E4-style precondition. It is the obvious
  next thing to try and the one thing that must not be started casually.
* **Output tiling at M=1.** Measured 2.3× slower and structurally
  anti-parallel. `INTFER_DECODE1_MR` ships default-OFF and exists only as the
  artifact backing that measurement.
  **Coverage, corrected 2026-10-04 (read this before adding an arm for it):** MR is
  guarded by `bits == 2`, and only the 27B model has INT2 tensors — 0.8B is
  `INT4 ×76 / FP16 ×93 / INT8 ×18`, **zero INT2**. So 9 of the 10 correctness
  gates cannot reach that branch, and the "bit-exact by construction" claim was
  until now supported only by reading the code. The one arm that can actually
  test it was added to **`scripts/test_tp2_27b.sh`**, which runs the INT2 27B
  model at `--tp 2` and now asserts the `INTFER_DECODE1_MR=1` stream against the
  **same fixed oracle** as its base run. It is **UNRUN** — it needs the box and a
  27B load, so treat it as pending, not as passing. Do **not** add 0.8B arms for
  this flag: they would be green because the path never runs.
* **Promoting either flag to default.** Both are opt-in until a counter verdict
  exists. Promoting on a reasoning argument rather than a measurement is the
  mistake this repo has retracted three times.