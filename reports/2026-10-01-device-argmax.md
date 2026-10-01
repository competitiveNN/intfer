# Device-argmax v1 (`INTFER_DEVICE_ARGMAX=1`) — per-row conditional lm_head argmax

**Status:** merged and correctness-gated. Wall-time A/B still OWED — needs a quiet box.
This report exists because the lever shipped in `d460eb7` + `c2be7f5` (2026-09-23/24)
without one, which is an AGENTS.md violation: a batch handed for benchmarking must
carry a report. Nothing about the code changed to produce this file.

Target model is **27B**; the counters below are 0.8B, the only thing loadable on a
1-GPU box. Read them as direction, not magnitude.

---

## E5 — `INTFER_DEVICE_ARGMAX=1`

### 1. Flag, kill switch, default

| | |
|---|---|
| Flag | `INTFER_DEVICE_ARGMAX=1` |
| Kill switch | **`env -u INTFER_DEVICE_ARGMAX`** — there is no `=0` form; the read is presence-only (`:1900`), so `INTFER_DEVICE_ARGMAX=0` still ENABLES it. Measured: `=0` emits 1 `[argmax]` line, `env -u` emits 0 |
| Default | **OFF** |
| Declined by | `INTFER_TRACE_LMHEAD`, TP>1 (`:1913`), prefill rows in a mixed batch, any non-greedy row in the M-row path |
| Bisection only | `INTFER_ARGMAX_PREFILL=1` restores prefill firing; `INTFER_ARGMAX_DBG=1` prints the `[argmax]` counter; `INTFER_ARGMAX_PARTS=1` dumps per-(device,tile) partials |

### 2. What it changes and why it should help

`lm_head` reads the full vocab row back to the host every token: an FP16 D2H of the
resident GEMM-out tile, a host FP32 convert, a TP-shard concat, then a host argmax
scan over `N` floats. At 0.8B `N`=248320, so ~0.5 MB of D2H per token per tile, plus
a 248320-iteration host loop.

The lever runs the argmax **on the device** over the already-resident FP16 tile
(`gemm::lm_head_argmax_tile`), per-row, chain-dep'd on that tile's GEMM event, and
D2Hs **8 bytes** (one `int` id + one FP32 val) instead. The host then reduces those
partials in `(device, tile-ts)` order using the identical strict-`>` loop.

It is **per-row conditional**, not global, because sampling rows need their full
logits. `run_lm_head_batch_best` takes an `is_greedy` mask; v1 requires *all* rows
greedy and falls back otherwise. Mixed-temperature v2 was analyzed and **rejected**
as a wash (`docs/seam_audit_b.md`) — it adds +1 submit and +2 small D2H per tile to
save part of the bulk D2H, netting ~±0.5 ms for high complexity.

Sites: `run_lm_head_best` (`main.cpp:1896`), `run_lm_head_batch_best` (`:2096`),
consumers at `:5165` (`step`, `greedy_only`), `:6188` (`score_row`, `all_greedy`),
`:6975` (`step_batch`, `greedy_mask` built `:6955`).

### 3. Benchmark command (quiet-box rule: load1 < 2)

```bash
# decode, TP=1, greedy (the only path that fires)

# OFF arm — the variable must be UNSET. `INTFER_DEVICE_ARGMAX=0` does NOT work:
# the flag is read as presence-only (`std::getenv(...) != nullptr`, :1900 and
# :2101), so `=0` still ENABLES the lever. Measured, not assumed: with
# INTFER_ARGMAX_DBG=1, `INTFER_DEVICE_ARGMAX=0` emits 1 `[argmax]` line and
# `env -u INTFER_DEVICE_ARGMAX` emits 0. A report that benchmarked `=0` would
# have measured flag-vs-flag and read as a wash.
env -u INTFER_DEVICE_ARGMAX ./build/examples/intfer_cli/intfer_cli --gpu --tp 1 \
  -m test/models/Qwen3.5-0.8B.intifer \
  --tokenizer-py scripts/bpe_tokenize.py --detok-py scripts/bpe_detokenize.py \
  -p "The capital of France is" -n 64

# ON arm
INTFER_DEVICE_ARGMAX=1 ./build/examples/intfer_cli/intfer_cli --gpu --tp 1 \
  -m test/models/Qwen3.5-0.8B.intifer \
  --tokenizer-py scripts/bpe_tokenize.py --detok-py scripts/bpe_detokenize.py \
  -p "The capital of France is" -n 64
```

**Verify the OFF arm actually is OFF before trusting any A/B.** Cheap
non-vacuity check, run it once:

```bash
env -u INTFER_DEVICE_ARGMAX INTFER_ARGMAX_DBG=1 ./build/examples/intfer_cli/intfer_cli \
  --gpu --tp 1 -m test/models/Qwen3.5-0.8B.intifer \
  --tokenizer-py scripts/bpe_tokenize.py --detok-py scripts/bpe_detokenize.py \
  -p "The capital of France is" -n 4 2>&1 | grep -c argmax    # expect 0
```

Note the two conventions coexist in this codebase and only one is right per
flag: `INTFER_DEVICE_ARGMAX` / `INTFER_ARGMAX_DBG` / `INTFER_ARGMAX_PARTS` /
`INTFER_ARGMAX_PREFILL` are presence-only, whereas `INTFER_LMHEAD_DPAS` tests
`[0]=='0'` (`:1903`). Do not generalize from one to the other — check the
`getenv` at the site you are toggling.

Interleave the two arms across ≥3 rounds; do not run them as blocks. For 27B use
`scripts/serve_27b.sh` (`--tp 2`), but note **TP>1 declines the flag outright** (`:1913`)
— on TP=2 this lever is currently a no-op, so a 27B TP=2 A/B measures nothing.
The 27B number that matters is TP=1-on-one-B580, which OOMs on the lm_head tile
(`AGENTS.md` AUDIT #11). State this explicitly when reporting.

### 4. Load-insensitive counters I measured (box loadavg 12.95 — NOT a timing run)

0.8B, TP=1, `-n 8`, `[GPU profile @decode]`:

OFF arm = `env -u INTFER_DEVICE_ARGMAX` (see §3 — `=0` would NOT be OFF).

| bucket | flag OFF | flag ON | delta |
|---|---|---|---|
| nk (launches) | 1680 | 1712 | **+32** |
| nw (waits) | 1104 | 1104 | 0 |
| nr (readbacks) | 576 | 576 | 0 |
| kernel | 81.3 ms | 81.3 ms | ~0 |
| submit | 92.9 ms | 94.9 ms | +2.0 ms |
| waitblock | 79.5 ms | 82.4 ms | +2.9 ms |
| readback | 39.8 ms | 39.4 ms | -0.4 ms |
| copy | 0.1 ms | 0.1 ms | 0 |

An earlier round on the same box gave kernel 82.3→86.4 ms, submit 93.6→97.5 ms.
**Caveat, and it is the main one:** at loadavg ~13 these millisecond buckets carry
more noise than the effect. The only signal I trust is `nk`, because it is an
integer counter and it reproduces exactly.

**`nk` +32 is the mechanism, stated as arithmetic:** 4 tiles × 8 tokens = 32
extra kernel launches — one argmax launch per (tile, token). The saving is *not*
a launch saving; it is a **bytes** saving (0.5 MB → 8 B per tile-token) that the
`nr`/`readback` counters do **not** resolve, because the D2H count is unchanged
(576 both ways). So the counters above can show the cost (+32 submits) but cannot
show the benefit. That asymmetry is the honest read, and it is why the wall-time
A/B is still owed rather than argued from this table.

Stream identity: base and flag both `279 6511 314 279 3046 13 198 760`, reproduced
across 2 rounds. `[argmax]` counter fires on flag, silent on base.

### 5. Expected direction, and what counts as what

- **Win:** decode tok/s up ≥2% at `-n 64`, with base/flag streams identical, and
  no `FATAL`/segfault in the output. The D2H is on the decode critical path, so a
  real saving should show; expect the gain to scale with vocab `N`, i.e. **larger on
  27B than on 0.8B**.
- **Neutral:** ±2% tok/s. Given +32 launches/token-scale on a submit-bound decode
  (submit ≈ 30% of decode per `AGENTS.md`), a wash is a plausible real result, and
  one that the load-insensitive table is consistent with.
- **Loss:** >3% down, or any stream divergence. A loss is expected-ish, because the
  argmax kernel adds one submit per tile per token; the design only pays if the D2H
  it removes is bigger than that submit.

Do **not** read the 0.8B `-n 8` numbers as a throughput verdict.

#### Why a quiet box is the wrong gate for E5 specifically

For E1-E4 a counters-only reading is enough to tell a mechanism from a no-op.
For E5 it is **structurally impossible**, and that is worth stating rather than
leaving as "owed". The saving is a *bytes* reduction on a path whose D2H
**count** does not change: the lever replaces `Nlt*2` bytes per tile-token with
8, but there is still exactly one D2H per tile either way. Every existing
bucket is a count or a duration of a kernel the counter already attributes:

| observable | flag OFF | flag ON | resolves the saving? |
|---|---|---|---|
| `nr` (readbacks) | 576 | 576 | no — count is identical by construction |
| `nk` (launches) | 1680 | 1712 | no — shows the +1 argmax *cost* only |
| `readback` ms | 39.8 | 39.4 | no — below noise at loadavg ~13 |

So no arrangement of the existing counters can judge E5; a bytes-transferred
bucket would have to be **added** to the `[GPU profile]` block (instrument the
`q.memcpy` byte counts at `:1775` and `:2313`), which touches profiling source
and therefore needs its own gate. That work is **scoped, not started** — see
§8. Until it exists, E5's verdict genuinely does require the quiet-box run, and
the honest statement stays "unmeasured", not "counters looked neutral".

### 6. Gates already passed

- `scripts/test_device_argmax.sh` — TP1/TP2 8-way matrix, `[argmax]` non-triviality
  lock (1 line on flag / 0 on base), sampling count parity, spec-verify streams,
  serve fence lock. On this 1-GPU box it is **PARTIAL (rc=2)**: every TP=1 arm runs
  and passes; its 5 TP=2 arms skip.
- `cpu_parity`, `http_batch`, `gpu_optin`, `serve_load` — rc=0 on this tree.
- `tp2_27b` — SKIPPED (needs 2 GPUs).
- Bit-contract on CPU is documented in `docs/seam_audit_b.md` §6: kernel scan ==
  host strict-`>` loop on adversarial tiles + 200k fuzz; multi-tile ts-order reduce
  == flat loop incl. cross-tile ties; half→float exact over all 65536 patterns.
  First-max-wins is preserved because the reduce uses strict `>` over
  `(device, tile-ts)` order (`main.cpp:2072`, `:2332`).

### 7. Compile status

`scripts/build.sh` clean on this tree. `main.cpp` also passes `-fsyntax-only` both
with and without `-DINTFER_PROFILE_GPU=1` (0 errors each), per the audit-#3c lesson
that a helper inside the GPU-backend guard can break a stripped configuration while
every gate still passes.

---

## 8. Blocked follow-on: a bytes-transferred bucket (E6, scoped, NOT implemented)

**Do not go looking for a flag — none exists and no code has been written.**

E5 cannot be judged from counters, for the structural reason in §5: the saving
is bytes on a D2H whose *count* is unchanged, so `nr` cannot see it. The fix is
one new counter in the `[GPU profile]` block — accumulate
`q.memcpy` byte counts at the two lm_head D2H sites (`main.cpp:1775` scalar,
`:2313` M-row) and print them beside `nr`. Expected signature if the lever
works: `d2h_bytes` drops from `~N*2` per token to `~8*ntiles` per token, with
`nr` unchanged.

Scoping notes, so this is not re-derived wrongly:

* It touches **profiling source**, so it needs its own gate before it can be
  trusted, per the build-first rule every other gate follows.
* It must be **load-insensitive by construction** (a byte count, not a duration)
  or it inherits E5's quiet-box dependency and achieves nothing.
* If added, re-run `scripts/test_device_argmax.sh` unchanged — the new bucket
  must not perturb the stream-identity arms, and the gate is what proves it.

`RESULT:` (bytes bucket, only after it exists): OFF ______ B/token, ON ______ B/token

---

## RESULT:

- 0.8B `-n 64` TP=1, quiet box, interleaved ≥3 rounds: OFF ______ tok/s, ON ______ tok/s
- Stream identity confirmed on the quiet-box run: YES / NO
- 27B: **not measurable here** — TP>1 declines the flag (`:1913`) and TP=1-on-one-B580
  OOMs on the lm_head tile. State this rather than reporting a 0.8B number as the
  27B result.
