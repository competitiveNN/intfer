# Batch verdicts — E1 / E2 / E3 measured on 0.8B TP=1 (27B BLOCKED)

Date: 2026-10-01. Author: maki (agent). Harness: `scripts/bench_reports_batch.sh`.

**Headline: none of the three experiments can be promoted to default-ON, and none
can be rejected either. All three are CORRECT (bit-exact) and cost nothing while
disabled, so the code stays. Only E3 has measured evidence in its favour.**

Every number below is a **load-insensitive counter** (`kernel` = device-side event
time; `nk`/`nw`/`nr` = exact launch/wait/readback counts). **No tok/s is claimed
anywhere in this file** — see "Why there is no wall-time" below. Those are counters
or nothing.

---

## BLOCKER 1 — only 1 of 2 GPUs enumerates (the reports' target is unreachable)

The reports target **27B `--tp 2`**. That cannot run. Verified, not assumed:

* Both B580s are physically present and healthy. `/sys/class/drm/card0` ->
  `0000:04:00.0`, `card1` -> `0000:86:00.0` (distinct PCI root complexes
  `pci0000:00` and `pci0000:80`), both `8086:E20B`, both `DRIVER=xe`, both
  `runtime_status=D0`, both with full `tile0` + `gt0/gt1` +
  `bcs/ccs/rcs/vcs/vecs` engine topology. `/dev/dri/{card0,card1,renderD128,renderD129}`
  are all present and **open `O_RDWR` fine** (`226:0`, `226:1`, `226:128`, `226:129`).
* The **compute runtime exposes one device**: `sycl-ls` prints 1
  `[level_zero:gpu]`, `xpu-smi --list-gpus` prints `GPU 0` only. This is
  independent of the container's L0 driver version — forcing the host's newer
  `libze_intel_gpu.so.1.17.39758` instead of the container's `1.15.38646` still
  yields **1** device. OpenCL (`intel64.icd`, a completely separate stack) also
  reports **1** GPU. Two independent driver stacks agreeing on 1 points at host
  enumeration, not a driver-library bug, and it is not fixable from inside the
  container.
* Consequence, reproduced: `--tp 2` on 27B prints
  `WARNING: TP=2 requested but only 1 Arc GPU(s) discovered`, preloads
  **18751 MiB** onto a **12216 MiB** card, then dies `signal 6`, **rc=139**.
  That is the documented AUDIT #11 signature, not a regression.

**Action for the operator:** this needs a host-side fix (rebind/rescan PCI
86:00.0, or check why the xe device dropped out of Level Zero enumeration) before
any 27B number in any of these reports can be produced.

## BLOCKER 2 — the box is contended, so wall-time is not obtainable

Quiet-box rule is `load1 < 2`. Measured `load1` during these runs: **10.65, 12.77,
8.57, 18.13, 10.18** — and 7.97 at the end. Contributors, none of them this
session:

* A **second maki agent session** ran `scripts/run_all_checks.sh` (the project's
  own gate suite), which spawns continuous `intfer_cli --gpu` runs on the same
  GPU 0. It has since finished.
* A **Planar.jl agent container** has two `julia` processes, one pinned at
  **99.7% CPU**, still running.

So a tok/s comparison here would be run variance, not signal — exactly the failure
mode this project has retracted claims for before. `INTFER_BENCH_TIMING=0`
(wall-time withheld) was used deliberately. **`RESULT:` lines stay blank.**

---

## E1 — `INTFER_DPAS_PREQ=1` — bit-exact, mechanism fires, but a measured LOSS at 0.8B

Stream identity, 0.8B TP=1, `-n 8`:

```
BASE    279 6511 314 279 3046 13 198 760
DPAS48  279 6511 314 279 5364 314 864 314
PREQ    279 6511 314 279 5364 314 864 314     <-- PREQ == DPAS48  PASS
```

The bit-exactness contract **holds**: PREQ reproduces the host GA64 quant exactly.

Counters, `-n 64` decode, load1=10.65:

| arm | nk | nw | nr | kernel (device) | copy |
|---|---|---|---|---|---|
| BASE | 13440 | 8832 | 4608 | 739.9 ms | 0.8 ms |
| DPAS48 | 17856 | 11904 | 8448 | 797.2 ms | 415.3 ms |
| DPAS48+PREQ | 23808 | 11904 | 8448 | **887.1 ms** | **450.9 ms** |

Read honestly:

* **The design works as specified.** `nk` +5952 (= 93 GEMM x 64 tok, the pre-pass
  really launches) while **`nw` and `nr` are unchanged** — it is dep-chained and
  adds no host round-trip. That is the report's central claim and it is confirmed.
* **But it costs, it does not save, at this scale.** Device kernel
  797.2 -> 887.1 ms (**+11.3%**), `copy` 415.3 -> 450.9 ms (**+8.6%**). The `copy`
  bucket rises because AUDIT #3/#5 deliberately kept the host O(M*K) max-abs
  reduction in order to ship the exact reciprocal bits — so PREQ *adds* the
  scale/inv H2D on top of a reduction it never removed.
* The report's own caveat is what bites: it promises "roughly half the originally
  promised saving" and targets the case where host act-quant is ~6 ms/token at 27B.
  At 0.8B the host quant is cheap, so there is nothing for PREQ to recover.

**Verdict: KEEP the code (default-OFF, bit-exact, zero cost when off). Do NOT
promote. It is currently a measured loss on every load-insensitive counter
available, and the theory that rescues it (27B host-quant cost) is UNVERIFIED.**

## E2 — `INTFER_INT8_KV=1` (+ `KV_Q8`, `KV_HAD`) — bit-exact, but traffic goes UP

Stream identity at **short ctx (5 tok)** — all three arms identical:

```
BASE / KV / KVQ8   279 6511 314 279 3046 13 198 760
```

Stream identity at **long ctx (3601 tok, `INTFER_KV_CAP=8192`)** — all three identical:

```
BASE / KV / KVQ8   561 3712 314 19820 10903 321 5484 6618
```

Long-ctx counters, `-n 64`, load1=18.13:

| arm | nk | nw | nr | kernel (device) | usm_write | readback |
|---|---|---|---|---|---|---|
| BASE | 13440 | 8832 | 4608 | 656.5 ms | 612.3 ms | 981.6 ms |
| KV | 13440 | 8832 | 4608 | 656.2 ms | 621.0 ms | 996.0 ms |
| KV+Q8+HAD | 13440 | 8832 | 4608 | 658.3 ms | 630.2 ms | 1022.2 ms |

* **`nk`/`nw`/`nr` are byte-identical across all three arms** — the flag adds no
  kernels and no round-trips. Device kernel time is flat (656.5 / 656.2 / 658.3).
* **The traffic counters move the wrong way**: `usm_write` +8.7 then +17.9 ms,
  `readback` +14.4 then +40.6 ms. The report describes `INTFER_KV_Q8` as an
  **append-only** G64 Q8 mirror, so it *adds* bytes beside the cache rather than
  replacing FP16 entries. On this model that is a net increase, not a reduction.
* Note this was tested at **long ctx**, not the 5-token case the report warns is
  noise — so "short ctx is noise" is not an available excuse here.

**Verdict: KEEP the code (default-OFF, bit-exact at both ctx lengths). Do NOT
promote. No decode win at long ctx on the only model available; the load-insensitive
counters move against it.** The report's premise — that KV is the traffic share
INT2 weights do not already shrink — is **not supported at 0.8B** and remains
untested at 27B.

### Capacity finding (report asked for this to be reported, not dropped)

A 3601-token prompt **aborts** on the default cache:

```
[intfer] FATAL: pos 1151 >= KV capacity 1024 (raise INTFER_KV_CAP, max 262144)
```

`INTFER_KV_CAP` defaults to **1024**. The E2 report's own long-context command
(*400 repetitions) cannot run without raising it. Any long-ctx test must set
`INTFER_KV_CAP` explicitly; this is a real usability/capacity constraint for the
long-context use case E2 is meant to serve.

## E3 — `INTFER_DPAS48_FOLD=1` — bit-exact, and its report's numbers REPLICATE

Stream identity:

```
BASE       279 6511 314 279 3046 13 198 760
DPAS48     279 6511 314 279 5364 314 864 314
FOLD       279 6511 314 279 5364 314 864 314     <-- FOLD == DPAS48      PASS
FOLDPREQ   279 6511 314 279 5364 314 864 314     <-- FOLDPREQ == DPAS48  PASS
```

Counters, `-n 64`, load1=10.18:

| arm | nk | nw | nr | kernel (device) |
|---|---|---|---|---|
| BASE | 13440 | 8832 | 4608 | 652.0 ms |
| DPAS48 | 17856 | 11904 | 8448 | 694.4 ms |
| DPAS48+FOLD | 21888 | 11904 | 8448 | **684.0 ms** |
| DPAS48+FOLD+PREQ | 27840 | 11904 | 8448 | 684.6 ms |

**This independently reproduces the E3 report's claimed figures:**

| quantity | E3 report claimed | measured here | match |
|---|---|---|---|
| extra launches | +504 per 8 tokens | +4032 per 64 tokens (= 504 x 8) | exact |
| device kernel | -1.6% | 694.4 -> 684.0 ms = **-1.5%** | yes |
| extra host waits | none | `nw`/`nr` unchanged (11904/8448) | yes |

* `nw` and `nr` are **unchanged** by FOLD — it is dep-chained, as designed.
* The hoist is a pure work hoist, and the stream identity confirms it changed no
  bits, which is what the exact-int-sum argument predicts.

**Verdict: KEEP, default-OFF. This is the only experiment of the three with a
positive, replicated device-side delta.** It is still small (-1.5% device kernel,
paid for with +4032 launches per 64 tokens), and its real target is wide-N 27B
prefill, which cannot be measured here. Keep the code; do not promote on this data.

---

## Summary table

| exp | flag | bit-exact | mechanism fires | device-kernel delta | traffic delta | verdict |
|---|---|---|---|---|---|---|
| E1 | `INTFER_DPAS_PREQ` | PASS | yes (nk +5952, nw/nr flat) | **+11.3% (worse)** | copy +8.6% | keep default-OFF, **do not promote** |
| E2 | `INTFER_INT8_KV`(+Q8/HAD) | PASS (short+long ctx) | no new kernels | ~0% | usm_write +3%, readback +4% | keep default-OFF, **do not promote** |
| E3 | `INTFER_DPAS48_FOLD` | PASS | yes (nk +4032, nw/nr flat) | **-1.5% (better)** | ~0 | **keep default-OFF, best evidence** |

**Nothing is promoted. Nothing is deleted.** All three are default-OFF,
bit-exact, cost nothing when disabled, and all three remain **UNVERIFIED at 27B**
because only 1 of 2 GPUs enumerates.

## Harness defect found and fixed while doing this (worth recording)

`scripts/bench_reports_batch.sh` initially printed **`OK  KV == BASE`** for a run
where **all three arms produced zero token ids** — an empty stream compared equal
to another empty stream, so the identity check passed *precisely when the run had
failed*. It was caught because the long-ctx run printed no `decode step` lines at
all (the KV-capacity abort). Fixed with an explicit empty-stream guard that exits
1; **negative-tested by re-running the same failing config, which now correctly
reports `identity check is VACUOUS for: BASE KV KVQ8` and rc=1.**

A second harness bug: the "BASE is expected to differ from an activating arm"
exemption was hard-coded to batch `E1`, so `E3` wrongly FAILED on a correct
result. Replaced with the derived rule (`arm == BASE && REF_ARM != BASE`).

Both are the same species as the vacuous-gate family in AGENTS.md (audits #3/#3c,
#14): a check that cannot fail, or a rule re-derived per call site instead of once.

## What to do next

1. **Host-side**: restore 2-GPU Level Zero enumeration (`sycl-ls` must print 2
   `[level_zero:gpu]` lines). Until then no 27B row in any report is obtainable.
2. **Quiet box**: get load1 < 2 (stop/finish the Planar.jl julia job; do not run
   two maki sessions' GPU work concurrently).
3. Then run these three batches for real with `--tp 2` on 27B, interleaved,
   3+ rounds, and fill the `RESULT:` lines in
   `reports/2026-09-30-pending-bench-backlog.md` and
   `reports/2026-09-30-dpas48-fold.md`.
4. E2's long-ctx command needs `INTFER_KV_CAP` raised above the prompt length.
