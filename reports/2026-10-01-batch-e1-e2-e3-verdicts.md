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
  `[level_zero:gpu]`, `xpu-smi --list-gpus` prints `GPU 0` only. Ruled out by
  measurement: the container's L0 driver version (forcing the host's newer
  `libze_intel_gpu.so.1.17.39758` instead of the container's `1.15.38646` still
  yields **1** device); cgroup device policy (no `devices.list`, and both render
  nodes open); NUMA (single node, both cards `numa=-1`); power state (both D0);
  and another process holding GPU 1 (`fuser` shows no userspace holder on
  `renderD129`).

> **LEADING HYPOTHESIS — NOT ESTABLISHED: host-side enumeration.** The evidence
> is that two *independent* driver stacks (Level Zero **and** OpenCL via the
> unrelated `intel64.icd`) both report 1 GPU while both cards are healthy at
> kernel level. That is *consistent with* a host/driver enumeration drop of PCI
> 86:00.0, but it is inference from agreement, not from reading the code that
> would have to be wrong — the exact mistake AUDIT #9 made and AUDIT #11 had to
> retract. **I did not find the mechanism.** What I can state from measurement is
> only the exclusion list above. The remedy ("host-side rebind/rescan of
> 86:00.0") is therefore a *suggestion to investigate*, not a diagnosed fix, and
> it may simply be wrong. Whoever picks this up should read the xe/Level Zero
> enumeration path rather than trusting this paragraph.

* Consequence, reproduced: `--tp 2` on 27B prints
  `WARNING: TP=2 requested but only 1 Arc GPU(s) discovered`, preloads
  **18751 MiB** onto a **12216 MiB** card, then dies `signal 6`, **rc=139**.
  That is the documented AUDIT #11 signature, not a regression.

**Action for the operator:** investigate why the xe device at PCI 86:00.0 dropped
out of Level Zero enumeration, before any 27B number in any of these reports can
be produced. See the hypothesis caveat above.

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

Stream identity at **short ctx** — all three arms identical:

```
BASE / KV / KVQ8   279 6511 314 279 3046 13 198 760
```

Stream identity at **long ctx** — all three arms identical:

```
BASE / KV / KVQ8   561 3712 314 19820 10903 321 5484 6618
```

**The long-ctx claim is backed by per-arm evidence, not by a command that
happened to print ids.** `scripts/verify_e2_ctx.sh` runs each arm and requires
rc=0, its own `prompt: N tokens` line, and 8 completed decode steps:

```
INTFER_KV_CAP=8192

BASE   rc=0  prompt: 3601 tokens    decode_steps=8
       ids: 561 3712 314 19820 10903 321 5484 6618
KV     rc=0  prompt: 3601 tokens    decode_steps=8
       ids: 561 3712 314 19820 10903 321 5484 6618
KVQ8   rc=0  prompt: 3601 tokens    decode_steps=8
       ids: 561 3712 314 19820 10903 321 5484 6618
```

So: all three arms genuinely processed **3601 prompt tokens** and each completed
its 8 decode steps with identical ids. Short-ctx arm prompt was 5 tokens
(`The capital of France is`). Both contexts are bit-exact for all three arms.

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

### Capacity finding — the E2 long-ctx command AS WRITTEN ABORTS (fix the command first)

A 3601-token prompt **aborts** on the default cache:

```
[intfer] FATAL: pos 1151 >= KV capacity 1024 (raise INTFER_KV_CAP, max 262144)
```

`INTFER_KV_CAP` defaults to **1024**. The long-context command in
`reports/2026-09-30-pending-bench-backlog.md` (the `*400` repetition) **cannot run
as written** — it aborts before the first decode step, and because my harness
initially compared empty id lists it even reported a false `OK`. This is a real
usability/capacity constraint for the long-context use case E2 exists to serve,
and it is not specific to 0.8B: the cap is a flat position limit.

**Corrected, verified-runnable long-ctx E2 command** (this is what produced the
long-ctx numbers above — 0.8B TP=1, and it needs `INTFER_KV_CAP`):

```bash
BIN=build/examples/intfer_cli/intfer_cli
M=test/models/Qwen3.5-0.8B.intifer
TOK="--tokenizer-py scripts/bpe_tokenize.py --detok-py scripts/bpe_detokenize.py"
LONG="$(python3 -c "print('The history of artificial intelligence and machine learning. '*400)")"
# PROMPT IS 3601 TOKENS -> the default INTFER_KV_CAP=1024 ABORTS. Raise it.
export INTFER_KV_CAP=8192
for arm in BASE KV KVQ8; do
  case $arm in
    BASE) unset INTFER_INT8_KV INTFER_KV_Q8 INTFER_KV_HAD ;;
    KV)   export INTFER_INT8_KV=1; unset INTFER_KV_Q8 INTFER_KV_HAD ;;
    KVQ8) export INTFER_INT8_KV=1 INTFER_KV_Q8=1 INTFER_KV_HAD=1 ;;
  esac
  "$BIN" --gpu --tp 1 -m "$M" -p "$LONG" -n 64 --no-stream $TOK 2>&1 \
    | grep -aE "prompt: [0-9]+ tokens|decode: [0-9]+ tokens in|@decode\]"
done
```

Always assert the `prompt: N tokens` line is present in the output. A run that
aborted on the KV cap produces no decode line, and that absence must not be
allowed to read as a pass.

### PRECONDITION on every `--tp 2` command in these reports

All three reports' 27B commands are `--tp 2`, and **none of them can run on a
1-GPU box**. Before spending any window on them:

```bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}
sycl-ls | grep -c '\[level_zero:gpu\]'     # MUST print 2, else STOP
```

If it prints **1**, `--tp 2` silently clamps to TP=1 and 27B (19.9 GB) preloads
18751 MiB onto a 12216 MiB card and dies `signal 6`/rc=139. Do not read that as
a regression, and do not "fix" it by relaxing the command.


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

### The derived rule is now exercised by a differential test (2026-10-01)

The derived rule had ONE data point (E1), which is not enough: the two batch
shapes differ in exactly the way the rule branches on. E2 has `REF_ARM=BASE`,
so BASE must be **identical**; E1/E3 have `REF_ARM=<quantizing arm>`, so BASE
must **differ**. Run in `INTFER_BENCH_TIMING=0` (counters only, no tok/s):

| batch | arms | BASE verdict | non-BASE invariant | rc |
|---|---|---|---|---|
| E1 | BASE DPAS48 PREQ | `OK BASE differs` | `PREQ == DPAS48` | 0 |
| E2 | BASE KV KVQ8 | *(no exemption — all identical)* | `KV == BASE`, `KVQ8 == BASE` | 0 |
| E3 | BASE DPAS48 FOLD FOLDPREQ | `OK BASE differs` | `FOLD == DPAS48`, `FOLDPREQ == DPAS48` | 0 |

The branch therefore fires on E1/E3 and correctly does **not** fire on E2, which
is the differential that a single E1 run could never have shown. A harness whose
exemption fired unconditionally would also pass E1 and fail E2, so this is the
test that distinguishes derived from hard-coded.

E2 identity (0.8B TP=1 `-n 8`): all three arms `279 6511 314 279 3046 13 198 760`.
E2 counters are flat across arms — `nk=13440 nw=8832 nr=4608` for BASE/KV/KVQ8 —
i.e. INT8-KV is launch-neutral at 0.8B, which is why it stays default-off
pending a quiet-box wall-time read. E3 counters: `nk` 13440 / 17856 / 21888 /
27840 across BASE/DPAS48/FOLD/FOLDPREQ, consistent with FOLD's one extra launch
per DPAS GEMM.

## Verification state of this batch

Measured after the reports were written, not asserted:

| check | result |
|---|---|
| `git status --porcelain` | 0 modified (tree clean) |
| check stamps vs HEAD | **5/5 CURRENT** (`device_precond_matrix`, `device_precond_matrix/mutate`, `fold_matrix`, `gate_inventory_matrix`, `run_all_mutate`) |
| `scripts/gate_inventory.sh` | **rc=0**, `sites_found=9`, `stale_tally_sites=0`; every gate builds its own target, no whole-gate skip strands 1-GPU arms |
| doc tally | `7 PASS, 2 PARTIAL, 1 SKIPPED` reconciles with the source classification for 1 visible GPU |

**How the stamps were refreshed — and a caveat worth stating.** The first
`run_all_checks.sh` invocation returned rc=1 with *"another run_all_checks.sh is
running"*. That was **correct behaviour, not a bug**: a second maki agent session
was already running the suite (lock taken 06:40:58, actively inside
`check_device_precond.sh`). I did **not** remove the lock — the mutation matrices
edit tracked source and restore with `git checkout --`, so two concurrent runs
would restore each other's mutations. I waited ~90 min for it to finish instead.
It completed, refreshed all 5 stamps, and committed further gate commits on top
of mine. `05024c9` and `a1ef1d9` are ancestors of the resulting HEAD and all
three of my files are present at it.

That session started ~5 min **before** my `a1ef1d9` commit, so its steps span
two HEADs. The stamps it wrote are CURRENT against the final HEAD, which is what
`run_all_checks.sh` step 3/3 enforces by enumeration.

**No gate was re-run by me** — these stamps are that suite's output. Per its own
printed caveat, the doc-tally reconciliation *"is not a re-run of the gates, so it
cannot detect a gate that is green-numbered but red."*

## What to do next

1. **Host-side**: investigate why the xe device at PCI 86:00.0 dropped out of
   Level Zero enumeration (`sycl-ls` must print 2 `[level_zero:gpu]` lines). Until
   then no 27B row in any report is obtainable. **The mechanism is a leading
   hypothesis, not a diagnosed fault** — see BLOCKER 1 and the AGENTS.md note.
2. **Quiet box**: get load1 < 2 (stop/finish the Planar.jl julia job; do not run
   two maki sessions' GPU work concurrently).
3. Then run these three batches for real with `--tp 2` on 27B, interleaved,
   3+ rounds, and fill the `RESULT:` lines in
   `reports/2026-09-30-pending-bench-backlog.md` and
   `reports/2026-09-30-dpas48-fold.md`.
4. E2's long-ctx command needs `INTFER_KV_CAP` raised above the prompt length.
