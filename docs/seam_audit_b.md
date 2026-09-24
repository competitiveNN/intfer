# Audit B: remaining GEMM round-trips per scalar decode token (0.8B, TP1)

Date: 2026-09-21. Load-insensitive (counters only, no wall-time claims;
box load ~4-5). Base = default path (fused MLP ON, wave ON).

## 1. Counter matrix (TP1 -n8 decode, this session)

| cfg | nk | nw | nr | submit | usm_write | readback |
|-----|----|----|----|--------|-----------|----------|
| base (fused+wave) | 1680 | 1104 | 576 | 86-87ms | ~42ms | ~38ms |
| NOWAVE | 1680 | 1104 | 1104 | 93ms | 43ms | 40ms |
| NOFUSE (host silu-mul) | 1488 | 1488 | 768 | 112ms | 50ms | 53ms |
| L4 | 1968 | 1104 | 576 | 62ms | 39ms | 41ms |

Per-token base: nk=210, nw=138, nr=72. Fused path verified firing
([l4] counters; streams identical).

## 2. Per-token GEMM census (24 layers: 18 LA + 6 full-attn; + lm_head)

LA layer (x18): wave-4 (qkv/z/b/a, 4 launches) + la_out (1) + gate/up
  wave-2 (2, fused path: on-device) + down (fused tail, 1) = 8 launches.
  Host-consumed outputs: mqkv (conv FIR), z/b/a (SSM prep + norm),
  core/normed_core (host norm, non-L4). Fused path already kills the
  gate/up->down seam (saves 2 D2H + host silu-mul + convert + H2D +
  1 wait/layer — the +13% decode win).
FA layer (x6): wave-3 (q/k/v) + o (1) + gate/up (2) + down (1) = 7.
  Host-consumed outputs: qp (split/norm/rope), k/v (norm/rope/cache),
  attn (o GEMM input — host attention, O(pos)), o_raw (rot), mg/mu
  (silu-mul, non-fused), md (rot).
MLP everywhere: covered by fused MLP on TP1 (gate/up->down seam dead).
lm_head (x1/token): run_lm_head_f16, DPAS INT8 path (XMX active here).

## 3. Seam candidates ranked (devact seam adds ZERO launches by design)

### #1 lm_head input (final-norm out -> head GEMM) — REJECTED on scoping
  (2026-09-21, INTFER_TRACE_LMHEAD): kernel ~700us, wall 1.4-4.7ms. The
  wall minus kernel is submit + D2H (0.5MB FP16 out) + host vocab convert
  (248K elems) + wait — and ALL of that stays (host argmax needs float
  logits). A device-norm seam would save only the 2KB H2D + K=1024 convert
  + norm (microseconds) while ADDING a ~25us+ norm submit: wash or loss.
  The audit's "largest single host-side convert" claim was wrong — the
  vocab-sized convert is on the OUTPUT side, which no input seam removes.
  The real lm_head lever (if any) is D2H-side: argmax-on-device returning
  one int (kills 0.5MB D2H + vocab convert), NOT input-side. That needs a
  new kernel + greedy/top-p contract change — scoped, not started.

### #2 FA q/k/v -> norm/rope/cache (x6 layers)
Blocker: consumers are host elementwise (rmsnorm/rope in-place) + cache
memcpy. A seam would need device norm+rope kernels first (new kernels,
new bit-contracts on rope sin/cos) — same cost class as the L5 dead end
(+launches/layer). NOT recommended standalone; only as part of a fused
qkv->prep kernel that also absorbs the attention input staging.
Saves/token if done: 3 converts + 3 H2D + 3 D2H + 3 converts x6 layers.
Costs: >=3 norm/rope submits x6. Net: likely wash or loss (L5 verdict).

### #3 LA qkv/z/b/a outputs -> conv/SSM prep (x18 layers)
Blocker: consumers are host conv FIR + q/k prep + SSM recurrence with
per-layer state. This IS the L5 front chain — measured dead (+144
launches/token, waitblock 5x). NOT recommended. Retained kernels
(fwht_rot_to_half, la_conv_prep) are the building blocks IF a future
fused fwht->GEMM->prep single kernel is built (one launch replacing
~8, not +8).

### #4 o_proj / la_out inputs (attn / normed_core -> GEMM)
attn is host-computed (attention O(pos) stays host); normed_core is
host-normed (non-L4) or device (L4 tail already covers la_out).
o_proj input H2D + convert per FA layer (6x/token) is the residual:
seam needs device-resident attn (device attention — the hard problem).
NOT recommended before device-side attention exists.

### #5 gate/up on TP2/DPAS48 (non-fused paths)
Fused MLP is TP1-only; TP2 pays wave + host silu-mul + down (2 D2H +
convert + H2D + wait per layer x24). A TP2-capable fused silu-mul would
need cross-device act visibility (shards are per-device) — structural
blocker, same reason TP=2 loses ~1.9x overall. NOT recommended.

## 4. Recommendation

NO input-side seam is recommended: #1 measured as wash/loss (see above),
#2/#3/#4 need device consumer kernels (L5 verdict stands), #5 is blocked
on TP2 cross-device visibility. The device-act pattern is EXHAUSTED on the
scalar decode path — fused MLP + L4 cover every GEMM->GEMM edge with no
host math between; all remaining edges have host consumers (conv/SSM/norm/
rope/attention/argmax) that cost more to move than the seam saves.

Remaining levers, in order:
(a) argmax-on-device for lm_head (kills 0.5MB D2H + vocab convert/token;
    new kernel + sampler contract — IMPLEMENTED 2026-09-23, d460eb7,
    default-off INTFER_DEVICE_ARGMAX=1; bit-contract proven on CPU,
    GPU stream-gates OWED — see §6);
(b) serve-side aggregate batching (highest documented value, quiet box);
(c) fused fwht->GEMM->prep single kernel (replaces ~8 launches with 1;
    session-scale build, only if (a) lands first and the ceiling holds).

## 5. Sampler-contract scope for (a) (2026-09-21, code-read, no behavior change)

Producers of `StepOut{best, logits}` (main.cpp): scalar `step()` (:4094),
`prefill_layer_outer()` last-row by default / all-rows via `all_out` (spec
verifier, :5593), serve `step_batch()` every row (:6306, M-row batched
lm_head when M>1).

Consumers split exactly on greedy-vs-sampling:
- Greedy (`temperature<=0 && top_k==0 && top_p<=0`) reads ONLY `.best`:
  `sched_iteration` (:6437), joint (:6541), CLI loop (:9190), spec-verify
  accept loop (`outs[i-1].best == drafts[i]`, :9124). Device-argmax
  returning one int serves all of these — no logits needed.
- Sampling reads the FULL `.logits`: `sample()` (:6708) runs softmax over
  the whole vocab + top-k nth_element + top-p full sort + multinomial with
  the per-request host `mt19937`. Device-argmax does NOT serve sampling
  rows — they keep the full D2H + host convert path.
- Spec decode needs `.best` per verify row (accept test) + the corrected/
  bonus token — greedy-compatible, no logits.

Consequence: device-argmax must be per-row conditional, not a global
switch. In mixed-temp `step_batch` batches, greedy rows skip the D2H
(consume the device int) while sampling rows still read back their full
logits — i.e. per-row masking (or a two-pass: argmax kernel over the
device logits buffer for greedy rows, D2H only sampling rows). The
prefill `all_out` spec path is greedy today (accept test on `.best`
only), so it can ride the int path too; any future sampling spec would
need full logits there. RNG stays host-side, untouched.
Bit-contract: device argmax must tie-break identically to the host
`if(s>bestv)` strict-greater scan (first-max-wins); NaN handling must
match (host strict-`>` ignores NaN — device kernel must too, else
spec-verify accept decisions diverge).

## 6. Device-argmax v1 status (2026-09-23, d460eb7, default-off)

Kernel `lm_head_argmax_tile` (quant_gemm.cpp:1542, decl quant_gemm.hpp:459):
per-row single-WI sequential scan, `best=-1/bestv=-1e30f` init, strict-`>`
compare, global vocab id via `global_col0`. Hooks:
`run_lm_head_best` (scalar `step()`) + `run_lm_head_batch_best` (M-row,
v1 requires ALL rows greedy — mixed-temp falls back to full D2H).
Any false (flag off, trace_lm, null/zero, N/K mismatch, alloc throw,
sampling row, LMHEAD_BATCH=0) keeps the full-D2H path.

Bit-contract PROVEN on CPU (throwaway harness, since deleted):
(a) `dev_half_to_float` == `sycl::half` cast over ALL 65536 half patterns;
(b) kernel scan == host loop on adversarial tiles (all-NaN, ties, NaN
placement, ±inf, ±0.0, subnormals, tile offsets) + 200k random fuzz tiles;
(c) multi-tile (65536 + ragged tail, 248320 vocab) ts-order host reduce ==
flat host loop incl. cross-tile tie -> first tile. Harness quirk found
during proving (NOT a kernel bug): raw-random fill can contain +inf which
legitimately beats the planted max — fill was finite-masked before the
tie-order assertion.

GPU stream-identity gates OWED on the fleet box (dev container has no Arc
L0 device — only CPU indifference verified: --no-gpu -n6 base==flag
279 6511 314 279 5364 314, expected since the flag is GPU-only).
Required before any use: scripts/test_device_argmax.sh (TP1 -n6 8-way
matrix base==flag==HOSTPREP==SSM==SSMxflag==FLAG+SSMxHOSTPREP==
TP2-base==TP2-flag, INTFER_ARGMAX_DBG non-triviality both ways, sampling
count smoke, spec-verify end-to-end, cpu parity + http batch + gpu-optin
10/10). NOTE 2026-09-23: spec-verify rows take the full-D2H path BY DESIGN
(spec-verify rows in prefill_layer_outer/score_row take the full-D2H path BY
DESIGN (StepOut carries full logits; the *_best hooks are never called
there), so the spec gate covers the end-to-end loop only, not verify rows.
UPDATE 2026-09-24 (v2, spec-verify rows ride the scalar hook): this note is
SUPERSEDED — score_row now tries run_lm_head_best per row when the caller
passes all_greedy=true (only the greedy spec-verifier does; use_spec implies
greedy, and verify consumers read .best only — accept test + corrected/bonus
— never .logits). Sequential M=1 calls = same launch pattern as scalar
decode (prefill-safe: the serve poison was batch-hook M>1 work only, and
serve never sets all_greedy). Any hook false falls back per-row (TP2/trace/
CPU decline inside). Gate: new §3b (repetitive prompt guarantees n-gram
verify rounds): spec base==flag streams identical, PARTS 32->40 (+8 = 2
verify rows x 4 tiles) deterministic 40/40, TP2 PARTS 0, TP2 base==flag.
The old §3 (non-repetitive prompt) gates only the scalar fallback loop —
kept as a smoke, §3b is the real verify-row gate.
No wall-time claims (quiet rule).

UPDATE 2026-09-24 (serve crash found + fixed, this box HAS 2x Arc B580):
TP1-only guard (2026-09-23): TP2+DPAS flakes intermittently (dev1 tiles
read +inf, torn partials, load-sensitive ~50% at -n6) — a DPAS-GEMM ->
argmax-link ordering flake; TP2 keeps the stable full-D2H path.
SERVE POISON (2026-09-24): ANY batch-hook (M>1) firing poisons a LATER
request's submit — SIGSEGV inside libze_intel_gpu, 8/8 repro on sequential
greedy-then-sampling AND concurrent mixed batches, crashing on the FIRST
all-greedy prefill chunk (M=17/22) in the mixed case. Bisect exonerated:
q.wait() device sync, DPAS vs non-DPAS GEMM, the argmax submit itself,
the small D2Hs, the unread-bulk-D2H theory, GEMM-only, oversize-capped
allocs. Scalar-hook-only and full-path-only runs survive. Driver/UR root
cause still open. FIX: run_lm_head_batch_best takes decode_only (caller
passes all-rows-non-prefill from step_batch BatchItem.prefill flags) and
declines prefill chunks to the full-D2H path; INTFER_ARGMAX_PREFILL=1
restores prefill firing (bisection only). Verified: sequential + concurrent
mixed + 4x-long-greedy (decode batches M=3/M=4 fire) all survive, no FATAL;
CLI matrix + full gate suite PASS (see below). Gate script fixed (2>&1:
decode lines go to stderr) and now PASSES on this box: TP1/TP2 8-way
279 6511 314 279 3046 13, counter fires, sampling 6/6, spec
279 6511 279 3046 13, cpu parity + http batch + gpu-optin PASS.

RE-RUN 2026-09-24 (HEAD 71e878d, binary current, load ~2.5 — stream-identity
only, no wall-time read): scripts/test_device_argmax.sh DEVICE-ARGMAX TEST:
PASS — TP1/TP2 8-way matrix all 279 6511 314 279 3046 13, [argmax] counter
fires 1 line on flag / 0 off, sampling count 6/6, spec base==flag
279 6511 279 3046 13 with [argmax] firing, cpu parity + http batch +
gpu-optin all PASS. 27B serve prefill-latency check still owed (needs quiet
box load1<2; load spiked to ~5-8 during this session — wall-time gated).
CONFIRM 2026-09-24 ~16:32 (judge re-verification): gate re-run reproduced
in-transcript DEVICE-ARGMAX TEST: PASS (8-way all 279 6511 314 279 3046 13,
counter 1/0, sampling 6/6, spec identical, standard gates PASS); 27B
deferral evidence: load1=6.20 (uptime 6.20/6.10/4.55), model present 19GB,
both B580 normal, no intfer procs — wall-time check stays blocked.

MIXED-TEMP v2 ANALYZED + REJECTED (2026-09-24, no code — submit arithmetic,
load-insensitive): extending the batch hook to mixed batches (greedy rows via
masked argmax kernel — the is_greedy mask field already exists in
LmArgmaxTileParams — + subset-D2H of sampling rows' tile slices) is a wash at
best. Tile sizing (main.cpp:1497: 128MB/(K·2B), K=1024 → 64K rows/tile;
vocab 248320 → T=4 tiles TP1): v2 adds +1 argmax submit + 2 small D2Hs per
tile (+12 submits ≈ +1ms at M=3) to save the greedy share of the bulk D2H
(~1MB ≈ 0.5-1ms) + a few host scans. Net ±0.5ms for HIGH complexity (strided
subset-D2H through the tile scatter, mask plumbing, new bit-contract surface)
AND it expands the batch-hook device surface just fenced for the serve
poison. The all-greedy v1 already captures the common case. Do NOT build;
revisit only if mixed-temp decode batches dominate a profiled workload AND
the poison root cause is found.
SPEC-VERIFY-ROW v2 IMPLEMENTED instead (same session, scalar-hook reuse, zero
new kernels/D2H patterns — see §6 UPDATE above): kills (K+1)×0.5MB D2H per
spec round for logits nobody reads; gate §3b PASS (PARTS 32->40, deterministic,
TP2-silent); full suite + serve-load PASS.

V3 AUDIT (2026-09-24, full-D2H site census — TRACK COMPLETE, nothing left to
build load-insensitively; suite reproduced exactly on HEAD c2be7f5: 8-way
279 6511 314 279 3046 13, PARTS 32/40/40, TP2 0, DEVICE-ARGMAX TEST: PASS):
scoring sites = step() scalar (v1) + step() sampling fallback (must stay) +
score_row all_greedy (v2) + score_row prefill-last-row + step_batch batch_best
(v1) + step_batch batch-full (must stay: sampling rows need logits) +
step_batch per-row scalar (v1, M==1/LMHEAD_BATCH=0 only) + one-time upload
probes (amortized). Devact seam coverage re-checked (fused MLP at all 5 GEMM
sites, L4 la_out scalar + batch): remaining run_linear_batch sites (o_proj,
la_out M-row, down-fallbacks) all have host consumers between — audit B
"exhausted" verdict CONFIRMED, no missed edge. REJECTED with reasons:
(a) prefill last-row scalar hook — value ~0 (1×0.5MB D2H inside GEMM-dominated
prefill) vs un-bisected risk (scalar device work inside bulk prefill-chunk
forwards; the poison involved prefill-chunk device work);
(b) TP2 / decode_only fence re-scoping — both fences guard load-sensitive
crash boundaries (DPAS-link flake ~50%, serve SIGSEGV 8/8); narrowing them is
NOT load-insensitive work;
(c) poison root-cause probes — zero perf payoff even if solved (prefill chunks
keep full D2H they mostly need; prefill is GEMM-dominated), and 7 probes
already failed to narrow it; curiosity only;
(d) per-row-hook-first in declined M>1 batches — duplicates GEMMs already run
by the batch path (strictly worse; the saving variant IS the rejected
mixed-temp v2);
(e) clearing stale logits on device rows — ~0.1ms memcpy saving for a sampler-
contract surface change; fail-loud safety noted, not worth the risk.
REMAINING LEVERS ALL GATED: quiet-box A/B of spec-verify D2H savings +
27B prefill latency (load1<2), TP2-flake repro (needs quiet), fused
fwht->GEMM->prep single kernel (session-scale build), serve aggregate
batching (highest value, quiet box).
