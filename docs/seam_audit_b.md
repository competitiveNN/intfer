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
    new kernel + sampler contract — the only unblocked device-side win);
(b) serve-side aggregate batching (highest documented value, quiet box);
(c) fused fwht->GEMM->prep single kernel (replaces ~8 launches with 1;
    session-scale build, only if (a) lands first and the ceiling holds).
