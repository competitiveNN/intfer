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

### #1 lm_head input (final-norm out -> head GEMM) — BEST ratio
Today: host rmsnorm -> FP32 fn -> FP16 convert + H2D (K=1024, tiny) ->
  DPAS GEMM -> D2H + convert (N=248320, HUGE: ~0.5MB FP16 out/token).
The output side dominates: vocab-sized readback + half_to_float_batch
every token. A devact head GEMM fed from a device-resident norm kernel
kills the vocab-sized D2H-side convert; input H2D is only 2KB.
Saves/token: 1 H2D (2KB) + 1 D2H-side host convert (~248K elems) +
  host norm. Costs: 1 norm submit. No new math (rmsnorm exact order).
Risk: LOW (same pattern as L4 norm kernel; lm_head DPAS path already
  takes FP16 acts). Note: readback of logits stays (argmax is host).

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

Implement #1 (device-resident final-norm -> devact lm_head) as the next
seam: only candidate with (a) no new consumer kernels beyond one exact
rmsnorm, (b) zero added GEMM launches, (c) the largest single host-side
convert killed per token (vocab-sized). Behind a default-off env flag,
stream-gated base==flag TP1/TP2, same contract as L4 (trace forces host).
Wall-time A/B on quiet box only.

Deliberately NOT recommended: #2/#3/#4 without fused consumer kernels
(L5 verdict stands), #5 (TP2 cross-device blocker).
