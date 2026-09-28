# Next plan: from ~27 tok/s TP1 toward 40 tok/s (0.8B decode)

Date: 2026-09-26. HEAD `34d3c54` (argmax 27B no-op check; prefix-cache
quiet-box complete; argmax scalar wall-time = measured loss).
Box contended (load ~19) — wall-time A/B gated on quiet box (load1<2).
Load-insensitive work (stream-identity + code) can proceed now.

## 1. Current ceiling (measured)

- Single-stream ceiling TP1: ~27 tok/s (fused MLP, quiet box).
- INT2 table-hoist PROMOTED 2026-09-28 (default-ON, kill switch
  INTFER_NO_INT2_TAB=1): quiet-box 27B TP=2 decode 3 rounds BASE vs TAB =
  2.63->2.78 (+5.7%), 2.67->2.83 (+6.0%), 2.68->2.82 (+5.2%) tok/s;
  kernel bucket (load-immune) 18856->16444ms (-12.8% R1); submit/transfer
  buckets unchanged; nk/nw/nr identical; streams identical TP1/TP2 +
  opt-out; cpu parity + http batch + gpu-optin 10/10 PASS.
  New 27B TP=2 decode ceiling: ~2.8 tok/s (was ~2.6).
- This session, contended box TP1 -n8: 29.79 tok/s => 33.6 ms/token.
  Per-token buckets: submit 9.6 / kernel 9.0 / waitblock 8.8 /
  usm_write 4.8 / readback 4.5 ms. nk=1680, nw=1104, nr=576 over 9 toks
  (~187 launches / ~64 readbacks per token).
- Documented 40 tok/s recipe (AGENTS.md): device-act seams kill transfers
  (->35 tok/s), batched L0 multi-launch halves submit (->31+), both = 45 tok/s.
  Engine-invasive; needs quiet box. Not attempted under load.

## 2. Cost model per run_linear_tp call (src/engine/engine.cpp ~L999-1109)

Every GEMM round-trip pays, in order:
1. host FP32->FP16 convert (`float_to_half_batch`, `copy` bucket)
2. H2D memcpy into ws act (`usm_write` bucket)
3. UR submit (`submit` bucket)
4. blocking event wait (`waitblock` bucket)
5. D2H readback + FP16->FP32 convert (`readback` bucket)

Proven pattern that kills 1+2+5 between chained GEMMs: the device-act seam
(`Engine::run_linear_tp_devact`, engine.hpp:106, engine.cpp:1260) —
fused MLP (gate/up -> down) and L4 (norm -> la_out) both use it.

## 3. Dead ends — do not retry as-is

- Faster submit: native L0 immediate (27.6 vs 25.2us, same floor),
  SYCL graphs (19.7 vs 11.9us, slower), no-prof queue (neutral).
  Lever is launch/sync COUNT, not cost.
- L5-front scalar as a whole: +144 launches/token, waitblock 5x.
  Salvageable pieces only (fwht_rot_to_half / half_to_float /
  la_conv_prep kernels retained for future fused kernels).
- DPAS48 default-ON at any shape; device-side act-quant alone;
  standalone INTFER_SSM_GPU=1 (still -9%, default-off).
- OpenVINO evaluated, NOT adopted (2026-09-18 spike): OV 2026.3 sees both
  B580s, but FP16 matmul M=361 2.6ms (~our EU kernel, no XMX unlock) and
  M=1 1.2ms vs our ~50us (20x worse, framework overhead). Native SYCL stays;
  XMX win must come from our own DPAS kernels with device-resident act-quant.

## 4. Ranked next steps

### Step 1 — Extend the device-act seam to remaining GEMM round-trips
Highest value, load-insensitive (stream-gated). For each GEMM whose output
feeds only the next GEMM's input with no host math between, keep it
device-resident via the existing `run_linear_tp_devact` seam (no new
launches added — unlike L5). Candidates to audit: qkv/z/b/a GEMM outputs
on the LA path, gate/up outputs on the non-fused (TP2/DPAS48) paths,
lm_head input (final-norm out -> head GEMM).
Deliverable: ranked seam-extension list with per-seam launch-count deltas
(nk/nw/nr), then implement top seam behind a default-off env flag,
stream-gated base==flag on TP1/TP2.

### Step 2 — Reduce sync count (not submit cost)
Wave grouping (convert+submit back-to-back, one wait per wave) is done.
Next: cross-layer wait collapsing wherever a consumer chain stays on-device
(the devact seam is the prerequisite — same work as Step 1 amortizes both).
No L0 scaffolding: raw lists measured same-floor; the multi-submit-single-wait
chain on in-order UR queues IS the achievable amortization.

### Step 3 — Serve-side aggregate throughput (the actual 40 tok/s path)
Single-stream ~27 tok/s ceiling stands; 40 tok/s single-stream needs Step 1
+ Step 2 together, quiet-box verified. The documented highest-value lever is
cross-request batching (--serve M-row batching + HTTP batching; C=8/MT=64
probe 9.02 tok/s aggregate). Requires quiet box for any wall-time claim.

## 5. Task options (pick one)

- A: Implement — scope + build the next device-act seam (Step 1 top item),
  stream-gated, default-off, bisectable commit. Do now (load-insensitive).
  NOTE 2026-09-23: device-argmax v1 (audit B lever (a)) is implemented,
  d460eb7, default-off INTFER_DEVICE_ARGMAX=1 — bit-contract proven on CPU
  (65536-pattern convert + adversarial/fuzz tiles + multi-tile reduce, ALL
  PASS), GPU stream-gates gated + fixed on THIS box 2026-09-24 (TP1-only
  guard, serve-poison decode_only fence, gate §3b PARTS 32->40; full suite
  + serve-load PASS). WALL-TIME CLOSED 2026-09-26 (e1e669b, quiet box):
  M=1 scalar is a MEASURED LOSS (OFF 26.5 vs ON 15.1 tok/s, -43%, buckets
  identical — GEMM re-run + 4 waits/tile + serial 62K scan). 27B serve is
  a silent no-op by guard (34d3c54: paired 4.9s vs 4.9s, zero [argmax]
  lines). Keep default-off; never promote v1. Only redemption path: fused
  GEMM+argmax single kernel (session-scale; TP2/poison fences stand).
  Full story: docs/seam_audit_b.md tail.
  UPDATE 2026-09-24 (v2): spec-verify rows ride the scalar hook
  (prefill_layer_outer all_greedy param, spec site only; serve untouched):
  gate §3b PASS (repetitive prompt, PARTS 32->40 deterministic, TP2-silent),
  full suite + serve-load PASS. Mixed-temp batch v2 analyzed + REJECTED
  (wash: +12 submits vs ~1MB D2H at T=4 tiles; expands fenced poison
  surface — see seam_audit_b.md tail).
- B: Audit only — ranked seam-extension list with nk/nw/nr deltas, no code.
- C: Serve-side — batching/scheduler work (wall-time gated on quiet box).

## 6. Standing rules

- Stream-identity gates before any perf claim: TP1 -n6
  base==flag==HOSTPREP==SSM==SSMxflag==TP2-base==TP2-flag,
  plus cpu parity + gpu-optin + http batch.
- Quiet rule: no tok/s quotes unless box load1<2 sustained.
- Bisectable commits; no /tmp litter (RAM is precious).
