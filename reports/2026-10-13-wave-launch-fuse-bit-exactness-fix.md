# Wave-Launch-Fuse Bit-Exactness Fix (2026-10-13)

## Problem
`INTFER_WAVE_LAUNCH_FUSE` fuses a wave's N independent GEMMs into one launch per device, removing N-1 submits + N-1 waits + N-1 D2H readbacks per device per wave. The gate (`test_tp2_27b.sh`) asserts token-stream identity (`mr_streams_agree`) between the fused and base paths — but the fused kernel's quantized-weight branch diverged from the base tiled kernel:

**Base** (`quant_gemm_forward_usm`, ~line 321):
```cpp
wval = s * canon_value(bits, code);
val = to_half_bits(wval);                 // FP16 pre-round
sW[e] = val;
...
acc += av * from_half_bits(val);          // av * (half)(s * canon_value)
```

**Fused** (`quant_gemm_forward_usm_wave`, line 2293 — before fix):
```cpp
acc += av * scale * canon_value(bits, code);   // no pre-round
```

For INT2/INT4/INT8, `canon_value` divides by 3/15/255 (not powers of two), so `scale * canon_value` is generically NOT exactly representable in FP16. The two accumulation orders round differently per term, producing logits that differ below the token oracle's resolution (argmax stable) but are not bit-identical.

## Fix
At `quant_gemm.cpp:2293`, the fused quantized branch now pre-rounds the weight term identically to the base:

```cpp
const float wval = scale * canon_value(bits, code);
const float wv = from_half_bits(to_half_bits(wval));
acc += av * wv;
```

## Why it is bit-exact
1. **Group width identity**: `Gw_i` (fused, line 2269) == `Gw` (base, line 372) — both derive from the item's `qcfg.group_size`.
2. **Scale indexing identity**: Both read `scale[row * ngroups + g]` where `g = kg / Gw`. Base: `wsk[ng * num_groups_w + g]` (line 304); Fused: `sptr_row[g]` with `sptr_row = sptr + n * ngroups_i` (line 2271). Same layout, same per-group scale.
3. **Summation order**: Both sum `kg` in strictly increasing order (base: k-blocks of 64, inner `kk=0..63`; fused: groups of `Gw`, inner `kg=kg0..kg1`). Same element sequence → same FP32 accumulation order.
4. **Activation identity**: Both expand the same FP16 activation bits via `from_half_bits(act[act_off+kg])` (fused) / `from_half_bits(sA[...])` (base).
5. **Per-term rounding**: Both apply `from_half_bits(to_half_bits(s_g * canon_value(code)))` per term before `av * wv`.
6. **Final rounding**: Both `to_half_bits(acc)` the final FP32 sum to FP16 output.

The FP16-weight branch (`bits >= 16`) needed no change — both paths expand raw half bits and `canon_value` is the identity there.

## Gate impact
The wave-launch-fuse arm in `test_tp2_27b.sh` compares token streams (`mr_streams_agree`). Before the fix, this passed via argmax-stability (divergence below oracle resolution). After the fix, the kernel is bit-exact to base, so the stream match is by true identity. The self-check predicates (`wf_wave_known`, `wf_wave_fused`, announcement line) are unchanged — they certify the LAUNCH COUNT, not the numerics.

## Status
- **Code change**: Applied (`quant_gemm.cpp:2293`).
- **Gate verification**: PENDING — next 22:00+ operator run required to re-derive `mr_streams_agree` PASS on real 27B `--tp 2` decode.
- **Flag state**: `INTFER_WAVE_LAUNCH_FUSE` remains default-OFF until the gate run confirms bit-exact PASS.
- **Promotion blocker**: The only remaining blocker is the 22:00 gate verification; the numerics argument is complete at the source level.

## Files changed
- `src/gemm/quant_gemm.cpp` (kernel fix)
- `scripts/test_tp2_27b.sh` (audit annotation updated)
- `reports/2026-10-13-wave-launch-fuse-bit-exactness-fix.md` (this report)