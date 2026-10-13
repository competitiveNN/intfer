# Gate-arm vacuity audit + kernel crash audit (2026-10-13)

Two real defects found by code reading (no model loaded), both in arms the
gate runs on a box with only 2 GPUs and a 128MB VRAM pressure window.
Neither was caught by the self-checks because the self-checks used
synthetic fixtures, never the engine's actual dispatch decisions.

## 1. INTFER_SSM_FUSE arm (a) -- impossible PRESENCE assertion (FIXED, 0129f4f)

Arm (a) ran `INTFER_SSM_FUSE=1` WITHOUT `INTFER_SSM_GPU=1` and asserted
that `[ssm-fuse]` appears on stderr. The announce lives inside
`gemm::ssm_forward` / `gemm::ssm_forward_chain`
(`quant_gemm.cpp:3263/3346`), reachable ONLY on the device recurrence path,
which is gated by `ssm_gpu_allowed()` (`main.cpp:2749` -- requires
`INTFER_SSM_GPU` set and `INTFER_DPAS48!=1`). Without that flag the SSM
recurrence is the HOST loop (`if(!ssm_done)` at main.cpp:5429/6492/7268,
pure C++ math over `S[]`), which never calls `gemm::ssm_forward` at all.

Consequence: the announce CANNOT print on arm (a)'s run, so the assertion
was unsatisfiable and the gate was structurally RED on every box. Measured
on this 2-GPU box: `GATE FAIL: TP=1 INTFER_SSM_FUSE=1: no [ssm-fuse] line
-- the fusion never took, stream match vacuous`.

Root cause: the arm's own comment claimed "the host path (main.cpp:5416-5425
...) calls gemm::ssm_forward". Those sites ARE the device path (guarded by
`ssm_gpu_allowed()` at main.cpp:5416). The host path is the `if(!ssm_done)`
branch BELOW them -- pure C++, no `gemm::` call.

Fix: invert the polarity. The arm now asserts the line is ABSENT on a
host-recurrence run (presence would mean the device kernel ran on a path
the caller never reached -- a defect). Reachability stays on arm (d),
which runs with `INTFER_SSM_GPU=1` and correctly asserts PRESENCE; that arm
printed the line on this box, proving the fused body is live on the device
path.

Self-check strengthened (b2)/(c): (b2) a fixture that carries the line
must FIRED the predicate (the refusal mechanism is live, not a dead
always-false grep), and (c) an empty log is ACCEPT silently (the empty
response is not "no announcement" treated as a match).

## 2. INTFER_TILE_SCHED self-check (b) -- fixture-content tautology (FIXED, c340e66)

The self-check built its "scheduler ignored" fixture by `sed`-rewriting the
real announcement's `INTFER_TILE_SCHED=on` to `=off`, then ran:

```
if grep -aq "INTFER_TILE_SCHED=off" "$TS_NOT_ERR"; then FAIL; fi
```

The fixture carries `off` BY CONSTRUCTION (the sed just built it that way),
so the grep always matches and `PASS=0` fires unconditionally. The arm
was structurally red on HEAD regardless of the engine's behaviour -- the
self-check certified its own fixture content, not the arm's verdict.

Fix (matches the wave-launch 8b shape, the correct precedent): apply the
arm's own predicate -- `grep "$TS_ANN_FULL"` (=`on`) -- to the fixture and
require REFUSAL (the grep must NOT match, because the fixture says `off`).
Now the self-check fails only if the arm would have ACCEPTED a
scheduler-ignored run.

## 3. INTFER_WAVE_LAUNCH_FUSE -- bit-exactness contract falsified (DEFERRED, NOT FIXED)

The gate runs the real arm at prefill (M=361) on the 0.8B model and asserts
stream identity vs base. The run PASSED on this box. The audit showed that
PASS is luck, not a contract:

* Base tiled kernel (`quant_gemm_forward_usm`, `quant_gemm.cpp:321-334`):
  pre-rounds the dequantized weight to FP16 in shared memory
  `wval = s * canon_value(bits, code); val = to_half_bits(wval); sW[e]=val;`
  then `acc += av * wv` -- the multiply is on the FP16-rounded weight.
* Fused wave kernel (`quant_gemm.cpp:2293`): folds the scale into the FMA
  without that rounding:
  `acc += av * scale * canon_value(bits, code)`.

`canon_value` for INT2/INT4/INT8 is `(2j-1)/3`, `(2j-15)/15`,
`(2j-255)/255` respectively -- none is a power of two, so
`scale * canon` is generically NOT FP16-exact and the two orders round
differently per term. The 0.8B gate fixture's waves carry the SAME scheme
mix as the 27B's (LA wave-4: FP16; MLP gate/up wave-2: INT4 -- parsed from
the INTFER1 header 2026-10-13), so the divergence is LIVE on the gate's own
fixture.

For FP16-scheme items (bits >= 16): BOTH paths read raw half bits and both
ignore the scale (base `:306-311`, fused `:2279-2285`) -- there the claim
holds and they are bit-exact. So the LA wave (FP16) is fine; the MLP
gate/up wave (INT4) is not.

Promotion gate (the fix before any default-ON): either (a) round the fused
path's weight term with `to_half_bits(...)` to match base exactly --
restoring bit-identity for every scheme, or (b) convert this arm to a
tolerance contract like ONEMKL's (never "bitwise"). The flag stays
default-OFF and is annotated as reaching-tolerance-gated in the gate
header. This comment is the contract of record until that lands.

## 4. Kernel crash audit (bounds verification, by reading)

`ssm_forward_kernel` / `ssm_forward_chain_kernel` (quant_gemm.cpp:3275-
3397): `block >= n_blocks` guard covers the rounded-up global; all index
maps verified against the packed row layout
`[qn(nk*kd) | kn(nk*kd) | v(nv*vd) | g(nv) | b(nv)]`, row stride
`rs = 2*nk*kd + nv*vd + 2*nv`; `core_all` indexed `m*n_v + hh)*vd + c`
<= `M*n_v*vd - 1`; `S` sized `kd*vd` by `hh*kd*vd`. In-bounds.

`quant_gemm_forward_usm_wave` (quant_gemm.cpp:2200): `start[17]` holds
under the engine's `live[16]` cap (`engine.cpp:2243-2245`, `nl < 16`);
`gid >= total` guards rounding; per-item `Ni` is the LOCAL column count
(engine.cpp:2093, the TP=2 2x-OOB fix is present); sub-8-bit and FP16
code indexing within shard sizes; `outp[m*Ni+n]` within `M*Nl_local`.
In-bounds.

`run_linear_wave_fused` Phase 3/4 (engine.cpp:2118-2164): the previously-
fatal stale-output defect is present and correct -- the fused path waits
its own events and drains the device work + readback itself rather than
relying on the caller's default-path Phase 3/4 (which the caller `return`s
around). SAFE.

`quant_gemm_dpas48_tiled` (E4): reviewed at `quant_gemm.cpp:3298-3327`
earlier (HISTORY.md) -- bounds and the 2-D-block cooperative layout
unchanged from the audited commit; no new crash vector.

`act_quant_ga64_dev` (quant_gemm.cpp:2575): `m < M` and `g < nGA`
guard; `k0 = g*GA`, `k1 = min(k0+GA, K)`; reads `x[m*K + kg]` and
`scale_inv[m * (K/GA... )]...` -- within M*K and M*nGA. SAFE.

`onemkl_probe` (examples/onemkl_probe/main.cpp): `fread(b,1,511,f)`
with `b[511]=0` is bounded; all three USM allocations freed on every
exit path including the freopen-failure branch. Minor robustness wart:
`freopen("/dev/tty","w",stderr)` at line 251 can fail under a pipe
(leaves stderr pointed at the capture file) -- observable, not a crash.

## Summary

2 gate-structural defects fixed (both made arms red-on-HEAD by construction;
the fixes are provable by reading, no model needed). 1 contract defect found
and deferred (wave-launch-fuse FP16 pre-rounding) -- flagged rather than
silenced, per audit #3 ("a guard whose stated rationale does not apply is
worse than none"). 5 kernels audited bounds-safe by code reading; 0 crash
vectors.
