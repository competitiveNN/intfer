# PLAN — intfer optimization backlog (2× Intel Arc Pro B580, TP=2)

Full historical record in `HISTORY.md` and exploration details in `docs/`.
**Objective (standing):** 27B `--tp 2`. Measured state: derive via
`scripts/model/model_byte_roofline.sh` or `INTFER_PROFILE_GPU` runs —
never quote from prose.

## Active priorities (biggest gain first)
1. Prefill kernel fusion: gate+up single launch + silumul (shipped);
   SSM fold (INTFER_SSM_FUSE); wave-launch fuse (INTFER_WAVE_LAUNCH_FUSE);
   M-row wave alignment. Launch-count reduction measured on nk/nw/nr & host buckets.
2. Decode launch-count: submit/wait reduction under --tp 2 via wave fusion and
   geometry knobs (INTFER_DECODE1_WG, INTFER_LA_CONV_PREP_WG, INTFER_XMX_WG).
3. Intra-device geometry knobs (UNMEASURED — operator A/B on counters only).
4. NPU path: host-USM doorbell probe (`examples/uncached_doorbell_probe`), Draft
   Source seam (`INTFER_DRAFT_SOURCE`, `INTFER_DRAFT_ASYNC`) for NPU-style draft.
   Hardware NPU via OpenVINO is NOT connected (lib in /run/host/usr/lib64 only).
5. Quantization: Greatjedi Arc-Tuned GGUF comparison BLOCKED (checkpoint absent).
   INTFER_QUANT_CODEBOOK/"fixed" and INTFER_QUANT_GROUP now selectable.
   Re-quantization effort depends on acquiring the checkpoint.

## Standing constraints
- Decide on load-insensitive counters (`nk`/`nw`/`nr` + host buckets), never tok/s.
- Retire wall-time benchmark queue on contended box; counters immune to load1.
- No 27B loads in agent sessions; operator runs A/B.

## Next steps not started
- Obtain Greatjedi checkpoint (HuggingFace) for quant comparison, OR remove the claim.
- Wire OpenVINO NPU (`dlopen libopenvino_c.so.2520` probe) as a decode/prefill device.
- Operator A/B of shipped flags (knob_sweep.sh) measuring counters before/after.
