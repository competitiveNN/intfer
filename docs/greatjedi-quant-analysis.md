# Greatjedi Intel Arc Tuned GGUF: Engineering Takeaways for intfer

> **STATUS: comparison BLOCKED.** The Greatjedi Qwen3.8-27B-Intel-Arc-Tuned-GGUF checkpoint
> is NOT present locally. The policy lines below are quoted from the upstream docs and are NOT
> measured on our weights — they have not been converted into a quantization-accuracy claim in
> this repo. The code-level change (INTFER_QUANT_CODEBOOK=fixed) selects intfer's `canonical_codebook()`,
> whose INT2 codes `{-1,-1/3,+1/3,+1}` equal the Greatjedi transposed linear codebook, but that
> is a parity of STORAGE FORMAT, not evidence of Greatjedi's accuracy.

## 1. Model availability
- The Greatjedi model `Greatjedi/Qwen3.8-27B-Intel-Arc-Tuned-GGUF` is **not present** in the local model cache or test fixtures.
- HuggingFace source: `https://huggingface.co/Greatjedi/Qwen3.8-27B-Intel-Arc-Tuned-GGUF`
- The model could not be downloaded due to size (~27B GGUF ≈ 15-20 GB) and container constraints.
- **Guidance:** To inspect the actual quantization parameters, download the model from HuggingFace and inspect its `metadata.gguf` header and weight files. The metadata contains `quant_type`, `codebook_type`, and block size fields.

## 2. Greatjedi quantization scheme (as described in repo)
Greatjedi transquantizes **non-linear I-Quants** (IQ3_S, IQ4_XS, IQ3_XXS) from the `qwen3.8-27b-intel-arc-tuned` GGUF into **linear** codebooks for Intel Arc DPAS/XMX paths. Key characteristics:

- **Quant types:** Q4_K, Q3_K, Q2_K — these are llama.cpp k-quants where the original non-linear IQ codes are linearly re-encoded.
- **Block size:** K-dimension blocks of 256 elements (standard k-quant block size).
- **Codebook structure:** Each group of G=16 or G=32 codes maps to a reconstruction value; the codes themselves are 2-bit indices.
- **Per-group scales:** FP16 scale per group, stored alongside the codes.
- **Linear dequant:** `dequant(code) = codebook_value * scale` — no lattice lookup at runtime.
- **Target paths:** DPAS (Direct Accumulation) and XMX matrix engines on Intel Arc, where the linear codebooks can be consumed directly by the matrix engine MMA units.
- **Relationship to IQ codes:** The original `IQ3_S`, `IQ4_XS`, `IQ3_XXS` formats use non-linear (GPTQ-style) quantization with per-tensor/per-layer scales. Greatjedi's transquantization step converts these into the linear `Q4_K`/`Q3_K`/`Q2_K` format that Intel Arc can consume without runtime dequantization beyond the per-group scale multiply.

## 3b. Measured comparison (0.8B, run via examples/quant_pipeline)

INTFER_QUANT_CODEBOOK=fixed vs learned on the real 0.8B-derived 16-tensor fixture, same
budget (41676 bytes, INT2 stress, ~99.18% used), same microscaling group sizes, same
INT4/INT8 candidate ladder, same total eff bits/weight (3.926 in both runs):

| codebook variant | total weighted error | notes |
|---|---|---|
| learned (intfer default) | 2.260855 | intfer DASH-Q non-uniform symmetric via learn_codebook_tensor |
| fixed (Greatjedi transposed canonical) | 2.603012 | INT2={-1,-1/3,+1/3,+1}, canonical_codebook (no learn) |

Conclusion: on this fixture the FIXED Greatjedi-shaped codebook is WORSE than the LEARNED
intfer codebook (~0.34 absolute WER in favour of learned). So the assumption that simply
adopting Greatjedi's transposed codebook improves quality is FALSIFIED for this data set.
The INTFER_QUANT_CODEBOOK=fixed option stays for interoperability (matching GGUF/Greatjedi
dequant bytes stream-identically) but is not an accuracy win. Operator decision: default ON =
learned.

## 3. Intfer DASH-Q quantization scheme (current implementation)
Intfer's pipeline uses **linear codebooks** with learned codebooks + microscaling, not k-quants. Key characteristics:

- **Quant schemes:** INT2, INT4, INT8, FP16 (canonical).
- **Codebooks:** Learned 4-entry symmetric codebook for INT2: `C = {−c₃, −c₁, +c₁, +c₃}`; fixed codebooks for INT4/INT8.
- **Grouping & microscaling:**
  - Canonical group size G=16, G=32 is budget variant.
  - Per-group FP16 scale `s = group_maxabs / codebook_maxabs`.
  - Per-tensor FP32 global divisor α (one float per tensor), folded into epilogue multiply.
- **Code storage:** 4 codes per byte (INT2), packed `uint8_t`.
- **Activation handling:** W2A8 (INT2 weights × INT8 activations via DPAS), W2A16-OTF (unpack→FP16 in registers via XMX FP16 MMA).
- **Effective bits/weight:** INT2-G16/FP16 = 3.0 eff bits/w (canonical), INT2-G32/FP16 = 2.5.

## 4. Comparison summary

| Aspect | Greatjedi (Q4_K/Q3_K/Q2_K) | intfer DASH-Q (INT2/INT4/INT8) |
|---|---|---|
| **Codebook type** | Fixed k-quant codebooks (transposed from IQ) | Learned 4-entry symmetric INT2 codebook; fixed for INT4/INT8 |
| **Block size** | K=256 per block | K padded to %64 per logical tensor |
| **Scale storage** | Per-group FP16 | Per-group FP16 + per-tensor FP32 global divisor α |
| **Codes per byte** | 2-bit codes, packing varies | 4 codes per byte (INT2) packed `uint8_t` |
| **Dequant at runtime** | code * scale (linear) | code * scale * α (linear + global) |
| **Primary target** | DPAS/XMX matrix engines (linear consume) | DPAS/XMX via W2A8/D2H or XMX FP16 MMA |
| **Relationship to IQ** | Starts from IQ3_S/IQ4_XS/IQ3_XXS (GPTQ-style) | Independent quantization; no IQ dependency |
| **VRAM efficiency** | ~2-4 bits/weight depending on type | INT2 = 3.0 eff bits/w (best accuracy/budget), INT4 = 4.25 eff bits/w |

## 5. Policy recommendation
- **Prefer native Q4_K/INT4 for latency on 24GB cards** (Greatjedi recommendation carries over): INT4 gives 4.25 eff bits/w with reasonable accuracy, fitting the ~18 GB/GPU budget after KV/cache reserve.
- **Use INT2 only when VRAM constraints demand it** (intfer already does this as canonical): 3.0 eff bits/w but with accuracy loss that RCO mitigates via mixed-precision.
- **Do not mix k-quants and learned codebooks in the same pipeline**: Greatjedi's Q4_K is a fixed codebook format, while intfer DASH-Q uses learned codebooks. Converting between them requires re-quantization, not just format swapping.
- **NPU discovery is now available** via `intfer::device::NpuBackend` (OpenVINO 2025.1 C API dlopen), separate from the Greatjedi/DashQ quantization path.

## 6. Unknowns (require model inspection)
Without the actual GGUF metadata, the following cannot be quantified:
- Exact codebook sizes and per-block structures for Q4_K/Q3_K/Q2_K on this model.
- Relative reconstruction error (est_error) of Greatjedi's transquantized weights vs FP16.
- Whether the linear transquantization introduces additional error beyond the original IQ quantization.
- Which Greatjedi quant type (Q4_K vs Q3_K vs Q2_K) gives the best accuracy/latency trade-off on Arc B580.