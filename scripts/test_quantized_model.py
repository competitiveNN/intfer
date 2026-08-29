#!/usr/bin/env python3
# Validate the intfer-quantized Qwen3.5-0.8B against the FP16 reference.
# Loads the real model in transformers, injects our dequantized Linear weights
# (from the .intfer dump), and compares:
#   * last-token logit cosine similarity (FP16 vs quantized)
#   * teacher-forced top-1 next-token agreement over the prompt
#   * greedy generation text from both
# This is an OUTPUT-level test of "does quantization break the model".
import struct, sys
import numpy as np
import torch
import transformers

MODEL = "/var/home/fra/data/models/safetensors/Qwen3.5-0.8B"
DUMP = sys.argv[1] if len(sys.argv) > 1 else "test/models/Qwen3.5-0.8B.dequant.bin"

def load_dump(path):
    d = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"DQMP"
        cnt = struct.unpack("<I", f.read(4))[0]
        for _ in range(cnt):
            nl = struct.unpack("<I", f.read(4))[0]
            name = f.read(nl).decode()
            N, K = struct.unpack("<II", f.read(8))
            buf = f.read(N * K * 4)
            d[name] = np.frombuffer(buf, dtype="<f4").reshape(N, K).copy()
    return d

print("loading dequant dump ...", flush=True)
dq = load_dump(DUMP)
print(f"  {len(dq)} quantized Linear tensors", flush=True)

print("loading reference model ...", flush=True)
tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
ref = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                        trust_remote_code=True)
ref.eval()
print("building quantized variant ...", flush=True)
q = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                      trust_remote_code=True)
q.eval()
sd = q.state_dict()
def map_key(k):
    if k in sd:
        return k
    k2 = k.replace("language_model.", "")
    if k2 in sd:
        return k2
    return None
matched = 0
missing = []
with torch.no_grad():
    for k, arr in dq.items():
        mk = map_key(k)
        if mk:
            sd[mk] = torch.from_numpy(arr).to(sd[mk].dtype)
            matched += 1
        else:
            missing.append(k)
q.load_state_dict(sd, strict=False)
# Re-tie lm_head to the (now quantized) embedding if the model ties weights.
if hasattr(q, "model") and hasattr(q.model, "embed_tokens") and hasattr(q, "lm_head"):
    try:
        q.lm_head.weight = q.model.embed_tokens.weight
    except Exception as e:
        print("  (tie note:", e, ")", flush=True)
print(f"  injected {matched} tensors; not-mapped: {len(missing)}", flush=True)
if missing:
    print("  sample not-mapped:", missing[:5], flush=True)

prompts = [
    "The capital of France is",
    "Quantum computing is a field that",
    "Once upon a time, there was a small village",
]

cos_list, agree_list = [], []
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids
    with torch.no_grad():
        lr = ref(ids).logits[0]            # [T, V]
        lq = q(ids).logits[0]
    # last-token logit cosine
    a = lr[-1].float().numpy(); b = lq[-1].float().numpy()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    cos_list.append(cos)
    # teacher-forced top-1 agreement over positions 1..T-1
    tr = lr[:-1].argmax(-1).numpy(); tq = lq[:-1].argmax(-1).numpy()
    agree = float((tr == tq).mean())
    agree_list.append(agree)
    print(f"\nPROMPT: {p!r}")
    print(f"  last-logit cosine = {cos:.4f} | top-1 agreement = {agree:.4f}")

print("\n=== greedy generation (24 tokens) ===")
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids
    with torch.no_grad():
        gr = ref.generate(ids, max_new_tokens=24, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        gq = q.generate(ids, max_new_tokens=24, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    print(f"\nPROMPT: {p!r}")
    print(f"  FP16 : {tok.decode(gr[0], skip_special_tokens=True)!r}")
    print(f"  QUANT: {tok.decode(gq[0], skip_special_tokens=True)!r}")

print("\n=== SUMMARY ===")
print(f"mean last-logit cosine : {np.mean(cos_list):.4f}")
print(f"mean top-1 agreement    : {np.mean(agree_list):.4f}")
print("VERDICT:", "PASS (quantized output matches FP16)" if np.mean(cos_list) > 0.95
      and np.mean(agree_list) > 0.9 else "REVIEW")
