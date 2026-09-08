#!/usr/bin/env python3
# scripts/oracle_qwen35.py
# Gold-reference oracle for the intfer Qwen3.5 text-only forward.
# Loads the real HF model (CPU, bf16) and dumps, for a fixed prompt:
#   - input_ids
#   - per-layer hidden states (after each of the 24 decoder layers, pre-final-norm)
#   - final normed hidden (last position)
#   - logits (last position) and greedy-decoded token ids
# The C++ engine is verified against these.
import sys, json, argparse
import numpy as np
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer

PATH = "/var/home/fra/data/models/safetensors/Qwen3.5-0.8B"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--out", default="test/models/qwen35_oracle.npz")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","float32"])
    args = ap.parse_args()

    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"loading model (CPU {args.dtype})...")
    tok = AutoTokenizer.from_pretrained(PATH)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(PATH, torch_dtype=dt)
    model.eval()
    lm = model.model.language_model

    ids = tok(args.prompt, return_tensors="pt").input_ids
    print("prompt tokens:", ids.tolist())

    # Prefill (full prompt) with per-layer hidden states for debugging.
    with torch.no_grad():
        out = lm(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # (input_embed, layer0..layer23)
    last = out.last_hidden_state  # after final norm, (1,S,H)
    print(f"[diag] len(hs)={len(hs)} hs[23][0] norm={np.linalg.norm(hs[23][0].float().numpy()):.4f} hs[24][0] norm={np.linalg.norm(hs[24][0].float().numpy()):.4f} last norm={np.linalg.norm(last[0].float().numpy()):.4f}")

    # Dump immediately (before any further module calls that may alias hidden-state storage).
    dump = {}
    dump["input_ids"] = ids[0].numpy().astype(np.int64)
    for i in range(1, len(hs)):
        arr = hs[i].detach().cpu().float().contiguous().numpy()[0].astype(np.float32).copy()
        dump[f"layer{i-1}"] = arr  # (S,H)
        if i>=22: print(f"[dump] layer{i-1} norm={np.linalg.norm(arr):.4f}")
    dump["final_norm_hidden"] = last.detach().cpu().float().contiguous().numpy()[0].astype(np.float32).copy()  # (S,H)

    # Save IMMEDIATELY: hidden-state arrays alias torch buffers reused by the generation loop.
    logits = model.lm_head(last).float()  # (1,S,V)
    dump["logits_prefill_last"] = logits[0, -1].detach().numpy().astype(np.float32)  # (V,)
    np.savez(args.out, **dump)
    print("wrote (prefill dump)", args.out)
    chk = np.load(args.out)
    print(f"[post-save] layer22={np.linalg.norm(chk['layer22'][0]):.4f} layer23={np.linalg.norm(chk['layer23'][0]):.4f} inmem dump layer22={np.linalg.norm(dump['layer22']):.4f}")

    # Greedy decode, re-running the full sequence each step (S small).
    generated = []
    cur = ids
    with torch.no_grad():
        for step in range(args.gen):
            o = lm(input_ids=cur, output_hidden_states=False, use_cache=False)
            lh = o.last_hidden_state
            lg = model.lm_head(lh).float()[0, -1]
            nxt = int(torch.argmax(lg).item())
            generated.append(nxt)
            if nxt == tok.eos_token_id:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    dump["generated"] = np.array(generated, dtype=np.int64)
    print("generated:", generated)
    print("decoded:", tok.decode(generated))

    print("generated:", generated)
    print("decoded:", tok.decode(generated))

if __name__ == "__main__":
    main()
