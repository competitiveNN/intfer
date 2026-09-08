#!/usr/bin/env python3
# hooks_l23.py: capture layer-23 internal components from the oracle and compare
# to the C++ reference forward (computed in numpy from the same unquantized weights).
import argparse, numpy as np, torch, sys, os
sys.path.insert(0, ".")
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

PATH="/var/home/fra/data/models/safetensors/Qwen3.5-0.8B"

def main():
    tok=AutoTokenizer.from_pretrained(PATH)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(PATH, torch_dtype=torch.float32)
    model.eval()
    lm=model.model.language_model
    ids=tok("The capital of France is", return_tensors="pt").input_ids
    # hooks on layer 23 (0-based)
    comps={}
    def make(name):
        def f(m, inp, out):
            if name=="hidden_in":
                comps[name]=inp[0][0].detach().float().numpy().astype(np.float32).copy()
            elif name=="o_in":
                comps[name]=inp[0][0].detach().float().numpy().astype(np.float32).copy()
            elif name=="o_out":
                comps[name]=out[0].detach().float().numpy().astype(np.float32).copy()
            elif name=="mlp_in":
                comps[name]=inp[0][0].detach().float().numpy().astype(np.float32).copy()
            elif name=="mlp_out":
                comps[name]=out[0].detach().float().numpy().astype(np.float32).copy()
        return f
    L=23
    h=lm.layers[L]
    handles=[
        h.register_forward_hook(make("hidden_in")),
        h.self_attn.o_proj.register_forward_hook(lambda m,i,o: make("o_in")(m,i,o)),
        h.self_attn.o_proj.register_forward_hook(lambda m,i,o: make("o_out")(m,i,o)),
        h.mlp.register_forward_hook(lambda m,i,o: make("mlp_in")(m,i,o)),
        h.mlp.register_forward_hook(lambda m,i,o: make("mlp_out")(m,i,o)),
    ]
    with torch.no_grad():
        out=lm(input_ids=ids, output_hidden_states=True, use_cache=False)
    for hh in handles: hh.remove()
    # save
    np.savez("test/models/l23_comps.npz", **{k:comps[k] for k in comps})
    print("captured keys:", list(comps.keys()))
    for k in comps:
        print(f"  {k}: shape={comps[k].shape} norm={np.linalg.norm(comps[k]):.3f}")
    # final hidden after layer 23 (before final norm) for pos0
    hs=out.hidden_states
    print("oracle layer22 pos0 norm:", np.linalg.norm(hs[23][0].float().numpy()))
    print("oracle layer23 pos0 norm:", np.linalg.norm(hs[24][0].float().numpy()))

if __name__=="__main__":
    main()
