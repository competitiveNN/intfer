#!/usr/bin/env python3
# hooks2.py: capture layer-3 o_proj INPUT (attn*gate) for ALL positions from the
# full forward, and layer-2 output, to compare against oracle and standalone.
import torch, numpy as np
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
PATH="/var/home/fra/data/models/safetensors/Qwen3.5-0.8B"
tok=AutoTokenizer.from_pretrained(PATH)
model=Qwen3_5ForConditionalGeneration.from_pretrained(PATH,torch_dtype=torch.bfloat16).eval()
lm=model.model.language_model
ids=tok("The capital of France is",return_tensors="pt").input_ids
S=ids.shape[1]

oins=[]
lays=[]
def hook_o(mod,inp,out):
    t=inp[0].detach().float()[0]  # (S,2048)
    oins.append(t.numpy().astype(np.float32))
def hook_lay(mod,inp,out):
    t=out.detach().float()[0]  # (S,1024)
    lays.append(t.numpy().astype(np.float32))
lm.layers[3].self_attn.o_proj.register_forward_hook(hook_o)
lm.layers[2].register_forward_hook(hook_lay)
with torch.no_grad():
    o=lm(input_ids=ids,use_cache=False)
oin=np.stack(oins[0])  # (S,2048)
lay2=np.stack(lays[0]) # (S,1024)
oracle=np.load("test/models/qwen35_oracle.npz")
print("pos : o_in_norm  lay2_norm  oracle_layer2_norm")
for p in range(S):
    print(f"  {p}: {np.linalg.norm(oin[p]):.4f}   {np.linalg.norm(lay2[p]):.4f}    {np.linalg.norm(oracle['layer2'][p]):.4f}")
print("oracle layer3 o_in-equivalent? compare o_in to v_concat*gate using oracle layer2:")
