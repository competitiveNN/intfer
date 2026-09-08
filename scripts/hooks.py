#!/usr/bin/env python3
# hooks.py: capture exact layer-3 self_attn o_proj input (attn*gate) and output
# for pos 0, to compare against my C++ / debug computation.
import torch, numpy as np
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
PATH="/var/home/fra/data/models/safetensors/Qwen3.5-0.8B"
tok=AutoTokenizer.from_pretrained(PATH)
model=Qwen3_5ForConditionalGeneration.from_pretrained(PATH,torch_dtype=torch.bfloat16).eval()
lm=model.model.language_model
ids=tok("The capital of France is",return_tensors="pt").input_ids

capt={}
def hook_o(mod,inp,out):
    capt["o_in"]=inp[0].detach().float()[0,0].numpy().astype(np.float32)
    capt["o_out"]=out.detach().float()[0,0].numpy().astype(np.float32)
lm.layers[3].self_attn.o_proj.register_forward_hook(hook_o)
def hook_q(mod,inp,out):
    capt["q_out"]=out.detach().float()[0,0].numpy().astype(np.float32)  # (4096,)
lm.layers[3].self_attn.q_proj.register_forward_hook(hook_q)
def hook_v(mod,inp,out):
    capt["v_out"]=out.detach().float()[0,0].numpy().astype(np.float32)  # (512,)
lm.layers[3].self_attn.v_proj.register_forward_hook(hook_v)
def hook_lay(mod,inp,out):
    capt["layer3_out"]=out.detach().float()[0,0].numpy().astype(np.float32)
lm.layers[3].register_forward_hook(hook_lay)

with torch.no_grad():
    o=lm(input_ids=ids,use_cache=False)
np.savez("test/models/hook_l3.npz", o_in=capt["o_in"], o_out=capt["o_out"], layer3_out=capt["layer3_out"], q_out=capt["q_out"], v_out=capt["v_out"])
print("saved test/models/hook_l3.npz : o_in",np.linalg.norm(capt["o_in"]),"o_out",np.linalg.norm(capt["o_out"]),"q",np.linalg.norm(capt["q_out"]),"v",np.linalg.norm(capt["v_out"]))
