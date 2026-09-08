#!/usr/bin/env python3
import torch, numpy as np
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
PATH="/var/home/fra/data/models/safetensors/Qwen3.5-0.8B"
tok=AutoTokenizer.from_pretrained(PATH)
model=Qwen3_5ForConditionalGeneration.from_pretrained(PATH,torch_dtype=torch.bfloat16).eval()
lm=model.model.language_model
oracle=np.load("test/models/qwen35_oracle.npz")
inp=torch.tensor(oracle["layer2"][0],dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).contiguous()  # (1,1,1024)
layer=lm.layers[3]
pos_emb=lm.rotary_emb(inp, torch.tensor([[0]],dtype=torch.long))
with torch.no_grad():
    out=layer.self_attn(hidden_states=inp, position_embeddings=pos_emb, attention_mask=None, past_key_values=None)
    # out[0] is attn_output AFTER o_proj (and gate). We want before o_proj -> hook.
capt={}
layer.self_attn.o_proj.register_forward_hook(lambda m,i,o: capt.update({'in':i[0].detach().float()[0,0].numpy()}))
with torch.no_grad():
    out2=layer.self_attn(hidden_states=inp, position_embeddings=pos_emb, attention_mask=None, past_key_values=None)
real_o_in=capt['in']
hook=np.load("test/models/hook_l3.npz")["o_in"]
print("real self_attn o_proj-input norm:",np.linalg.norm(real_o_in),"hook norm:",np.linalg.norm(hook))
print("cos(real, hook):",float(np.dot(real_o_in,hook)/(np.linalg.norm(real_o_in)*np.linalg.norm(hook))))
# Now reconstruct v_concat*gate in numpy using REAL submodule weights to see what it SHOULD be
# get qkv projections
wQ=layer.self_attn.q_proj.weight.float().detach().numpy()  # (4096,1024)
wV=layer.self_attn.v_proj.weight.float().detach().numpy()  # (512,1024)
x=oracle["layer2"][0].astype(np.float32)
qp=wQ@x; vp=wV@x
gate=sigmoid(qp[2048:]) if False else 1/(1+np.exp(-qp[2048:]))
v=vp.reshape(2,256)
# GQA blocked
vc=np.concatenate([v[0]]*4+[v[1]]*4)
pred=(vc*gate).reshape(-1)
print("numpy replay (blocked) norm:",np.linalg.norm(pred),"cos vs real:",float(np.dot(pred,real_o_in)/(np.linalg.norm(pred)*np.linalg.norm(real_o_in))))
# also try without any gate (raw attn output)
print("raw vc norm:",np.linalg.norm(vc),"cos raw vs real:",float(np.dot(vc,real_o_in)/(np.linalg.norm(vc)*np.linalg.norm(real_o_in))))
