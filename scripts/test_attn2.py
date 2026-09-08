#!/usr/bin/env python3
import torch, numpy as np
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
PATH="/var/home/fra/data/models/safetensors/Qwen3.5-0.8B"
tok=AutoTokenizer.from_pretrained(PATH)
model=Qwen3_5ForConditionalGeometry if False else Qwen3_5ForConditionalGeneration
model=Qwen3_5ForConditionalGeneration.from_pretrained(PATH,torch_dtype=torch.bfloat16).eval()
lm=model.model.language_model
oracle=np.load("test/models/qwen35_oracle.npz")
hook=np.load("test/models/hook_l3.npz")
layer=lm.layers[3]
# normed input to self_attn = input_layernorm(layer2)
in_ln=np.load  # not needed; use HF module
ly0=layer.input_layernorm
x0=torch.tensor(oracle["layer2"][0],dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).contiguous()
xn=ly0(x0)  # (1,1,1024) normed
pos_emb=lm.rotary_emb(xn, torch.tensor([[0]],dtype=torch.long))
capt={}
layer.self_attn.o_proj.register_forward_hook(lambda m,i,o: capt.update({'in':i[0].detach().float()[0,0].numpy()}))
with torch.no_grad():
    out=layer.self_attn(hidden_states=xn, position_embeddings=pos_emb, attention_mask=None, past_key_values=None)
print("normed-input self_attn o_in norm:",np.linalg.norm(capt['in']),"  hook o_in norm:",np.linalg.norm(hook['o_in']))
print("cos:",float(np.dot(capt['in'],hook['o_in'])/(np.linalg.norm(capt['in'])*np.linalg.norm(hook['o_in']))))
# also feed raw input
capt2={}
layer.self_attn.o_proj.register_forward_hook(lambda m,i,o: capt2.update({'in':i[0].detach().float()[0,0].numpy()}))
with torch.no_grad():
    out2=layer.self_attn(hidden_states=x0, position_embeddings=pos_emb, attention_mask=None, past_key_values=None)
print("RAW-input self_attn o_in norm:",np.linalg.norm(capt2['in']))
