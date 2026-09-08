#!/usr/bin/env python3
# verify_inproc.py: per-layer cosine between the transformers oracle (in-process hidden_states)
# and the C++ reference forward dump (test/models/qwen35_cpp.bin). No npz round-trip.
import numpy as np, torch, sys
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

PATH="/var/home/fra/data/models/safetensors/Qwen3.5-0.8B"

def main():
    tok=AutoTokenizer.from_pretrained(PATH)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(PATH, torch_dtype=torch.float32); model.eval()
    lm=model.model.language_model
    ids=tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        out=lm(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs=out.hidden_states  # hs[1..24] = after layer 0..23
    S=ids.shape[1]; H=1024; nL=24
    orc=np.zeros((nL,S,H),dtype=np.float64)
    for L in range(nL):
        orc[L]=hs[L+1].float().numpy()[0].astype(np.float64)

    # C++ dump
    f=open("test/models/qwen35_cpp.bin","rb")
    hdr=np.frombuffer(f.read(12),dtype=np.int32)
    cnL,cs,ch=int(hdr[0]),int(hdr[1]),int(hdr[2])
    cpp=np.frombuffer(f.read(),dtype=np.float32).reshape(cnL,cs,ch).astype(np.float64)

    def cos(a,b):
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    print(f"oracle nL={nL} S={S}  cpp nL={cnL} S={cs}")
    worst=(2.0,-1)
    for L in range(nL):
        cs_=[cos(orc[L,p],cpp[L,p]) for p in range(S)]
        if min(cs_)<worst[0]: worst=(min(cs_),L)
        print(f"layer{L:2d}: mincos={min(cs_):.6f} pos0={cs_[0]:.6f} pos{S-1}={cs_[-1]:.6f}")
    print("WORST", worst)

if __name__=="__main__":
    main()
