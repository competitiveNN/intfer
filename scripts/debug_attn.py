#!/usr/bin/env python3
# debug_attn.py: replay layer-3 full-attention (pos 0) with my exact formulas,
# compare to oracle layer3[pos0]. Toggle ROPE / GATE to find the culprit.
import numpy as np, json, sys
d="/var/home/fra/data/models/safetensors/Qwen3.5-0.8B"
import os
idx=json.load(open(d+"/model.safetensors.index.json"))["weight_map"]
# memory-map the single shard
import mmap
fp=open(d+"/model.safetensors-00001-of-00001.safetensors","rb")
mm=mmap.mmap(fp.fileno(),0,access=mmap.ACCESS_READ)
hdr=json.loads(mm[8:8+int.from_bytes(mm[:8],'little')])
base=8+int.from_bytes(mm[:8],'little')
def rd(name):
    meta=hdr[name]
    assert meta["dtype"]=="BF16", meta["dtype"]
    off=meta["data_offsets"][0]; ln=meta["data_offsets"][1]-off
    raw=np.frombuffer(mm[base+off:base+off+ln],dtype=np.uint16).astype(np.uint32)
    return raw.astype(np.float32).view(np.float32) if False else raw.astype(np.float32)  # bf16->fp32 approx
# bf16 decode properly
def rd2(name):
    meta=hdr[name]
    off=meta["data_offsets"][0]; ln=meta["data_offsets"][1]-off
    raw=np.frombuffer(mm[base+off:base+off+ln],dtype=np.uint16)
    # bf16 to fp32
    u=raw.astype(np.uint32)
    f=(u<<16).view(np.float32)
    return f.reshape(meta["shape"])

H=1024
oracle=np.load("test/models/qwen35_oracle.npz")
inp=oracle["layer2"][0].astype(np.float64)   # hidden after layer2, pos0 (input to layer3)
tgt=oracle["layer3"][0].astype(np.float64)   # oracle layer3 pos0

p="model.language_model.layers.3."
in_ln=rd2(p+"input_layernorm.weight")
post_ln=rd2(p+"post_attention_layernorm.weight")
q_proj=rd2(p+"self_attn.q_proj.weight")  # (4096,1024)
k_proj=rd2(p+"self_attn.k_proj.weight")  # (512,1024)
v_proj=rd2(p+"self_attn.v_proj.weight")  # (512,1024)
o_proj=rd2(p+"self_attn.o_proj.weight")  # (1024,2048)
q_norm_w=rd2(p+"self_attn.q_norm.weight")# (256,)
k_norm_w=rd2(p+"self_attn.k_norm.weight")# (256,)
mg=rd2(p+"mlp.gate_proj.weight"); mu=rd2(p+"mlp.up_proj.weight"); md=rd2(p+"mlp.down_proj.weight")

USE_ROPE = "--norope" not in sys.argv
USE_GATE = "--nogate" not in sys.argv
ROTARY_DIM=64; THETA=1e7

def rms(x,w):
    v=np.mean(x*x); return x/np.sqrt(v+1e-6)*(1+w)
def rope(h,pos):
    out=h.copy()
    inv=1.0/THETA**(np.arange(0,ROTARY_DIM,2)/ROTARY_DIM)
    for dd in range(ROTARY_DIM//2):
        f=inv[dd]*pos; c=np.cos(f); s=np.sin(f)
        a=h[dd]; b=h[dd+32]
        out[dd]=a*c-b*s; out[dd+32]=a*s+b*c
    return out

h=inp.copy()
x=rms(h,in_ln)
qp=q_proj@x   # (4096,)
qp2=qp.reshape(8,512)  # per-head interleaved: [head, :256]=query, [head,256:]=gate
qpart=qp2[:, :256].copy(); gate=qp2[:, 256:].copy()
k=k_proj@x; k=k.reshape(2,256)
v=v_proj@x; v=v.reshape(2,256)
qn=np.array([rms(qpart[hh],q_norm_w) for hh in range(8)])
kn=np.array([rms(k[hh],k_norm_w) for hh in range(2)])
if USE_ROPE:
    qn=np.array([rope(qn[hh],0) for hh in range(8)])
    kn=np.array([rope(kn[hh],0) for hh in range(2)])
# attention pos0: only t=0
scale=1/np.sqrt(256)
attn=np.zeros((8,256))
for hh in range(8):
    kvh=hh//4
    src = v
    if "--useq" in sys.argv: src=qn
    if "--usek" in sys.argv: src=kn
    if "--noattn" in sys.argv:
        attn[hh]=0; continue
    for dd in range(256): attn[hh,dd]=src[kvh,dd]   # softmax over 1
if USE_GATE and "--nogate" not in sys.argv:
    attn=attn*(1/(1+np.exp(-gate)))
attn=attn.reshape(-1)
o=o_proj@attn
h=h+o
# mlp
x=rms(h,post_ln)
gx=mg@x; g=gx*(1/(1+np.exp(-gx))); u=mu@x
a=g*u  # silu(g)*u
mlp=md@a
h=h+mlp
cos=float(np.dot(h,tgt)/(np.linalg.norm(h)*np.linalg.norm(tgt)))
hook=np.load("test/models/hook_l3.npz")
o_in=hook["o_in"].astype(np.float64); o_out=hook["o_out"].astype(np.float64)
attn2d=attn.reshape(8,256)
my_o_in=(attn2d*(1/(1+np.exp(-gate)))).reshape(-1) if USE_GATE else attn2d.reshape(-1)
my_o_out=o_proj@my_o_in
def c(a,b):
    a=a.reshape(-1); b=b.reshape(-1)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
print(f"cos(layer3 pos0)={cos:.5f}")
print(f"  qp cos={c(qp,hook['q_out']):.5f} |mine|={np.linalg.norm(qp):.3f} |hook|={np.linalg.norm(hook['q_out']):.3f}")
print(f"  v  cos={c(v.reshape(-1),hook['v_out']):.5f} |mine|={np.linalg.norm(v):.3f} |hook|={np.linalg.norm(hook['v_out']):.3f}")
vc=np.concatenate([v[0]]*4+[v[1]]*4)  # v_concat blocked (2048,)
# true gate = o_in / v_concat (where both are correct)
trueg=o_in.reshape(8,256)/vc.reshape(8,256)
mask=np.abs(vc.reshape(8,256))>0.05
tg=trueg.copy(); tg[~mask]=np.nan
print(f"  TRUE GATE mean={np.nanmean(tg):.4f} min={np.nanmin(tg):.4f} max={np.nanmax(tg):.4f}")
# compare to sigmoid of qp halves
gA=1/(1+np.exp(-qp2[:, :256])); gB=1/(1+np.exp(-qp2[:, 256:]))
print(f"  cos(true_gate, sigmoid(qp[:2048]))={c(trueg, gA):.4f}")
print(f"  cos(true_gate, sigmoid(qp[2048:]))={c(trueg, gB):.4f}")
# maybe gate is sigmoid of something with different scale; check correlation of true_gate with qp[:2048] linearly
# try: true_gate vs raw qp[:2048] and qp[2048:]
print(f"  cos(true_gate, qp[:, :256])     ={c(trueg, qp2[:, :256]):.4f}")
print(f"  cos(true_gate, qp[:, 256:])     ={c(trueg, qp2[:, 256:]):.4f}")
print(f"  o_in  cos={c(my_o_in,o_in):.5f} |mine|={np.linalg.norm(my_o_in):.4f} |hook|={np.linalg.norm(o_in):.4f}")
