#!/usr/bin/env python3
# compare.py: oracle npz vs C++ binary dump, per-layer cosine over prefill positions.
import numpy as np, sys

npz = np.load("test/models/qwen35_oracle.npz")
layers = int(sys.argv[1]) if len(sys.argv)>1 else 24
f = open("test/models/qwen35_cpp.bin","rb")
hdr = np.frombuffer(f.read(12), dtype=np.int32)
nL, S, H = int(hdr[0]), int(hdr[1]), int(hdr[2])
cpp = np.frombuffer(f.read(), dtype=np.float32).reshape(nL, S, H)

def cos(a,b):
    a=a.astype(np.float64); b=b.astype(np.float64)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

print(f"oracle layers={layers} cpp nL={nL} S={S} H={H}")
for L in range(layers):
    o = npz[f"layer{L}"]  # (S,H)
    c = cpp[L]
    coses = [cos(o[p], c[p]) for p in range(S)]
    print(f"layer{L:2d}: mincos={min(coses):.5f}  cos[pos0]={coses[0]:.5f}  cos[pos{S-1}]={coses[-1]:.5f}")
