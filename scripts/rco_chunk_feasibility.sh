#!/usr/bin/env bash
# DERIVE the RCO feasibility edge for the 27B's 2-layer chunking: for a given
# chunk (which layers, which budget fraction) is ANY allocation allowed by the
# per-role bit floors affordable, and what does the cheapest one cost?
#
# WHY THIS EXISTS. The 27B artifact was not quantized in one pass. It was
# quantized as 32 chunks of 2 layers (scripts/archive/run_27b_int2_chunks.sh),
# each with its own budget = (that chunk's FP16 bytes) x budget_frac
# (examples/quant_model_real/main.cpp:293), and the chunks were then concatenated
# by examples/quant_merge. Two budget fractions are used -- 0.29 and 0.305 -- and
# the archived driver's header explains the split only as "LIN+LIN chunks need
# 0.305", citing four probe logs. This script DERIVES the split from the code's
# own floors and cost function instead, and reproduces every observed outcome
# including the exact "selected" byte count of a chunk RCO had no freedom in.
#
# The derivation has three inputs, all read from source or from the artifact:
#   * floors      src/quant/rco.cpp:38 floor_scheme(role, layer)
#   * the ladder  src/quant_spec.cpp:36 default_candidates(role)   <- the file the
#                 real quantizer compiles (scripts/build_quant_real.sh), NOT
#                 src/quant/quant_spec.cpp, which offers more and is not built
#                 into this binary. The LINEAR_ATTN_* roles are absent from the
#                 compiled ladder, so their only candidate is FP16.
#   * cost        src/quant/dashq.cpp:638-640 -- codes = N*K*bits/8, and the
#                 scale array is n_groups*2 bytes, i.e. ONE FP16 scale per
#                 K-group for the WHOLE tensor (not per row).
# Shapes are read from the model's own metadata, so they cannot drift.
#
# WHAT IT SETTLES. Two things that are otherwise folklore:
#   1. Why 0.29 and 0.305. A LIN+LIN chunk's floor (2 layers x 5 FP16 linear-attn
#      tensors + lm_head at INT4 + 6 MLP at INT2 g16) is 1175.6 MiB against a
#      3886.9 MiB FP16 denominator, i.e. the floor itself is 30.2% of FP16. 0.29
#      and even 0.302 are therefore infeasible, and 0.305 clears it by 0.8%.
#   2. That the LIN+LIN chunks have NO ALLOCATIVE FREEDOM. Their selected total
#      equals their floor to the byte, so every MLP in those chunks sits at
#      INT2 g16 (or INT2 g32 when a few hundred spare bytes allow it). The
#      per-layer precision map is therefore partly an artifact of WHICH CHUNK
#      CLASS a layer landed in, not only of its error.
#
# Read-only. No GPU, no build, no model beyond reading its metadata header.
# Usage: scripts/rco_chunk_feasibility.sh [model.intifer] [layers_csv] [frac ...]
#   layers_csv  comma-separated layer indices, default 2,3
# Exit: 0 = derived and printed; 1 = model unreadable; 2 = usage.
set -u

MODEL="${1:-${INTFER_MODEL:-/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer}}"
shift || true
LAYERS="${1:-2,3}"
shift || true
FRACS="${*:-0.29 0.302 0.305}"

[ -f "$MODEL" ] || { echo "FAIL: model not found: $MODEL" >&2; exit 1; }

MODEL="$MODEL" LAYERS="$LAYERS" FRACS="$FRACS" python3 - <<'PYEOF'
import json, os, sys

model = os.environ["MODEL"]
layers = [int(x) for x in os.environ["LAYERS"].split(",") if x.strip()]
fracs = [float(x) for x in os.environ["FRACS"].split()]
if not layers or not fracs:
    sys.exit("usage: rco_chunk_feasibility.sh [model] [layers_csv] [frac ...]")

with open(model, "rb") as f:
    head = f.read(8 * 1024 * 1024)
i = head.find(b'{"model_id"')
if i < 0:
    sys.exit(f"FAIL: no metadata object in {model}")
depth, j = 0, i
while j < len(head):
    c = head[j:j+1]
    if c == b'{':
        depth += 1
    elif c == b'}':
        depth -= 1
        if depth == 0:
            break
    j += 1
else:
    sys.exit("FAIL: metadata object truncated")
meta = json.loads(head[i:j+1].decode("utf-8", "replace"))
tensors = meta.get("tensors") or []
if not tensors:
    sys.exit("FAIL: model declares zero tensors")

# ---- shapes, straight from the artifact -----------------------------------
shape = {}
for t in tensors:
    shape.setdefault(t["role"], (int(t["N"]), int(t["K"])))
need = {"LM_HEAD", "MLP_GATE", "MLP_UP", "MLP_DOWN", "ATTN_Q", "ATTN_K",
        "ATTN_V", "ATTN_O", "LINEAR_ATTN_QKV", "LINEAR_ATTN_Z",
        "LINEAR_ATTN_A", "LINEAR_ATTN_B", "LINEAR_ATTN_OUT"}
missing = sorted(need - set(shape))
if missing:
    sys.exit(f"FAIL: model has no tensor for role(s): {', '.join(missing)} -- "
             f"this derivation is specific to the hybrid Qwen3 architecture")

# ---- cost function (src/quant/dashq.cpp:638-640) ---------------------------
CODE_BITS = {"FP16": 16, "INT8": 8, "INT4": 4, "INT2": 2}


def stored(N, K, scheme, G):
    if scheme == "FP16":
        return N * K * 2
    codes = (N * K * CODE_BITS[scheme]) // 8
    ng = (K + G - 1) // G
    return codes + ng * 2 + 4          # + per-tensor FP32 alpha, always 1.0f


# ---- ladder offered by the COMPILED default_candidates --------------------
# src/quant_spec.cpp:36. Absent roles fall to `default:` and get FP16 only.
A8 = [("INT2", 16), ("INT2", 32), ("INT4", 64), ("INT8", 32)]
LADDER = {
    "EMBED": [("INT4", 64), ("INT8", 32), ("FP16", 0)],
    "LM_HEAD": [("INT4", 64), ("INT8", 32), ("FP16", 0)],
    "ATTN_Q": A8 + [("FP16", 0)], "ATTN_K": A8 + [("FP16", 0)],
    "ATTN_V": A8 + [("FP16", 0)], "ATTN_O": list(A8),
    "MLP_GATE": list(A8), "MLP_UP": list(A8), "MLP_DOWN": list(A8),
}
RANK = {"INT2": 0, "INT4": 1, "INT8": 2, "FP16": 3}


def floor_for(role, layer):
    """src/quant/rco.cpp:38 floor_scheme."""
    if role in ("EMBED", "LM_HEAD"):
        return "INT4"
    if role in ("ATTN_Q", "ATTN_K", "ATTN_V", "LINEAR_ATTN_QKV"):
        return "INT4" if layer == 0 else "INT2"
    return "INT2"


def cheapest(role, layer):
    """Floor-filtered argmin of stored bytes. Absent role => FP16 only."""
    N, K = shape[role]
    lad = LADDER.get(role)
    if lad is None:
        return ("FP16", 0, stored(N, K, "FP16", 0))
    fl = floor_for(role, layer)
    ok = [(s, g) for s, g in lad if RANK[s] >= RANK[fl]]
    if not ok:
        sys.exit(f"FAIL: floor {fl} excludes every candidate for {role}")
    s, g = min(ok, key=lambda sg: stored(N, K, sg[0], sg[1]))
    return (s, g, stored(N, K, s, g))


def roles_of(layer):
    """Hybrid arch: layer % 4 == 3 is full attention, everything else is the
    gated-delta / linear-attention stack. Verified against the artifact, which
    carries ATTN_Q on exactly the layers where layer % 4 == 3."""
    full = (layer % 4) == 3
    attn = (["ATTN_Q", "ATTN_K", "ATTN_V", "ATTN_O"] if full else
            ["LINEAR_ATTN_QKV", "LINEAR_ATTN_Z", "LINEAR_ATTN_A",
             "LINEAR_ATTN_B", "LINEAR_ATTN_OUT"])
    return attn + ["MLP_GATE", "MLP_UP", "MLP_DOWN"]


# Vacuity guard: the artifact must actually agree with the arch assumption, or
# the derivation below is answering a different question (AGENTS.md rule 2).
have_q = {int(t["name"].split("layers.")[1].split(".")[0])
          for t in tensors if ".layers." in t["name"] and "self_attn.q_proj" in t["name"]}
pred_q = {L for L in range(int(meta.get("n_layers") or 0)) if L % 4 == 3}
if have_q and have_q != pred_q:
    sys.exit(f"FAIL: artifact's full-attention layers {sorted(have_q)} do not match "
             f"the layer%4==3 rule {sorted(pred_q)} the derivation assumes")

MiB = float(2**20)
rows = []
for L in layers:
    for r in roles_of(L):
        N, K = shape[r]
        s, g, b = cheapest(r, L)
        rows.append((L, r, s, g, N * K * 2, b))
# lm_head travels with EVERY chunk that does not pass --no-embed, and embed only
# with the chunk that does. Both are non-layer tensors, so their floor is INT4.
N, K = shape["LM_HEAD"]
s, g, b = cheapest("LM_HEAD", -1)
rows.append((-1, "LM_HEAD", s, g, N * K * 2, b))
if 0 in layers:
    N, K = shape["EMBED"]
    s, g, b = cheapest("EMBED", -1)
    rows.append((-1, "EMBED", s, g, N * K * 2, b))

fp16_total = sum(r[4] for r in rows)
floor_total = sum(r[5] for r in rows)

kind = "LIN+LIN" if all(L % 4 != 3 for L in layers) else "MIXED"
print(f"model   : {os.path.basename(model)}  (n_layers={meta.get('n_layers')})")
print(f"chunk   : layers {layers}  -> {kind}"
      f"{' + embed' if 0 in layers else ''}")
print()
print(f"{'tensor':22s} {'cheapest':>12s} {'FP16 MiB':>10s} {'floor MiB':>10s}")
for L, r, s, g, f, b in sorted(rows, key=lambda r: (r[0], r[1])):
    tag = f"{s}/g{g}" if g else s
    print(f"{'L' + str(L) + ' ' if L >= 0 else '   '}{r:19s} {tag:>12s} "
          f"{f/MiB:10.1f} {b/MiB:10.1f}")

print()
print("The 'cheapest' column is the FEASIBILITY BOUND, not RCO's pick: it is the "
      "strictly\nleast expensive floor-legal option per tensor. RCO picks "
      "minimally value + lambda*cost,")
print("so within the headroom below it may choose a dearer option (INT2 g16 over "
      "g32, or")
print("INT4 g64) when the error reduction is worth it. What it can never do is go "
      "BELOW this.")
print()
print(f"FP16 footprint of THIS chunk (the budget denominator): {fp16_total/MiB:9.1f} MiB")
print(f"cheapest allocation any RCO run could ever select    : {floor_total/MiB:9.1f} MiB")
print(f"=> the floor is {100.0*floor_total/fp16_total:.2f}% of FP16; a budget "
      f"fraction below that is INFEASIBLE by construction,")
print(f"   because lagrange_feasible() falls back to cheapest() "
      f"(src/quant/rco.cpp:183) and")
print(f"   run_pipeline then throws (src/quant/pipeline.cpp:209).")
print()
print(f"{'frac':>7s} {'budget MiB':>12s} {'vs floor':>10s} {'headroom':>10s}  verdict")
for fr in fracs:
    budget = fp16_total * fr
    slack = 100.0 * (budget - floor_total) / budget
    verdict = "FEASIBLE" if floor_total <= budget else "INFEASIBLE"
    print(f"{fr:7.3f} {budget/MiB:12.1f} {slack:+9.1f}% "
          f"{(budget-floor_total)/MiB:9.1f}M  {verdict}")

# How much freedom does the headroom actually buy? Price the jump to the next
# BIT WIDTH for the family that dominates the parameter count. Pricing the next
# rung by raw bytes would find INT2 g16, which differs from INT2 g32 by a few
# hundred bytes and is not a decision anyone makes on error grounds.
up = []
for r in ("MLP_GATE", "MLP_UP", "MLP_DOWN"):
    N, K = shape[r]
    lo_s, lo_g, lo_b = cheapest(r, 1)
    nxt = min(((s, g) for s, g in LADDER[r]
               if RANK[s] > RANK[lo_s]),
              key=lambda sg: stored(N, K, sg[0], sg[1]), default=None)
    if nxt:
        nb = stored(N, K, nxt[0], nxt[1])
        up.append((r, lo_s, lo_g, lo_b, nxt, nb, nb - lo_b))
print()
if up:
    cheapest_up = min(u[6] for u in up)
    n_mlp = len(layers) * 3
    print(f"cheapest BIT-WIDTH upgrade above the floor, over the {n_mlp} MLP tensors")
    for r, lo_s, lo_g, lo_b, nxt, nb, d in up:
        print(f"  {r:10s} floor {lo_s}/g{lo_g} {lo_b/MiB:7.2f} MiB -> "
              f"{nxt[0]}/g{nxt[1]} {nb/MiB:7.2f} MiB (+{d/MiB:6.2f} MiB, "
              f"x{nb/lo_b:.2f})")
    print(f"smallest bit-width upgrade costs {cheapest_up/MiB:.2f} MiB.")
    for fr in fracs:
        head_bytes = fp16_total * fr - floor_total
        n = min(n_mlp, int(head_bytes // cheapest_up)) if head_bytes > 0 else 0
        print(f"  at frac {fr:6.3f} headroom {head_bytes/MiB:7.2f} MiB -> at most "
              f"{n} of {n_mlp} MLP tensors can leave INT2")
print()
print(f"smallest budget fraction that clears the floor: "
      f"{floor_total/fp16_total:.5f}  (this is the feasibility edge; the archived "
      f"probes sit either side of it)")
PYEOF

cat <<'EOF'

Compare against the observed logs before trusting this. The archived probes
(test/quant_int2/probe_4_6_b30.log, _b302.log, _b305.log and the per-chunk logs
beside each chunks_int2/*.intifer) are the measurement; this is the derivation.
They must agree, and if they do not, the model above is wrong.
EOF
