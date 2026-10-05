#!/usr/bin/env bash
# DERIVE the per-layer precision map of a .intifer model: which scheme/group each
# tensor actually carries, and what that costs in bytes and parameters.
#
# WHY THIS EXISTS. "At which precision is each layer quantized?" is the one
# quantization question that cannot be answered from the source tree, because the
# answer is not in the source: it is the OUTPUT of SpinQuant -> DASH-Q -> RCO
# under a specific budget, for one specific checkpoint. Every statement of the
# form "the MLP is INT2" is a citation of a run, and AGENTS.md rule 4 says a
# hand-written total is a citation, not a claim.
#
# It is also not derivable by reading the role enum, because the ladder a role
# OFFERS is not the scheme it RECEIVES:
#   * src/quant_spec.cpp (compiled by scripts/build_quant_real.sh, and the file
#     that produced the 27B artifact) has no case for LINEAR_ATTN_*, so those
#     roles fall through to the FP16-only default. 240 of the 27B's 498 tensors
#     are FP16 for that reason alone.
#   * src/quant/quant_spec.cpp (compiled by six examples/* targets) DOES extend
#     the ladder to LINEAR_ATTN_*. Two ladders, one repo, different results.
# So this script reads the model's own metadata header. That cannot drift.
#
# The script also prints the FP16-by-omission census separately, because "FP16
# because RCO spent the budget on it" and "FP16 because no INT2 candidate was
# ever offered" are different claims about the same file. The distinction is
# checked by reading the two default_candidates definitions against the build
# lists that select between them (src/quant_spec.cpp vs src/quant/quant_spec.cpp);
# this script only reports WHICH result the artifact carries.
#
# CITATION for every hardware/box number: none used here. This script is pure
# metadata arithmetic; the only numbers are the file's own bytes.
#
# Usage:
#   scripts/quant_precision_map.sh [model.intifer] [--no-map]
# Defaults: $INTFER_MODEL or the on-disk 27B.
#
# Exit: 0 = derived and printed; 1 = model unreadable / no tensors / zero bytes;
#       2 = usage.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${INTFER_MODEL:-/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer}"
WANT_MAP=1
for a in "$@"; do
  case "$a" in
    --no-map) WANT_MAP=0 ;;
    '') ;;
    -*) echo "usage: $0 [model.intifer] [--no-map]" >&2; exit 2 ;;
    *) MODEL="$a" ;;
  esac
done

[ -f "$MODEL" ] || { echo "FAIL: model not found: $MODEL" >&2; exit 1; }

# The metadata object is a few hundred KB at most (498 tensors -> 183 KB on the
# 27B). 8 MiB is generous and costs one read.
HEAD_BYTES=$(( 8 * 1024 * 1024 ))

MODEL="$MODEL" HEAD_BYTES="$HEAD_BYTES" WANT_MAP="$WANT_MAP" ROOT="$ROOT" \
python3 - <<'PYEOF'
import json, os, sys, collections

model  = os.environ["MODEL"]
nbytes = int(os.environ["HEAD_BYTES"])
want_map = os.environ["WANT_MAP"] == "1"

with open(model, "rb") as f:
    head = f.read(nbytes)

# Vacuity guard: assert the harvest found the metadata object BEFORE reporting a
# verdict on it (AGENTS.md rule 2). An unreadable header must not print an empty
# table that reads as "the model has no quantized layers".
i = head.find(b'{"model_id"')
if i < 0:
    sys.exit(f"FAIL: no metadata object in {model} (first {len(head)} bytes)")

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
    sys.exit(f"FAIL: metadata object truncated in {nbytes} bytes; raise HEAD_BYTES")

try:
    d = json.loads(head[i:j+1].decode("utf-8", "replace"))
except Exception as e:                                   # noqa: BLE001
    sys.exit(f"FAIL: metadata is not JSON: {e}")

tensors = d.get("tensors") or []
if not tensors:
    sys.exit("FAIL: model declares zero tensors -- nothing to derive from")

GiB = float(2**30)

def layernum(name):
    """Canonical HF prefix, matching rco::parse_layer_index (src/quant/rco.cpp:20)."""
    key = "model.language_model.layers."
    p = name.find(key)
    if p < 0:
        return -1
    p += len(key)
    v, any_ = 0, False
    while p < len(name) and name[p].isdigit():
        v, any_, p = v * 10 + int(name[p]), True, p + 1
    return v if any_ else -1

def tag(q):
    s = q.get("scheme")
    if s == "FP16":
        return "FP16"
    return f"{s[3:]}g{q.get('group_size')}"

def stored_bytes(t):
    """AUTHORITATIVE size: what the file actually holds, not what N/K implies."""
    return int(t.get("code_bytes", 0)) + int(t.get("scale_bytes", 0)) + \
           int(t.get("global_bytes", 0))

def params(t):
    return int(t["N"]) * int(t["K"])

total_bytes = sum(stored_bytes(t) for t in tensors)
if total_bytes <= 0:
    sys.exit("FAIL: every stored_bytes is 0 -- the harvest found nothing to total")
total_params = sum(params(t) for t in tensors)
if total_params <= 0:
    sys.exit("FAIL: zero parameters across all tensors")

n_layers = int(d.get("n_layers") or 0)

print(f"model        : {os.path.basename(model)}")
print(f"  model_id   : {d.get('model_id')}")
print(f"  n_layers   : {d.get('n_layers')}   tensors: {len(tensors)}")
print(f"  rotations  : {len(d.get('rotations') or [])} "
      f"({', '.join(sorted({str(r.get('site')) for r in (d.get('rotations') or [])}))})")
print(f"  payload    : {total_bytes/GiB:.3f} GiB over {total_params/1e9:.3f} G params "
      f"=> {8.0*total_bytes/total_params:.3f} stored bits/weight")

# ---------------------------------------------------------------------------
# 1. Scheme census, by role
# ---------------------------------------------------------------------------
agg = collections.defaultdict(lambda: {"n": 0, "bytes": 0, "params": 0,
                                       "tags": collections.Counter(),
                                       "N": 0, "K": 0, "err": []})
for t in tensors:
    r = t["role"]
    a = agg[r]
    a["n"] += 1
    a["bytes"] += stored_bytes(t)
    a["params"] += params(t)
    a["tags"][tag(t["qcfg"])] += 1
    a["N"] = int(t["N"])              # last seen; roles are shape-homogeneous
    a["K"] = int(t["K"])
    if t.get("est_error") is not None:
        a["err"].append(float(t["est_error"]))

print()
print("=" * 96)
print("1. ROLE x SCHEME  (bytes and params are AUTHORITATIVE: read from the file)")
print("=" * 96)
print(f"{'role':16s} {'tensors':>7s} {'N':>7s} {'K':>6s} {'schemes':>26s} "
      f"{'GiB':>7s} {'%bytes':>7s} {'%params':>8s} {'bits/w':>7s} {'mean err':>9s}")

for r, a in sorted(agg.items(), key=lambda kv: -kv[1]["bytes"]):
    mix = " ".join(f"{k}x{v}" for k, v in sorted(a["tags"].items(),
                                                 key=lambda kv: -kv[1]))
    e = sum(a["err"]) / len(a["err"]) if a["err"] else float("nan")
    print(f"{r:16s} {a['n']:7d} {a['N']:7d} {a['K']:6d} {mix:>26s} "
          f"{a['bytes']/GiB:7.3f} {100*a['bytes']/total_bytes:6.1f}% "
          f"{100*a['params']/total_params:7.1f}% "
          f"{8.0*a['bytes']/a['params']:7.3f} {e:9.4f}")
print(f"{'TOTAL':16s} {len(tensors):7d} {'':7s} {'':6s} {'':>26s} "
      f"{total_bytes/GiB:7.3f} {100.0:6.1f}% {100.0:7.1f}% "
      f"{8.0*total_bytes/total_params:7.3f}")

# ---------------------------------------------------------------------------
# 2. Scheme mix, both ways
# ---------------------------------------------------------------------------
mix_b = collections.defaultdict(int)
mix_p = collections.defaultdict(int)
for t in tensors:
    mix_b[tag(t["qcfg"])] += stored_bytes(t)
    mix_p[tag(t["qcfg"])] += params(t)

print()
print("scheme mix by BYTES (what decode has to stream):")
for k, b in sorted(mix_b.items(), key=lambda kv: -kv[1]):
    print(f"  {k:8s} {b/GiB:8.3f} GiB  {100*b/total_bytes:5.1f}%   "
          f"{mix_p[k]/1e9:7.3f} G params  {100*mix_p[k]/total_params:5.1f}% of params")
print()
print("scheme mix by PARAM COUNT (what the quality problem is about):")
for k, p in sorted(mix_p.items(), key=lambda kv: -kv[1]):
    print(f"  {k:8s} {p/1e9:8.3f} G params  {100*p/total_params:5.1f}%   "
          f"{mix_b[k]/GiB:8.3f} GiB  {100*mix_b[k]/total_bytes:5.1f}% of bytes")

# ---------------------------------------------------------------------------
# 3. Activation scheme + scale/global overhead
# ---------------------------------------------------------------------------
print()
print("per-tensor metadata, distinct combos actually present:")
combos = collections.Counter(
    (t["qcfg"].get("scheme"), t["qcfg"].get("group_size"),
     t["qcfg"].get("scale_dtype"), t["qcfg"].get("act"),
     bool(t["qcfg"].get("has_global_divisor")), int(t["qcfg"].get("rotation_id") or 0))
    for t in tensors)
print(f"  {'scheme':7s} {'G':>4s} {'scale':>6s} {'act':>9s} {'gdiv':>5s} {'rot':>4s} {'count':>6s}")
for (sch, g, sd, act, gd, rot), c in sorted(combos.items(), key=lambda kv: -kv[1]):
    print(f"  {str(sch):7s} {str(g):>4s} {str(sd):>6s} {str(act):>9s} {str(gd):>5s} "
          f"{rot:4d} {c:6d}")

scale_bytes = sum(int(t.get("scale_bytes", 0)) for t in tensors)
glob_bytes = sum(int(t.get("global_bytes", 0)) for t in tensors)
code_bytes = sum(int(t.get("code_bytes", 0)) for t in tensors)
print(f"\noverhead: code {code_bytes/GiB:.3f} GiB | scales {scale_bytes/1024:.1f} KiB "
      f"({100*scale_bytes/total_bytes:.4f}%) | globals {glob_bytes} B "
      f"({100*glob_bytes/total_bytes:.5f}%)")

# ---------------------------------------------------------------------------
# 4. Per-layer matrix
# ---------------------------------------------------------------------------
rows = collections.defaultdict(dict)
full = set()
for t in tensors:
    L = layernum(t["name"])
    if L < 0:
        continue
    rows[L][t["role"]] = tag(t["qcfg"])
    if t["role"] == "ATTN_Q":
        full.add(L)

lin_only = [L for L in sorted(rows) if L not in full]
print()
print(f"layer types: {len(full)} full-attention (every 4th, starting at L3), "
      f"{len(lin_only)} linear-attention (gated-delta/SSM)")

if want_map:
    roles = [("ATTN_Q", "Q"), ("ATTN_K", "K"), ("ATTN_V", "V"), ("ATTN_O", "O"),
             ("LINEAR_ATTN_QKV", "QKV"), ("LINEAR_ATTN_Z", "Z"),
             ("LINEAR_ATTN_A", "A"), ("LINEAR_ATTN_B", "B"),
             ("LINEAR_ATTN_OUT", "o"), ("MLP_GATE", "gate"),
             ("MLP_UP", "up"), ("MLP_DOWN", "down")]
    print()
    print("=" * 96)
    print("2. PER-LAYER PRECISION MAP   (INT2g16 / INT2g32 / INT4g64 / INT8g32 / FP16)")
    print("   '-' = tensor absent in this layer (hybrid architecture)")
    print("=" * 96)
    print(f"{'L':>3s} {'type':>5s} " + " ".join(f"{lab:>5s}" for _, lab in roles))
    for L in sorted(rows):
        r = rows[L]
        ty = "full" if L in full else "lin"
        print(f"{L:3d} {ty:>5s} " + " ".join(f"{r.get(role,'-'):>5s}" for role, _ in roles))

    # MLP is the only mixed family; show its distribution rather than 64 rows.
    print()
    print("MLP distribution over the 64 layers (RCO picks per TENSOR, not per role):")
    for role in ("MLP_GATE", "MLP_UP", "MLP_DOWN"):
        c = collections.Counter(rows[L].get(role, "?") for L in sorted(rows))
        mix = "  ".join(f"{k} x{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
        print(f"  {role:10s} {mix}")

# ---------------------------------------------------------------------------
# 5. The FP16-by-omission check
# ---------------------------------------------------------------------------
# Roles whose ONLY present tag is FP16. For those, ask whether the compiled
# ladder (src/quant_spec.cpp) offered anything else: it has no case for them, so
# default_candidates returns the FP16-only branch. Verified by the derivation in
# scripts/check_quant_ladder_divergence.sh, not asserted here.
fp16_only = sorted(r for r, a in agg.items() if set(a["tags"]) == {"FP16"})
fp16_bytes = sum(agg[r]["bytes"] for r in fp16_only)
fp16_params = sum(agg[r]["params"] for r in fp16_only)
print()
print("=" * 96)
print("3. UNIFORM-FP16 ROLES (single scheme across every tensor of the role)")
print("=" * 96)
for r in fp16_only:
    print(f"  {r:16s} {agg[r]['n']:4d} tensors  {agg[r]['bytes']/GiB:7.3f} GiB  "
          f"{100*agg[r]['bytes']/total_bytes:5.1f}% of bytes  "
          f"{100*agg[r]['params']/total_params:5.1f}% of params")
print(f"  {'TOTAL':16s} {sum(agg[r]['n'] for r in fp16_only):4d} tensors  "
      f"{fp16_bytes/GiB:7.3f} GiB  {100*fp16_bytes/total_bytes:5.1f}% of bytes  "
      f"{100*fp16_params/total_params:5.1f}% of params")

mixed = sorted(r for r, a in agg.items() if len(a["tags"]) > 1)
print(f"\nroles whose scheme VARIES tensor-to-tensor (RCO chose per tensor): "
      f"{', '.join(mixed) if mixed else 'none'}")
uniform_q = sorted(r for r, a in agg.items()
                   if len(a["tags"]) == 1 and set(a["tags"]) != {"FP16"})
print(f"roles that are UNIFORM and QUANTIZED: "
      f"{', '.join(f'{r}={list(agg[r]['tags'])[0]}' for r in uniform_q) if uniform_q else 'none'}")

# ---------------------------------------------------------------------------
# 5b. Reconstruction error, params-weighted, split by whether the tensor was
#     quantized at all. quant_model_real's PASS/REVIEW verdict thresholds
#     (examples/quant_model_real/main.cpp:398-399) are computed over ALL params,
#     so the FP16 tensors -- which have est_error ~0 by construction -- dilute the
#     mean. Report both numbers so the threshold can be read against the
#     population it actually describes.
# ---------------------------------------------------------------------------
def wmean(pred):
    num = den = 0.0
    for t in tensors:
        if not pred(t):
            continue
        p = float(params(t))
        num += p * float(t.get("est_error") or 0.0)
        den += p
    return (num / den if den else float("nan")), den / 1e9

all_m, all_p = wmean(lambda t: True)
q_m, q_p = wmean(lambda t: t["qcfg"].get("scheme") != "FP16")
fp_m, fp_p = wmean(lambda t: t["qcfg"].get("scheme") == "FP16")
print()
print("est_error, params-weighted (the number the artifact stores per tensor):")
print(f"  all tensors          : {all_m:.4f}  over {all_p:6.3f} G params "
      f"<- the population the <0.25 / <0.40 PASS thresholds are applied to")
print(f"  QUANTIZED tensors only: {q_m:.4f}  over {q_p:6.3f} G params "
      f"({100*q_p/(total_params/1e9):.1f}% of params)")
print(f"  FP16 tensors         : {fp_m:.4f}  over {fp_p:6.3f} G params "
      f"({100*fp_p/(total_params/1e9):.1f}% of params)")
print(f"  -> FP16 tensors pull the all-tensors mean DOWN by a factor of "
      f"{q_m/all_m:.2f}x relative to the quantized population "
      f"(a threshold written against the former is not a statement about the "
      f"latter)")

# ---------------------------------------------------------------------------
# 6. Coverage assertions. A script that prints a table it did not verify is a
#    table nobody should cite.
# ---------------------------------------------------------------------------
layered = sum(a["n"] for r, a in agg.items()
              if any(layernum(t["name"]) >= 0 for t in tensors if t["role"] == r))
nonlayered = len(tensors) - layered
print()
print(f"coverage: {layered} tensors carry a layer index, {nonlayered} do not "
      f"(embed / lm_head), layers present: "
      f"{min(rows) if rows else '-'}..{max(rows) if rows else '-'} "
      f"({len(rows)} distinct, n_layers={n_layers})")
if n_layers and rows and len(rows) != n_layers:
    print(f"WARN: metadata n_layers={n_layers} but {len(rows)} distinct layer "
          f"indices found -- the map above is not the whole model")

assert sum(mix_b.values()) == total_bytes, "scheme byte mix lost bytes"
assert sum(mix_p.values()) == total_params, "scheme param mix lost params"
assert len(fp16_only) + len(mixed) + len(uniform_q) == len(agg), "role partition broken"
print("OK: byte and param partitions are exact, and every role is classified "
      "(uniform-FP16 / uniform-quantized / mixed).")
PYEOF
rc=$?
[ "$rc" -eq 0 ] || exit "$rc"

cat <<'EOF'

Cite the output, do not retype it. If this script and a document disagree, the
document is wrong -- that is the whole point of deriving it.
EOF
