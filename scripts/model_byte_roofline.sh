#!/usr/bin/env bash
# DERIVE the per-token weight-byte roofline for a .intifer model, at a given TP.
#
# WHY THIS EXISTS. The roofline every decode optimization is argued against was
# hand-written, and it was WRONG twice:
#
#   * AGENTS.md / quant_gemm.cpp carried "N ~ 2048 per rank over 512 Xe cores"
#     and "35.5 GiB/s ... about 21% of the B580's ~179 GB/s". The B580 has 20 Xe
#     cores and 456 GB/s (CITATION: docs/Intel Arc B580 Specs _ TechPowerUp GPU
#     Database.html, saved from techpowerup.com/gpu-specs/arc-b580.c4244; cross-
#     checked against Intel ARK sku/241598). So the kernel was at ~9% of peak,
#     not ~21%, and "EU occupancy" was the wrong model of the bottleneck.
#   * docs/next_plan.md's per-role VRAM table sums to 25.5 GB against its own
#     stated 19.5 GB total.
#
# Both are the same defect AGENTS.md rule 4 names: "A hand-written total is a
# citation, not a claim. Derive it, or mark it HISTORICAL/CITATION/re-derive."
# This script is the derivation. It reads code_bytes straight out of the model's
# own metadata header, so it cannot drift from the model it describes.
#
# CITATION(techpowerup arc-b580.c4244 + Intel ARK sku/241598, both agree):
#   Xe cores 20 | vector engines 160 | XMX 160 | mem 12GB GDDR6 192-bit 19Gbps
#   bandwidth 456.0 GB/s | L2 18 MB | L1 256KB/EU | FP32 13.67 TFLOPS @2670MHz
# Derived from those: 456e9/2 (FP32 is 2 flop/FMA) / 2.67e9 = ~2560 FP32 lanes.
# scripts/check_hw_constants.sh enforces that no doc re-states these unmarked.
#
# Usage:
#   scripts/model_byte_roofline.sh [model.intifer] [tp]
# Defaults: $INTFER_MODEL or the on-disk 27B, tp=2.
#
# Exit: 0 = derived and printed; 1 = model unreadable / no tensors; 2 = usage.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${1:-${INTFER_MODEL:-/var/home/fra/data/models/intfer/Qwen3.8-27B-int2.intifer}}"
TP="${2:-${INTFER_TP:-2}}"

[ -f "$MODEL" ] || { echo "FAIL: model not found: $MODEL" >&2; exit 1; }
case "$TP" in
  ''|*[!0-9]*) echo "usage: $0 [model.intifer] [tp]" >&2; exit 2 ;;
esac

# Read enough of the header to hold the whole metadata object. The header is a
# few hundred KB at most (496 tensors); 8 MiB is generous and costs one read.
HEAD_BYTES=$(( 8 * 1024 * 1024 ))

MODEL="$MODEL" TP="$TP" HEAD_BYTES="$HEAD_BYTES" ROOT="$ROOT" \
python3 - <<'PYEOF'
import json, os, sys, collections

model  = os.environ["MODEL"]
tp     = int(os.environ["TP"])
root   = os.environ["ROOT"]
nbytes = int(os.environ["HEAD_BYTES"])

# CITATION(techpowerup arc-b580.c4244 + Intel ARK sku/241598). Keep in sync with
# scripts/check_hw_constants.sh, which fails if a doc states one of these
# without a CITATION marker.
PEAK_BW = 456.0e9
XE_CORES = 20
VEC_ENGINES = 160
FP32_TFLOPS = 13.67
CLOCK_HZ = 2.67e9

with open(model, "rb") as f:
    head = f.read(nbytes)
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
except Exception as e:                      # noqa: BLE001
    sys.exit(f"FAIL: metadata is not JSON: {e}")

tensors = d.get("tensors") or []
if not tensors:
    sys.exit("FAIL: model declares zero tensors -- nothing to derive from")

# code_bytes is the AUTHORITATIVE packed weight size: it is what the engine
# actually copies into USM per shard. Recomputing it from N/K/scheme would be a
# second source of truth that can disagree with the file (and, historically, did).
agg = collections.defaultdict(lambda: {"bytes": 0, "count": 0})
shape = {}
for t in tensors:
    r = t["role"]
    agg[r]["bytes"] += int(t.get("code_bytes", 0))
    agg[r]["count"] += 1
    shape.setdefault(r, t)

total = sum(v["bytes"] for v in agg.values())
if total <= 0:
    sys.exit("FAIL: every code_bytes is 0 -- the harvest found nothing to total")

# Per-token traffic. TP splits N across devices, K stays whole, so a device
# reads code_bytes/tp of each tensor exactly once per token at M=1.
share = 1.0 / tp

def n_local(t):
    return int(t["N"]) // tp

print(f"model            : {os.path.basename(model)}")
print(f"                  model_id={d.get('model_id')}  layers={d.get('n_layers')}")
print(f"tensors          : {len(tensors)}  roles={len(agg)}")
print(f"packed weights   : {total/1e9:.2f} GB total   ({total/2**30:.2f} GiB)")
print()
print(f"device (CITATION(techpowerup arc-b580.c4244 / ARK 241598)):")
print(f"  Xe cores {XE_CORES}   vector engines {VEC_ENGINES}   peak BW {PEAK_BW/1e9:.1f} GB/s")
print(f"  FP32 {FP32_TFLOPS} TFLOPS @ {CLOCK_HZ/1e9:.2f} GHz "
      f"=> ~{int(FP32_TFLOPS*1e12/2/CLOCK_HZ)} FP32 lanes")
print()
print(f"{'role':22s} {'n':>4s} {'N_loc':>7s} {'K':>6s} {'scheme':>7s} "
      f"{'GB total':>9s} {'GB/dev/tok':>11s} {'share':>7s}")

by_share = sorted(agg.items(), key=lambda kv: -kv[1]["bytes"])
for role, v in by_share:
    t = shape[role]
    q = t["qcfg"]
    gb = v["bytes"] / 1e9
    per_dev = gb * share
    print(f"{role:22s} {v['count']:4d} {n_local(t):7d} {t['K']:6d} "
          f"{q['scheme']:>7s} {gb:9.3f} {per_dev:11.3f} {100*gb/(total/1e9):6.1f}%")

per_dev_total = (total / 1e9) * share
print(f"{'TOTAL':22s} {'':4s} {'':7s} {'':6s} {'':>7s} "
      f"{total/1e9:9.3f} {per_dev_total:11.3f} {100.0:6.1f}%")

# Scheme mix -- the finding that redirected this batch. AGENTS.md's whole decode
# narrative is about the INT2 path; for the 27B it is not the biggest one.
mix = collections.defaultdict(float)
for role, v in agg.items():
    mix[shape[role]["qcfg"]["scheme"]] += v["bytes"] * share
print()
print("scheme mix, per device per token:")
for sch, b in sorted(mix.items(), key=lambda kv: -kv[1]):
    print(f"  {sch:6s} {b/2**30:7.3f} GiB  {100*b/(total*share):5.1f}%")

print()
print("achieved vs roofline (fill in kernel_ms from [GPU profile] @decode):")
for ms in (237.5,):
    bytes = per_dev_total * 1e9
    gbs = bytes / (ms * 1e-3) / 1e9
    print(f"  kernel {ms:6.1f} ms/token -> {gbs:6.1f} GB/s = "
          f"{100*gbs/(PEAK_BW/1e9):4.1f}% of {PEAK_BW/1e9:.0f} GB/s")
print(f"  (the 237.5 ms figure is the CITATION(perf_base.err) baseline: "
      f"total_kernel_us 1187601.2 / 5 tokens)")
PYEOF
rc=$?
[ "$rc" -eq 0 ] || exit "$rc"

cat <<'EOF'

Cite the output, do not retype it. If this script and a document disagree, the
document is wrong -- that is the whole point of deriving it.
EOF
