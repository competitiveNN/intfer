#!/usr/bin/env bash
# _dpas_pred.sh -- the ONE home of the DPAS shape eligibility predicate.
#
# Consumed by check_dpas_shape_audit.sh (the audit) and by the self-check
# fixtures (which source this file alone, so they never touch the model).
# AGENTS rule 4: a predicate fixed at one site and re-derived at another is
# a remembered constant that decays into a guess. This is the single home.
#
# Quoted verbatim from the engine's own dpas48_eligible (engine.cpp:201) plus
# the lm_head s8-weight gate (examples/intfer_cli/main.cpp:1548-1605):
#   bits != 16; combo_ok per (bits, G); (N_local % 16 == 0) && (K % 32 == 0)
set -u

# Kernel shape constants, read off the source, not remembered.
TN=16   # quant_gemm.cpp:2707 constexpr uint32_t TN = 16
OP=32   # quant_gemm.cpp:2707 constexpr uint32_t OP = 32

dpas_eligible() {
  # $1=scheme $2=group_size $3=N_local $4=K  -> 0 eligible, 1 not
  local scheme="$1" gs="$2" Nl="$3" K="$4"
  local bits
  case "$scheme" in
    FP16) return 1 ;;            # engine.cpp:210  if (bits == 16) return false
    INT2) bits=2 ;;
    INT4) bits=4 ;;
    INT8) bits=8 ;;
    *) return 1 ;;
  esac
  local G="$gs"; [ "$G" = "0" ] && G="$K"   # engine.cpp:211
  local combo=1
  case "$bits:$G" in
    4:64|8:32|2:32|2:16) combo=0 ;;  # engine.cpp:212-215
  esac
  [ "$combo" = "1" ] && return 1
  [ $((Nl % TN)) -ne 0 ] && return 1    # engine.cpp:216
  [ $((K  % OP)) -ne 0 ] && return 1    # engine.cpp:216
  return 0
}

# The engine's OWN shard formula, quoted verbatim (engine.cpp:524-533):
#   for i in 0..n: start = total*i/n; end = total*(i+1)/n
# INTEGER division, so a total that is %16==0 can still yield a shard that
# is not. The audit therefore checks every SHARD, not only the full tensor.
shard_size() {  # $1=total $2=i $3=n  -> prints shard size for rank i
  local total="$1" i="$2" n="$3"
  local start=$(( total * i / n ))
  local end=$(( total * (i + 1) / n ))
  echo $(( end - start ))
}
