#!/usr/bin/env bash
# Operator-run knob-sweep A/B for the intra-device geometry knobs:
#   INTFER_DECODE1_WG, INTFER_XMX_WG, INTFER_LA_CONV_PREP_WG
#
# PURPOSE. AGENTS.md (GRIMOIRE survey, reminder #9) records a real gap:
#   "intfer owns every knob ... but has no harness that enumerates a product
#   over one and prints a table, so a multi-knob interaction ... is only
#   observable by running both and reading two logs."
# This script closes that gap: it enumerates a product over the three launch-
# geometry knobs, runs the 27B --tp 2 target for each combo, parses the CLI's
# [GPU profile @decode] line for the load-insensitive counters (nk/nw/nr exact;
# submit/waitblock/copy/readback/usm_write host wall-clock), prints a baseline-
# relative table, and applies the spread-floor rule (a delta must CLEAR the
# max spread of the two arms compared, across >= 2 paired rounds).
#
# DECIDE ON COUNTERS, NEVER ON A TOK/S. No tok/s is printed or quoted here.
# A round whose load1 crosses the quiet threshold for any part of its run is
# recorded r<k> SKIPPED-CONTENDED and publishes NO figure: loadlog_arm returns
# the CLI's exit code, so its verdict is read separately via loadlog_usable
# (_gate_loadlog.sh). Withholding is recorded, never silent, so the table shows
# why a combo lost a round; both the per-arm and the baseline filter exclude
# SKIPPED-CONTENDED rounds from the usable set.
#
# Usage: scripts/knob_sweep.sh [model.intifer] [rounds]
#   model   default $INTFER_MODEL or the on-disk 27B
#   rounds  default 2 (each combo run this many times, paired on round index)
# Env:
#   INTFER_BUILD_DIR   build tree (default $ROOT/build)
#   INTFER_08B         override model path (same default as other gates)
#   INTFER_QUIET_LOAD  load1 quiet threshold (default 2.0)
#   INTFER_SWEEP_KNOBS comma list "KNOB:v1,v2|KNOB:v1,v2" (default the 3 geometry knobs)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${INTFER_BUILD_DIR:-$ROOT/build}}"
BIN="$BUILD_DIR/examples/intfer_cli/intfer_cli"
MODEL="${1:-${INTFER_MODEL:-${INTFER_08B:-$ROOT/test/models/Qwen3.8-27B-int2.intifer}}}"
ROUNDS="${2:-2}"

[ -x "$BIN" ] || { echo "FAIL: cli binary missing: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model missing: $MODEL" >&2; exit 1; }
if ! python3 -c "open('/dev/dri/renderD128','rb').close()" 2>/dev/null; then
  echo "SKIP: no render-node access from this shell." >&2; exit 2
fi

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1

# ---- shared helpers (sourced, never executed) ----
. "$SCRIPT_DIR/_gate_build.sh"
. "$SCRIPT_DIR/_gate_precond.sh"
. "$SCRIPT_DIR/_gate_loadlog.sh"

gate_require_build intfer_cli || exit 1
gate_require_device || exit 2
gate_require_gpus 2 || exit 2

LOGDIR="$BUILD_DIR/knob_sweep_$(date -u +%Y%m%dT%H%M%SZ)"
loadlog_init "$LOGDIR" >/dev/null

# ---- knob catalogue (product is enumerated below) ----
# Each entry: KNOB:v1,v2,...  A bare "-" means unset/shipped default.
DEFAULT_KNOBS='INTFER_DECODE1_WG:-,16,64,256|INTFER_XMX_WG:-,16|INTFER_LA_CONV_PREP_WG:-,64'
KNOB_SPEC="${INTFER_SWEEP_KNOBS:-$DEFAULT_KNOBS}"

declare -a KNOB_NAMES=()
declare -A KNOB_VALS   # KNOB -> newline-separated values ("-" = unset/shipped)
KNOBS=()
IFS='|' read -ra PAIRS <<<"$KNOB_SPEC"
for p in "${PAIRS[@]}"; do
  [ -n "$p" ] || continue
  knob="${p%%:*}"; vals="${p#*:}"
  [ -n "$knob" ] || { echo "FAIL: empty knob name in INTFER_SWEEP_KNOBS" >&2; exit 1; }
  [ -n "$vals" ] || { echo "FAIL: no values for knob $knob" >&2; exit 1; }
  KNOBS+=("$knob")
  KNOB_NAMES+=("$knob")
  KNOB_VALS["$knob"]="$vals"
done
NKNOBS=${#KNOBS[@]}

# Cartesian product over knob values -> array of "KNOB=v KNOB=v ..." strings.
# A "-" value contributes nothing (knob left unset => shipped default).
PRODUCT=()
walk() { # $1 = depth, $2 = accumulated env string
  local depth="$1" acc="$2"
  if [ "$depth" -eq "$NKNOBS" ]; then
    PRODUCT+=("$acc"); return
  fi
  local knob="${KNOBS[$depth]}"
  local vals="${KNOB_VALS["$knob"]}"
  local v
  while IFS= read -r v; do
    [ -n "$v" ] || continue
    if [ "$v" = "-" ]; then
      walk $((depth + 1)) "$acc"
    else
      walk $((depth + 1)) "$acc $knob=$v"
    fi
  done <<<"$vals"
}
walk 0 ""

BASELINE="${PRODUCT[0]}"   # first entry = all knobs unset/shipped default

# ---- result storage ----
# result_lines[label] = newline list of "round kernel submit waitblock rebuild usm_write copy readback nk nw nr"
declare -A result_lines

run_combo() { # $1 = label; remaining args = "KNOB=v ..." env prefix
  local label="$1"; shift
  local r log prof rc
  local knob
  for r in $(seq 1 "$ROUNDS"); do
    log="$LOADLOG_DIR/${label}_r${r}.log"
    # Export the knobs for this combo explicitly (no eval).
    local kv
    for kv in "$@"; do
      export "$kv"
    done
    loadlog_arm "${label}_r${r}" \
      "$BIN" -m "$MODEL" --gpu --tp 2 --ids \
        -p "The history of artificial intelligence and machine learning." -n 4 \
        --no-stream
    rc=$?
    # Unset the knobs so later combos start clean.
    for knob in "${KNOBS[@]}"; do unset "$knob"; done

    if [ $rc -ne 0 ]; then
      result_lines["$label"]+="r$r FAILED rc=$rc log=$log"$'\n'
      continue
    fi
    # The load verdict is a SEPARATE axis from rc and MUST be read separately:
    # loadlog_arm returns the CLI's exit code, so a CONTENDED round arrives here
    # with rc=0 AND a scrapeable profile line. Publishing that line would report a
    # delta measured on a contended box -- the failure mode the load gate exists to
    # prevent. Withholding is RECORDED, not silent, so the table shows why a combo
    # lost a round. The SKIPPED round is then excluded by both filters below.
    #
    # THE POSITION OF THIS GUARD IS NOT AN ACCIDENT, and neither half is optional.
    # It sits AFTER the rc check, because a round that died must be recorded
    # FAILED rc=N so the operator still learns WHY: loadlog_arm also returns 1 for
    # its own "call loadlog_init first" wiring failure (_gate_loadlog.sh:68), which
    # has nothing to do with load, and a contended AND crashed round read this way
    # round would be relabelled SKIPPED-CONTENDED and the exit code lost. It sits
    # BEFORE the line that publishes the round, because after that the contended
    # arm's figure is already in the usable set and withholding it is impossible.
    # Both orderings are asserted by check_knob_sweep_contract.sh's
    # p_guard_before_publish, and the "move the guard after the publication" state
    # (mutate_knob_sweep_contract.sh KS-guardafter) is what keeps that assertion
    # from being a comment about a comment.
    if ! loadlog_usable; then
      result_lines["$label"]+="r$r SKIPPED-CONTENDED max_load1=${_LL_MAX:-NA}"$'\n'
      continue
    fi
    prof="$(grep -a '\[GPU profile @decode\]' "$log" | head -1)"
    if [ -z "$prof" ]; then
      result_lines["$label"]+="r$r NO_PROFILE log=$log"$'\n'
      continue
    fi
    result_lines["$label"]+="r$r $prof"$'\n'
  done
}

# ---- run baseline first (it is the reference for every delta) ----
run_combo "BASELINE($BASELINE)" $BASELINE

# ---- run remaining combos ----
local combo
for combo in "${PRODUCT[@]:1}"; do
  run_combo "$combo" $combo
done

# ---- parse helpers ----
# field(line, name) -> value or empty
field() { # $1 = profile line, $2 = field name
  local line="$1" name="$2"
  local re
  case "$name" in
    kernel)   re='kernel=([0-9.]+)ms' ;;
    submit)   re='submit=([0-9.]+)ms' ;;
    waitblock)re='waitblock=([0-9.]+)ms' ;;
    rebuild)  re='rebuild=([0-9.]+)ms' ;;
    usm_write)re='usm_write=([0-9.]+)ms' ;;
    readback) re='readback=([0-9.]+)ms' ;;
    copy)     re='copy=([0-9.]+)ms' ;;
    nk)       re='nk=([0-9]+)' ;;
    nw)       re='nw=([0-9]+)' ;;
    nr)       re='nr=([0-9]+)' ;;
    *)        return 1 ;;
  esac
  [[ $line =~ $re ]] && printf '%s' "${BASH_REMATCH[1]}"
}

median() { # stdin = numbers
  awk '{a[NR]=$1} END{
    n=NR; for(i=1;i<=n;i++) for(j=i+1;j<=n;j++) if(a[i]>a[j]){t=a[i];a[i]=a[j];a[j]=t};
    if(n%2) print a[(n+1)/2]; else print (a[n/2]+a[n/2+1])/2 }'
}

spread() { # stdin = numbers -> max-min
  awk 'BEGIN{mn=1e9;mx=-1e9;n=0} {if($1<mn)mn=$1;if($1>mx)mx=$1;n++} END{if(n<2)print 0; else printf "%.4f", mx-mn}' 
}

# ---- emit table ----
{
echo "knob_sweep: model=$MODEL rounds=$ROUNDS build=$BUILD_DIR"
echo "knobs: $KNOB_SPEC"
echo "baseline: $BASELINE"
echo
printf '%-34s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n' \
  'combo' 'kernel' 'submit' 'waitblock' 'rebuild' 'usm_write' 'readback' 'copy' 'nk' 'nw' 'nr'
printf '%-34s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n' \
  '----------' '--------' '--------' '--------' '--------' '--------' '--------' '--------' '--------' '--------' '--------'

emit_row() { # $1 = label
  local label="$1" vals="${result_lines[$label]:-}"
  local -a rows=()
  local line
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    rows+=("$line")
  done <<<"$vals"
  local n=${#rows[@]}
  if [ "$n" -eq 0 ]; then
    printf '%-34s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n' "$label" '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-'
    return
  fi
  # collect per-field values across usable rounds (non-FAILED, non-NO_PROFILE)
  local -a k s wb rb uw rbc cp nk nw nr=()
  local usable=0
  local line
  for line in "${rows[@]}"; do
    case "$line" in
      *FAILED*|*NO_PROFILE*|*SKIPPED*) continue ;;
    esac
    usable=$((usable+1))
    k+=("$(field "$line" kernel)")
    s+=("$(field "$line" submit)")
    wb+=("$(field "$line" waitblock)")
    rb+=("$(field "$line" rebuild)")
    uw+=("$(field "$line" usm_write)")
    rbc+=("$(field "$line" readback)")
    cp+=("$(field "$line" copy)")
    nk=("${nk[@]}" "$(field "$line" nk)")
    nw=("${nw[@]}" "$(field "$line" nw)")
    nr=("${nr[@]}" "$(field "$line" nr)")
  done
  if [ "$usable" -eq 0 ]; then
    printf '%-34s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n' "$label" '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-'
    return
  fi
  printf '%-34s ' "$label"
  printf '%8s' "$(printf '%s\n' "${k[@]}" | median)"
  printf '%8s' "$(printf '%s\n' "${s[@]}" | median)"
  printf '%8s' "$(printf '%s\n' "${wb[@]}" | median)"
  printf '%8s' "$(printf '%s\n' "${rb[@]}" | median)"
  printf '%8s' "$(printf '%s\n' "${uw[@]}" | median)"
  printf '%8s' "$(printf '%s\n' "${rbc[@]}" | median)"
  printf '%8s' "$(printf '%s\n' "${cp[@]}" | median)"
  printf '%8s' "$(printf '%s\n' "${nk[@]}" | head -1)"
  printf '%8s' "$(printf '%s\n' "${nw[@]}" | head -1)"
  printf '%8s' "$(printf '%s\n' "${nr[@]}" | head -1)"
  printf '\n'
  # ---- spread-floor deltas vs baseline ----
  if [ "$label" != "BASELINE($BASELINE)" ] && [ "$usable" -ge 2 ]; then
    local -a bkarr bsarr bwbarr brbarr buwarr brbcarr bcparr
    local bline
    local bused=0
    while IFS= read -r bline; do
      [ -n "$bline" ] || continue
      case "$bline" in *FAILED*|*NO_PROFILE*|*SKIPPED*) continue ;; esac
      bused=$((bused+1))
      bkarr+=("$(field "$bline" kernel)")
      bsarr+=("$(field "$bline" submit)")
      bwbarr+=("$(field "$bline" waitblock)")
      brbarr+=("$(field "$bline" rebuild)")
      buwarr+=("$(field "$bline" usm_write)")
      brbcarr+=("$(field "$bline" readback)")
      bcparr+=("$(field "$bline" copy)")
    done <<<"${result_lines["BASELINE($BASELINE)"]:-}"
    if [ "$bused" -ge 2 ]; then
      local bkmed bsmed
      bkmed="$(printf '%s\n' "${bkarr[@]}" | median)"
      bsmed="$(printf '%s\n' "${bsarr[@]}" | median)"
      echo "  vs BASELINE: kernel=$(awk -v a="$(printf '%s\n' "${k[@]}" | median)" -v b="$bkmed" 'BEGIN{printf "%+.2f (%.1f%%)", a-b, 100*(a-b)/b}')ms  submit=$(awk -v a="$(printf '%s\n' "${s[@]}" | median)" -v b="$bsmed" 'BEGIN{printf "%+.2f (%.1f%%)", a-b, 100*(a-b)/b}')ms"
      local kspr sspr
      kspr="$(printf '%s\n' "${k[@]}" | spread)"
      sspr="$(printf '%s\n' "${bsarr[@]}" | spread)"
      local floor
      floor="$(awk -v a="$kspr" -v b="$sspr" 'BEGIN{printf "%.4f", (a>b?a:b)}')"
      local kdelta
      kdelta="$(awk -v a="$(printf '%s\n' "${k[@]}" | median)" -v b="$bkmed" 'BEGIN{printf "%+.2f", a-b}')"
      # Delta direction: negative = speedup (lower kernel time), positive = regression.
      local dir
      if awk -v d="$kdelta" 'BEGIN{exit !(d<0)}'; then dir="speedup"; else dir="regression"; fi
      local adelta
      adelta="$(awk -v d="$kdelta" 'BEGIN{printf "%+.2f", d}')"
      # Floor check uses magnitude: a genuine improvement (or regression) must CLEAR
      # max(spread of the two arms) across >= 2 paired rounds. A speedup is the
      # direction the PARAMOUNT DIRECTIVE optimizes for, so it qualifies too.
      if awk -v d="$kdelta" -v f="$floor" 'BEGIN{exit !(d<0 && -d>f || d>0 && d>f)}'; then
        echo "  $dir KERNEL CLEARS SPREAD FLOOR: delta=$adelta ms floor=$floor ms (usable rounds=$usable)"
      else
        echo "  $dir kernel INSIDE spread floor: delta=$adelta ms floor=$floor ms (not a result)"
      fi
    fi
  fi
}

emit_row "BASELINE($BASELINE)"
for combo in "${PRODUCT[@]:1}"; do
  emit_row "$combo"
done

} | tee "$LOGDIR/sweep_table.txt"

echo
echo "raw logs: $LOGDIR"
echo "table:    $LOGDIR/sweep_table.txt"
