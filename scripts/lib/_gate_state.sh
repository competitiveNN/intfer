#!/usr/bin/env bash
# Shared recorder for mutation-matrix state verdicts.
#
# WHY A SHARED RECORDER. Several mutation matrices grade a planted state and then
# append a one-line verdict to a per-run log (STATES_LOG / summary). Inlining
# that printf at every call site lets a state record a verdict on only one
# branch (the M-I defect that started this work) and drifts the column layout
# between matrices. Centralising the append keeps the format identical and makes
# "record on every branch" the path of least resistance.
#
# API:
#   gate_state_record <log> <id> <verdict> <want> [absent]          # standard verdict line
#   gate_state_record --raw <log> <text...>                          # arbitrary line
#   gate_state_record --format <fmt> <log> <value...>                # printf template
#   gate_state_record --stdin <log>                                  # read stdin, echo + append (tee semantics)
#   gate_state_record --stdout ...                                    # also echo to stdout
#   gate_state_record --dry-run ...                                   # preview; do not write
#   gate_state_record --json ...                                      # emit a JSON line to stdout (no file write)
#   gate_state_record --help                                          # print this contract
#
# Exit codes:
#   0  success (or dry-run preview printed)
#   1  missing required argument (log, or id in std mode)
#   2  usage error (unknown option)
#
# A missing log directory is created automatically (except --dry-run/--json,
# which never touch the filesystem). --dry-run and --json are mutually exclusive
# with file writes: they preview/emit only.
gate_state_record() {
  local stdout=0 dry=0 json=0 no_log=0 fmt="" mode="std"
  while [ $# -gt 0 ]; do
    case "$1" in
      --stdout) stdout=1; shift ;;
      --dry-run) dry=1; shift ;;
      --json) json=1; no_log=1; shift ;;
      --raw) mode="raw"; shift ;;
      --format) mode="fmt"; fmt="${2:-}"; shift 2 ;;
      --stdin) mode="stdin"; shift ;;
      --help) sed -n '3,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; return 0 ;;
      --*) echo "gate_state_record: unknown option '$1'" >&2; return 2 ;;
      *) break ;;
    esac
  done
  local log=""
  if [ "$no_log" -eq 0 ]; then
    log="${1:-}"; shift 2>/dev/null || true
    [ -z "$log" ] && return 1
  fi

  # json mode: emit a JSON line to stdout, no file write
  if [ "$json" -eq 1 ]; then
    case "$mode" in
      std)
        local id="${1:-}" verdict="${2:-}" want="${3:-}" absent="${4:-none}"
        [ -z "$id" ] && return 1
        printf '{"id":"%s","verdict":"%s","want":"%s","absent":"%s"}\n' "$id" "$verdict" "$want" "$absent"
        ;;
      raw) printf '{"raw":"%s"}\n' "$*" ;;
      fmt) printf '{"formatted":"%s"}\n' "$(printf "$fmt" "$@")" ;;
      stdin) printf '{"stdin":"%s"}\n' "$(cat)" ;;
    esac
    return 0
  fi

  # dry-run: print the would-be line(s) without writing
  if [ "$dry" -eq 1 ]; then
    case "$mode" in
      std)
        local id="${1:-}" verdict="${2:-}" want="${3:-}" absent="${4:-none}"
        [ -z "$id" ] && return 1
        verdict="$(printf '%s' "$verdict" | tr '[:upper:]' '[:lower:]')"
        printf '%-4s verdict=%-6s want=%s absent=%s\n' "$id" "$verdict" "$want" "$absent"
        ;;
      raw) printf '%s\n' "$*" ;;
      fmt) printf "$fmt" "$@" ;;
      stdin) cat ;;
    esac
    return 0
  fi

  if [ "$no_log" -eq 0 ]; then mkdir -p "$(dirname "$log")" 2>/dev/null || true; fi
  case "$mode" in
    raw)
      if [ "$stdout" -eq 1 ]; then printf '%s\n' "$*" | tee -a "$log"; else printf '%s\n' "$*" >>"$log"; fi
      ;;
    fmt)
      if [ "$stdout" -eq 1 ]; then printf "$fmt" "$@" | tee -a "$log"; else printf "$fmt" "$@" >>"$log"; fi
      ;;
    stdin)
      # Preserve tee semantics: echo to stdout AND append to log.
      tee -a "$log"
      ;;
    std)
      local id="${1:-}" verdict="${2:-}" want="${3:-}" absent="${4:-none}"
      [ -z "$id" ] && return 1
      verdict="$(printf '%s' "$verdict" | tr '[:upper:]' '[:lower:]')"
      local line
      line="$(printf '%-4s verdict=%-6s want=%s absent=%s' "$id" "$verdict" "$want" "$absent")"
      if [ "$stdout" -eq 1 ]; then printf '%s\n' "$line" | tee -a "$log"; else printf '%s\n' "$line" >>"$log"; fi
      ;;
  esac
}
