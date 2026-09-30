# Shared preconditions for gate scripts. Sourced, never executed.
#
# Usage, after the gate has defined SCRIPT_DIR/ROOT:
#     . "$SCRIPT_DIR/_gate_precond.sh"
#     gate_require_gpus 2 || exit 2
#
# WHY. A gate that needs two GPUs and runs on a one-GPU box does not report a
# defect, it reports the BOX. But `--gpu --tp 2` does not fail on a 1-GPU box:
# Engine::load clamps to min(tp_devices, dm.size()) (AUDIT #11), so every TP=2 arm
# silently runs as TP=1 and the gate then fails an assertion about a path it
# never exercised. That is a false FAIL, and it is the reason AGENTS.md carries
# "device_argmax FAIL" as a standing item that is really box state.
#
# The two failure modes to avoid:
#   * bare FAIL  -- indistinguishable from a real regression, so it trains
#                   everyone to ignore red gates.
#   * exit 0     -- a silent skip reads as a pass, which is worse: the gate
#                   claims coverage it did not provide.
# So a blocked precondition exits 2, LOUDLY, and the convention is already
# established by test_device_argmax.sh ("Exit 0 PASS, 1 FAIL, 2 SKIP").
#
# Counted from `sycl-ls` `[level_zero:gpu]` lines, which is the same adapter the
# engine enumerates -- deliberately NOT opencl:gpu, which can report a device
# the Level Zero backend will not use. A missing libumf on LD_LIBRARY_PATH
# makes sycl-ls report ZERO devices (memory: sycl_env_libumf.md), so a 0 here
# means "cannot verify" and skips rather than passing.

gate_gpu_count() {
  if ! command -v sycl-ls >/dev/null 2>&1; then
    echo "SKIP: sycl-ls not on PATH; cannot count GPUs" >&2
    return 1
  fi
  sycl-ls 2>/dev/null | grep -c "\[level_zero:gpu\]"
}

# $1 = required device count. Exits 2 (SKIP) unless that many are visible.
gate_require_gpus() {
  local need="${1:-}" n
  if [ -z "$need" ]; then
    echo "FAIL: gate_require_gpus called with no count" >&2
    return 1
  fi
  n="$(gate_gpu_count)" || exit 2
  if [ "$n" -lt "$need" ]; then
    echo "SKIPPED (needs $need GPU(s), $n visible): $(basename "${BASH_SOURCE[1]:-$0}")" >&2
    echo "  This is BOX STATE, not a code failure. Re-run on a 2-GPU box." >&2
    echo "  Nothing was verified and nothing regressed." >&2
    exit 2
  fi
  echo "precondition: $n GPU(s) visible (need $need)" >&2
  return 0
}
