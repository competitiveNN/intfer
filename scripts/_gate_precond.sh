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

# Prints EXACTLY ONE integer line, and the return code carries the distinction:
#   0 = counted successfully (the integer may legitimately be 0)
#   1 = could NOT count (sycl-ls unavailable); stdout is EMPTY
#
# WHY THE SHAPE MATTERS (audit #14). The previous body was a bare
# `sycl-ls | grep -c`. `grep -c` prints "0" AND exits 1 when nothing matches, so
# every caller's `n="$(gate_gpu_count || echo 0)"` captured the TWO-LINE string
# "0\n0" on exactly the case this helper exists to handle: sycl-ls present,
# Level Zero reporting no device (the libumf LD_LIBRARY_PATH failure, memory
# sycl_env_libumf.md). The following `[ "$n" -ge 2 ]` then printed
# `[: 0
# 0: integer expected` to stderr. The verdict came out right by accident -- a
# failed test is non-zero -- but gate_inventory's `NDEV -eq 0` loud-skip branch
# could never be taken, so on the one box state it exists for it silently fell
# into the "GPUs are visible, run the tally" path instead.
#
# So: the number goes to stdout as one clean line, and "could not count" is
# signalled ONLY by rc, never by text mixed into the value. Callers should use
# `N="$(gate_gpu_count)" || N=0` -- NOT `|| echo 0`, which cannot distinguish a
# failed substitution from a failed count.
gate_gpu_count() {
  if ! command -v sycl-ls >/dev/null 2>&1; then
    echo "SKIP: sycl-ls not on PATH; cannot count GPUs" >&2
    return 1
  fi
  local n
  n="$(sycl-ls 2>/dev/null | grep -c "\[level_zero:gpu\]")" || true
  printf '%s\n' "${n:-0}"
  return 0
}

# The cheaper, stricter precondition: can this shell OPEN THE RENDER NODE at all?
# Stricter than the sycl-ls count, because a missing libumf makes sycl-ls report
# zero devices on a box whose GPUs are perfectly healthy -- the count would then
# refuse coverage that would actually run. Opening the node is a direct syscall,
# so it cannot be fooled by an adapter that failed to load.
#
# rc 0 = the node opened, 1 = cannot open. Exits 2 (SKIP) when it cannot, with
# the same "this is BOX STATE" framing as gate_require_gpus. Callers that need a
# device COUNT as well (the TP=2 arms) use both, never this alone.
gate_require_render_node() {
  if python3 -c "open('/dev/dri/renderD128','rb').close()" 2>/dev/null; then
    return 0
  fi
  echo "SKIPPED: cannot open /dev/dri/renderD128 from this shell." >&2
  echo "  BOX STATE, not a code failure. Nothing was verified and nothing" >&2
  echo "  regressed. A GPU gate reaching this point NEEDS a device, so it must" >&2
  echo "  skip rather than fail red -- see gate_inventory.sh's NEEDS-GPU column." >&2
  exit 2
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
