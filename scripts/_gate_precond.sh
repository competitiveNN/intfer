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
# Env: INTFER_SYCL_LS  override the sycl-ls binary (default: sycl-ls on PATH).
#   Needed because `source setvars.sh` PREPENDS the oneAPI bin dir, so a PATH stub
#   for sycl-ls is shadowed by the real one inside every gate -- the 0-device
#   direction was untestable end-to-end until this seam existed. Same rationale as
#   the INTFER_BUILD_DIR override: an assumption baked into a probe that the
#   harness cannot redirect is an assumption the harness cannot test.
# ----------------------------------------------------------------------------
# ENVIRONMENT vs BOX STATE. Keep these two apart, LOUDLY.
#
# MEASURED, TWICE, IN ONE SESSION, AND BOTH TIMES I GOT IT WRONG. A previous
# revision of this work reported "the box's in-process SYCL discovery returns no
# device -- the documented container/host driver mismatch" and told the operator
# the 27B handoff could not run. It was a MISSING `source setvars.sh` in my own
# shell. The second time, check_device_precond.sh sat until a timeout (rc=124)
# for the same reason. In both cases the tool printed a line that reads like a
# statement about the HARDWARE, and I believed it:
#
#     SKIP: sycl-ls not available; cannot count GPUs
#
# That line is about PATH. Nothing about it is evidence about the machine. It is
# also the single most expensive class of mistake available here, because it
# turns "I forgot to set up my environment" into "the hardware is broken" -- a
# claim that is unfalsifiable from the session that made it and that sends the
# operator to debug a box that is fine.
#
# So: every "cannot measure" path below now names ENVIRONMENT FIRST and says
# what to do about it, and says explicitly that it is NOT box state. Exit codes
# are UNCHANGED (rc 1 = unmeasurable, exit 2 = skip) because callers and the
# mutation matrices depend on them; only the text is different. The audit #2 rule
# -- "0 findings" and "found nothing to look at" are different claims -- has a
# sibling here: "the tool is missing" and "the tool ran and found nothing" are
# different claims, and they must never share a sentence.
#
# The diagnostic is a PROBE, not an assumption: it checks for a oneAPI env var
# and for sycl-ls itself, and says which one is missing rather than guessing.
gate_env_diagnose() {
  local reason="${1:-}"
  echo "" >&2
  echo "  ================ ENVIRONMENT, NOT BOX STATE ================" >&2
  echo "  This shell cannot MEASURE the device population. That is a" >&2
  echo "  statement about your SHELL, not about this machine." >&2
  [ -n "$reason" ] && echo "  Reason: $reason" >&2
  if [ -z "${ONEAPI_ROOT:-}${CMPLR_CLANG_80_SPIRV_64_ENV:-}" ]; then
    echo "  No oneAPI environment variables are set (ONEAPI_ROOT is unset)," >&2
    echo "  which means setvars.sh was never sourced." >&2
  fi
  if ! command -v sycl-ls >/dev/null 2>&1 && [ -z "${INTFER_SYCL_LS:-}" ]; then
    echo "  'sycl-ls' is not on PATH." >&2
  fi
  echo "  FIX:  source /opt/intel/oneapi/setvars.sh     (then re-run)" >&2
  echo "  Verify with: scripts/l0_enumerate.sh  -> TOTAL LEVEL ZERO DEVICES: 2" >&2
  echo "  Do NOT report this as a hardware or driver fault. It is not." >&2
  echo "  ===========================================================" >&2
  echo "" >&2
}

gate_gpu_count() {
  local sycl_ls="${INTFER_SYCL_LS:-sycl-ls}"
  if [ ! -x "$sycl_ls" ] && ! command -v "$sycl_ls" >/dev/null 2>&1; then
    echo "SKIP: $sycl_ls not available; cannot count GPUs" >&2
    gate_env_diagnose "'$sycl_ls' is not executable on PATH."
    return 1
  fi
  local n
  n="$("$sycl_ls" 2>/dev/null | grep -c "\[level_zero:gpu\]")" || true
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
  echo "  Nothing was verified and nothing regressed. A GPU gate reaching" >&2
  echo "  this point NEEDS a device, so it must skip rather than fail red --" >&2
  echo "  see gate_inventory.sh's NEEDS-GPU column." >&2
  # The render node is usually box state, but it is ALSO reachable from an
  # unprepared shell (no /dev/dri in the container's view, wrong device cgroup),
  # and this session already misattributed that class once. So the diagnostic is
  # offered rather than asserted: it reports what it can see and stops short of
  # claiming to know which of the two this is.
  gate_env_diagnose "/dev/dri/renderD128 could not be opened (group membership,
  device cgroup, or an unprepared container are the usual causes)."
  exit 2
}

# A gate that needs a DEVICE has two independent ways to be unable to get one,
# and audit #15 closed only the first:
#
#   (a) the render node cannot be opened from this shell   -> covered below
#   (b) the node opens fine but Level Zero enumerates ZERO devices, because
#       libumf is missing from LD_LIBRARY_PATH (memory: sycl_env_libumf.md)
#
# (b) is the common one on this exact box and it is a RED line, not a skip:
# test_dpas_fold.sh / test_http_batch.sh / test_serve_load.sh all carried only
# the render-node check, so on a box with (b) and not (a) they would have gone
# red and the red would be indistinguishable from a regression. So a gate that
# needs a device wants BOTH, and this is the helper that means both.
#
# Order matters and is deliberate: probe the node FIRST. The node probe is a
# direct syscall and costs nothing; the sycl-ls count spawns a process. And if
# the node is unreachable the count is meaningless anyway.
#
# rc 0 = a device is reachable, 1 = not. Exits 2 (SKIP) when not, with the same
# "this is BOX STATE" framing as the other two.
gate_require_device() {
  if ! python3 -c "open('/dev/dri/renderD128','rb').close()" 2>/dev/null; then
    echo "SKIPPED: cannot open /dev/dri/renderD128 from this shell." >&2
    echo "  BOX STATE, not a code failure. Nothing was verified and nothing" >&2
    echo "  regressed." >&2
    exit 2
  fi
  # "CANNOT COUNT" is not a pass, and the distinction is load-bearing.
  # gate_gpu_count returns rc=1 with EMPTY stdout when sycl-ls is not on PATH
  # (that is how it signals "unmeasurable", vs rc=0 with an integer). An empty
  # n makes `[ "" -lt 1 ]` a bash INTEGER ERROR whose result is FALSE -- so the
  # helper would return 0, i.e. a SILENT NO-OP, on exactly the box it exists to
  # catch. Caught by running the fixture directions rather than reading the
  # function: the audit-#14 "two failure modes in one value" class, recurring one
  # function down. Hence the explicit rc and emptiness tests, not a bare -lt.
  local n rc=0
  n="$(gate_gpu_count)" || rc=$?
  if [ "$rc" != "0" ] || [ -z "$n" ]; then
    echo "SKIPPED: render node opens but sycl-ls cannot count devices." >&2
    echo "  Not a pass and not a device: the precondition is UNMEASURABLE, so" >&2
    echo "  nothing was verified." >&2
    gate_env_diagnose "sycl-ls cannot count devices."
    exit 2
  fi
  if [ "$n" -lt 1 ]; then
    # Distinguish "sycl-ls works and found none" from "sycl-ls is unusable" --
    # both mean no device here, but only one is the libumf LD_LIBRARY_PATH
    # failure, and that is the one worth naming.
    echo "SKIPPED: render node opens but Level Zero enumerates 0 device(s)." >&2
    echo "  BOX STATE, not a code failure. This is the libumf LD_LIBRARY_PATH" >&2
    echo "  failure mode (memory: sycl_env_libumf.md): source" >&2
    echo "  /opt/intel/oneapi/setvars.sh, which puts umf on the path." >&2
    gate_env_diagnose "Level Zero enumerated 0 device(s) -- umf/LD_LIBRARY_PATH."
    exit 2
  fi
  return 0
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
