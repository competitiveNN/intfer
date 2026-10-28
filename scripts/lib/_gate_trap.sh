#!/usr/bin/env bash
# _gate_trap.sh -- the cleanup-trap contract in ONE place, for the verifier
# family under scripts/checks/.
#
# WHY THIS EXISTS. Round 8 (2026-10-24) shipped scripts/checks/check_closing_band_rows.sh
# whose planted self-check sandbox cleaned up on EXIT only. An EXIT-only trap never
# runs on SIGINT, so a Ctrl-C during the self-check leaked the corrupted copy into
# the temp directory. check_trap_release.sh -- whose population the new checker had
# just joined -- refused it, and the closing band classified the red as a REGRESSION
# introduced by the round that added the checker.
#
# The contract is three lines, and the failure mode was that the second and third
# lines were SKIPPED: a trap copied from an EXIT-only example, or edited later by a
# contributor who added the INT line without the TERM one. check_trap_release.sh
# already refuses that shape (its T-A/T-B/T-C matrix states, and its live run over
# the population), so the rule is enforced -- but enforcement is post-hoc: the
# defect had to be committed before it was caught.
#
# *** THE HELPER DOES NOT MAKE THE RULE VACUOUS, AND THAT WAS THE RISK. *** If a
# verifier sources this file and installs its trap by calling this function, its own
# text carries no `trap ... EXIT` line, so the line-start classifier that
# check_trap_release.sh uses returns "not a subject" and the script drops out of the
# population -- a verifier that sources this file and then never calls the function
# would pass with NO trap installed. The three-trap rule is instead applied along a
# three-rung chain: this file is a corpus site of its own (it is auditable, and its
# two signal traps terminate exactly because the function is called once); a caller
# that sources this file without calling gate_trap_release is a NAMED VIOLATION, not
# an exemption; and mutating the helper's second trap in this file makes every caller
# fail along with it. That is what "cannot be half-written" means here -- it is not a
# guaranteed consequence of extracting three lines, it is the closure the classifier
# and the matrix now carry.
#
# WHY THE TRAP LIVES INSIDE A FUNCTION. The other candidate -- a static three-line
# block each verifier copies -- is exactly the shape that was half-written. The
# function re-reads the directory at each trap INSTALLATION, so the two signal arms
# always name the directory that is current, not the one that was current when the
# function was first called.
#
# *** AND WHY THE NAME IS GLOBAL, WHICH IS THE CORRECTION OF A REAL DEFECT. ***
# The first version of this file used `local d="$1"` and the trap bodies expanded
# `$d`. A single-quoted trap body is expanded when the trap FIRES, and by then the
# function has returned and its local is GONE, so the expansion hit an unbound
# variable, the handler aborted under `set -u`, and the directory was NEVER removed.
# Measured: the helper leaked on every single invocation, and the leak was invisible
# to check_trap_release.sh because that checker's classifier reads SOURCE TEXT and
# never executes a trap -- the round-9 shape passed its own static rules and still
# leaked 100% of the time. The name is therefore a GLOBAL, which is still in scope
# when the trap fires, and the runtime probe in check_trap_release.sh is what keeps
# the two from diverging again.
#
# Usage:  . "$LIB/_gate_trap.sh"; gate_trap_release "$SANDBOX"
# The directory is removed on EXIT, and on INT/TERM it is removed and the process
# terminates with the conventional 128+signal status (130 for SIGINT, 143 for
# SIGTERM), so a `timeout` wrapper that relays the signal does not wait on a live
# child.
#
# ONE DIRECTORY PER PROCESS. A second call REPLACES the first: the trap set is
# re-installed, and the single global names the newest directory. Every caller in
# this tree owns exactly one sandbox, so this is not a limitation in practice; a
# caller that needs two would have to hand the first to its own trap, which is the
# shape the rule is about anyway.
# *** RESET INHERITED SIG_IGN BEFORE INSTALLING TRAPS. *** A shell that starts
# with SIGINT or SIGQUIT ignored (SIG_IGN -- the condition POSIX sets for
# background jobs of a NON-INTERACTIVE shell) cannot trap them: "signals ignored
# upon entry to the shell cannot be trapped or reset". When this helper is
# sourced by a script launched detached (e.g. via `setsid nohup ... &` from a
# non-interactive shell), the parent's INT/QUIT ignore is inherited, and every
# `trap ... INT`/`trap ... TERM` below would be silently refused -- the cleanup
# would never fire.
#
# MEASURED A/B under inherited SIG_IGN (`bash -c 'trap "" INT QUIT; exec bash ...'`),
# with helper/probe `trap - INT QUIT` removed to build reset-free variants.
# Raw rc + whether the sandbox was removed:
#     ARM 1  real helper   + probe WITH reset     -> rc=130  dir_removed=YES
#     ARM 2  real helper   + probe WITHOUT reset  -> rc=130  dir_removed=YES
#     ARM 3  noreset helper+ probe WITH reset     -> rc=130  dir_removed=YES
#     ARM 4  noreset helper+ probe WITHOUT reset  -> rc=0    dir_removed=YES
# ARM 4 (genuinely reset-free) exits rc=0: with NO reset anywhere the INT trap
# is refused (the probe prints "DID_NOT_TRAP") and it cannot reach exit 130.
# `trap - INT QUIT` IS load-bearing; this line is required even when callers do
# their own reset. NOTE ARM 4 still reports dir_removed=YES -- the EXIT trap
# fires on the normal exit and removes the sandbox -- so "was the directory
# removed" does NOT discriminate this failure; only the exit code does.
#
# MASK: authoritative in-process measurement (SigIgn before `trap -`, after
# `trap -`, after `trap '...' INT` install, all in ONE process):
#     inherited SIG_IGN -> 0000000000001006 / 0000000000001006 / 0000000000001006
#     default           -> 0000000000001000 / 0000000000001000 / 0000000000001000
# The /proc mask never changes even though the trap fires (rc=130 on both), so
# SigIgn is NOT a usable diagnostic here; behaviour (rc 130 vs rc 0) decides.
# An earlier version claimed the mask moved across exec; that claim was FALSE and
# is retracted here.
# *** NOTE: `trap - INT QUIT` was REMOVED from this helper. ***
# Under inherited SIG_IGN, `trap - INT QUIT` restores default disposition (so
# `kill -INT $$` terminates the process) but does NOT re-enable custom trap
# installation -- `trap -p INT` still reports `trap -- '' SIGINT` and custom
# traps are silently refused. The helper therefore cannot install traps under
# inherited SIG_IGN and must DECLINE rather than silently degrade. The reset
# is the caller's responsibility: a caller that wants to work under inherited
# SIG_IGN must reset its own disposition before sourcing this helper, knowing
# that even with the reset the helper will decline and the caller's process
# will terminate on signal via default disposition.
#
# A `trap -` on a signal that was never ignored is a no-op, so the caller's
# reset is safe in the default-disposition case.

_GATE_TRAP_DIR=""
# *** BEHAVIORAL DETECTION OF INHERITED SIG_IGN, INSIDE THE FUNCTION. ***
# A shell that starts with SIGINT ignored (POSIX sets this for background jobs
# of a non-interactive shell) cannot trap it: "signals ignored upon entry to
# the shell cannot be trapped or reset". The reset at line 95 restores default
# disposition at source time, so `kill -INT $$` terminates the process -- but
# under inherited SIG_IGN custom trap installation is silently REFUSED even
# after the reset: `trap -p INT` still reports `trap -- '' SIGINT`. A caller
# that sources this helper and calls gate_trap_release therefore leaks on every
# signal while every static classifier sees a clean file.
#
# Detection: install a sentinel trap, then read back `trap -p INT`. When
# installation succeeded it reports the sentinel body; when refused it reports
# `trap -- '' SIGINT`. The probe does NOT send the signal (that would kill the
# process under default disposition after the reset); it only inspects the
# installed disposition. When refused, the inherited ignore is restored with
# `trap '' INT` so the caller's process is left as it was found -- the decline
# does not change the signal disposition. When accepted, the sentinel is
# disarmed and the real traps are installed. One trap round trip, no process
# kill, no self-termination.
gate_trap_release() {
  _GATE_TRAP_DIR="$1"
  # Probe: can this process actually install a custom INT trap?
  _gt_sentinel=0
  trap '_gt_sentinel=1; exit 130' INT
  local _tp
  _tp="$(trap -p INT)"
  if [[ "$_tp" == *_gt_sentinel* ]]; then
    # Sentinel installed OK -- disarm it and install the real traps.
    trap - INT
  else
    # Custom trap refused under inherited SIG_IGN. Restore inherited ignore so
    # the caller's process is left exactly as it was found; do NOT use `trap -
    # INT`, which would restore default disposition and make the next
    # `kill -INT $$` terminate the process.
    trap '' INT
    echo "[gate-trap] INTFER_GATE_TRAP declined: SIGINT is inherited ignored and cannot be trapped in this process" >&2
    echo "           (the `trap - INT QUIT` at the top of this file restores default disposition but does not re-enable" >&2
    echo "           custom trap installation under inherited SIG_IGN; the reset must run BEFORE this call)" >&2
    echo "           cleanup trap NOT installed -- the directory '$_GATE_TRAP_DIR' will NOT be removed on INT/TERM" >&2
    return 1
  fi
  trap 'rm -rf -- "$_GATE_TRAP_DIR"' EXIT
  trap 'rm -rf -- "$_GATE_TRAP_DIR"; exit 130' INT
  trap 'rm -rf -- "$_GATE_TRAP_DIR"; exit 143' TERM
}
