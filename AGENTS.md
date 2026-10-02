Port ninfer (https://github.com/Neroued/ninfer) to our local setup made of intel arc pro b580 (2x gpus)

## READ FIRST — current gate state, and how to read the history below

**The box currently enumerates ONE Arc B580** (`sycl-ls` prints a single
`[level_zero:gpu]` line). The design target is still 2 GPUs, but the second one
is not currently visible — see AUDIT #11, where it vanished mid-session. That
single fact governs how every gate result in this file must be read:

  === 2026-10-01: THE HARDWARE IS FINE; ONLY THE *ENUMERATION* IS DOWN ===
  Re-measured from scratch. This is NOT a "1-GPU machine" and must not be
  described as one — that phrasing caused a wrong conclusion in this session.
  BOTH B580s are present and healthy at kernel level:
    * card0 -> 0000:04:00.0 and card1 -> 0000:86:00.0, on DIFFERENT PCI roots
      (pci0000:00 and pci0000:80); both 8086:E20B, both DRIVER=xe, both
      runtime_status=D0; both expose tile0 + gt0/gt1 + bcs/ccs/rcs/vcs/vecs
    * /dev/dri/{card0,card1,renderD128,renderD129} all present (226:0, 226:1,
      226:128, 226:129) and all open O_RDWR successfully
  But the COMPUTE RUNTIME exposes one device: `sycl-ls` -> 1 [level_zero:gpu],
  `xpu-smi --list-gpus` -> "GPU 0" only.
  RULED OUT BY MEASUREMENT (do not re-try these): the L0 driver library version
  — the container has libze_intel_gpu 1.15.38646 and the host has 1.17.39758, and
  forcing the host's newer one onto LD_LIBRARY_PATH still yields 1 device;
  cgroup device policy (no devices.list, and both render nodes open); NUMA (one
  node, both cards numa=-1); power state (both D0); another process holding
  GPU 1 (`fuser` shows no userspace holder on renderD129).
  *** LEADING HYPOTHESIS, NOT ESTABLISHED: host-side enumeration. *** The only
  real evidence is that Level Zero AND OpenCL (via the unrelated intel64.icd)
  both report 1 GPU while both cards are kernel-healthy. That is CONSISTENT
  WITH a host/driver enumeration drop of 86:00.0, but it is inference from
  agreement — precisely the mistake AUDIT #9 made that AUDIT #11 had to retract.
  **I did not find the mechanism.** Any "fix" (rebind/rescan 86:00.0) is a
  suggestion to investigate, not a diagnosed remedy, and may be wrong. Read the
  xe/Level Zero enumeration path before trusting any of it. Details + the full
  exclusion list: reports/2026-10-01-batch-e1-e2-e3-verdicts.md.
  *** LOCALIZED, 2026-10-01 (later, and stronger than the line above). *** A raw
  Level Zero probe bypassing SYCL/UR (zeInit -> zeDriverGet -> zeDeviceGet on
  libze_loader) returns **1 driver, 1 device**, uuid
  86800be2-0000-0000-0400-000000000000. Group 4 of an Intel L0 UUID encodes
  PCI bus:dev:function, so that is **04:00.0 = card0**; xpu-smi reports the same
  uuid. A card1 device would carry group 4 == 8600 and is **ABSENT**. So the
  loss is NOT in SYCL, NOT in UR, and NOT in device_manager.cpp:37 (same path
  sycl-ls uses) — **the L0 driver's own zeDeviceGet does not return card1.**
  REPRODUCE: `scripts/l0_enumerate.sh` (and with
  `INTFER_L0_DRIVER_LIB=/run/host/usr/lib64` to test a different driver lib).
  *** DRIVER-LIBRARY VERSION RULED OUT, measured. *** Container
  libze_intel_gpu 1.15.38646 (driverVersion 0x10396f6) and host 1.17.39758
  (0x1039b4e) BOTH return exactly 1 device, same uuid. The probe prints
  driverVersion precisely so a non-vacuous swap is visible.
  *** ONE card1-ONLY KERNEL ANOMALY — REAL, BUT NOW REFUTED AS THE CAUSE. ***
  `scripts/gpu_health_check.sh` diffs the two cards' sysfs: exactly four
  attributes fail on card1 and only card1 —
  tile0/gt{0,1}/freq0/{min,max}_freq return **ECANCELED (errno 125)**, 8/8
  reproducible, while cur_freq/rp0_freq/num_csiles/ccs_mode match card0 exactly.
  **It cannot drop a device.** Reading compute-runtime's discovery code
  (drm_neo.cpp) shows that exact file is read by getMaxGpuFrequency() and the
  **return value is DISCARDED** at product_helper_drm.cpp:85 — a sysfs read
  whose result is thrown away cannot gate enumeration. The companion GuC theory
  is refuted too: `grep -rniE 'GuC' shared/source/` finds only UAPI constant
  names, i.e. **no GuC load or check exists in the user-space driver**, and
  DRM_XE_DEVICE_QUERY_HWCONFIG's return value is ignored. Kept recorded as a
  genuine card0/card1 difference; do NOT re-offer it as the cause.
  *** WHERE THE DROP ACTUALLY HAPPENS (code-read, 2026-10-01). *** The driver
  scans **/dev/dri/by-path** (not /sys/class/drm), open(O_RDWR) +
  DRM_IOCTL_VERSION, then a per-card root-device build. A card is SILENTLY
  dropped on: (a) open() fails, (b) DRM_IOCTL_VERSION name not exactly
  "i915"/"xe", (c) DRM_IOCTL_XE_DEVICE_QUERY fails, (d) setupHardwareInfo() fails
  (unknown id / caps / **small BAR** / topology / EUCount==0), (e) root-device
  creation fails or ZE_AFFINITY_MASK filters it. (b) is already excluded by
  measurement (our driver is xe).
  *** STRONGEST CANDIDATE: SMALL BAR on card1. *** drm_neo.cpp:582-590 returns
  -1 (drop) when `isSmallBarDetected()` (any VRAM region with
  cpu_visible_size < total_size) **and** !isSmallBarConfigAllowed() — and
  `IoctlHelperXe::isSmallBarConfigAllowed()` returns **false** (i915's returns
  true). This is a real upstream defect, not a guess: compute-runtime issue
  #905 ("silently rejected the DG2 device when using the xe driver ... while
  i915 worked fine with small BAR"). Our cards are on DIFFERENT PCI roots,
  exactly where ReBAR often differs. **NOT YET CHECKED — the check is
  host-side and CANNOT be done here (measured):** lspci/setpci not installed;
  /sys/.../resource0,2 open -> EACCES; /sys/.../config readable but only the
  first 64 bytes, and ReBAR is PCIe ext cap 0x0015 at offset 0x100+.
  *** ALSO NOT EXCLUDED: a BMG-specific defect. *** 0xE20B IS IGFX_BMG
  (devices_base.inl:62), so the B580 is BMG. compute-runtime issue #921 reports
  exactly 2x B-series enumerating as 1 with raw zeDeviceGet affected, fixed by
  restoring initializeInternalEngineImmediately() to true
  (bmg/os_agnostic_product_helper_bmg.inl:73, now false). **But** our driver
  1.15.38646 (~NEO 26.22) is AFTER the 25.44 regression point, and in that
  thread the raw-L0 probe saw 2 devices while UR failed — so it is NOT
  established as our mechanism. Do not assume it.
  *** WHY WE CANNOT GET THE DRIVER TO SAY WHICH ONE. ***
  `scripts/l0_enum_trace.sh` runs the real driver with
  `NEOReadDebugKeys=1 PrintDebugMessages=1` (the flags that print skip (c)/(d)).
  Measured: it prints unrelated banners but **no skip diagnostic** — the release
  build has PrintDebugMessages compiled out, so the skip is silent by
  construction on this build.
  *** CANNOT BE DIAGNOSED OR FIXED FROM THIS CONTAINER (all measured). ***
  dmesg = EPERM (no kernel log, so no xe BAR/GuC/reset message is visible);
  writing remove/rescan/reset = EACCES (unprivileged), so a rebind of 86:00.0
  cannot even be TESTED here.
  *** HOST-SIDE PLAN, IN ORDER (this is a PLAN, not a diagnosis). ***
  1. `dmesg | grep -iE 'xe.*86:00|resizable|rebar'` — a BAR/ReBAR complaint
     confirms or refutes the strongest candidate.
  2. Compare BAR sizes for the two slots: `lspci -vv -s 04:00.0` vs
     `-s 86:00.0`, Region 2 / Resizable BAR vs VRAM size.
     **Equal BAR sizes refutes the small-BAR candidate outright.**
  3. Only then try a rebind/rescan of 86:00.0 and re-run
     `scripts/l0_enumerate.sh`. **The rebind is an untested CANDIDATE, not a
     remedy** — nothing here shows it would help.
  4. If the BARs are fine, the next step is a DEBUG build of compute-runtime to
     get the skip reason printed — not more guessing from this container.
  REPRODUCE ALL OF IT (committed scripts, not prose):

**The vacuous-green tooling (AUDIT #20).** Shared guard, its self-check, its
matrix, and the shared patch helper — all dispatched steps of
`run_all_checks.sh`, so section 0b fails if any loses its arm:
  - `scripts/_gate_enum.sh` — `enum_require_nonempty` / `enum_require_min`, the
    ONE home for "a checker that cannot harvest its subject cannot certify it".
    Sourced by `check_doc_counts.sh`, `check_build_dir_consumers.sh` and
    `gate_inventory.sh`.
  - `scripts/check_gate_enum.sh` — behavioural self-check for that guard, with
    the required CONVERSE: must REJECT 0/empty/non-numeric/below-minimum, and
    must ACCEPT well-formed input SILENTLY.
  - `scripts/mutate_gate_enum.sh` — mutation matrix for both, and for the two
    CONSUMERS, because both historical vacuous-greens lived in the consumers
    rather than in the helper. Patches sandbox copies under the build tree.
  - `scripts/_gate_patch.sh` — `patch_literal` (exact, must occur exactly once),
    `verify_patched`, `residue_scan`. New matrices source this instead of
    adding another copy; the three remaining duplicates are a recorded decision
    with explicit conversion triggers, see the `patch_literal` note below.
  - `scripts/_gate_pred.sh` — `pred_expect_absent` / `pred_expect_present`, so a
    comparison NAMES its polarity at the call site. See the inverted-predicate
    rule below.
  - `scripts/_gate_pop.sh` — `pop_dirs` / `known_dirs` / `pop_unknown_dirs` /
    `pop_assert_no_leak`. The set of stamp directories a matrix legitimately
    owns is **derived** from each `mutate_*.sh`'s own `WORK=`/`LOG_DIR=`/`STAMP=`
    assignment, never hand-listed: an omission is silent, and the correct fix
    (add the entry) is indistinguishable from the wrong one. It also covers the
    three prefix forms real matrices write (`${INTFER_BUILD_DIR:-$ROOT/build}/`,
    `$BUILD_DIR/`, `$PWD/build/`) and accepts EITHER `INTFER_BUILD_DIR` or a
    local `BUILD_DIR`. `scripts/check_gate_pop.sh` is its own behavioural
    self-check, written because the helper had already produced two defects
    during conversion — a **false LEAK** on a clean tree from assuming
    `$PWD/build`, and a **vacuous** assertion from demanding an exported
    variable — and case (e) of that check then found a third: with neither
    variable set it enumerated an empty path and still returned rc=0.
  - `scripts/l0_enumerate.sh` — device count + uuid at the driver API. Honours
    INTFER_L0_DRIVER_LIB and prints driverVersion so a no-op lib swap is
    visible. Deletes its own output and refuses to run a stale binary.
  - `scripts/gpu_recover.sh` — **RUN THIS ON THE HOST, AS ROOT.** It exists
    because two decisive measurements were blocked by the CONTAINER, not by the
    problem: PCI config space reads only the first 64 bytes here and Resizable
    BAR lives at offset 0x100+, and writes to `/sys/bus/pci/...` are EACCES, so
    the remedy could not even be attempted. Subcommands: `diag` (read-only, no
    root), `rebind` (root; unbind+rebind, recording the BAR layout before and
    after), `verify` (re-count at the driver API).
    | exit | meaning |
    |---|---|
    | 0 | 2+ devices enumerated — the fault is gone; follow the trigger |
    | 1 | MEASURED, still 1 device — the fault persists |
    | 2 | **UNMEASURABLE** — the probe could not count; *not* "1 device" |
    | 3 | needs root (`rebind` only); nothing was touched |
    `scripts/check_gpu_recover.sh` pins all four with a STUBBED probe.
    *** `rebind` IS AN UNTESTED CANDIDATE, NOT A DIAGNOSED REMEDY. *** Nothing
    here shows a rebind would help. And the strongest lead — a small VRAM BAR on
    card1, which compute-runtime silently drops on xe because
    `IoctlHelperXe::isSmallBarConfigAllowed()` returns false — **has NO
    in-software fix at all**: it is a BIOS/slot setting (Above 4G Decoding +
    Resizable BAR) or a different slot. `diag` prints `DECISIVE TEST NOT
    MEASURED` when `lspci` is absent, because a check that cannot harvest the
    BAR sizes must not certify that the theory was refuted (AUDIT #20).
  - `scripts/gpu_driver_lib_ab.sh` — the committed A/B over driver-lib versions.
    **Exits 1 "INCONCLUSIVE" if both arms report the same driverVersion**, since
    then the libraries were never actually swapped and the comparison proves
    nothing. Negative-tested by pointing both arms at one lib. Measured
    NEGATIVE: container 0x10396f6 and host 0x1039b4e both enumerate 1 device.
  - `scripts/gpu_health_check.sh` — card0-vs-card1 sysfs diff. Runs a GATING
    self-check first (seeds a `card0-DP-1` connector, requires it rejected),
    and prints HEAD so its output is attributable to a tree.
  - `scripts/l0_enum_trace.sh` — the debug-flag attempt. **Exits 3
    = INCONCLUSIVE** when the driver prints no skip reason, so "no output" can
    never be misread as "no information". Measured here: exit 3.
  - `scripts/check_l0_cite.sh` — verifies every compute-runtime `file:line`
    cited in AGENTS.md / the verdicts report against a **pinned commit
    63647bb8214cdc581d2bd5ed3822f157f7d34679**, so the citations are checkable
    rather than a research-subagent transcript. Clones sparse on demand;
    if the pin cannot be resolved it exits 2 and the docs must then be read as
    UNVERIFIED.
  - `scripts/check_checker_matrix_coverage.sh` — DERIVES the "every checker has
    a dispatched mutation matrix" claim from `scripts/` instead of trusting a
    hand-written total. It exists because that claim had only ever been prose,
    and the property is load-bearing: three vacuous-greens shipped green
    because no matrix made them fail. Deliberately has **no** "orphan matrix"
    rule — `mutate_fold_matrix.sh`, `mutate_gate_inventory.sh` and
    `mutate_run_all_checks.sh` have no same-named checker because they check
    OTHER files, and flagging them produced three false alarms on a healthy
    tree. A check that cries wolf is the mirror defect.
    *** NO TOTAL IS QUOTED HERE, ON PURPOSE. *** Per "every hand-counted total
    is a citation", the checker/matrix counts are DERIVED at run time and
    deliberately not transcribed into prose -- a number written here goes stale
    the moment a script is added. Re-derive on demand:
        scripts/check_checker_matrix_coverage.sh
    It enumerates `check_*.sh` and `mutate_*.sh` (the same population
    run_all_checks.sh section 0b uses) and exits 1 on any drift.
  - `scripts/mutate_checker_matrix_coverage.sh` — its matrix, 3/3. Each state
    does BOTH things: it plants a checker with no matrix in the sandbox tree AND
    removes the requirement that would catch it. Weakening alone is
    UNOBSERVABLE while the real tree has full coverage (AUDIT #21) — the first
    attempt at this matrix scored 0/3 for exactly that reason and was reverted.
    States judge on the VERDICT line, not on the table: the table still prints
    "MISSING" for the planted checker even with the requirement removed, so only
    the `bad` increment and exit code disappear.
  - `scripts/mutate_l0_cite.sh` — mutation matrix for that checker, 7/7 states
    caught each for its own reason, stamped `build/l0_cite_mutate/last_pass.stamp`.
    It also carries a **stamp-population leak assertion** (`leak_assert()`),
    called THREE times against ONE predicate — as-found (before any mutation),
    poisoned-by-M-H (fixture still on disk), and after-states. Measured: clean
    tree exit 0; a fixture pre-planted in `$BUILD_DIR` exit 1.
  - `scripts/check_e2_ctx.sh` — the E2 long-context identity checker. The
    `check_*` prefix is load-bearing: `gate_inventory.sh` derives the GATE
    population from `ls scripts/test_*.sh`, so a `check_*` name cannot be
    misclassified as a gate carrying no device guard (AUDIT #17), while
    `run_all_checks.sh` 0b enumerates `check_*`/`mutate_*` and requires a
    dispatch arm for each. **Both new checkers and their matrix honor
    `INTFER_BUILD_DIR`** (AUDIT #8: an artifact consumer that ignores the
    override silently validates the WRONG tree); verified by pointing the
    override at an empty directory, which yields rc=2 SKIP rather than silently
    falling back to the default tree.
  - `scripts/gpu_driver_lib_ab.sh` — committed driver-lib A/B; exits 1
    INCONCLUSIVE if both arms report the same driverVersion.
  - `scripts/l0_enum_trace.sh` — debug-flag attempt; exits 3 INCONCLUSIVE when
    the driver prints no skip reason.
  Exit codes for all of the above are tabled in
  `reports/2026-10-01-batch-e1-e2-e3-verdicts.md`.
  *** NEXT-SESSION TRIGGER — the one thing worth reopening. *** If
  `scripts/l0_enumerate.sh` ever prints **2** devices, the enumeration fault is
  gone. Immediately close the ONLY remaining unverified coverage, in order:
  1. `sycl-ls | grep -c '\[level_zero:gpu\]'` must print 2, and
     `scripts/run_all_checks.sh` must be re-run end-to-end so the stamps and
     the doc tally are re-derived against the new box state. Expect `tp2_27b`
     to stop skipping and `gpu_optin`/`device_argmax` to stop reporting
     PARTIAL — i.e. the two PARTIAL arms gain exactly the TP=2 coverage they
     currently lack.
  2. Re-run `scripts/test_tp2_27b.sh`: its sole arm is `--tp 2` on 27B, so it is
     the ONLY gate that can certify TP=2 coverage at all right now.
  3. Then, and only then, fill the `RESULT:` lines in
     `reports/2026-09-30-pending-bench-backlog.md` and
     `reports/2026-09-30-dpas48-fold.md`, and re-decide the E1/E2/E3 verdicts at
     27B — they currently rest on 0.8B counters alone.
  Also negative, so they are not retried: ONEAPI_DEVICE_SELECTOR in
  {level_zero:gpu, level_zero:*, *:gpu, level_zero:0, *} and SYCL_DEVICE_FILTER
  in {level_zero:gpu, *} all still yield 1; /dev/accel/accel0 is the 265KF's
  INTEGRATED gpu at 0000:00:0b.0, not a B580.
  CONSEQUENCE (reproduced 2026-10-01): 27B at --tp 2 clamps to TP=1, preloads
  18751 MiB onto a 12216 MiB card, dies `signal 6` / rc=139. Every 27B `--tp 2`
  command in reports/ is unrunnable until enumeration is restored.

  ==> **2026-10-01: BOTH GPUs ENUMERATE AGAIN.** A host-side REBIND of the
  missing card (PCI 86:00.0) by the operator restored the second device;
  `sycl-ls` prints 2 `[level_zero:gpu]` entries and the L0 UUIDs are
  `...-0400-...` (04:00.0) and `...-8600-...` (86:00.0). `scripts/gpu_recover.sh
  rebind` is the tool that does it; `verify` now exits 0.
  **THIS REFUTES THE SMALL-BAR HYPOTHESIS.** The leading theory for weeks was
  that card1 was dropped because its VRAM BAR is smaller than its VRAM
  (`IoctlHelperXe::isSmallBarConfigAllowed()` returns false; upstream #905). A
  plain rebind fixed it, so that was NOT the cause — it was driver state. The
  `min/max_freq` ECANCELED refutation (non-gating path) still stands.
  ==> **RUNBOOK IF IT DROPS AGAIN — DRIVER STATE FIRST.** A plain rebind of the
  missing card fixed it; nothing about the hardware, driver-library version,
  cgroup, NUMA, power or BAR sizes had to change.
    1. `scripts/gpu_recover.sh diag` — read-only. Confirms BOTH cards are still
       xe-bound and that only the ENUMERATION dropped. A card that is unbound
       needs a rebind and nothing else.
    2. `sudo scripts/gpu_recover.sh rebind` — unbind + rebind, recording the PCI
       `resource` layout before and after, then `verify`. **This is what worked.**
    3. `verify`: 0 = 2 devices, 1 = MEASURED 1 device, 2 = UNMEASURABLE. A box
       that cannot count its devices is not a box with one device.
  *** DO NOT LEAD WITH THE BAR THEORY. *** Small BAR is a real drop path in
  compute-runtime and worth knowing, but it was NOT this failure — a rebind
  fixed it with no BAR change. Investigate BARs only if a rebind stops working.
- **Current full-suite result: 10 PASS, 0 PARTIAL, 0 SKIPPED** (rc = 0 for all
  ten). The ten gates: `cpu_parity`, `device_argmax`, `dpas_fold`, `ga64_recip`,
  `gguf_roundtrip`, `gpu_optin`, `http_batch`, `oracle_equiv`, `serve_load`,
  `tp2_27b`. This is derived by `gate_inventory.sh` from each gate's own
  precondition plus the visible device count, not hand-counted — with 2 GPUs
  every precondition is met, so nothing is PARTIAL and nothing is SKIPPED.
  `test_tp2_27b` now runs for real: **TP2-27B TEST: PASS**, with
  `GPU backend: 2 device(s) (TP=sharded)`.
- **TP=2 coverage is no longer absent.** `test_gpu_optin` and
  `test_device_argmax` now execute their `--tp 2` arms instead of skipping them,
  and `test_tp2_27b` covers `--tp 2` on 27B directly.
- *** HISTORICAL — DO NOT QUOTE AS PRESENT-DAY COVERAGE. *** Throughout this file
  are older lines reading "7 PASS, 2 PARTIAL, 1 SKIPPED [SUPERSEDED 2026-10-01: now 10 PASS, 0 PARTIAL, 0 SKIPPED]", "PASS (20/20 arms, TP=1
  and TP=2)", "x TP=1/TP=2", "10/10", and "There is currently NO TP=2 coverage
  at all". Those were true of the tree WHILE ONLY ONE GPU ENUMERATED, and are
  now wrong about the current box. They are **records of the 1-GPU era**, not
  statements about this tree. Do not "correct" them either — rewriting a true
  historical record to match today's box is the same laundering failure in the
  other direction.
- The correct present-tense sentence is: *"10 PASS, 0 PARTIAL, 0 SKIPPED; both
  B580s enumerate and TP=2 coverage is live."* Anything shorter is an overclaim
  in one direction; quoting the old tally is an overclaim in the other.
- All 7 rc=0 gates were re-run after the audit #13 tooling/doc commits, to
  confirm that pass was cosmetic-only (no engine source touched since ce2eb05):
  ga64_recip, gguf_roundtrip, dpas_fold, cpu_parity, oracle_equiv, http_batch,
  serve_load — all rc=0. The two TP=2-dependent gates and tp2_27b were not
  re-run; their class is unchanged (still blocked on the second GPU).
  correctness gate from source, and flags a whole-gate skip on a gate that has
  1-GPU arms (the audit #13 over-skip bug, made mechanically detectable). It also
  **reconciles the doc tallies**: it derives the expected
  `N PASS / N PARTIAL / N SKIPPED` from each gate's own precondition plus the
  visible device count, and fails if any tally in `AGENTS.md` or `reports/*.md`
  disagrees — so "the docs said 8 instead of 7" is a build refusal now, not
  something a reviewer has to notice. Mutation-tested both ways: reverting one
  site to 8 gives rc=1, and making the tally regex match nothing gives rc=1 via
  its own `sites_found=0` vacuity guard (which is not theoretical — the first
  version of that regex was `^`-anchored, matched 0 sites, and cheerfully
  reported "0 stale" while verifying nothing).
  `scripts/fold_report_counts.sh` does the same job for the fold matrix table,
  printing every cell from `build/fold_matrix/*.log` and naming the counting
  basis, so the report's numbers cannot be silently transcribed onto the wrong
  basis again. Both are read-only: no mutation, no GPU, no build.


Stage 1.
- choose specific model quantization analyze what makes the quantization format chosen by ninfer really fast
- Construct our own int2 quantization based on this schema:
[ FP16 Model ] 
      │
      ▼
[ 1. SpinQuant Preprocessing ]  ──► Rotate W & A to eliminate outliers
      │
      ▼
[ 2. DASH-Q Solver ]            ──► Generate ultra-low error quantized tensor variants
      │
      ▼
[ 3. RCO Bit-Budget Search ]    ──► Allocate DASH-Q variants per tensor under exact VRAM budget
      │
      ▼
[ Standard GGUF / Native SYCL ]

SpinQuant (Orthogonal Rotation Preprocessing): A spatial transformation step. By multiplying weight matrices and activations by learned/randomized orthogonal rotation matrices ($W' = W \cdot R$), activation/weight outliers are suppressed and spread evenly across dimensions.
DASH-Q (Diagonal Hessian & Iterative Least Squares): A non-heuristic PTQ solver. Instead of relying on off-diagonal Hessian error propagation (like GPTQ, which suffers from sampling noise at extremely low bits), DASH-Q uses diagonal Hessian approximations to solve independent weighted least-squares problems for weight assignments.

Stage 2. (tensor parallel since we have 2x gpus)
- Implement the engine (based on oneapi level zero and sycl)
1. Low-Level Native C++/SYCL Kernels (Direct Engine Port)

If you were to adapt ninfer's exact philosophy—bypassing heavy frameworks for lean C++ execution—you would target Intel's hardware primitives directly via oneAPI SYCL and Level Zero:

XMX MMA Intrinsics: Replace NVIDIA mma.sync / wmma PTX calls with Intel DPAS (Dot Product Accumulate Systolic) or XeTile SYCL primitives for INT8/INT4 matrix multiplications.

Kernel Fusion & Memory Layout: Port the fused FlashAttention and quantized GEMM kernels directly to SYCL. Xe matrix engines rely on specific register layouts (e.g., 2D block loads), so layout transposition must match Arc's L1/L2 cache and subgroup sizes (typically 16 or 32 threads per SIMD lane).

SYCL Command Graphs: Replace CUDA Graphs with sycl::ext::oneapi::experimental::command_graph to record token execution sequences and submission queues, keeping CPU launch overhead near zero during generation loops.

| ninfer Technique | Intel Arc Hardware / API Equivalent | Implementation Feasibility |
| :--- | :--- | :--- |
| **Low-Bit Quantization (NVFP4 / W4A4 / INT8)** | **Intel XMX (Xe Matrix Extensions)** & DPAS instructions via SYCL / oneAPI | **Fully Possible:** Intel Arc Alchemist and Battlemage GPUs feature XMX AI engines that natively support INT8, INT4, FP16, and BF16 matrix multiply operations with high throughput. |
| **CUDA Graphs** | **SYCL Graph Extensions** | **Fully Possible:** SYCL supports workload graph recording and submission, eliminating host CPU-to-GPU queue submission overhead during decode loops. |
| **Fused FlashAttention / W4A4 MMA Kernels** | **Intel XeTile / Intel Extension for PyTorch (IPEX)** custom SYCL kernels | **Possible:** Hand-tuned SYCL / Intel Xe-Assembly kernels can match low-level memory layout optimizations for fused operations. |
| **Multi-Token Prediction (MTP) / Speculative Decoding** | **SYCL / Level Zero C++ Runtime** | **Fully Possible:** MTP is an algorithmic framework (running speculative draft tokens parallel to main heads). It is vendor-agnostic and relies purely on batch memory bandwidth. |
| **Groupwise-INT8 KV Cache** | **Custom SYCL Quantization Kernels** | **Fully Possible:** Groupwise INT8 KV cache management reduces VRAM footprint and memory bandwidth usage on Intel Arc GPUs. |

Stage 3.
Implement features for maximizing throughput for concurrent requests, starting from continuous batching.

## Performance targets (2x Arc Pro B580)

> FOCUS (2026-09-24): our optimization target is the **27B model** — decode
> and prefill work, benchmarks, and perf claims are scoped to 27B unless
> explicitly stated otherwise. The 0.8B numbers below are legacy reference
> only, not the target.

- Token generation (decode): **40 tok/s** (per-stream, 27B model)
- Prefill: **4000 tok/s** (27B model)
- Current (2026-09-18, scripts/bench_gpu_ab.sh, DPAS48 default OFF,
  fused MLP default-ON, quiet box loadavg ~1.9):
  0.8B GPU decode TP=1 27.24 tok/s (-n 64), TP=2 14.14 tok/s.
  Pre-fusion quiet baseline was TP=1 ~24.4 (3x 24.37/24.41/24.45)
  / TP=2 ~14.3: quiet-box fused gain = +11.6% TP1, TP=2 unchanged
  as designed (fused path is TP=1 only). TP=1 still wins ~1.9x:
  2x submits/transfers/readbacks outweigh halved kernel
  time at tiny shapes. DPAS48 fast path defaulted OFF (was default
  ON): host GA64 act-quant ~6ms/token dominated, XMX kernel no faster
  than tiled legacy at M=1/128; greedy streams bit-identical either
  way (opt back in with INTFER_DPAS48=1). Decode +16-18%, prefill
  +5% vs DPAS-on (pre-fusion baselines). Prefill TP=1 (361-tok,
  2026-09-18 quiet box, fused default-ON):
  chunk8 54.7, chunk32 111.8, chunk128 149.2, chunk361 156.6 tok/s;
  TP=2: 42.3 / 89.0 / 122.4 / 134.0 tok/s. Chunk 361 (whole prompt,
  one pass) still best (fewer layer passes → fewer submits).
  Steady-state chunk128 TP=1 ~= 176 tok/s excluding the one-time 466ms
  rebuild (186 shards); recorded ~195 was lower-load.
  Prefill
  attribution (2026-09-18, chunk128 TP=1, 361 tok in 2800ms): rebuild
  924ms = preload_all one-time (186 shards, amortized to zero in
  serving); steady-state ~195 tok/s = kernel 475 + waitblock 504 +
  submit 303 + transfers ~214 + host ~380ms. Submit inflates
  84us/launch (M=1) to ~600us/launch (M=128): UR submit blocks when
  the device is saturated — same serialization wall as decode, only
  L0 immediate lists or cross-call fusion move it. 27B GPU decode ~2.8 tok/s pinned TP=2
  (kernel 66% of decode — TP tradeoff reverses vs 0.8B); TP=1 OOMs on 27B
  (lm_head INT8 tile alloc fails: ~19GB on one ~12GB B580).
  Per-token profile (0.8B TP=2): submit ~34%, host act/copy ~33%,
  kernel ~20%, wait ~11%; run-to-run variance is +/-10%.
- XMX verdict (2026-09-18, source-level + xmxmon runbook): **zero DPAS
  instructions execute in the default path** — decode M=1
  (gemm_usm_decode1_vec, quant_gemm.cpp:388) and prefill M>1 tiled
  (quant_gemm.cpp:526) are pure EU/ALU dot-products (dequant-on-the-fly
  + FP32 FMA); xmxmon would read all-zero XVE_INST_EXECUTED_XMX_* on
  every workload. Matrix engines 100% idle. Only DPAS-issuing kernels
  in the tree: quant_gemm_dpas48 (u4/u2/u8, opt-in INTFER_DPAS48=1,
  default OFF — host GA64 act-quant dominated) and quant_gemm_dpas_s8w
  (lm_head-only). Per xmxmon docs, mixed-precision DPAS counts under
  the WIDER operand (W2A8 -> INT8 counter, not INT2). Blocker:
  metric streamer needs `sudo sysctl -w dev.xe.observation_paranoid=0`
  (this box: paranoid=1, no sudo — zetMetricStreamerOpen 0x7ffffffe on
  both GPUs; counters verified present via --list-metrics).
  Harness: scripts/xmx_profile.sh joins xmxmon capture + internal
  [GPU profile] buckets, degrades gracefully to buckets-only when the
  streamer is blocked. Implication: prefill 4000 tok/s needs XMX (EU
  pipes are the whole prefill story today); decode 40 tok/s needs
  launch-count/transfer reduction (XMX at M=1 barely helps — shapes
  are memory-bound).
- XMX CORRECTION (2026-09-18, empirical via xmxmon after sysctl unlock):
  the "zero DPAS" verdict above is WRONG for lm_head — lm_head runs
  INT8 DPAS (quant_gemm_dpas_s8w, main.cpp:1231 use_dpas, default-ON
  whenever the repack exists, no env gate). Measured: prefill-only
  361tok capture = 180M INT8 slots; decode-only M=1 = 5.5M INT8 slots
  (~0.8M/token, consistent with vocab/16 tiles x K/32 ops). Layer GEMMs
  (99% of launches) remain EU/ALU, XMX idle there — that half stands.
  DEVICE ACT-QUANT PRE-PASS (2026-09-29 49b21d3; AUDITED + CORRECTED
  2026-09-30, INTFER_DPAS_PREQ=1, default-OFF): moves the GA64 act-quant
  STORE half off the host (it made DPAS a net +6% wall loss on 27B despite
  a -15% device-side kernel win — copy bucket 269us/GEMM + submit 127us).
  Quant store moves to a standalone pre-pass kernel, dep-chained into the DPAS launch
  (trailing `sycl::event` PARAMETER, NOT a Dpas48Params member) so it adds
  NO host wait. One launch PER DEVICE, never one for both — each rank owns
  its own sycl::context (device_manager.cpp), so a kernel on rank 0's queue
  cannot legally write rank 1's malloc_shared buffer.

   === 2026-09-30 AUDIT: THE ORIGINAL GATE EVIDENCE WAS VACUOUS ===
  PREQ is NESTED INSIDE the DPAS48 path, and DPAS48 is default-OFF. The
  gates cited in 49b21d3 (cpu parity, http batch, gpu-optin 10/10) all ran
  WITHOUT INTFER_DPAS48=1, so run_linear_tp_dpas48 was never entered and
  the pre-pass NEVER FIRED in any of them. They prove nothing about PREQ.
  Worse, the cited invariant "streams bit-identical base vs PREQ" is
  impossible: DPAS48's own INT8/GA64 act-quant legitimately differs from
  the FP16-act base. The correct invariant is PREQ == DPAS48 (the pre-pass
  reproduces the host GA64 quant).

   === 2026-09-30 AUDIT #2: THE NEWTON "FIX" WAS BACKWARDS. DO NOT RE-ADD IT ===
  An earlier revision of this audit concluded "device 1.0f/s is not IEEE,
  add an fma Newton step". WRONG IN DIRECTION, and reverted. Truth, from a
  direct device probe (9,689,600 realistic GA64 elements incl. adversarial
  exact-.5 tie zones: 0 mismatches for the bare kernel expression vs the
  host AVX2 lane path) plus exact rational arithmetic on the bit patterns:
    * `q = trunc(x*inv +- 0.5)` is PATHOLOGICALLY ILL-CONDITIONED in `inv`:
      a 1-ulp reciprocal difference moves the product across a .5 tie and
      flips the code by a full +-1. Two implementations that do not share
      the SAME reciprocal bits disagree on ~2 elements per ~1950 GEMM
      calls. Correctness here means sharing `inv`, NOT being correctly
      rounded.
    * A DEVICE `1.0f/s` lowers to rcp.approx: 15,868/151,400 mismatches vs
      host divss over legal s.
    * HOST `1.0f/s` is not a portable bit-source either: same expression,
      same -O3 flags, no fast-math, yet host call sites in different
      codegen contexts were observed to disagree (that is the whole reason
      one site now exists). Never assume a second site agrees.
  => "Correcting" the device moved it AWAY from the reference and CREATED
  the decode-step-14 divergence this same commit then documented as an
  unfixable "KNOWN, UNRESOLVED tie rule". The Newton step is GONE.

   === THE FIX: ship the host's (scale, inverse) to the device ===
  dashq::act_quant_ga64_scale_inv_batch() computes per 64-slice
  (s, inv) = (max(1e-6,max|x|)/127, 1.0f/s) ON THE HOST; the pre-pass
  consumes both verbatim via ActQuantDevParams::scale_inv_in (2*M*nGA FP32
  interleaved, per-device `ws48_[d].si` landing zone, tiny H2D).
  Bit-exact BY CONSTRUCTION. Bonus: the device no longer redoes the O(64)
  max-abs reduction, removing the only way the scale it WRITES could
  disagree with the inverse it used. Cost: the O(M*K) max-abs reduction is
  back (the pre-pass was built to delete the full host quant); it is a
  reduction plus M*K/64 divides instead of a full host quant, and stays off
  the GEMM critical path (dep-chained, no added wait).

   === AUDIT #3: ONE reciprocal site, and a numeric gate for it ===
  The fix above shipped the host's bits but still computed them at a SECOND
  host call site (`1.0f/s` inside the helper vs inside the AVX2 oracle).
  That is the same hazard one layer down: correct only by luck of codegen,
  while the commit message claimed "by construction". Measured with a probe
  that RECOVERS the oracle's effective reciprocal (re-quantizing with
  inv-1ulp/inv/inv+1ulp): 288k slices / 18.4M elements agreed in the
  7f005fe binary — luck, not a guarantee.
  FIX: dashq now has exactly ONE max-abs site (ga64_slice_max{,_avx2,_scalar})
  and ONE reciprocal site, `ga64_recip()`, deliberately NOINLINE so it
  compiles to a single body/instruction sequence that every GA64 producer
  executes (AVX2 oracle, scalar fallback, scale_inv helper). Side benefits:
  the helper stopped duplicating a SCALAR max-abs loop (it now vectorizes via
  the shared site, so the restored host reduction is cheaper than 7f005fe's),
  the helper's NaN handling now matches the oracle's, and the helper derives
  nGA from K itself (a caller-supplied nGA could silently leave tail slices'
  (s,inv) uninitialized). Also: the pre-pass loop now stops submitting once
  any device declines instead of launching on the remaining ranks.
  The shared site currently happens to be correctly rounded (64000/64000
  legal scales) — that is a property to PRESERVE, not to re-derive; a
  Newton-style "fix" at a second site would move it away from the oracle.
  NEW GATE: scripts/test_ga64_recip.sh -> examples/ga64_recip_test (CPU-only,
  <1s, no model/GPU). It asserts the helper's (s, inv) reproduce the oracle's
  scales bit-exactly AND that the device quant rule applied to the shipped inv
  reproduces the oracle's CODES, over random / tie-adversarial / edge
  (zeros, denormals, saturation, outliers) / K-not-multiple-of-64 datasets and
  M=3 multi-row batches; on failure it reports the ulp distance, not just
  "streams differ". NEGATIVE TEST: injecting a 1-ulp drift
  (nextafterf) into the helper makes it FAIL — tie dataset 64000/64000
  slices caught, random dataset 21 — so the gate is not vacuous (the earlier
  vacuous-gate bug was the nested PREQ flag).

   === 2026-09-30 AUDIT #4: THE GATE CHECKED 1 OF M ROWS, AND READ A
  STALE BINARY. BOTH FIXED. ===
  (a) COVERAGE HOLE, PROVEN. check_batch() called the helper with M=3 but
  only ever validated ROW 0 against the oracle, while the helper writes
  2*M*nGA (s, inv) pairs at a row stride and the engine feeds it M rows
  (decode M=1, prefill M=8..128, serve M>1). A row-index bug therefore left
  every prefill/serve row with row 0's bits and the gate still reported
  PASS. Demonstrated with a mutation that keeps row 0 bit-exact
  (`o = out + 2*m*nGA + (m ? 2 : 0)`, in bounds): OLD row-0-only gate
  PASSED (64000/64000 ok on all four datasets), FIXED gate FAILED with
  467289 diagnostics. Now every row is checked against its OWN oracle call
  (192000 slices instead of 64000).
  (b) VACUITY NOW ASSERTED, NOT ASSUMED. Each slice also records whether a
  1-ulp reciprocal drift WOULD have changed the answer, and the gate FAILS
  if the tie-adversarial dataset contains no such slice. It contains
  192000/192000 — the tie construction is genuinely adversarial. (A gate
  that cannot fail proves nothing; the nested-PREQ bug was a vacuous gate,
  this is the second one in the same family.)
  (c) THE GATE READ A STALE BINARY. test_ga64_recip.sh ran whatever
  $BIN/ga64_recip_test was on disk with no build step, so a source edit
  that failed to compile left the old binary in place and the script printed
  PASS for code that no longer existed — observed live during this audit
  (a compile error scrolled past, then "GA64 RECIPROCAL TEST: PASS"). The
  script now BUILDS its target first and refuses to run on build failure
  (opt out with INTFER_GATE_NO_BUILD=1). Verified: an injected
  `static_assert(false)` now yields rc=1 instead of a stale PASS. Same class
  as audit #3c; the sibling GPU/parity/http scripts had the same gap and were
  closed in audit #6, then re-audited tree-wide in audit #7 — see
  `scripts/_gate_build.sh`.
  (d) Housekeeping: engine.cpp's host fallback had been left as a second
  unindented `if (!preq_ok) {` after the pre-pass block's closing brace —
  merged into a single `else`; a stray 2-space reindent inside
  quant_gemm_dpas4_row (unrelated to the audit) reverted.
  UNCHANGED: the perf story. The O(M*K) max-abs reduction is still on the
  HOST (see the "Cost:" note under AUDIT #3) — only the quantize+store half
  moved to the device, so expect roughly half the host act-quant saving the
  49b21d3 design promised, and read the `copy` bucket as including that
  reduction.
  (e) M>1 DEVICE COVERAGE ADDED. Every PREQ stream assertion in
  test_gpu_optin.sh used the 5-token PROMPT, whose prefill is a single ~5-row
  batch — so nothing in the suite exercised the helper's 2*M*nGA row stride on
  a device-sized batch (decode is M=1 by definition, which is why (a) shipped
  unnoticed for three commits). New WIDE-PREFILL arm uses a ~97-token prompt
  that the default chunk_M=128 forwards as ONE ~97-row batch, asserting
  DPAS48+PREQ == DPAS48 at TP=1 and TP=2 ('198 760 7171 40132 369 264 5054
  852'). Non-vacuity proven by mutation: re-injecting the row bug from (a)
  makes the wide arm diverge on the FIRST decode token ('1292 557 264 5054 852
  557 264 5054'). A temporary INTFER_PREQ_TRACE probe (not committed)
  confirmed the pre-pass really is fed M=97 (x93 GEMM calls) and M=1 (x744) on
  that run, so the arm is not accidentally running an M=1 prefill.

   === 2026-09-30 AUDIT #5: async-lifetime (checked, no bug) + dispatch hoist ===
   SUSPECTED use-after-free, DISPROVED by reading the code: if the DPAS
   kernel declined after a successful pre-pass, nothing would ever wait on
   preq_events, and the in-flight pre-pass would race the next call's
   std::memcpy into ws48_[d].act — and `si48_staging_`/`x` (host memory the
   async q.memcpy reads) could be freed under the copy. NOT reachable:
   quant_gemm_dpas48() has exactly ONE return statement (the q.submit) and
   never declines, and use_row (which CAN decline) is mutually exclusive with
   preq_ok, so whenever preq_ok is true a real DPAS event dep-chains on
   preq_events[d] and the kernel_events wait at the end of the call covers the
   memcpys transitively. No code change; recorded so the next reader does not
   re-open it or "fix" it with a redundant drain.
   REAL FIX (host cost): act_quant_ga64_scale_inv_batch called
   avx2_available() — a function-local static, so a guard-variable load —
   once PER SLICE, i.e. M*nGA times per GEMM on the host pass that audit #3
   had to bring back. Hoisted to once per call (ga64_slice_max_sel). Provably
   bit-neutral (the probe result is a process constant): test_ga64_recip.sh
   is unchanged at 480645 slices / 30.7M elements with IDENTICAL per-dataset
   discriminating-slice counts (random 103, tie 192000, edge 11, tail 0)
   before and after.
   SCOPED, NOT ATTEMPTED: deleting the host O(M*K) reduction outright. The
   max-abs IS order-independent and therefore exactly reproducible on device
   (IEEE max is exact and associative; only NaN differs, and NaN activations
   mean a broken model anyway), so the device could own the reduction. The
   blocker is still the reciprocal: shipping host `inv` requires the host to
   already have `s`, so the reduction cannot leave the host without a per-GEMM
   D2H/H2D round trip that costs far more than the pass it removes. The only
   way out is a correctly-rounded reciprocal computed on device with integer
   arithmetic, which WOULD match the host site today (ga64_recip is verified
   correctly rounded, 64000/64000 legal scales) — but that silently couples
   device correctness to the host's divss codegen, so any future compiler
   change to ga64_recip breaks it invisibly. Not worth it for a default-OFF
   path; recorded, not built.

   === 2026-09-30 AUDIT #7: gate hygiene, family 3 tree-wide + shared helper ===
   Audit #6's fix (74e80a0) closed the stale-artifact hole in the FOUR gates it
   had audited and the memory note then declared family 3 "FIXED EVERYWHERE".
   That was wrong, and it broke the note's own scope corollary: grepping every
   script for a build-artifact consumer found FIVE more with no build step —
   test_serve_load.sh (the Stage-3 continuous-batching load gate),
   test_device_argmax.sh, test_oracle_equiv.sh, test_tp2_27b.sh (all
   intfer_cli) and test_gguf_roundtrip.sh (intfer_gguf). Nine gates total;
   audit #6 touched four. Same damage as http_batch: a failed build leaves the
   previous binary runnable, so the gate judges code that no longer exists.
   (HISTORICAL COUNT: nine was correct for audits #7/#8. There are TEN gates now
   — scripts/test_dpas_fold.sh was added later; see the EXPERIMENT SHIPPED
   block. Do not read "nine" below as the current count.)
   Fix: ONE shared, sourced helper `scripts/_gate_build.sh` with
   `gate_require_build <cmake-target>`, called by all nine gates
   (`. "$SCRIPT_DIR/_gate_build.sh"; gate_require_build <target> || exit 1`).
   Rationale: nine hand-rolled copies of the block is precisely how the family
   reopened — a per-gate copy is a per-gate chance to forget it. The helper
   fails closed on an empty target name, keeps INTFER_GATE_NO_BUILD=1, and
   writes a PER-GATE log ($BUILD_DIR/gate_<gate>_build.log): audit #6's version
   had all four gates sharing intfer_cli_build.log, so a concurrent pair
   overwrote each other's compiler error and `tail -20` could print the wrong
   failure. No engine/quant/schedule code touched.
   Anti-vacuity, measured (not asserted): `static_assert(false)` appended to
   src/common/logger.cpp — a 707-byte TU that both intfer_cli (via
   intfer_common) and intfer_gguf compile — while the on-disk binary was
   confirmed still present and still runnable (`--help` rc=0). Result: all NINE
   gates rc=1 with "did not build (refusing to run a stale binary)", and
   nothing downstream of the guard executed. Helper unit checks: bogus target
   -> rc=1 with ninja's "unknown target" (so the target name is really passed
   through, not a blanket build), empty target -> rc=1, real target -> rc=0,
   INTFER_GATE_NO_BUILD=1 -> bypasses. Mutation reverted, logger.cpp
   byte-identical to db6e98c, binary relinked.
   Clean-tree re-run of all nine (box load 8-14, CORRECTNESS ONLY — no wall-time
   or throughput claim): ga64-recip PASS (480645 slices / 30,753,153 elements
   bit-identical, 192000 tie slices discriminating), gguf-roundtrip PASS,
   cpu-parity PASS, oracle-equiv PASS, gpu-optin PASS (TP=1+TP=2, wide M=97
   prefill arm, PREQ-alone inert), http-batch PASS, device-argmax PASS (its
   nested cpu-parity/http-batch/gpu-optin all PASS through the helper too),
   serve-load PASS, tp2-27b PASS.
   Process lesson, now in memories/gate-hygiene-three-families.md: a family is
   closed by an exhaustive ENUMERATION of its instances (grep every script for
   the artifact it consumes), never by the count of files the current fix
   happened to touch.
   AUDIT #8 (re-verification of #7 + scope correction). #7's guard is sound;
   re-derived independently rather than trusted: guard ordering re-checked in
   all 9 (every guard precedes the first EXECUTION of the binary — the earlier
   "hole" reading was an assignment line, not a call), helper edge cases
   re-tested (empty/no target -> rc=1, bogus target -> rc=1 with ninja's
   "unknown target" so the name really reaches cmake, missing build tree ->
   rc=1, real target -> rc=0, INTFER_GATE_NO_BUILD=1 -> bypass, `sh` -> still
   rc=0), and the mutation RE-RUN: static_assert(false) into
   src/common/logger.cpp with the stale binary confirmed still runnable
   (--help rc=0) => 9/9 gates rc=1, nothing past the guard. Reverted; logger.cpp
   byte-identical; all 9 green again on the rebuilt tree.
   SCOPE CORRECTION — "closed tree-wide" was still an overclaim, the same error
   in a new place. Grepping for build-artifact consumers finds SEVENTEEN scripts,
   not nine: the 9 correctness gates (guarded) plus 8 measurement/serve
   harnesses — bench_gpu_ab.sh, bench_quiet_ab.sh, bench_quiet_l4_ab.sh,
   bench_serve.sh, bench_chat.sh, serve_27b.sh, stream_ids.sh, xmx_profile.sh
   — which consume intfer_cli with NO build step. Family 3 is therefore closed
   for CORRECTNESS ORACLES ONLY; those 8 are measurement scripts whose whole
   purpose is timing a specific binary on a quiet box, so an implicit rebuild
   would (a) perturb the very quiet-loadavg window they gate on and (b) silently
   substitute code for the A/B under test. They are deliberately NOT guarded —
   but that is now stated instead of being papered over by "tree-wide".
   REAL BUG FOUND AND FIXED in that sweep: 5 of those harnesses hardcoded
   BIN="$ROOT/build/...", silently IGNORING the documented INTFER_BUILD_DIR
   override (README.md:96) that 12 other scripts honor. Consequence: pointing
   INTFER_BUILD_DIR at another tree — the natural way to benchmark an
   experimental build — left all five benchmarking the DEFAULT tree instead,
   i.e. exactly the "measured the wrong binary" failure family 3 exists to
   prevent, arrived at from the other direction. Now
   BIN="${INTFER_BUILD_DIR:-$ROOT/build}/..." in all five; verified both
   directions (override set -> resolves into that tree and fails closed when it
   has no binary; override unset -> resolves to the real default binary).
   All 17 artifact consumers now honor the override; `bash -n` clean on each.
   Benchmarks re-verified CORRECTNESS-only; no wall-time claim is made here.

   === AUDIT #9: test_tp2_27b FAILING = BOX STATE, NOT A REGRESSION ===
   *** SUPERSEDED IN PART BY AUDIT #11 (further below). Keep this section's
   CONCLUSION ("box state, not a code regression") and DISREGARD its
   MECHANISM — the "ranges[d] wild pointer" story below was TESTED AND
   DISPROVED, and the follow-up action item it spawned was aimed at a bug
   that does not exist. Do not "fix" the TP clamp; it is correct. ***
   The audit-#8 9-gate re-run came back 8 PASS / 1 FAIL, the failure being
   test_tp2_27b: intfer_cli exits rc=139 (SIGSEGV) after preload, with the
   backtrace resolving to intfer::Engine::run_linear_wave <- GpuBackend::
   run_linear_wave <- step <- main (addr2line on the shipped binary; no -g,
   hence no line numbers). NOT a code regression — this audit touched no
   engine/quant/schedule source (diff is 5 harness scripts + docs), and the
   same binary passed tp2_27b earlier in the session.
   ROOT CAUSE, measured not inferred: the box now exposes ONE GPU to the
   compute runtime. The CLI's own banner says `Engine: 1 Arc device(s)`, and
   independently: `xpu-smi --list-gpus` -> GPU 0 only; `sycl-ls` (compiler
   2026.0, umf on LD_LIBRARY_PATH) -> exactly one [level_zero:gpu] entry.
   Meanwhile /dev/dri still has card0/renderD128 AND card1/renderD129 and all
   four open OK, i.e. the kernel still has both nodes and only the runtime's
   enumeration dropped PCI 86:00.0. Shell env is clean (no SYCL_DEVICE_FILTER
   /ZE_AFFINITY_MASK/ONEAPI_DEVICE_SELECTOR), so this is NOT the libumf
   LD_LIBRARY_PATH gotcha (that one yields ZERO devices, not one).
   ~~TP=2 then slices every tensor into 2 shard ranges while only rank 0
   has a device, and the wave path walks ranges[d] -> wild pointer ->
   SIGSEGV.~~  STRUCK BY AUDIT #11: FALSE. engine.cpp:361 already clamps
   `n = min(tp_devices, dm.size())` and shards come from devices_.size(), so
   --tp 2 on a 1-GPU box is simply TP=1. Verified: 0.8B with `--gpu --tp 2`
   exits rc=0 with the expected ids. Real cause is 27B VRAM exhaustion
   (~19GB vs ~12GB), aborting inside Level Zero's page-fault manager.
   Timeline fits the loss: gpu_optin (TP=1+TP=2 arms) PASSED earlier in the
   same run, then the second GPU vanished before the last gate.
   ACTION: re-run scripts/test_tp2_27b.sh once BOTH GPUs enumerate again
   (`sycl-ls` must print 2 [level_zero:gpu] lines). Until then treat the
   9-gate result as 8/9 with TP=2 coverage unverified, not as a code failure.
   ~~SCOPED, NOT ATTEMPTED: the engine should reject `--tp N` when
   devices_.size() < N with a clear error instead of segfaulting deeper in
   the wave path.~~  WITHDRAWN BY AUDIT #11: this task is aimed at a bug
   that DOES NOT EXIST — there is no missing-rank walk, because the clamp is
   correct. Do NOT implement it. What was actually missing is the opposite:
   the clamp was SILENT, and that is now fixed (a loud, deliberately
   NON-fatal warning when dm.size() < tp_devices, so a real TP shortfall can
   never be mistaken for a driver fault).

   === AUDIT #11: AUDIT #9's ROOT CAUSE WAS WRONG (supersedes it) ===
   #9 attributed the tp2_27b rc=139 to TP sharding walking a missing device
   ("ranges[d] -> wild pointer"). TESTED AND DISPROVED: the clamp is real and
   correct — engine.cpp:361 `n = (tp_devices==0) ? dm.size() :
   std::min(tp_devices, dm.size())` and shards are built from devices_.size(),
   so --tp 2 on a 1-GPU box is simply TP=1. Verified: 0.8B with `--gpu --tp 2`
   on this 1-GPU box exits rc=0 with the expected ids and the banner
   "Engine: 1 Arc device(s)" / "GPU backend: 1 device(s) (TP=single)". There
   is no missing-rank wild pointer. #9's conclusion (box state, not a code
   regression) still holds; its MECHANISM did not, and the mechanism is what
   the follow-up action item was built on — so #9's "reject --tp N when
   devices < N" task was aimed at a bug that does not exist.
   ACTUAL ROOT CAUSE, reproduced and read out of the log: 27B simply does not
   fit one B580. Order of events in the failing run: "preloading all shards
   into VRAM..." -> "starting prefill..." -> prompt ids computed fine ->
   "FATAL: signal 6" with the abort raised INSIDE the Intel Level Zero runtime
   (`level_zero/core/source/memory/cpu_page_fault_memory_manager.cpp`), rc=139.
   So it is a device-OOM at page-fault time: pool_acquire's FATAL guard
   (engine.cpp:91-101) only fires when malloc_shared returns null, and shared
   USM under pressure RESERVES successfully and fails later when the GPU
   faults the pages in — which is why the guard never saw it and the death
   looked like a driver fault. Consistent with the standing 27B note that TP=1
   needs ~19GB for the lm_head INT8 tile against a ~12GB B580.
   FIX SHIPPED (diagnosability only, the OOM itself untouched): the min() clamp
   was SILENT — nothing ever said "you asked for 2, you got 1". Engine::load
   now prints a loud WARNING when dm.size() < tp_devices, naming the shortfall
   and the ~19GB-vs-~12GB consequence. Deliberately NON-fatal: a legitimate
   deliberate TP=1 run must not be blocked, so warn and continue. Verified on
   the exact trigger box: --tp 1 -> 0 warnings (no false positive), --tp 2 and
   --tp 4 -> exactly 1 warning each, and all three produce IDENTICAL ids
   ("...capital of the country"), i.e. zero behavior change. On 27B the warning
   lands at log line 11 vs the abort at line 50, so the cause now reaches the
   operator before the failure instead of after. cpu-parity PASS after the
   change. The 27B rc=139 is unchanged and expected on a 1-GPU box — a real
   preflight VRAM check (footprint vs device_free_bytes) is the only way to
   turn it into a clean refusal, and that is NOT attempted: it is heuristic,
   a false positive would block runs that work today, and it cannot be
   validated on the TP=2 configuration it exists to protect (family "gate
   cannot exercise the configuration", AUDIT #3c).
   LESSON (same family as #10, one level up): #9 recorded a mechanism from
   a backtrace symbol without reading the clamping code. A plausible-sounding
   mechanism is not a verified one — read the code that would have to be wrong,
   or reproduce it. Reproducing was cheap (the 0.8B run is seconds) and would
   have caught this immediately.

   === AUDIT #12: THE FOLD ROW-STRIDE RULE WAS NOT APPLIED AT THE READER ===
   E3 (INTFER_DPAS48_FOLD=1, aeb65b5) hoists the per-(row,op) activation
   column sums out of quant_gemm_dpas48's 16-N tile loop. Its commit message
   insists the row stride must be >= 2*nOp (each row owns [S0][Sfull]; a
   shorter stride lets row m's Sfull overwrite row m+1's S0 — silent at M=1).
   That rule was enforced ONLY in the WRITER (quant_gemm_dpas48_rowfold). BOTH
   READERS resolved it independently and got it wrong — `? nOp : p.fold_stride`
   in quant_gemm_dpas48 AND verbatim again in quant_gemm_dpas4_row — and the
   header DOCUMENTED the bug as the contract (`fold_stride = 0; // 0 => nOp`),
   so the public API invited any caller to set fold_usm and leave the stride
   defaulted. Not reachable today (engine.cpp sets both fields), but latent and
   wrong, and the header invited it.
   TWO REASONS IT SURVIVED ITS OWN FEATURE'S MUTATIONS:
   (a) VACUOUS COVERAGE. The probe gated the writer's VALUES; the engine always
       sets fold_usm AND fold_stride together, so all four test_gpu_optin FOLD
       arms — which DO exercise M>1 and the row stride — are structurally
       incapable of reaching the 0 default. A path nothing reaches is a path
       nothing can break.
   (b) MUTUAL MASKING, the interesting half. Wrong default is harmless while
       the reader VALIDATES; missing validation is harmless while the default
       is RIGHT. Under the ENGINE's call pattern that means neither bug
       produces wrong numbers: the engine always passes a stride that clears
       2*nOp, and validation absorbs the bad default — so the four
       end-to-end arms could not have shown either one. PRECISION, measured
       (see the matrix below): M2 alone is NOT harmless to a gate, because a
       probe can feed the too-small stride the engine never sends — S2 fails
       the readback at 30/64. The masking is real in the ENGINE, not in the
       probe. Same species as the GA64 reciprocal (#3) and "1 of M rows" (#4):
       a rule fixed at one site and merely assumed at another.
   FIX: ONE shared rule in quant_gemm.hpp — `fold_row_stride()` (0 => 2*nOp)
   and `fold_readable()` (refuses any stride < 2*nOp, returns nullptr so the
   kernel falls back to the INLINE fold = bit-identical to pre-FOLD, just no
   hoist). Writer and both readers now consume the same two functions, so the
   reader can never index a layout the writer refuses to produce.
   COVERAGE (same probe, so still one gate): contract check over 80
   (nOp,stride) pairs with expectations restated INDEPENDENTLY of the rule it
   checks (so it cannot pass by sharing code with it), plus a DEVICE READBACK
   that runs the real quant_gemm_dpas48 against the inline fold in three stride
   regimes and demands bit-identical FP16 outputs. Activations are built so S0
   and Sfull are deterministically distinct, making a mis-stride catchable BY
   CONSTRUCTION rather than by luck.
   MUTATION TABLE — each state applied and run IN ISOLATION from a clean
   checkout, full output captured per state (an earlier draft asserted the
   M2-alone row by inference; it was WRONG, corrected below, and S4 added
   because the real question is whether the readback earns its place).
   RE-RUNNABLE: ./scripts/mutate_fold_matrix.sh (7/7 as characterized;
   S5+S6 added in AUDIT #13 — see the 7-state table there, this one is the
   original 5):
     S0 clean                contract pass    index pass  readback pass          rc=0
     S1 M1 default 2*nOp->nOp contract FAIL(20) index pass  readback PASS (masked) rc=1
     S2 M2 no-validation     contract FAIL(30) index pass  readback FAIL 30/64    rc=1
     S3 M1+M2 (historical)   contract FAIL(42) index pass  readback FAIL 30/64    rc=1
     S4 fold_index_sfull drops +nOp
                              contract PASS(BLIND) index FAIL readback FAIL 60/64  rc=1
   (S1) = (fold_stride==0) ? nOp : fold_stride;  (S2) = return usm;  (S3) = both;
   (S4) = fold_index_sfull loses the `+ nOp` (its single home, post-refactor).
   CORRECTION to the first writeup: M2 is NOT masked. With the default correct
   but validation gone, the deliberately-too-small stride=1 regime is accepted
   and mis-reads; that fail-closed regime is what catches it. Only M1 is masked
   from the readback. S4 is the load-bearing one: the contract check reasons
   about stride arithmetic and is structurally BLIND to the kernel's indexing, so
   it reports 80/80 while 60/64 outputs change. => no single check covers the
   class: only the contract sees S1; only index+readback see S4.
   ALSO FIXED: the probe printed its one-line PASS/FAIL verdict BEFORE the E4
   section ran, so a reader could see PASS and then a failure that flipped the
   exit code; the verdict now prints last. And `bad` did not accumulate the
   reader-contract failures, so a failing run printed the self-contradictory
   "[dpas-fold-probe] FAIL (... 0 violations)" — the line now carries the real
   count (the probe's aggregate, which sums contract + index + readback:
S1=20, S2=31, S3=42, S4=722, S5=4, S6=810 — note this file previously said
"S4=2", a pre-index-check figure; per-COLUMN counts are in the report and differ,
e.g. S3's contract column is 40 not 42).
   THE MATRIX IS NOW A SCRIPT, NOT PROSE: scripts/mutate_fold_matrix.sh
   re-derives S0-S6 (each state alone from a clean header, per-state logs,
   restore + residue re-verified) and ASSERTS each state's character. 7/7 as
   characterized (S5/S6 added in AUDIT #13), stamped with the HEAD sha it was
   proven against
   (build/fold_matrix/last_pass.stamp). It REFUSES to start on a dirty tree
   because it edits tracked files and restores with `git checkout --` — which
   is a real hazard, not a hypothetical one: during this work an ad-hoc
   `git checkout --` on the header destroyed uncommitted helpers and broke the
   build, exactly as the guard predicts. Negative-tested on all three columns
   (contract, readback, index) — each reports UNEXPECTED, rc=1.
   THE STALE-ANCHOR HOLE, CLOSED: a mutation harness has artifact assumptions a
   refactor silently invalidates — the index refactor DELETED the text apply_m4
   corrupted, and the matrix only failed closed by luck of ordering. An
   ambiguous (not absent) anchor would have mis-targeted a state and still said
   "pass". preflight_anchors() now asserts every M1/M2/M4 target occurs
   EXACTLY ONCE plus two structural invariants, before any state runs; all
   three drift classes verified (rc=1 each). `--anchors-only` runs it in <1s
   with no build and no mutation. Re-point the apply_* if it fires; NEVER loosen
   the check — a loosened anchor is how a mutation silently stops mutating.
   AND THE SELF-CHECK WAS VACUOUS AT FIRST: it grepped its own source for the
   summary strings, so the pattern sat INSIDE the check and it could never fail
   — it reported OK with the S1 line deleted. Same failure class, one level
   down; found by mutation-testing the check instead of reading it. It is now
   BEHAVIOURAL: compares what the script RENDERS against a literal, and proves
   the residue scan detects each marker by seeding a temp file. Rewriting it
   that way immediately exposed a real bug it had introduced — the scan had
   become `grep -F '/\*M1\*/'`, where the backslash is a LITERAL under -F, so
   it could never match the actual '/*M1*/' marker (the old `git grep -e` form
   was a regex, where `\*` is correct). Moral: -F and regex spell the same
   pattern differently, and a "cleanup" between them is a silent no-op gate.
   Both the PASS summary and the residue scan now route through emit_summary()
   and has_residue(), so the strings the conclusions depend on have one home.
   S4's GAP CLOSED AT SOURCE (not just by a slower device check): every fold
   read now goes through fold_index_sfull/fold_index_s0 in quant_gemm.hpp, so
   the index expression has ONE home, unit-gated over 360 (m,stride,nOp,o)
   tuples (Sfull-S0 == nOp; S0 at the row base; neither escapes its row). Drop
   `+ nOp` from fold_index_sfull and the index check reports "halves WRONG"
   while the stride contract still says 80/80 — the blindness, caught for free.
   This is the lesson to carry into E4: reordering the work means rewriting
   INDEX expressions, and a stride contract cannot see indexing.
   GATES (correctness only, load-insensitive, NO wall-time claim): quant_gemm.hpp
   is a SHARED header pulled in by intfer_gemm, so the audit enumerated the
   consumers instead of stopping at the 4 that are obviously GPU-path.
   On the CURRENT 1-GPU box, re-verified at AUDIT #13: dpas_fold, cpu_parity,
   http_batch, ga64_recip, gguf_roundtrip, oracle_equiv, serve_load all PASS;
   gpu_optin and device_argmax are PARTIAL (their TP=1 arms all pass, their
   TP=2 arms do not run — **TP=2 is UNVERIFIED**); tp2_27b is SKIPPED.
   *** The line this replaces read "gpu_optin PASS (20/20 arms TP=1+TP=2)". It
   was true of the 2-GPU era (AUDIT #11: the second GPU later vanished) and is
   NOT a claim about the tree you are reading. ***
   LESSON: when a fix lands at ONE call site of a layout rule, grep for the
   OTHER sites that re-derive it — and check whether any of them document the
   wrong value as the contract, because a header that says "0 => nOp" is worse
   than no header: it launders the bug into the API. SECOND LESSON, from the
   bad first mutation table: do not generalize a row you did not run in
   isolation. S2 was asserted from a run where M1 was still applied; measuring
   it alone inverted the conclusion, and the S4 control then showed why BOTH
   checks are needed instead of a hand-wave.

   === AUDIT #13: THE 2-GPU PRECONDITION WAS BOTH OVER- AND UNDER-APPLIED ===
   AUDIT #12's follow-up wired `gate_require_gpus 2` into test_device_argmax.sh
   and test_tp2_27b.sh. Enumerating the family (not the files the fix happened
   to touch -- the audit #7 lesson) and re-reading what each gate actually
   contains found BOTH errors at once.
   (a) OVER-APPLIED in test_device_argmax.sh. That gate is MOSTLY 1-GPU:
   sections 1, 1b, 2, 3, 4 and 5 are TP=1 (5 runs in section 4 are
   cpu_parity/http_batch/gpu_optin, all 1-GPU), and only FIVE arms need a
   second device. A top-level `gate_require_gpus 2 || exit 2` therefore skipped
   the WHOLE gate on a 1-GPU box -- silently dropping section 5, the serve
   fence-lock test, which exists precisely because that crash "only manifests on
   a LATER sampling request" and is invisible to every stream gate. A gate that
   always skips is indistinguishable from a gate nobody runs, and it took the
   coverage of a known-dangerous path with it. FIXED: the guard is now PER ARM
   (HAVE_TP2), the five TP=2 arms are skipped, and the verdict reports
   PARTIAL PASS with exit 2 -- still never 0, still never a bare FAIL.
   Knock-on: section 4 ran its sub-gates with a bare `if ...; then`, so
   gpu-optin's new exit 2 would have been counted as a failure. sub_gate() now
   maps 0 -> PASS, 2 -> PARTIAL (counted, not failed), else -> FAIL.
   (b) UNDER-APPLIED: test_gpu_optin.sh was MISSED. It runs
   `for tp in 1 2` with ~14 assertions each, and had no precondition at all.
   This is the worst of the three outcomes the helper exists to prevent, and
   worse than the false FAIL it was introduced to fix: because its assertions
   are stream-IDENTITY comparisons, the tp=2 pass on a 1-GPU box re-ran tp=1
   and every assertion HELD, so the gate printed PASS while certifying nothing
   about tensor parallelism. A false PASS is strictly more dangerous than a false
   FAIL. FIXED: TPS is "1" or "1 2" by device count, verdict exits 2 when the
   tp=2 pass was skipped.
   (c) test_tp2_27b.sh's top-level skip is CORRECT and unchanged: its one arm
   is `--tp 2` on 27B and it asserts TP=sharded, so there is no 1-GPU coverage
   to preserve.
   FAMILY, EXHAUSTIVELY ENUMERATED (grep every script for a TP>1 arm): 8 scripts
   match -- bench_gpu_ab.sh, bench_quiet_l4_ab.sh, bench_serve.sh, serve_27b.sh
   (measurement/serve harnesses, deliberately unguarded per the audit #8 scope
   note), and test_device_argmax.sh, test_gpu_optin.sh, test_tp2_27b.sh
   (correctness gates, all three now guarded -- per-arm where 1-GPU coverage
   exists, whole-gate only where none does).
   THIRD FOLD COVERAGE HOLE, FOUND WHILE DOING (a)/(b): the matrix never
   mutated the WRITER, and the writer open-coded the stride rule as a bare
   `if (fold_stride < 2*nOp) return dep;` -- a SECOND copy of a rule whose own
   header says "Do NOT resolve or validate this by hand at a call site", i.e.
   the exact condition AUDIT #12 exists to eliminate, recreated 800 lines away.
   Invisible to the contract check (which only calls fold_readable) and to the
   whole matrix. FIXED: one shared decision, fold_stride_ok(), consumed by the
   writer and by fold_readable; the writer also gained the null check that
   mirrors fold_readable's (it dereferences fold_usm unguarded, so null was a
   device-side null WRITE rather than a decline). Two new states prove it:
     S5 writer gate removed       -> rc=1, and contract/index/readback ALL stay
                                      green. Only the probe's two NEG stride
                                      rows see it, so that copy was verified by
                                      exactly two hand-picked rows and nothing else.
     S6 fold_index_s0 row offset  -> rc=1, and ONLY the index check sees it.
     S6 is the sharpest result in the matrix: fold_index_s0 is called from the
     u2/G16 split branch ONLY, and every device readback in the tree runs
     bits=4 (the u4 path, which reads just fold_index_sfull). So
     fold_index_s0 has NO device coverage anywhere -- the host-side index check
     added in AUDIT #12 is its sole gate. Matrix is now 7/7 (S0..S6) and the
     claim is "all FOUR checks are load-bearing", each invisible to the others.

   === FULL 10-GATE SUITE ON THIS TREE — 7 PASS, 2 PARTIAL, 1 SKIPPED [SUPERSEDED 2026-10-01: now 10 PASS, 0 PARTIAL, 0 SKIPPED] ===
   Re-run after test_dpas_fold.sh was added. PASS: cpu_parity, dpas_fold,
   ga64_recip, gguf_roundtrip, http_batch, oracle_equiv, serve_load.
   PARTIAL (exit 2, not FAIL): device_argmax and gpu_optin — every arm that
   runs on 1 GPU is green, and only their TP=2 arms are skipped, because the
   box exposes one GPU. tp2_27b is SKIPPED outright (exit 2): its single arm is
   `--tp 2` on 27B, so there is nothing it can still check here.
   Those gates used to report a BARE FAIL, which was the whole problem: a
   clamped TP=1 run (Engine::load min()s tp_devices to dm.size()) fails an
   assertion about a path it never took, and a red gate that means "the box"
   teaches everyone to ignore red gates. And gpu_optin was worse still — it
   reported PASS, because identity assertions on a re-run tp=1 trivially hold.
   All three now go through scripts/_gate_precond.sh, PER ARM where 1-GPU
   coverage exists. Never exit 0 (that would claim coverage nobody got) and
   never exit 1 (indistinguishable from a real regression). Helper paths
   unit-checked: require-met -> 0, require-unmet -> 2, empty arg -> 1,
   sycl-ls absent -> 2. See AUDIT #13 for why whole-gate skipping was itself a
   defect.
   *** OUTSTANDING, NOT DONE: the TP=2 arms of device_argmax and gpu_optin,
   and all of tp2_27b, are UNVERIFIED ***
   RE-RUN AFTER AUDIT #13 (this box, 1 GPU, correctness only, no timing claim).
   All ten gates executed; nothing skipped silently. Tally by EXIT CODE, which
   is the only non-double-countable measure:
     rc=0 (PASS), 7: cpu_parity, dpas_fold, ga64_recip, gguf_roundtrip,
       http_batch, oracle_equiv, serve_load.
     rc=2 (PARTIAL), 2: gpu_optin and device_argmax. Every tp=1 / TP=1 arm
       inside them ran and passed -- gpu_optin's KV/Q8/HAD/SSM_GPU, the whole
       DPAS48 family, PREQ x ROW/PREW/SSM_GPU, the wide M=97 arm, the n=24 FOLD
       arms and FOLD-alone inert; device_argmax's sections 1, 1b, 2, 3, 3b with
       PARTS 32/40/40 exactly as locked, plus the serve fence lock (server alive
       through greedy+sampling+mixed). Their TP=2 arms did not run.
     rc=2 (SKIPPED), 1: tp2_27b.
   *** COUNTING TRAP, hit while writing this very block: it first read "8 PASS"
   by listing device_argmax's green TP=1 sections alongside the seven rc=0
   gates. The GATE exits 2, so that is 7, not 8. An aggregate that does not
   reconcile against per-gate exit codes is the same defect class as a report
   that claims coverage it does not have. ***
   *** THE WRONG NUMBER ALSO LIVES IN COMMIT 793b880's SUBJECT LINE, and it is
   left there on purpose. Do not "fix" it by amending: 530cc81 is the commit
   that records the error and its correction, so the history explains itself,
   and rewriting a published message to hide a mistake that the next commit
   already documents destroys the only record that the mistake happened. The
   same rule as the era-marked TP=2 lines above, applied to commits. ***
   Two things this run actually PROVED about the fix, not just exercised:
     - device_argmax's section 4 reported `gpu-optin: PARTIAL/SKIP (rc=2, box
       state -- not counted as a failure)`. Without sub_gate()'s rc map, the
       gpu_optin fix would have turned a legitimate skip into a red line here —
       the same false-FAIL class one level up. Caught in the same commit that
       caused it.
     - The serve fence lock ran and passed. Under the previous top-level skip it
       never ran at all, so that crash-only-on-a-later-sampling-request path had
       been silently uncovered on every 1-GPU box.
   Fold matrix 7/7 (S0..S6) on the single-sourced rule; anchors-only rc=0;
   residue 0/0; 36/36 scripts `bash -n` clean; all 10 gates carry a build
   guard; all 29 artifact consumers still honor INTFER_BUILD_DIR (re-enumerated
   *** EVERY HAND-COUNTED TOTAL IN THIS FILE IS A CITATION, NOT A COUNT ***
   "36/36 scripts", "39/39 scripts", "29 artifact consumers" etc. above were
   true when written and are NOT re-derived by anything. Scripts have been added
   since (they are added, not replaced), so today's figure is higher. Do NOT
   "correct" one of these to match a present-day `ls` -- rewriting a true
   historical record to fit today's box is the same laundering failure in the
   other direction, and it destroys the only record of what the tree looked like.
   Conversely do not quote one as present-day coverage. Derive the live numbers
   from source when they are needed: `ls scripts/*.sh | wc -l` for the syntax
   sweep, and `scripts/gate_inventory.sh` for gate/consumer classification
   (re-enumerated from the CMake `add_executable` list, not from a remembered
   name list). **`gate_inventory.sh` now ENFORCES this rule**: it fails when a
   hand-counted total is presented as a live claim, excluding paragraph-scoped
   history, explicitly-cited lines, and quoted claims the doc is refuting.
   (882a0c2). A number in prose is a citation.
   They are blocked on a second GPU, not on missing work. Do not let any
   future summary fold "skipped" into "covered" — the correct phrasing is
   "7 PASS, 2 PARTIAL, 1 SKIPPED [SUPERSEDED 2026-10-01: now 10 PASS, 0 PARTIAL, 0 SKIPPED], E1-E5 unmeasured". Itemized table of exactly
   what is
   unverified and how to close it: reports/2026-09-30-pending-bench-backlog.md,
   "OUTSTANDING — blocked on the second GPU".
  === 2026-09-30 AUDIT #14: THE AUDIT-#13 CHECKER HAD A PROVEN FALSE
  NEGATIVE, AND THE COUNTING HELPER CORRUPTED ITS OWN VALUE AT 0 GPUs ===
  AUDIT #13 shipped scripts/gate_inventory.sh as the machine check that the
  doc tallies and the skip shapes reconcile, plus the fold-matrix report cells
  and the 7/7 stamp. Auditing that machinery found four defects, all in the
  checkers — none in the engine, and the fold matrix itself still holds 7/7.
   (1) THE HEADLINE INVARIANT HAD A FALSE NEGATIVE (proved by mutation, not
   argued). Granularity was classified by grepping for the PRESENCE of
   per-arm marker text (`HAVE_TP2|SKIPPED_TP2`), so a gate carrying a
   top-level `gate_require_gpus 2` *while keeping* the per-arm machinery was
   still read as "per-arm" — the exact audit-#13 defect, re-added, and
   gate_inventory exited 0. Measured: RC=0 with the defect present, RC=1
   after the fix. Root cause: a whole-gate skip was detected by the ABSENCE of
   something, which is not a detection. Now derived from an actual
   line-start `gate_require_gpus` statement.
   (2) BOTH CLASSIFICATION COLUMNS READ PROSE, NOT CODE. test_device_argmax.sh's
   ONLY `gate_require_gpus` occurrence is inside the comment explaining the
   audit-#13 defect it must never re-introduce. So the PRECONDITION column
   labelled it `gate_require_gpus` while it actually uses `gate_gpu_count`,
   and reported test_gpu_optin.sh — the identical mechanism — under a
   different name. Two gates, same shape, two labels, in the one column a
   reader would act on. Every classification now runs on a comment-stripped
   copy of the source.
   (3) THE COUNTING HELPER CORRUPTED ITS OWN VALUE ON THE CASE IT EXISTS FOR.
   `gate_gpu_count` was a bare `sycl-ls | grep -c`, and `grep -c` prints "0"
   *and exits 1* when nothing matches. So every `n="$(gate_gpu_count || echo
   0)"` captured the TWO-LINE string "0\n0" precisely when Level Zero reports no
   device — the libumf LD_LIBRARY_PATH case the helper is written about. The
   following `[ "$n" -ge 2 ]` printed `[: 0
   0: integer expected`. The verdict survived by luck (a failed test is
   non-zero) but gate_inventory's `NDEV -eq 0` loud-skip branch could never be
   taken, so on the one box state it exists for it fell into the "GPUs are
   visible, run the tally" path instead. Fixed at the source: the helper now
   prints exactly one integer line and signals "could not count" by RETURN CODE
   ONLY, never by text mixed into the value. Callers use `|| N_GPUS=0` (a
   substitution failure), not `|| echo 0`.
   (4) THE CHECKER EXEMPTED ITSELF FROM THE RULE IT ENFORCES, AND THE NEW
   REPORT HAD NO PROVENANCE. gate_inventory.sh did not `source setvars.sh`
   (every sibling gate does), so on a bare shell `sycl-ls` was absent, the count
   was 0, the doc-tally check SKIPPED — and the script still exited 0. It now
   sources setvars itself and, if it still cannot see a GPU, exits 2 (never 0).
   Separately, scripts/fold_report_counts.sh printed "7/7 as characterized"
   from logs stamped at a806ef3 while HEAD was ddea91d and 4b31d25 had touched
   the very header five of the six mutations target: family 3 (stale artifact)
   in a brand-new script. It now prints the stamp-vs-HEAD delta, names any
   mutated file changed since, refuses the unqualified verdict, and exits 2.
   The honest reading of that state: the cells are still VALID (4b31d25 was
   comment-only, and `--anchors-only` re-proves the anchors still resolve) but
   they are evidence for a806ef3, not for HEAD — a citation, not a guess.
   RESOLVED, not argued away: after audit #15 the full matrix was re-run on the
   final tree (7/7, with real device readback regimes), so the stamp is current
   and `fold_report_counts.sh` certifies rc=0 with "logs match HEAD". The
   STALE-qualifying path is RETAINED, not deleted to make a report green: it is
   the behaviour that makes the rc=0 mean something.
   MUTATION-TESTED, because the previous checker was not. The evidence is
   COMMITTED as a re-runnable script, scripts/mutate_gate_inventory.sh, NOT
   left in a transcript: a mutation that exists only in a commit message cannot
   be re-run, so the next session re-derives (or re-asserts) it. It mirrors
   scripts/mutate_fold_matrix.sh — dirty-tree guard, EXIT trap + `git checkout
   --` restore, an anchor pre-flight asserting each of the 6 targets occurs
   EXACTLY ONCE, per-state logs + summary.txt, a byte-identical restore check,
   a residue scan, and a sha/date stamp. 6/6 mutations refused the checker
   (M1–M5 via the checker's own self-check, M6 via its exit-2 loud-skip path):
     M1 un-anchor the line-start require test — the LOAD-BEARING row. With
        comment-stripping in place this is an EQUIVALENT mutation on the real
        gates, so it is observable only through the BROKEN self-check case
        (an unplaceable `gate_require_gpus` wrapped in a helper). Proves the
        anchoring is tested rather than merely present.
     M2 drop comment stripping, M3 reintroduce the stdin-consuming grep chain,
        M4 invert a self-check expectation, M5 disable the NO-SKIP-REPORT check.
     M6 let a 0-GPU box exit 0 again — the defect (4) above, replayed with a
        stub `sycl-ls` so it needs no special hardware.
   TWO SCORING RULES THE HARNESS ENFORCED ON ITSELF, both from it refusing to
   launder a red state green: a mutation that BREAKS SYNTAX is scored MISSED,
   not caught (a syntax error is not evidence a mechanism is load-bearing) —
   which caught three of the first draft's six mutations, written as shell
   strings with a `|||` delimiter that left trailing garbage and unbalanced
   quotes; and the first M3 attempt, which patched only the elif with
   `- <"$1"`, reopened the file for that one grep and starved nothing, so it
   changed no verdict and was scored MISSED. Reproducing the real bug needs
   both greps on a shared fd, hence the multi-hunk patch support. Harness
   guards proven non-vacuous: dirty tree -> rc=1 without mutating; duplicated
   anchor -> "ANCHOR STALE ... found 2 occurrence(s), want 1", rc=1.
   RE-RUN ON THIS TREE AFTER THE FIX (1 GPU, correctness only, no timing
   claim). All ten gates have MEASURED exit codes on the final bytes — the
   device gates were re-run with a generous timeout specifically so their rc is
   observed rather than a timeout artefact:
     rc=0 (PASS), 7: ga64_recip, dpas_fold (3 real device readback regimes),
       gguf_roundtrip, cpu_parity, oracle_equiv, http_batch, serve_load.
     rc=2 (PARTIAL), 2: gpu_optin "tp=1 arms all green; tp=2 pass SKIPPED",
       and device_argmax "TP=1 arms all green; skipped: TP2 arms=1,
       sub-gates=1" — measured, NOT killed on a timeout this time. Every TP=1
       arm inside it ran and passed: the 6-way TP=1 stream matrix, the
       [argmax] counter non-triviality lock (1 on flag, 0 on base), the
       sampling count parity, both spec-verify sections, PARTS 32/40/40 exactly
       as locked, cpu parity + http batch PASS, and the serve fence lock with
       the server alive through greedy+sampling+mixed.
     rc=2 (SKIPPED), 1: tp2_27b.
   Also on the final bytes: gate_inventory rc=0 with the tally actually RUNNING
   (sites_found=8, 0 stale) and the corrected PRECONDITION column;
   mutate_gate_inventory 6/6; the audit-#13 mutation re-added -> rc=1 with the
   OVER-SKIP marker; 0-GPU box -> rc=2 loud; fold_report_counts rc=2 (stale
   stamp, as it should be); fold-matrix anchors-only rc=0; 39/39 scripts
   (historical citation -- see the "EVERY HAND-COUNTED TOTAL" note)
   `bash -n` clean. No engine, quant or schedule code touched.
  === 2026-09-30 AUDIT #15: THREE GPU GATES HAD NO DEVICE GUARD AT ALL ===
  The follow-up audit to #14 asked one question AUDIT #8 had already answered
  the hard way: is the instance list still right? #14 hardened the checker
  without re-deriving what the checker checks. Re-deriving — from the gates
  themselves, not from the 10 names the docs remembered — found three gates that
  `gate_inventory.sh` had been reporting as precondition "none", counted in the
  among the seven gates counted as exits-0, and which are 100%
  device-dependent:
   test_dpas_fold.sh   runs dpas_fold_probe, whose 3 device readback regimes are
     the whole point of the gate. It said "needs a GPU" in its HEADER COMMENT
     (line 2) and carried no guard whatsoever — the comment was the entire
     precondition. This is the sharpest instance of the class #14 found: a
     comment read as a mechanism.
   test_http_batch.sh  launches the server `--gpu --tp 1` (line 45).
   test_serve_load.sh  launches the server `--gpu --tp 1` (line 62).
  All three PASS on a 1-GPU box, which is exactly why the gap survived a day of
  gate work: the failure is not visible until the second GPU goes away, at which
  point they go RED and a red line is indistinguishable from a regression. The
  old `sycl-ls | grep -c` render check in test_device_argmax.sh/test_gpu_optin.sh
  is the same idiom these three lack.
  ROOT CAUSE, precisely: gate_inventory's GRANULARITY column classified
  "needs no device" and "needs a device and forgot to say so" IDENTICALLY, both
  as "none". Two different things, one label, and on a healthy box both exit 0.
   FIX. gate_require_render_node() in scripts/_gate_precond.sh, sibling of
   gate_require_gpus, exits 2 (SKIP) with the same "BOX STATE, not a code
   failure" framing. It is the STRICTER and more correct of the two
   preconditions: a missing libumf makes sycl-ls report zero devices on a
   perfectly healthy box, so a device COUNT would refuse coverage that would in
   fact run, whereas opening /dev/dri/renderD128 is a direct syscall that no
   failed adapter can fake. All three gates already sourced setvars.sh, so the
   guard sits beside it. Verified BOTH directions with a stubbed python3: node
   open -> rc=0 and the gates still pass; node unopenable -> each of the three
   exits 2 with the loud message.
   CLOSE THE CLASS, not the instance. gate_inventory gains a RENDER-NODE column
   that re-derives GPU dependence from each gate's own code (--gpu / sycl-ls /
   level_zero / xpu-smi / dpas_fold_probe) and fails a GPU gate carrying no
   guard, so a newly added GPU gate cannot reopen this. M7 in
   scripts/mutate_gate_inventory.sh proves it can fail, and M7's own history is
   the argument: the FIRST M7 neutered the loop's `rnode="MISSING"` branch and
   was scored MISSED, correctly, because every GPU gate on the real tree is
   guarded so that branch is UNREACHABLE. Fixed by hoisting the verdict into
   gpu_guard_state() and adding a self-check case that seeds an unguarded GPU
   gate. That is the audit-#14 equivalent-mutation lesson recurring one level
   down, and it was caught by the harness rather than argued about.
   The doc tally is deliberately UNCHANGED at 7 PASS / 2 PARTIAL / 1 SKIPPED [SUPERSEDED 2026-10-01: now 10 PASS, 0 PARTIAL, 0 SKIPPED].
   The render-node guard exits 2 only on a box with no device, so on a 1-GPU box
   these three are the same rc=0 they always were: the guard is protective, not a
   behaviour change. A tally that moved because a no-op guard was added would be
   the same overclaim in a new place.
   Matrix now 7/7 (M1–M5, M7 via the checker's self-check; M6 via its exit-2
   path). No engine, quant or schedule code touched.
  === AUDIT #15 (b): THE FOLD MUTATION HARNESS LEFT A MUTATED BINARY ON DISK ===
  Found by the final verification pass, and it is the same class as everything
  else above — family 3, stale artifact — except that the harness that CAUSES it
  is the one written to prove the checks work.
  `scripts/mutate_fold_matrix.sh` restores the SOURCE with `git checkout --` and
  nothing else. Its last state is S6, so the `dpas_fold_probe` binary left in the
  build tree is the MUTATED one. Any later gate that reuses the build without
  rebuilding (`INTFER_GATE_NO_BUILD=1`) then judges mutated code. Observed live:
  the parallel 7-gate re-run reported `test_dpas_fold rc=1` with the S6
  signature — 360 index tuples WRONG, m=4095 reading S0=224 — while `git status`
  was clean, and a one-line rebuild turned it PASS. The gate was RIGHT and the
  harness was wrong, which is the only acceptable way for this to surface.
  Note what the old post-run check could not see: it asserted the SOURCE was
  byte-identical to HEAD, which was true, and said nothing about the 4 MB binary
  next to it. Restoring text and restoring a build are different acts.
  FIX. The matrix's last act is now restore -> REBUILD `dpas_fold_probe` through
  the shared `gate_require_build` helper -> run `test_dpas_fold.sh` with the
  build DISABLED and require rc=0. A green result is the real proof that the tree
  was left in the state the matrix claims, and it also catches an artifact
  poisoned by an EARLIER run, which no source-diff check can see.
  Proven non-vacuous both ways: poison the binary, then run the matrix with the
  rebuild opted out -> FAIL, 7 states UNEXPECTED plus the explicit "the build
  tree is poisoned" message; run it normally -> the rebuild self-heals and it
  reports 7/7 with "post-restore rc=0 on the LEFT-BEHIND artifact".
  GENERALISATION for the next matrix: a mutation harness that edits tracked
  source must treat the build tree as part of the state it mutates. A
  byte-identical restore of the sources is not a restore.
  === 2026-09-30 AUDIT #16: THE DEVICE GUARD COVERED HALF THE FAILURE MODES,
  AND THE ANCHOR PRE-FLIGHT WAS CHECKING LESS THAN IT CLAIMED ===
  (1) `gate_require_render_node` protected only ONE of the two ways a gate can
  fail to get a device. It exits 2 when /dev/dri/renderD128 will not open, but a
  box where the node opens fine and Level Zero enumerates ZERO devices — the
  libumf LD_LIBRARY_PATH failure documented in this repo's own notes, and the
  common one on THIS box — still produced a RED line from test_dpas_fold /
  test_http_batch / test_serve_load. The helper's own comment already said
  callers needing a count "use both", and no gate did.
    FIX: `gate_require_device()` in scripts/_gate_precond.sh checks BOTH, node
  first (a direct syscall, free) then the count, exit 2 on either. All three
  gates now route through it, and test_gpu_optin / test_device_argmax had their
  INLINE render-node probes replaced with the shared helper — two fewer
  hand-rolled copies of the same probe, which is how this family reopened the
  first time. Zero hand-rolled `renderD128` probes remain in scripts/test_*.sh.
    Verified in ALL THREE directions with in-tree fixtures
  (scripts/fixtures/{no-device,no-render-node}): node open + 1 GPU -> rc=0;
  node unreachable -> every device gate rc=2; node open + 0 devices -> every
  device gate rc=2. (Measured by hand first; since AUDIT #17 this is a COMMITTED
  HARNESS, not a transcript claim — run `scripts/check_device_precond.sh` for the
  12-cell matrix and `scripts/mutate_device_precond.sh` for its 3/3
  non-vacuity proof. Note there are SIX GPU gates, not five: test_tp2_27b.sh is
  count-guarded and was outside the original five.) The fixtures exist because a
  proof that lives in a transcript cannot be re-run (audit #13/#14), and because
  `source setvars.sh` PREPENDS the oneAPI bin dir, so a PATH stub for sycl-ls is
  shadowed inside every gate. Hence the `INTFER_SYCL_LS` override — an assumption
  baked into a probe that the harness cannot redirect is an assumption the
  harness cannot test. Same rationale as INTFER_BUILD_DIR.
    TWO BUGS FOUND BY RUNNING THE DIRECTIONS RATHER THAN READING THE CODE, both
  introduced by this very fix:
    * `gate_require_device` was a SILENT NO-OP when sycl-ls was absent. It took
      `n="$(gate_gpu_count)"` with no rc check; that returns EMPTY stdout for
      "unmeasurable", and `[ "" -lt 1 ]` is a bash integer error whose result is
      FALSE — so the helper returned 0, i.e. certified a device it could not
      see. That is audit #14's "two failure modes in one value" class recurring
      one function down. Now gated on rc AND emptiness, and "cannot count" exits
      2 loudly.
    * The two per-arm gates called the whole-gate guard BEFORE sourcing setvars,
      so the count was always empty and (by the bug above) always passed. The
      guard now runs after setvars, which is the only point at which sycl-ls
      exists to be counted.
  (2) The anchor pre-flight in scripts/mutate_gate_inventory.sh parsed its table
  with `IFS='|'` while the anchor PATTERNS themselves contain `|` — ERE
  alternation and `||`. Every such row was silently TRUNCATED at the first `|`,
  and because a truncated prefix is still a substring, the pre-flight went on
  printing "found exactly once" while checking strictly LESS than it claimed.
  Four of nine anchors were affected. That is the failure the pre-flight exists
  to prevent — a loosened anchor — arriving through the pre-flight's OWN parser.
  Surfaced only because the M7 patch went stale after gpu_guard_state() learned
  about gate_require_device and the two disagreed; `patch_once` reported 0
  matches while the pre-flight said 1.
    FIX: the patterns now live in a QUOTED ARRAY, where a pattern may contain
  any character, and a parallel ANCHOR_FILE array says which file each lives in
  (M9's target is in a GATE, not the checker, and a single-file pre-flight
  silently checked zero of it). A behavioural self-check now seeds a DUPLICATED
  anchor into a copy of the target and requires the pre-flight to REJECT it —
  a pre-flight that cannot fail manufactures confidence rather than checking.
  (3) The RENDER-NODE column emitted "-" both for "not applicable" and for "not
  checked", which a reader cannot tell apart. It now emits one of four tokens:
  `via-count` / `render-node` / `not-gpu` / `MISSING`.
  (4) Audit #15's end-to-end proof — seed an unguarded GPU gate, watch the
  column catch it — lived only in a transcript, the exact defect this matrix
  exists to stop. It is now state M9, whose target is a GATE rather than the
  checker: strip the device guard from test_dpas_fold and the column MUST report
  MISSING and the checker MUST exit 1. Proves both that the guard is
  load-bearing and that the column is live. Matrix 9/9.
  === TOOLING NOTE: THE BULK-EDIT TOOL LOST AN EDIT A THIRD TIME ===
  The `gpu_guard_state()` function definition was inserted with the bulk edit
  tool, which reported success; the four CALL SITES (written by a later
  python3 patch) persisted while the DEFINITION did not, and the failure only
  surfaced because the self-check I had just added exercised the function and
  printed `gpu_guard_state: command not found`. AGENTS.md already records two
  silent losses of the same kind. The generalisation: a tool reporting success
  is not evidence the bytes are on disk, and a self-check that CALLS the new code
  is what converts a silent loss into a loud one. Verify with grep for the
  DEFINITION, not for its uses.

- **AUDIT #17: THE FIXTURES WERE COMMITTED, THE DRIVER THAT RUNS THEM WAS NOT**
  (scripts/check_device_precond.sh + scripts/mutate_device_precond.sh; no
  engine/quant/schedule code touched). Audit #16 added two stubs —
  `scripts/fixtures/{no-device,no-render-node}` — and recorded the measured
  result at AGENTS.md:946-948. Enumerating the tree, not the transcript:
  `$ grep -rn 'fixtures' scripts/*.sh` -> **no matches**. The stubs were
  committed; the wiring that drives them (prepend the dir to PATH, set
  `INTFER_SYCL_LS`) existed only in the stub's own header comments. So the
  device-direction proof still lived in a transcript, with a decorative fixture
  attached to it — the exact defect the fixture was added to fix, one level up.
  A stub with no driver is a comment with a shebang. Meanwhile
  `mutate_gate_inventory.sh` (audit #16) was exactly the pattern that was
  already right in this repo, so the omission was not a judgement call.
    FIX: `scripts/check_device_precond.sh` runs 6 GPU gates x 2 failure
  directions (12 cells) and asserts rc=2 AND the correct DIAGNOSTIC per cell;
  `scripts/mutate_device_precond.sh` proves it can fail (3/3). Both stamp the
  sha. Four load-bearing pieces, each added because a weaker version was
  measurably vacuous:
    * SEAM DISCRIMINATION. `gate_require_device` turns BOTH "0 devices" and
      "sycl-ls unmeasurable" into exit 2, so if the `INTFER_SYCL_LS` seam rots
      the whole zero-device direction silently degrades into "sycl-ls is
      missing" while still reporting green. The harness requires the two causes
      to produce DIFFERENT diagnostics and fails if they stop being
      distinguishable — otherwise direction 2 is a tautology that cannot go red.
    * PER-GATE DIAGNOSTIC, not just rc. rc=2 for the WRONG reason is still a
      defect: it reports box state that is not the box state, and it is how a
      guard quietly degrades to a weaker mechanism. M-C below is exactly this.
    * CLASSIFICATION CROSS-CHECK. The device-gate list is derived in the harness
      AND cross-checked against the RENDER-NODE column `gate_inventory.sh`
      reports; disagreement is a hard fail. Two derivations that must agree,
      deliberately not a third hand-maintained table (the copy-drift family).
    * BEHAVIOURAL SELF-CHECK, run FIRST, gating the matrix. Three synthetic
      gates: one that exits 0 under a failure direction (must be REJECTED), one
      that skips for the wrong cause (must be REJECTED), one with the correct
      cause (must be ACCEPTED, so the first two cannot pass by an always-red
      harness). Executable scripts, never a grep of the harness's own source.
  MEASURED, all 12 cells rc=2: 5 gates report `node` + `count0`;
  **test_tp2_27b.sh reports the COUNT cause in BOTH directions** — correct, and
  not a gap: for a >=2 GPU requirement both failure modes collapse to
  `count < need`, so a node probe would be redundant, and demanding a node
  message would have been force-fitting. This closes the audit-#16 claim's
  coverage hole too: it said "all FIVE device gates"; there are SIX GPU gates
  and test_tp2_27b was never exercised in either direction.
  NAMING IS LOAD-BEARING. The checker must NOT be named `test_*.sh`.
  `ls scripts/test_*.sh` IS the definition of the gate population
  `gate_inventory.sh` classifies, and this checker's own CODE names the stub
  path (`NODEV_STUB=".../fixtures/no-device/sycl-ls"`), so as a `test_*` it
  would be classified a GPU gate carrying no device guard — a false MISSING row,
  against a checker that by design needs no device. Caught before it happened by
  asking what the glob would classify, not after.
  MUTATION MATRIX 3/3, 0 unattributed (scripts/mutate_device_precond.sh):
    * M-A strip a real gate's device guard -> classification cross-check
      catches "needs a device but carries NO guard at all".
    * M-B clear a fixture's EXECUTE BIT. The nastiest shape: PATH lookup skips
      a non-executable file, so a 644 stub hands every probe back to the real
      tool and BOTH directions become no-ops that still report green. It does
      not fail, it stops testing. Caught by a fixture pre-flight that stats the
      mode. (Verified by measurement that `git checkout --` restores the
      recorded mode, 644 -> 755, so M-B restores cleanly.)
    * M-C weaken `gate_require_device` -> `gate_require_render_node`, a HALF
      guard: still correct for node-unreachable, blind to the 0-device libumf
      failure this box actually hits. The gate then runs and exits 0. This is
      the audit-#16 defect in its purest form and the reason the matrix needs
      BOTH directions: a single-direction matrix scores M-C as a PASS.
  === BUGS FOUND BY RUNNING IT, INCLUDING SEVERAL I INTRODUCED ===
    * A `fi` closing a `for` loop. It was MASKED because I chained
      `bash -n f && echo OK; echo RUN; run` with `;` instead of `&&`, so the
      script ran anyway and produced plausible partial output. Gate your own
      verification commands; a syntax check that is not load-bearing is a
      comment.
    * THE BULK-EDIT TOOL MERGED THREE SELF-CHECK CASES INTO ONE WRONG BLOCK
      (cases a/b/c collapsed; a later edit overwrote the real case c). The
      FOURTH occurrence of the silent-loss family already recorded three times
      above — reproduced inside the fix for that very family, by a different
      tool. Detected by counting gate DEFINITIONS with grep, then by a
      self-check that actually CALLS the code.
    * The self-check's INTENDED rejections each incremented the global failure
      counter through `expect_skip`'s own `bad()`, so the run ended FAIL(2) with
      every real assertion green — the always-red twin of the vacuous check.
      Only the self-check's explicit verdicts may count (`SUPPRESS`).
    * CROSS-CONTAMINATION, the one that nearly produced a fake result: M-B and
      M-C were first run in a PARALLEL tool batch sharing one working tree, so
      M-C inherited M-B's `chmod` and "passed" by aborting on M-B's pre-flight.
      That result was DISCARDED and M-C re-run serially, where it was caught
      for the right reason. Two of these states touch the same files, so the
      matrix now takes a `mkdir` lock and re-asserts a clean pre-state before
      EVERY state (asserting inside `run_state` was itself wrong: M-B's mutation
      IS the chmod, so it would always abort).
    * The `via:` attribution pattern `^ *FAIL` dropped M-C's reason, because the
      matrix prints a per-cell prefix BEFORE the FAIL — a real catch was
      reporting as unattributed. Now matches FAIL anywhere, and an
      UNATTRIBUTED catch FAILS the run: a checker that exits non-zero without
      naming an assertion may be failing for an unrelated reason.
    * The M-B anchor `python3` matched 3x in the fixture (two comments + the
      exec) and the pre-flight refused to score. Correct refusal: an anchor
      matching several sites proves nothing about which is mutated. Re-pointed
      to `exec /usr/bin/python3`.
  === TOOLING CAUTION: TWO OF MY "FINDINGS" WERE OUTPUT ARTIFACTS ===
  Before the real work I nearly filed two bogus defects, both from trusting a
  tool's rendering rather than the system:
    * `ls -l` in this shell DROPS the permission columns, so both fixtures
      appeared to be mode 644. `stat -c %a` says 755 and git records 100755.
    * `ls /dev/dri` showed only `by-path/`, suggesting no render node — yet
      `python3 -c "open('/dev/dri/renderD128','rb').close()` returned 0, which
      is exactly what the gate probes, and all three render-node-guarded gates
      passed. `/dev/dri` listing is not a device-existence test.
  Rule: for permissions use `stat`, for device existence use the same syscall
  the gate uses. Never report a defect read off an `ls` rendering.

- **MUTATION-ANCHOR WRITING RULE (2026-10-01, learned twice in one session while
  adding state N-K).** An `A_*` anchor is a literal that must occur EXACTLY ONCE
  in the target so `patch_literal` can swap it. Three rules, each bought with a
  failed state:
  1. **Span the whole syntactic unit being removed.** The first N-K anchor stopped
     at `HEAD_MOVED=1` and orphaned the closing `fi`; `verify_patched` aborted on a
     syntax error. Correct behaviour — a syntax error is not evidence a mechanism
     is load-bearing — but the state scored nothing. If the replacement is a
     one-liner, the anchor must cover every line the one-liner absorbs.
  2. **Build it with a quoted heredoc, never hand-escaped inside single quotes.**
     `read -r -d '' A_X <<'EOF' ... EOF`. That block contains `'STALE'`, and
     escaping those quotes into a single-quoted shell literal produced a syntax
     error in the MATRIX SCRIPT ITSELF — the same anchor-covers-the-wrong-bytes
     failure, one level out, and it broke the file before any state ran.
  3. **Assert uniqueness before committing, and before the baselines.** The
     pre-flight (`check_anchor`) now runs BEFORE the baselines, not after: an
     anchor that has drifted should fail in the first second, not after eleven
     baselines have each executed a full mutant run.
  The unifying lesson: an anchor that covers the wrong bytes does not fail
  quietly, but it fails in a way that reads as a *harness* problem rather than a
  *state* problem — so it is always worth confirming the anchor form mechanically
  (`occurs exactly 1x`, first line, last line) instead of by eye. And note the
  family: this is the same shape as audits #3 (rule fixed at one site, re-derived
  at another), #4 (checked on 1 of M rows) and #12 (stride rule enforced only in
  the writer), recurring in the tooling that verifies those very fixes.

- **AUDIT #17b: THREE BLIND SPOTS IN THE DEVICE-PRECOND PROOF, AND A DRIVER FOR
  THE WHOLE STACK** (scripts/check_device_precond.sh healthy column,
  scripts/mutate_device_precond.sh now 6 states, NEW scripts/run_all_checks.sh;
  still no engine/quant/schedule code touched). Audit #17 closed the "fixtures
  with no driver" gap. This pass asked the question #17 should have asked first:
  **which claimed mechanism is actually exercised by a state that exists?**
    (1) **M-A NEVER EXERCISED THE DRIFT COMPARISON.** Stripping a gate's device
  guard tripped the per-file "carries NO guard at all" message, so the
  `own_devices != inv_devices` branch was never the assertion that fired — a
  claimed load-bearing check that was STRUCTURALLY BLIND, which is the audit-#12
  S4 shape. New state **M-D** narrows ONLY this checker's GPU-token regex, so its
  derived list shrinks (dropping test_dpas_fold.sh, which names dpas_fold_probe
  and carries no `--gpu`) while gate_inventory.sh still lists that gate, and
  neither gate is broken. "classification drift" must fire.
    (2) **4 CLAIMED MECHANISMS, 3 STATES.** Added:
      * **M-E** rots the `INTFER_SYCL_LS` seam: `gate_gpu_count` reports "0
        devices, counted fine" when sycl-ls is absent instead of "unmeasurable",
        so both paths exit 2 with the SAME diagnostic and the zero-device
        direction degrades into "sycl-ls is missing" while still reporting green.
        Only the seam-discrimination assertion can see this. Patched as a
        two-line block with python3, because a sloppy `sed` would hit the wrong
        branch.
      * **M-F** rewords ONLY self-check case (c)'s synthetic gate, so the
        always-red case genuinely rejects the correct cause. Without it, a harness
        that rejected EVERYTHING would still read as a passing non-vacuity proof.
    (3) **EACH STATE NOW DECLARES THE REASON IT MUST FAIL FOR.** A non-zero exit
  is not enough: a checker can exit non-zero for an unrelated reason, which is
  exactly how the contaminated hand-run of M-C scored a false catch. A state
  caught for the wrong reason is now MISSED, and the log distinguishes the three
  kinds (not load-bearing / wrong reason / unattributed).
  **THREE BUGS FOUND BY RUNNING IT — TWO OF THEM SCORED MY OWN MATRIX WRONG**
    * **M-D's `sed` never applied.** `sed -i "s|$A_TOKENS|(--gpu)|"` used `|` as
  the delimiter on an anchor that CONTAINS `|` (ERE alternation), so sed died
  with "unknown option to `s'". This is the audit-#16 collision recurring in a new
  form: a `pat|desc` table and an `IFS='|'` parser truncating at a `|` inside a
  pattern — here a sed DELIMITER. All text patches now go through
  `patch_literal()`, a quoted python exact-literal replacement with no delimiter
  to collide with.
    * **M-E's rot was too weak, and the matrix blamed the checker.** The patch
  kept `gate_gpu_count`'s `echo "SKIP: ... not available"` line, and `probe_diag`
  takes the FIRST `SKIPPED|SKIP:` line, which that surviving echo precedes — so
  the two diagnostics stayed distinguishable and the seam assertion correctly did
  NOT fire. The state then reported "the mechanism is NOT load-bearing", a
  confident false conclusion about code that was never exercised. A seam that
  still announces itself is not rotted; both lines go.
    * **Consequence, now structural: `verify_patched()`.** If a patch does not
  change its file, the run ABORTS naming the STATE, not the checker. An
  unverified patch is worse than no state: it converts a fault in the harness into
  a verdict about the harness. A mutation that fails to REPRODUCE its defect is
  worse than none, because it launders red green.
  **THE THIRD DIRECTION, WHICH THE HARNESS ITSELF DISCLAIMED.** It printed "NOT
  asserted here: that the gates PASS on a healthy box (AGENTS.md records those
  runs)" — a proof in a transcript, the defect audit #17 closed one level up. A
  **healthy column** now runs all 6 device gates and records the observed rc in
  the stamp: committed state, not prose. It asserts rc is 0 or 2, that an rc=2 is
  EXPLAINED by a skip diagnostic, and that nothing crashes, times out, or goes
  red. It deliberately does NOT demand rc=0 — measured, 3 gates return 0 and 3
  return 2 (tp2_27b needs 2 devices; gpu_optin/device_argmax report PARTIAL when
  their TP=2 arms skip). Over-asserting here is how a correctness gate becomes a
  flaky gate, and a timeout (rc=124) is explicitly not scored as a pass. Measured
  on this box (loadavg 7-8, ~6.5 min): dpas_fold 0, http_batch 0, serve_load 0,
  gpu_optin 2, device_argmax 2, tp2_27b 2.
    Two defects in this addition, caught by running it:
    * The PASS line claimed "on this healthy box no gate went red or silently
  skipped" even when the column was SKIPPED — the same defect as `ddea91d` ("the
  OK line must not claim the tally check ran when it skipped"), committed into
  this harness by this very change. The healthy claim now follows the healthy RUN,
  and the stamp's `healthy=` line is labelled STALE when the column did not run.
    * The per-gate rc display used `${HEALTHY_RCS##* =}` with a space before the
  `=`, which can never match `name=rc`, so it printed the whole accumulator.
  The mutation matrix sets `INTFER_DEVICE_PRECOND_SKIP_HEALTHY=1` for a STATED
  reason (M-A/M-C break the guards, so on a healthy box the gate would run to
  completion and its rc would change for reasons unrelated to the mutation). A
  stated purpose, not a silent default-off; the checker runs the column by default.
  **`scripts/run_all_checks.sh` — MECHANISED THE STAMP STANDARD.** "All stamps
  equal HEAD" was a SESSION RITUAL: a human re-ran three checkers and three
  matrices serially and read four stamp files, i.e. the "enumerate, never
  remember" rule performed BY MEMORY. It now does the enumeration, the ordering,
  the locking, and the staleness assertion. The stamp list comes from
  `find -name '*last_pass.stamp'`, never a table — a hand-written list quietly
  stops covering a matrix when one is added, which is how audits #7 and #8
  overclaimed "closed". Fails closed on a dirty tracked tree, because the matrices
  restore with `git checkout --`. `INTFER_QUICK=1` runs checkers only.
  TWO MORE DEFECTS FOUND IN IT, both by running it:
    * It called `fold_report_counts.sh` rc=2 a "real regression". That script
  exits 2 when its stamp PREDATES HEAD — "Nothing is wrong with the code; the
  EVIDENCE is about an older tree." Conflating the two makes a red meaning
  "re-run the matrix" indistinguishable from a red meaning "fix the code", which
  is the doctrine violation this repo keeps auditing for. rc=2 is now UNPROVEN,
  counted separately, and only a real failure exits 1; a stale stamp exits 2.
    * The full run left all four stamps CURRENT and still exited 2. The verdict
  was computed from a snapshot the run itself invalidated: checkers run BEFORE the
  matrices, so `fold_report_counts` correctly reported rc=2 on a tree whose stamps
  were stale, the matrices then re-stamped everything, and nothing re-checked.
  Unproven steps are now RE-RUN after the matrices, and the set is DERIVED
  (whichever steps returned rc=2) rather than hard-coded as "this checker reads
  that stamp" — a future checker that reads a matrix stamp is then re-validated
  automatically, where a hand-written dependency table would quietly omit it.
  Step dispatch moved into one `dispatch()` so the first pass and the re-run
  cannot drift apart, and the re-run is reported, never silent.
  Verified end-to-end at ea955bb: 6/6 device-precond states caught for their own
  reasons, fold 7/7, gate-inventory 9/9, 4/4 stamps current, `run_all_checks`
  rc=0 with the 2b re-run visibly reporting "ok (evidence now current)".
- **AUDIT #17c: THE ENTRY POINT HAD NO MATRIX, NO SELF-CHECK, AND THE SEAM THAT
  MADE IT TESTABLE BROKE IT** (NEW scripts/mutate_run_all_checks.sh, 4/4;
  run_all_checks.sh gains a gating self-check + three seams; no engine/quant/
  schedule code touched). `run_all_checks.sh` is the ENTRY POINT for the whole
  verification stack and was the one script in it with neither a mutation matrix
  nor a self-check, so a state that made it always-red, or that broke its
  verdict, would have looked healthy.
    **It could not be matrixed naively** — it runs the full ~45 minute stack, so
  four states would cost three hours. Hence three seams
  (`INTFER_RUN_ALL_STEPS` / `INTFER_RUN_ALL_EXTRA_STEPS_DIR` /
  `INTFER_RUN_ALL_STAMP_DIR`), for the same reason `INTFER_SYCL_LS` exists: an
  assumption the harness cannot redirect is one it cannot test. Every case runs in
  seconds against synthetic steps and stamps; none runs another matrix, so there
  is no recursion.
  **TWO DESIGN POINTS WORTH KEEPING**
    * A state is judged by a **DEFECT PREDICATE**, not by the exit code, because
  three of the four breaks produce rc=0 or rc=2 — exactly what a HEALTHY run
  produces. A matrix watching only the exit code scores those as passes forever.
      * Every predicate is a **POSITIVE** assertion ("this specific evidence is
    ABSENT/BAD"), never the mere absence of a good signal. The first N-C
    predicate was `NOT grep STALE`, which a top-of-run abort trivially satisfies.
    * Every case is **BASELINED** first, and the baseline must show the correct
    marker PRESENT as well as the right rc. A baseline that checks only the rc is
    exactly what let that false catch through.
    States: **N-A** rc=2 no longer UNPROVEN (only the self-check sees it);
  **N-B** the 2b re-run deleted; **N-C** `find` replaced by a hand-written
  2-item list so a third, stale stamp goes unreported (the audits #7/#8 shape);
  **N-D** the dirty-tree guard deleted, detected behaviourally via a sentinel.
  **THE FIRST VERSION OF THIS MATRIX WAS MASKING ITS OWN DEFECTS.** It mutated
  `run_all_checks.sh` IN PLACE and then ran it — but that script fails closed on
  any dirty tracked tree, so the MUTATION ITSELF tripped its own target's guard
  and every case aborted at the top before the defect could express itself. Both
  outcomes were bad and one was worse than a miss: N-A/N-B scored MISSED when the
  states never ran, and **N-C scored CAUGHT and that was a FALSE CATCH** — its
  loose predicate was satisfied by the abort, so the stamp-enumeration defect it
  claims to test was never exercised. A mask that hides the defect under test is
  worse than no state, and a predicate loose enough to be satisfied by an early
  abort is a check that cannot fail. FIX: the mutation applies to an **UNTRACKED
  copy** at `scripts/_rac_mutant.sh` (`git diff` ignores untracked files, so the
  tree stays clean and the target's guard stays armed); the trap always removes
  the mutant, since a mutated copy left in `scripts/` would poison every later
  check that globs that directory.
  **THREE MORE DEFECTS FOUND BY RUNNING THE MATRIX AS A STEP** (not in isolation)
    * **The seam broke EVERY real step.** `dispatch()` had a `case` with no `*)`
  arm and the external-dir fallback placed AFTER the `esac`, so a MATCHING arm
  ran its command and then fell through to "unknown step" and returned 97. Every
  real step was broken by that edit — and it went unnoticed because the only
  tests in the same commit drove SYNTHETIC steps, which take the fallback path and
  work. A seam that changes control flow needs a REAL-PATH test, not only a seam
  test. Now the seam is the `*)` arm and every arm returns.
    * **The matrix polluted the stamp population it was measuring.** Its synthetic
  stamp files lived under the build tree, and the real enumeration is
  `find $BUILD_DIR -name '*last_pass.stamp'` — so the stack's real stamp set grew
  by SEVEN fixture files. The enumerator was silently measuring its own test data.
  Adding a fourth matrix is exactly the population change that exposed it. The
  synthetic world now lives outside the build tree, and a leak check FAILS the
  matrix if any fixture stamp is found under `BUILD_DIR`. This is the audit #7/#8
  lesson one level down: an enumerator needs its own population audited, because
  "enumerate, never remember" does not mean "enumerate only what you meant to".
    * **The inner runs inherited the outer run's lock.** The mutant tried to take
  the lock `run_all_checks.sh` already held, so every inner run exited 1 with
  "another run_all_checks.sh is running" and the matrix reported that as a broken
  CASE. Inner runs now get their own `INTFER_BUILD_DIR`.
  **ITEM #2 RE-DERIVED, NOT ASSUMED.** `verify_patched()` was added to
  `mutate_device_precond.sh` in #17b, so the question became whether the other two
  matrices had the same hole. They did NOT, by a different mechanism:
  `mutate_fold_matrix.sh` re-greps for the marker each `apply_mN` injects and
  every call site is `|| { echo "FAIL: MN did not apply"; exit 1; }`;
  `mutate_gate_inventory.sh`'s `patch_once` requires every hunk to match EXACTLY
  ONCE, errors on `applied == 0`, and `run_state` reports PATCH ERROR. Both were
  already sound, so the only real gap was diagnostic: the fold M1/M2 appliers
  signalled failure as a bare trailing `grep` exit code with no message, unlike
  M4/M5/M6. They now name what was not mutated. Reported rather than "fixed",
  because the improvement is cosmetic and the substance was already correct.
  Verified at d7a5ab6: `mutate_run_all_checks` 4/4 with 4/4 baselines; run AS A
  STEP of run_all_checks it reports `ok` and `gate_inventory.sh` also reports
  `ok` (proving the rc=97 fall-through is gone), the stamp population is exactly
  5 with the new `run_all_mutate/last_pass.stamp` enumerated and no fixtures.
- **AUDIT #17d: FIVE MORE ITEMS, AND THE HYGIENE GATE EXPOSED THREE MORE BUGS**
  (mutate_fold_matrix.sh uses patch_literal; NEW tree-hygiene gate in
  run_all_checks.sh; mutate_run_all_checks now 7/7 in a SANDBOX repo; no
  engine/quant/schedule code touched).
    (1) **Item 2 — fold's `sed` delimiter hazard removed.** M5's anchor CONTAINS
  `||`, so it only worked because the delimiter was escaped to `\|`: the
  audit-#17b collision class sitting live inside a state that PASSES. A small edit
  to that line could silently stop the sed matching, and the state would then
  report "not load-bearing" — a verdict about the checker when the fault is in the
  state. `sed` also replaces the FIRST match and reports nothing about how many
  there were, and **M6's anchor is a PREFIX of M4's**, so after M4 ran, M6's
  pattern still matched the M4-mutated line. Nothing caught that; states happen to
  restore in between. All five appliers now use `patch_literal()`, which requires
  EXACTLY ONE match, so an ambiguous anchor is a hard stop rather than a silent
  wrong-site patch. Fold still 7/7.
    (2) **Item 1 — the tree-hygiene invariant asserted ONCE, in the driver.**
  `run_all_checks` already refuses to START dirty; that does not prove a step left
  the tree as it found it. Each matrix verified only the files it declared as
  targets, so a step that dirtied anything else left no trace in its own report,
  and a mutated `_rac_mutant.sh` left behind would poison every later check that
  globs `scripts/`. Placed in the driver, not copied into four matrices: "nine
  hand-rolled copies of the block is precisely how the family reopened" (#7). The
  driver ran them all, so the driver checks they cleaned up.
    (3) **Items 3+4 — three more states (7/7, all 7/7 baselines correct):**
      * **N-E** disarms the classifier SELF-CHECK while `classify_rc` is ALSO
  wrong. N-A alone cannot prove the self-check can fail — its detection mechanism
  IS the self-check, so disarming it is invisible while `classify_rc` is correct.
  Compound, and the always-red lesson applied one level in.
      * **N-F** keeps the 2b re-run block and breaks the `$UNPROVEN_LABELS`
  derivation it iterates, so it finds nothing to do. Same observable as N-B,
  different mechanism: "enumerate, never remember" proven load-bearing in the
  re-run path, not merely present.
      * **N-G** deletes the new hygiene assertion, so untracked residue in
  `scripts/` goes unnoticed.
    **THE HYGIENE GATE IMMEDIATELY EARNED ITS KEEP, AND FORCED A REDESIGN.**
  On its first run it flagged the matrix's OWN untracked mutant as leftover
  residue — the second version had moved the mutant to an untracked copy at
  `scripts/_rac_mutant.sh`, which is untracked residue in precisely the directory
  the new assertion polices. Exempting that filename would have been a
  hand-maintained list, i.e. the copy-drift family again. So the mutant now lives
  in a **throwaway git repo**: its `git diff`/`git rev-parse` resolve in the
  sandbox, its stamps are built from the sandbox head, and the real repository is
  never touched at all — not the mutant, not the probe files.
  **FIVE MORE DEFECTS, all found by running it (three of them by the new gate)**
    * Patching the sandbox mutant left the sandbox tree dirty, so the mutant's own
  start-of-run guard aborted every case — the original masking, reappearing one
  level down. Each patched state is now COMMITTED in the sandbox, so the guard
  stays armed against what it is meant to catch (N-D's deliberate dirtiness).
    * `reset_mutant` used `git commit -a`, which committed the deliberately dirty
  probe as well. That silently **disarmed N-D**: the tree came back clean, the
  guard never fired, and the state scored as a broken CASE. Commits are now
  path-limited to the mutant.
    * Stamps were baked from the sandbox's INITIAL head, but every patch creates a
  commit, so the head advanced and every state after the first read as STALE —
  the family-3 stale-artifact lesson in miniature. Stamps are rewritten per run
  from the head in force at that moment.
    * N-G's residue probe was `git add -N`'d, making it intent-to-add and
  therefore STAGED dirt, so the START guard rejected it before the end-of-run
  assertion it exists to exercise. Untracked means untracked. (It also
  demonstrates the intended split: the start guard rejects TRACKED dirt, the
  end-of-run assertion catches untracked leftovers.)
    * **N-E and N-G predicates could never be true.** N-E looked for the absence
  of a SELFCHECK line, but those print DURING the loop and appear even when the
  self-check's return value is neutralised; the real defect is that the run
  PROCEEDS past it. N-G grepped `untracked residue`, which also matches the
  SUCCESS branch ("ok  no untracked residue"). Both were states that reported the
  checker as fine while being broken. N-E's predicate is now only sound because
  the CONVERSE is asserted too: N-A shows a misclassification stops the run before
  the steps, so "reached the steps" cannot pass vacuously on a healthy run.
  A mutation state whose predicate cannot be true is indistinguishable from a
  checker that is not load-bearing.
  ALSO CORRECTED: the matrix header and PASS message still described the removed
  untracked-copy design — stale prose about a mechanism that no longer exists is
  the same defect as a stamp that does not match the code.
- **AUDIT #18: THE ENTRY POINT REPORTED A REAL FAILURE AS HEALTHY.** (run_all_checks.sh
  exit-code class + residue scope + dispatch coverage; mutate_run_all_checks 7→**10/10**
  states; two diagnostic fixes; no engine/quant/schedule code touched.)
  Read #17's work back with the question "what does this claim, and can it be
  false?", rather than re-running the same checks.
    (1) **THE SERIOUS ONE — the 2b re-run classified a regression as missing
  evidence.** The re-run branched only on success/failure and left `FAILED` at 0
  either way. Measured with a synthetic step that answers rc=2 on the first pass
  and rc=1 on the re-run: it printed `still rc=1` and then the final verdict
  **"RUN ALL CHECKS: … NOT a regression: Nothing is wrong with the code"**,
  exit 2 — the script calling a checker failure healthy, in a script whose whole
  reason for existing is keeping that apart. The re-run now routes through the
  SAME `classify_rc` as the main loop and escalates `fail` → `FAILED`, printing
  `REGRESSED rc=… -- was unproven, is now a real failure`. Post-fix: exit 1,
  "FAIL (1 step(s) FAILED)". This is the 1-vs-2 doctrine violation measured on
  the doctrine's own enforcement point.
    (2) **The residue assertion was scoped to ONE DIRECTORY.** The end-of-run
  check ran `git status -- scripts/` and called it "no untracked residue in
  scripts/". `scripts/` is where the *previous* version of the matrix kept its
  mutant; the scope was never revisited after the sandbox removed that file.
  Measured: a step that littered the **repository root** left the run printing
  "ok no untracked residue" and **exiting 0 = PASS**. Now the whole work tree is
  SNAPSHOT before step 1 and DIFFed after — a diff, not a tally, so pre-existing
  untracked files (which the start-of-run message promises are fine) still do not
  fail the run. Ignored paths are excluded by git, so the build tree and the lock
  never appear.
    (3) **dispatch()/STEPS were hand lists.** A new `check_*`/`mutate_*` nobody
  remembered to add is not a step that FAILS, it is a step that NEVER RUNS — and
  this entry point would still print PASS. Section `0b` now enumerates
  `check_*.sh`/`mutate_*.sh` and requires a dispatch arm for each, extracting the
  arm list from `dispatch()` itself (so the two cannot drift) and failing LOUDLY
  if the extraction yields nothing.
    (4) **Three new states, each with the pre-fix defect verbatim as its patch,
  not a proxy for it.** N-H patches the 2b rc=1 arm back to its old shape; N-I
  patches the residue check back to `git status -- scripts/` and litters the
  sandbox root; N-J neuters the coverage enumeration and drops an undispatched
  `mutate_dummy.sh` into the sandbox's `scripts/`, COMMITTED so the residue diff
  cannot be what fails.
    **FIVE HARNESS DEFECTS THE NEW STATES EXPOSED — every one of them a state
  reporting on a broken harness rather than on the checker**
      * **A stray `rm` disarmed four states.** The N-J baseline removed the
  sandbox dummy with `rm -f`, leaving a DELETION in the sandbox working tree, so
  every later state aborted at the top on the mutant's own dirty-tree guard:
  N-A, N-B, N-E and N-F all scored MISSED. The removal is committed, and a
  derived `sandbox is clean after the baselines` assertion (`git status`, not a
  file list) now catches the class independently.
      * **A converse assertion satisfied by an abort.** The N-A converse only
  required `'1/2 checkers'` to be ABSENT — which an early abort satisfies
  trivially — so it printed `ok` while four states were failing. It now first
      requires the self-check to have actually run, pinning the absence to the
  gating decision and nothing else.
      * **The litter had to be created BY THE STEP.** The end-of-run check is a
  diff, so a file the harness created *beforehand* is in both snapshots and
  correctly reads as "not left behind" — the N-G baseline scored rc=0 and caught
  it. Littering now happens during the run, via a synthetic step.
      * **Two anchors covered less than the code they patched.** `A_RESIDUE`
  matched line 1 of a 4-line continuation and `A_ESCALATE` one line of a 3-line
  arm; the patches left dangling fragments and `verify_patched` correctly aborted
  on a syntax error. Anchors are now whole blocks.
      * **An argument accepted and never forwarded.** `baseline()` took a
  litter-site parameter and dropped it, so N-I's litter landed in `scripts/` —
  the very directory the pre-fix `git status -- scripts/` watched — which made
  N-I a duplicate of N-G rather than the root-litter case it claims to be. The
  state still scored correctly (the narrowed check missed it, so the defect WAS
  reproduced), so nothing went red: the fault was **invisible, not absent**. The
  baseline line now echoes the litter site, and baselines and states write
  different logs (`$WORK/$id.baseline.log`) because a state was overwriting its
  own baseline's evidence — the transcript-vs-tree distinction this repo keeps
  failing on, in the harness that exists to enforce it. Measured from the tree:
  N-G's baseline flags `?? scripts/_rac_residue_probe.md`, N-I's flags
  `?? _rac_root_litter.md`.
      * **The pre-flight counted LINES, the patch counts OCCURRENCES.**
  `check_anchor` used `grep -cF`, so a multi-line anchor reported "4 times" and
  refused to start, while `patch_literal` — the thing that actually has to find
  it — counts substrings. A guard and the thing it guards must agree on what
  "exactly once" means, so both now count occurrences in python. (Earlier,
  `grep -c ... || echo 0` appended a second `0` on every miss, since `grep -c`
  prints 0 *and* exits 1.) Also: `run_state` printed an `ok` line directly under
  MISSED verdicts — a harness's diagnostic is part of what it certifies.
    **VERIFICATION, and why this line carries no sha.** The certified fact is:
  `scripts/run_all_checks.sh` (full, NOT `INTFER_QUICK=1`) exits **0**, with 3
  checkers + 4 matrices green, **5/5 enumerated stamps naming HEAD**, tracked
  tree clean, and no new untracked file anywhere in the work tree;
  `mutate_run_all_checks.sh` is **11/11 caught with 11/11 baselines correct**
  *** HISTORICAL CITATION — superseded. As of 2026-10-01 the matrix has 12
  baselines: N-Kq and N-L add the CONVERSE half of the mid-run HEAD-moved
  detector (it must stay silent when HEAD holds still, and an always-firing
  detector must be rejected). So the current figures are 11/11 caught, 12/12
  baselines. Read them from `build/run_all_mutate/last_pass.stamp`
  (`states=`, `caught=`, `baselines=`), never from here — this paragraph is a
  citation and cannot track the tree. ***
  (each baseline requires the correct marker PRESENT, so no predicate can be
  satisfied by an early abort). Two earlier drafts of this paragraph quoted
  `c693f95` and then `baf6a85` — **both stale by construction**, since
  documenting a sha in prose moves HEAD the instant the prose is committed. The
  single authoritative record is the stamp set itself:
  `find "$INTFER_BUILD_DIR" -name '*last_pass.stamp'` must have every entry's
  `sha=` equal to `git rev-parse HEAD`, and `run_all_checks.sh` step 3/3 enforces
  exactly that by enumeration. Read the sha from there, not from this paragraph.
  No wall-time or throughput claim is made anywhere above: the box was loaded
  (loadavg ~9) and this audit changed no engine code.

  - **THE CONVERSE FAMILY IS ALREADY CLOSED TREE-WIDE (2026-10-01, measured —
    a suggestion to extend it was DISPROVEN, do not re-open).** The always-red
    lesson was found in `mutate_run_all_checks.sh` (N-Kq/N-L, 2026-10-01), and
    the obvious next move is to extend the same converse treatment to the other
    three matrices. Measured instead of assumed: they already have it, in the
    structures those matrices actually use (which differ — `run_state` with an
    expected-reason argument, not `def_` predicates, so grepping for `def_`
    returns zero and wrongly reads as "no coverage").
    * `check_device_precond.sh` (driven by `mutate_device_precond.sh`): the
      strongest instance in the tree — a three-case behavioural self-check at
      `:189-237` whose case (c) requires the **CORRECT cause to be ACCEPTED**,
      which is precisely what makes the always-red cases (a)/(b) meaningful.
      Exercised by mutation M-F, which rewords only that case.
    * `mutate_gate_inventory.sh`: `selfcheck()` asserts `rc_class` is not a
      constant function and that the anchor pre-flight must both REJECT a
      duplicated anchor AND still ACCEPT the real target; plus M4/M9/M7 covering
      the self-check's own failure modes.
    * `mutate_fold_matrix.sh`: `selfcheck()` seeds each mutation marker to prove
      the residue scan detects it, then asserts it does **not** false-positive on
      clean input; plus the `S0_clean` state, which is the healthy direction.
    The general form, worth reusing rather than re-deriving: a converse needs a
    case where the check must fire AND a case where it must stay quiet, and the
    quiet case is the one that gets forgotten. `grep -c '^def_'` returning 0 is
    evidence about the SEARCH, not about the coverage.

  - **AUDIT #19 (2026-10-01): A SELF-PROVING MUTATION STATE HIDES A BROKEN
    ASSERTION — and the only cure is running the NEGATIVE direction by hand.**
    Added a stamp-population leak check to `mutate_l0_cite.sh` so the matrix's
    own fixtures cannot join the `find $BUILD_DIR -name '*last_pass.stamp'`
    population that `run_all_checks.sh` step 3/3 enumerates (the AUDIT #17c leak
    class). It took **three** attempts, and every one of them PASSED, because the
    state that was supposed to prove the check could fail was itself keeping the
    population clean. The three defects, in order:
      * (a) The first version was a **print, not an assertion** — it echoed a
        LEAK line and then exited 0 on the states, so a leaking run still read
        as PASS. A comment with a shebang.
      * (b) It probed for stamps **after `rm -rf`**, so even a real leak was
        unobservable. A check that cannot fail.
      * (c) It lived **after** the mutation states, and the state deleted its own
        fixture before the assertion read the population. M-H planted a stamp,
        counted, **deleted it**, then compared against a constant — i.e. against
        its own doing rather than a property of the tree.
    (d) Separately, `pop_total` and `known_dirs` were defined **after** their
    callers, so M-H died with `pop_total: command not found` and then
    `known_dirs: unbound variable`. Under `set -u` a helper referenced before it
    exists turns into a FAILED state, which is the shell's version of the family-3
    stale binary: the mutation appears to run and proves nothing.
    (e) An inline **duplicate** of the predicate sat next to the late assertion,
    so M-H proved the COPY while the shipped check stayed unproven — the AUDIT #3
    "rule fixed at one site, re-derived at another" shape, in shell.
    FIX, all of it: `leak_assert()` is now a **function called three times against
    one predicate** (as-found, poisoned-by-M-H, after-states); it compares the
    SET of stamp DIRECTORIES against a known-matrices list rather than a COUNT
    (counting fired on a healthy tree, because `mutate_device_precond.sh` writes
    TWO stamps; and a count-only fixture would cancel against a missing stamp);
    and the fixture now STAYS planted while the assertion runs.
    MEASURED both directions: clean tree **exit 0**; fixture pre-planted in
    `$BUILD_DIR` **exit 1** ("[as found] FAIL ... LEAK").
    *** THE GENERAL LESSON, which is the whole point. *** A self-proving
    mutation state is not evidence: it validates the state, and a state that
    cleans up after itself will happily "prove" a check that cannot fail. Every
    time, the defect was found by **running the negative direction by hand** —
    planting the fixture BEFORE invoking the script. A self-consistent M-H passed
    all three times. When a mutation state cannot fail, prove the harness with a
    case the harness did not set up. Related: AUDIT #17b's `verify_patched()`
    (a mutation that does not reproduce its defect is worse than none).

  - **AUDIT #20 (2026-10-01): THE VACUOUS-GREEN CLASS RECURRED THREE TIMES, AND
    ONCE INSIDE THE CHECKER WRITTEN TO CATCH IT.** AUDIT #14 recorded the shape —
    a harvest that matches nothing reports "nothing to fix" while verifying
    nothing. It is not retired, and the proof is that it arrived again:
    * (a) `check_doc_counts.sh` printed **`WARN no 'agree at **N**' sentence
      found to check (skip, not silently pass)`** and exited **0** when its
      subject sentence was absent. MEASURED by rewording the sentence. The
      wording of the warning *was* the defect restated: it WAS silently passing.
      A missing subject is now rc=1 citing this audit.
    * (b) `check_build_dir_consumers.sh` printed
      **`PASS (0 consumer(s), all honor INTFER_BUILD_DIR)`** at exit 0 when its
      self-exclusion was widened until it skipped every script — a green verdict
      over an empty census. Found by its OWN matrix state M-E, i.e. inside the
      checker written to prevent exactly this. Now closed by a guard asserting
      the classifier inspected something.
    * (c) `check_build_dir_consumers.sh` had already shipped once with `awk
      '{print $2}'` on `add_executable(name ...` — awk does not split on `(` —
      so the CMake harvest was **0** and the checker exited 2 looking like a
      working check that simply found no consumers.
    FIX, structural rather than three separate patches: **one shared guard**,
    `scripts/_gate_enum.sh` (`enum_require_nonempty` / `enum_require_min`),
    now used by `check_doc_counts.sh`, `check_build_dir_consumers.sh` AND
    `gate_inventory.sh` — the last of which previously carried its own hand-rolled
    copy of the `sites_found == 0` rule, i.e. the AUDIT #7/#8 "rule fixed at one
    site, re-derived at another" shape. **RULES FOR THE NEXT ONE:**
      * A checker must assert it harvested something BEFORE reporting a verdict
        on it. "0 findings" and "found nothing to look at" are different claims.
      * **A `WARN ... skip` path in a correctness checker is a defect.** If the
        subject is missing the checker is broken, not the input optional.
      * Never write a numeric test that a non-numeric value can satisfy.
        `scripts/check_gate_enum.sh` asserts `""` and `"abc"` FAIL CLOSED, because
        `[ "" -gt 0 ]` is FALSE and a naive implementation would pass them.
    *** AND THE MUTATION-TESTING RESULT, which is the argument for the rest. ***
    Every checker added this session failed on its first run, and every failure
    was in the CHECKER, never in the engine — and case (b) was AUDIT #14's exact
    shape reproduced *inside* the checker written to catch it, found only because
    that checker had a mutation state designed to make it fail. A checker with no
    matrix would have shipped all three.

  - **AUDIT #21 (2026-10-01): A STUB-DRIVEN CHECKER MUST BE MUTATED BY
    BREAKING THE FIXTURE, NOT THE CHECKER.** Writing the matrix for
    `check_gpu_recover.sh` cost one wasted pass and the reason generalizes to
    every fixture-driven checker in the tree.
  My first four states made the CHECKER more permissive — drop an assertion, or
  replace a condition with `if true`. Every stub the checker drives was CORRECT,
  so the checker kept passing and **all four scored MISSED (0/4)**. The mutation
  was unobservable. That is not a weak state, it is a no-op with extra steps, and
  a matrix that reports it as "caught" would be a lie.
  **THE RULE: if the artifact under test is correct by construction, weakening
  the checker is unobservable. Mutate the ARTIFACT and require the checker to
  REJECT it.** Then the predicate is "exits non-zero AND stops printing its ok
  line", which is a real difference.
  * checker-weakening states ARE valid when the artifact is itself defective —
    that is what `mutate_gate_enum.sh` M-E and `mutate_doc_counts.sh` M-B do
    (poison the artifact, checker untouched);
  * checker-weakening states are VOID when the artifact is a correct run — so
    `mutate_e2_ctx.sh` tests only the SELF-CHECK half, whose inputs are
    synthetic. Its real-run half (prompt line, step count, id equality against
    the actual model) is **not** mutation-testable at all, and the matrix says so
    rather than pretending.
  *** A STATE THAT CANNOT BE OBSERVED MUST BE EXCLUDED, NOT SCORED. *** The
  gate-removal state in that matrix (`if ! selfcheck` -> `if false`) only becomes
  visible if the run reaches the real-run half and produces a verdict, which
  needs a GPU and the model; measured, it does not. It is excluded with the
  reason in the file, and a note says where it could be added properly.
  *** A DELETED CASE IS SILENT; AN INVERTED ONE IS NOT. *** My first M-B/M-C
  removed a self-check case outright, which changes nothing observable. Inverting
  it (`= "ACCEPT"` -> `!= "ACCEPT"`, or evaluating the case against the wrong
  input) produces a real failure. Prefer inverting.
  *** AND EXPECT THE PREDICATE YOU MEASURED, NOT THE ONE YOU EXPECTED. *** M-A
  made the accept/reject predicate always-reject; cases (a) and (b) still
  passed, because they WANT a rejection — only the CONVERSE broke. The state was
  scored MISSED against an ok-line that was never going to move. Run the
  mutated checker, read which line actually changed, and write the predicate
  from THAT. This is the same discipline as AUDIT #3c's stale-anchor rule.

- **AUDIT #22 (2026-10-02): THE REPORTED 27B WALL-TIME WAS NEVER MEASURED, AND
  NO CHECKER COULD HAVE TOLD ME.** The third retracted wall-time claim in this
  repo, and the first one where the *checker meant to prevent it* was structurally
  incapable of seeing it. Read this before quoting ANY tok/s from
  `reports/2026-09-30-pending-bench-backlog.md`.
    - **THE FINDING, stated as measured rather than inferred.** Commits `bc18e15`
      ("2 runs, agreeing to ~1%") and `26a6c4b` ("3 quiet runs") added a per-arm
      tok/s table (BASE 2.78 / 2.81 / 2.79, DPAS48 2.44, PREQ 2.50). **No artifact
      produced any of them.** The runs' own logs are
      `build/bench_E1_27b.log.counters` and `build/bench_E3_27b.log.counters`, and
      both record `TPS=SKIPPED-CONTENDED` on **every** line with `LOAD1=4.67 /
      4.41 / 4.71` (E1) and `17.04 / 4.41 / 5.50 / 18.26` (E3) — the harness
      correctly REFUSED to emit a figure and recorded why.
      `build/bench_when_quiet.log` holds **no measurement at all** (287 bytes,
      ending at the round-1 header). The claimed "loads at the start of each run:
      1.84 / 1.98 / 1.72" directly contradicts the run's own `LOAD1`.
      The only real 2.78 on the box is `build_tmp/ab27b_tab.err` (2026-09-28) — a
      DIFFERENT experiment, four days earlier, where 2.78 is the **`tab` arm** and
      BASE is 2.63 / 2.67 / 2.68. So the table is not a rounding of a real run
      either. Withdrawn; the `RESULT:` lines stay BLANK.
    - **WHAT SURVIVES, because it is load-insensitive and sourced:** DPAS48's host
      act-quant costs 2559.2 ms per 64-token decode (11.7 ms -> 2559.2 ms) and
      +30,720 launches; **PREQ recovers 52.7 ms of that and pays another +30,720**,
      so it cannot win at 27B and must not be promoted. E3's FOLD removes 331.8 ms
      of device kernel (-1.8%), replicating the 0.8B -1.6% at 27B scale. The
      "PREQ cannot win" conclusion never needed a tok/s.
    - **ROOT CAUSE 1 — THE QUIET GATE SAMPLED THE WRONG INSTANT.** It read `load1`
      at the START of a round and then measured for minutes. This box moves from
      `load1` 1.94 to 10.69 in under two minutes; a run was observed in flight at
      `load1` 10.69 whose round had passed at 1.94. **A start-of-round pass
      certified nothing about the measurement.** Fixed: every arm samples `load1`
      every second for its whole duration and is judged on the MAXIMUM.
    - **ROOT CAUSE 2 — NO ARM WROTE A LOG.** Output went to a pipe, so a figure
      could only be read off a terminal and nothing on disk could contradict it.
      Fixed: every arm tees to `build/bench_logs/<run>/<arm>.log` with a per-arm
      load trace, and a contended arm's figure is emitted TAGGED
      (`[CONTENDED -- NOT quotable]`) so it cannot be copied as a result.
    - **ROOT CAUSE 3 — THE CHECKER'S SCOPE WAS THE FENCE, NOT THE REPORT.**
      `check_result_block.sh` (b) greps for a numeric tok/s **inside the fenced
      bash block**. The table was written to the report BODY, ninety lines away,
      and the checker stayed green through the entire incident. The property
      wanted is "this report asserts no unmeasured wall-time"; the property
      implemented was "this fenced region contains no number". **Scope, not logic.**
      It also could not tell a FILLED `RESULT:` stanza from a blank one, so the
      protocol's core invariant ("a partially filled RESULT is worse than a blank
      one") was UNCHECKABLE. Both closed: stanzas are now checked where the
      numbers go, and every arm is required to route through `arm_run`.
    - **THE CHECKS ARE NOW MUTATION-PROVEN 9/9** (`mutate_result_block.sh`,
      states R-A..R-I), including **R-H, which pastes a figure into a `RESULT:`
      stanza and is the retracted defect reproduced**. `R-E` is the sharp one: it
      reverts the criterion to the FIRST load sample while LEAVING the max line in
      place, so the text check stays green and only the BEHAVIOURAL test sees it —
      the AUDIT #12 S4 shape. The behavioural test extracts `arm_run` from the
      block and RUNS it against a fixture loadavg that **drifts** (0.5 -> 7.5
      mid-arm), in three cases: quiet-then-contended must reject, quiet must
      accept, contended must reject with the figure tagged. `arm_run` reads
      `$LOADAVG` (default `/proc/loadavg`) for exactly this — the same
      redirectable-seam rationale as `INTFER_SYCL_LS`: a test that depends on the
      box being busy is a flaky gate.
    - **FOUR DEFECTS FOUND BY RUNNING THAT MATRIX, THREE IN CODE I HAD JUST
      WRITTEN.** (1) The matrix body was **DUPLICATED verbatim**, so the four
      states ran twice, `total` counted 8 while the summary listed 4 — a
      self-contradicting report, the AUDIT #12 shape — and the second `mktemp -d`
      reassigned `SB` without removing the first, **leaking a sandbox into /tmp on
      every run**. (2) `fake_measure` read `"${4:-}"`, but `arm_run` invokes its
      command as `"$@"` after shifting its own label, so `$4` is
      `fake_measure`'s fourth argument and always unset: the fixture never rose
      and the case passed **VACUOUSLY**, unable to detect the defect it was
      written for. (3) "No arm pipes `$BIN` into grep" was a same-line regex, but
      every arm wraps with a trailing backslash so the `|` is on the next line —
      it matched **ZERO lines even on the healthy block** and could not fail. (4)
      The call-site count was printed by an UNCONDITIONAL `echo "  ok  $n_arms
      arm_run call sites"` after the min-check, so a run where the minimum FAILED
      still printed an `ok` line naming the failing count — AUDIT #12's
      self-contradicting verdict line, committed a second time.
    - **A FIFTH CLASS OF MISCLASSIFIED STATE, distinct from the three the harness
      distinguishes (not load-bearing / wrong reason / unattributed): R-I was
      scored MISSED on a MARKER TYPO.** The checker did catch it (`FAIL 4 arm(s)
      invoke $BIN directly but only 3 go through arm_run`) but the expected-regex
      said `unlogged, ungated` while the message reads `unlogged and ungated`. It
      fails in the SAFE direction — a false MISSED, never a false catch — but it
      costs a full ~4 minute matrix run to find, and the ONLY reason it surfaced
      is that I read the per-state log instead of re-hypothesising the mechanism.
      **The expected-regex is a CONTRACT with the checker's message wording:
      rename a FAIL message and you can silently un-score a state. Edit both
      together or not at all.**
    - **WHY THIS ONE GOT THROUGH WHERE AUDITS #14/#20/#21 WERE CAUGHT:** those
      audits hardened checks against defects the CHECKER could detect. This defect
      lived in the gap between the check's SCOPE and the artifact, so no amount of
      tightening inside the fence would have found it. **Ask what a checker cannot
      SEE, not only whether its assertions are right — and when a number is at
      stake, require that a number be traceable to an artifact on disk.**
    - **STALE STAMPS ARE A FINDING TOO.** On arrival, **14 of 15** enumerated
      stamps cited `bc18e15` — the very commit that carried the fabricated table —
      because that session committed without re-running the stack. Per this file's
      own rule that is UNPROVEN, not wrong; re-running is the remedy, and
      `mutate_fold_matrix.sh` re-stamped 7/7 at the new HEAD.

- **AUDIT #23 (2026-10-02): AUDIT #22'S TWO ROOT CAUSES HAD REOPENED ACROSS THE
  WALL-TIME HARNESS FAMILY, AND THE FIX WAS A SINGLE HOME.** The quiet gate that
  sampled load1 once at the START of a round, and the "no arm writes a log" gap,
  were closed for the backlog block only. Every OTHER harness that can publish a
  tok/s still had the original defect, because the family was never enumerated.
    - **THE FAMILY, DERIVED FROM scripts/ AND NOT REMEMBERED.** A non-gate script
      that scrapes a timing figure out of CLI output. `check_/test_/mutate_` are
      excluded, the helper excludes itself, and **comment lines are stripped
      first**, because a prose mention of "tok/s" is not a figure — the AUDIT #21
      rule; without the strip the derivation would pull in `run_all_checks.sh`
      over a comment. That yields exactly six: `bench_gpu_ab.sh`,
      `bench_quiet_ab.sh`, `bench_quiet_l4_ab.sh`, `bench_reports_batch.sh`,
      `bench_serve.sh`, `xmx_profile.sh`. The derivation carries an anti-vacuity
      floor (`enum_require_min ... 3`, AUDIT #20): a population of zero would
      make every assertion below vacuously true and report PASS over an empty
      census.
    - **THE THREE HAND-ROLLED QUIET GATES, ALL DOING THE SAME WRONG THING.**
      `bench_quiet_ab.sh`, `bench_quiet_l4_ab.sh` and `bench_reports_batch.sh`
      each printed `WARN: box never quieted (load1=$L >= $QLOAD)` — and then
      measured and published anyway. Named by script, NOT by line: the edits
      moved the lines, and a citation pointing at a line that no longer holds the
      sentence is the stale-citation trap this file keeps recording.
      `bench_reports_batch.sh` was the worst of the three: its TIMING loop
      scraped tok/s UNCONDITIONALLY into a machine-readable
      `ARM=... TPS=... LOAD1=...` field — exactly the route the retracted 27B
      table took, a figure on a record a human then pasted into a report.
      **A warning that does not change behaviour is a comment.** All three keep
      the wait, then route every arm through the shared gate, and the warn text
      now states the consequence that actually happens (the leg's `TPS` is
      recorded `SKIPPED-CONTENDED`).
    - **THE SINGLE HOME: `scripts/_gate_loadlog.sh`.** `loadlog_arm <label> <cmd…>`
      samples load1 every `$INTFER_LOAD_SAMPLE` sec for the arm's WHOLE duration
      into `<dir>/<label>.load`, runs the command tee'd to `<dir>/<label>.log`, and
      publishes `_LL_LOG` / `_LL_MAX` / `_LL_USABLE`; usable iff the MAXIMUM over
      the arm is under `$INTFER_QUIET_LOAD` (default 2). `loadlog_tps_field`
      returns `SKIPPED-CONTENDED` for a contended arm — **the same token the
      counters-only mode has always written**, so the two modes agree by
      construction instead of by convention. Seams `$LOADAVG`,
      `$INTFER_QUIET_LOAD`, `$INTFER_LOAD_SAMPLE`, `$LOADLOG_DIR`, so a test can
      drive them (the `INTFER_SYCL_LS` rationale: an assumption the harness cannot
      redirect is one it cannot test). *"Nine hand-rolled copies of the block is
      precisely how the family reopened"* (AUDIT #7) — **and the family DID
      reopen, three times**, so the three copies were DELETED rather than
      documented as accepted drift. `bench_serve.sh` additionally skips its
      throughput FLOOR on a contended run (a stall guard that on a loaded box can
      only raise a false alarm) while its token-count and per-request-id
      assertions still run and still mean something.
    - **`bench_when_quiet.sh` NO LONGER DEPENDS ON THE OPERATOR REMEMBERING
      `nohup ... > log 2>&1 &`.** AUDIT #22 root cause 2 was operator-enforced
      here: the run's stdout only reached a file if someone remembered the
      redirect, which is why the one `bench_when_quiet` log on the box held no
      measurement. It now captures ITSELF to a per-run artifact named
      `<utc>_HEAD_<pid>.log`, with a header carrying HEAD, the threshold and the
      starting load1, and **refuses to report a run whose artifact is missing or
      empty** (AUDIT #20: "found nothing to look at" and "found nothing wrong" are
      different claims — without it the tee could silently no-op and the run would
      look measured exactly as it did on arrival).
      **TWO DEFECTS FOUND BY TESTING, NOT READING.** (a) `exec > >(tee)` was the
      fragile way to capture; it is now `| tee -a` with `PIPESTATUS[0]`, which
      preserves the BLOCK's exit code instead of tee's. (b) a second-resolution
      timestamp let two runs in the same second produce the SAME path, so the
      second APPENDED to the first and the artifact could no longer say which run
      produced which line — an ambiguous log is the artifact problem all over
      again. Hence HEAD and PID in the name.
    - **CHECKER + MATRIX, and the population is derived so a new harness is
      covered automatically.** `scripts/check_bench_load_gate.sh` asserts four
      things per derived harness: it routes through the one helper; its figure
      comes from `$(loadlog_tps_field …)`; it does not REDEFINE
      `loadlog_arm`/`loadlog_tps_field` (a redefinition would leave the first
      property saying "yes" while the criterion has a second home — the
      copy-drift family caught at the DEFINITION, AUDITs #3/#12); and the
      warn-and-continue line is gone. Plus one **BEHAVIOURAL** property: the
      helper is RUN against a fixture loadavg that drifts 0.5 → 7.5 mid-arm and
      must WITHHOLD, and against a quiet fixture and must PUBLISH — both
      directions required, since a gate that can only say USABLE is the bug and one
      that can only say CONTENDED is useless (always-red converse).
      `scripts/mutate_bench_load_gate.sh` is **5/5** (states B-A..B-E; read live
      figures from `build/bench_load_gate_mutate/last_pass.stamp`, never from
      here). The model is AUDIT #21 applied to the ARTIFACT: the harnesses are
      CORRECT after this change, so weakening the CHECKER is unobservable — each
      state damages a harness in a sandbox COPY of `scripts/` (never a symlink,
      which would edit the real tree) and requires the checker to reject it.
      **On its FIRST run the checker found `xmx_profile.sh` ungated**, which is
      the argument for deriving the population instead of naming it.
      `B-E` is the sharp state — the helper stops withholding, so every TEXT
      property still passes and only the behavioural probe can see it (the
      AUDIT #12 S4 shape, with the always-red converse already in place).
    - **THE "NO BARE THROUGHPUT WRITE" CHECK WAS A FALSE-POSITIVE MACHINE, RECORDED
      SO NOBODY REINSTATES IT.** The first draft tried to detect a bare throughput
      write line-by-line and flagged every LEGITIMATE shape: the scrape itself
      (`tps="$(grep -aoE tok/s …)"`), a printf whose format says `TPS=%s` while the
      VALUE arrives from `loadlog_tps_field` on the NEXT line, and an
      already-gated `tok/s=$TPS_FIELD`. Replaced by the robust composition: **a
      harness can only obtain its figure from the function that withholds, and the
      behavioural property proves that function withholds.** A check that cries
      wolf is the mirror defect.
    - **TWO DEFECTS IN THE NEW MATRIX ITSELF, both found by running it.** (a)
      state B-A's anchor was INDENTED and matched 0 times, so the state scored
      MISSED with `mutation fn failed` — a message that reads as "the harness
      cannot be damaged" when it means "the anchor is wrong". Same lesson as the
      AUDIT #17 anchor parser: confirm the anchor form mechanically, never by eye.
      (b) `summary.txt` was never truncated between runs, so it ACCUMULATED and the
      printed summary listed `B-A MISSED (fn)` from the PREVIOUS run directly above
      `B-A caught`, under a `PASS (5/5)` line — the self-contradicting-report
      class of AUDIT #22, and the worst place for it: the in-memory counters make
      the VERDICT correct, so only the EVIDENCE was wrong.
    - **AND THE STACK CAUGHT A REGRESSION OF MINE, WHICH IS THE POINT OF IT.**
      The first full run after this change reported `RUN ALL CHECKS: FAIL (2
      step(s) FAILED)` — `check_build_dir_consumers.sh` rc=1, with
      `mutate_build_dir_consumers.sh` failing its baseline behind it — and
      correctly classified it as a real regression, not unproven evidence. Cause:
      `_gate_loadlog.sh` consumes NO build artifact (the caller picks the
      directory), but its usage comment showed `loadlog_init
      build/bench_logs/run1`, and the consumer classifier matches `build/`
      WITHOUT stripping comments, so a prose example was graded as a consumer.
      All six real callers write `${INTFER_BUILD_DIR:-$ROOT/build}/bench_logs/…`
      and all six passed, which is what localised it to the comment. Fixed by
      making the example a variable and stating the real contract: the CALLER owns
      the path and must honour `INTFER_BUILD_DIR`.
      The tempting fix — teach the classifier to strip comments — was **DECLINED**:
      that checker and its matrix are green, and this is a documentation bug, not a
      classification bug. Note the two checks now disagree on principle:
      `check_bench_load_gate.sh` DOES strip comments before matching, because its
      population derivation would otherwise demand a load gate from
      `run_all_checks.sh` over a prose mention. Both choices are recorded at their
      call sites.
    - **SCOPED, NOT DONE: `scripts/bench_chat.sh`.** It measures HTTP TTFT / TPOT
      / E2E in seconds off the SSE stream rather than scraping a CLI tok/s, so it
      falls outside the derived population (its only `tok/s` is prose in comments,
      which the comment strip removes) and was NOT wired this round. Precise
      wording: **out of scope of the derivation, not exempt by judgement** — it is
      a wall-time measurement harness and wiring it is a mechanical follow-up.

- **SIXTH RECURRENCE OF THE SAME CLASS: I WROTE THE PREDICATE FOR THE OUTPUT I
    EXPECTED, NOT THE ONE I OBSERVED.** Two more instances this session, both
    found by a state reporting MISSED while its defect had reproduced perfectly:
      * `mutate_checker_matrix_coverage.sh` M-A asserted "the table must not print
        MISSING". It still does — the table row is independent of the
        requirement; only the `bad` increment and the exit code disappear. The
        observable is the VERDICT line, so the predicate now judges on that.
      * `mutate_run_all_checks.sh` N-M asserted the run must contain "No such
        file or directory". That string is printed by the STEP's own log, not by
        the stack. N-M's actual evidence was `step_current FAIL rc=97` and
        `RUN ALL CHECKS: FAIL (1 step(s) FAILED)`, and it reproduced on the first
        run — the predicate was simply looking in the wrong file.
    **RULE: when a state reports MISSED, read the state log and write the
    predicate from what is ACTUALLY THERE, before touching the mechanism.**
    A MISSED state with a reproduced defect is nearly always a predicate aimed
    at the wrong evidence, not a state that fails to mutate. Two of the last
    three "unobservable mutation" conclusions were this mistake.
    N-M itself closes the class the full stack exposed on 2026-10-01: a `STEPS`
    edit adding a matrix name ALSO matched inside a `dispatch()` arm, so the arm
    ran as `bash $SCRIPT_DIR/mutate_gate_pop.sh mutate_checker_matrix_coverage.sh`
    (rc=127, stale stamp). Every DIRECT invocation of that matrix passed; only the
    end-to-end stack failed. N-M hands a dispatch arm a stray argument and
    requires the run to go loud with that specific failure — note it leaves the
    arm's LOGIC intact, so 0b's coverage enumeration and step 3/3 must NOT notice,
    because the arm still exists and is dispatched; only its EXECUTION fails.
    Read live figures from `build/run_all_mutate/last_pass.stamp`.

- **AN INVERTED PREDICATE SCORES CORRECT CODE AS BROKEN — NAMED RULE, 3 FOR 3.**
    A mutation state that asserts "this marker is ABSENT" while the marker is
    the checker reporting the defect correctly is a state that can never pass.
  MEASURED THREE TIMES IN TWO SESSIONS, always in *working* code:
    1. `verify_patched()` failed when the marker WAS present — i.e. "the mutation
       did not apply" for every mutation that applied. Five of six states scored
       MISSED for an unrelated reason, and because they returned early they
       skipped their restore, so a mutated helper leaked into the next run.
    2. `residue_scan` returns **0 when clean**, and the call site branched on
       success. Inverted, same commit.
    3. State M-I asserted the guard's own FAIL line while the scorer checked
       absence — but for that state the guard firing **IS** the success.
  **RULE: a bare `grep -q "$expect"` is not an assertion, it is a convention
  nobody can read.** Use `scripts/_gate_pred.sh`'s `pred_expect_absent` /
  `pred_expect_present`, which NAME the polarity at the call site and print it
  in the verdict (`verdict-polarity=expect-absent`). A state written for the
  wrong polarity then fails LOUDLY with the polarity named instead of silently
  reporting MISSED. Converted: `mutate_gate_enum.sh`,
  `mutate_build_dir_consumers.sh`, `mutate_doc_counts.sh`, each re-run and
  re-stamped individually. **Recognise this by the symptom, not by re-deriving
  it:** "correct checker + MISSED state" is almost always an inverted
  predicate, and the fix is to check which side the marker is on before
  touching the checker.

- **`patch_literal` IS STILL DUPLICATED IN THREE MATRICES — DELIBERATE, WITH A
    TRIGGER.** `scripts/_gate_patch.sh` (added AUDIT #20) owns `patch_literal`,
    `verify_patched` and `residue_scan` for NEW matrices. `mutate_fold_matrix.sh`,
    `mutate_device_precond.sh` and `mutate_run_all_checks.sh` still carry their own
    copies. **This is a known, accepted duplication, not an oversight**, and the
    reason is worth stating so the next session does not "fix" it casually:
    converting them means editing the source of three PASSING, mutation-proven
    matrices, and a refactor that breaks a matrix is strictly worse than the
    drift it removes — the drift is bounded (one function, ~10 lines, three
    copies) and every copy is exercised whenever its matrix runs.
    **CONVERT WHEN ANY OF THESE IS TRUE, one matrix at a time, each with its own
    re-run and stamp:**
      * a copy is edited and the edit is made to only one of them (then the
        copies have genuinely diverged and the shared helper is now a *bug*
        source, not a dedup opportunity);
      * a fifth matrix is added (the count is what makes the hand-maintenance
        cost real);
      * one of the three is next being modified anyway, so the review cost is
        already being paid.
    `mutate_gate_inventory.sh` uses its own `patch_once`, which is a DIFFERENT
    function (per-hunk, not whole-block) and is not a candidate for this helper.
    Do NOT convert it "for consistency".
    *** `patch_literal` TAKES EXACTLY THREE ARGS — and now ASSERTS IT. ***
    `file`, `exact-old`, `exact-new`. This was violated and cost a long hunt:
    a matrix author passed FIVE arguments — one single-quoted string per LINE of
    a two-line case — expecting them to concatenate. **They do not: each quoted
    string is a separate ARGUMENT.** The helper therefore replaced line 1 with
    line 2, wrote no mutation marker, and **returned success**, so five states
    reported "PATCH VERIFY FAIL: NOT present" with no error anywhere. Found only
    with `bash -x`, comparing the argv the function actually built against a
    hand-verified invocation.
    **To patch a MULTI-LINE block, put the newline INSIDE ONE single-quoted
    argument** — a single-quoted bash string may span lines:
        patch_literal F 'line1
      line2' ': # MARKER'
        patch_literal F 'line1
      line2' 'repl1
      repl2'           && verify_patched F 'MARKER'
    `patch_literal` now returns 2 with that message on a wrong argument count, so
    the silent mis-patch cannot recur. **When a matrix state reports "PATCH
    VERIFY FAIL", run `bash -x` and read the argv — do not re-hypothesise.**
    *** THE TRIGGERS ARE CHECKED, NOT ASSUMED. *** As of 2026-10-01 none has
    fired: the three are untouched by recent work (`mutate_fold_matrix.sh` last
    changed at `b49e164`, `mutate_device_precond.sh` at `8d6ac55`,
    `mutate_run_all_checks.sh` at `1adf06f`), and no copy has been edited
    without the others, so there is no divergence to repair. To re-check before
    assuming the decision still holds:
        git log -1 --format=%h -- scripts/mutate_fold_matrix.sh   # etc.
    If ANY of those shas has moved recently, trigger 2 has fired and the
    conversion is due.

- **A GUARD SOURCED INSIDE A CONDITIONAL IS A GUARD THAT IS USUALLY ABSENT.**
    While routing `gate_inventory.sh` through `_gate_enum.sh`, the `.` landed
    inside its `if ! command -v sycl-ls` block, so the helper existed **only when
    sycl-ls was missing from PATH** — and the call site 400 lines below failed
    with `enum_require_nonempty: command not found` on the NORMAL path, turning
    a passing suite red. Sourced helpers go at top level, after any env
    bootstrap, and never inside a conditional. (The repo already records a
    related one: AUDIT #3c, a helper hoisted outside `#ifdef` because its caller
    was not.)

  - **A HELPER THAT RECORDS ITS VERDICT IN A GLOBAL SILENTLY LOSES IT IN
    `$(...)`.** `_gate_enum.sh` sets `_enum_bad`; the self-check that drives it
    originally called it inside command substitution, so the assignment died in
    the subshell and the parent read `0` — the helper's own stdout said FAIL
    while the harness recorded it as accepted. Four cases scored FAIL for a
    reason that had nothing to do with the helper. Capture to a file so the call
    runs in the current shell.

  - **SINGLE WRITER: do not commit to this clone while `run_all_checks.sh` is
    running.** MEASURED, not hypothetical — a second agent session committed
    three times during one ~45 min stack here. HEAD then moved between the
    start-of-run snapshot and the stamp assertion, so stamps naming the NEW
    commit were compared against the OLD `$HEAD_SHA` and reported STALE: a
    verdict that looks like missing evidence when the evidence is current, and
    whose printed remedy ("re-run the named matrix") cannot possibly help.
    Three ~45 minute re-runs were burned on this. The stamp standard is only
    meaningful if the tree holds still while it is measured.
    The failure is now LOUD rather than silent: step **2c** re-reads HEAD
    immediately before the comparison and reports a move as its own condition
    (counted FAILED, exit 1 — deliberately NOT folded into STALE and NOT
    laundered into the rc=2 "unproven" tally, because a race is neither stale
    evidence nor a regression, and the two call for opposite actions). The
    comparison uses `$HEAD_NOW`, so a moved HEAD is not additionally misreported
    as 5 phantom STALE stamps. Mutated by state **N-K** in
    `scripts/mutate_run_all_checks.sh` (11/11 caught, 12 baselines — the extra two
   are the N-K converse; read the live figures from its stamp; a synthetic
    `step_commit` makes the move reproducible without disarming the mutant's
    dirty-tree guard). If you see "HEAD MOVED under this run", finish the other
    commit and re-run — do not debug the stamps.
    Related contention on this box, not in this clone: a Planar.jl container
    holds a `julia` process near 100% CPU, which is why wall-time is unobtainable
    (quiet rule `load1 < 2`).





   THE FAILURE IS BOX STATE, NOT A REGRESSION — established, not assumed:
     GATE FAIL: hook fired on TP2 (40 lines)   [PARTS TP2 spec flag: 40, expect 0]
   The gate's TP2 arm asserts the argmax hook stays OFF under --tp 2, because
   the hook is TP1-only. The hook's guard is `is_distributed()` =
   `num_devices_ > 1` (main.cpp:989). With ONE GPU the engine correctly clamps
   --tp 2 down to a single device, so it is correctly NOT distributed, so the
   hook correctly fires, so the gate's expectation of 0 lines is wrong for a
   1-GPU box. Verified NOT MINE: `git log 7efed06..HEAD -- scripts/
   test_device_argmax.sh examples/intfer_cli/main.cpp` is EMPTY, and
   `is_distributed()` last changed in 22431d5, long before this session.
   DO NOT "fix" this by relaxing the gate to expect 40 lines — that would break
   it the moment a second GPU enumerates. It is the same class as AUDIT #9/#11
   (a gate whose expectation encodes a 2-GPU box configuration). Re-run once
   both GPUs enumerate. Recorded here so the next session does not read 9/10 as
   a code defect or "fix" the gate into a permanent no-op.
   serve_load numbers in that run (c1=8.04 -> c8=5.07 tok/s aggregate, x0.63)
   are CONTENDED-BOX throughput, not a measurement: the box was not quiet and
   no round was interleaved. Recorded only to show the gate ran; no perf claim.

   === EXPERIMENT SHIPPED: INTFER_DPAS48_FOLD=1 (default OFF) ===
   The "rewritten 2D-tiled DPAS kernel" was blocked on one specific thing, and
   that thing is now removed. quant_gemm_dpas48 runs one WI per (row, 16-N
   tile) and, INSIDE its K loop, recomputed the activation column sums
   `S = sum_{k<32} va[k]` (and S0/Sf for the INT2/G16 split). Those depend only
   on (row, op), never on the tile — so all nT = N/16 tiles per row redid the
   same O(K) integer fold. That replication is the documented reason the wide-N
   DPAS kernel loses to the legacy tiled path despite the multiply being on XMX.
   New plain-SYCL kernel `quant_gemm_dpas48_rowfold` computes them once per
   (row, op) into a grow-only per-device workspace (2*M*nOp INT32); the DPAS
   kernel reads them. Dep-chains on the pre-pass, adds no host wait.
   WHY IT IS SAFE, precisely: the hoisted values are EXACT int32 sums of int8
   codes (|sum| <= 4096). Integer addition is exact and associative, so hoisting
   cannot change a bit — no rounding-order exposure, unlike the FP32 folds
   elsewhere in that file. DPAS math, scales and accumulation order untouched.
   A WORK hoist, not a numerics change. Do not "simplify" it into a device sum
   in another order or a wider type; the exactness IS the safety argument.
   ROW-STRIDE TRAP (cost a real bug, caught only by the new probe): each row
   owns 2*nOp ints, `[S0][Sfull]`, so the stride must be >= 2*nOp, NOT >= nOp.
   With a shorter stride, row m's Sfull block OVERWRITES row m+1's S0 block.
   This is SILENT AT M=1 (only row 0 exists) and only fires when the grow-only
   workspace stride is below the call's 2*nOp — i.e. a smaller-K GEMM after a
   larger one. Guarded in the kernel (`fold_stride < 2*nOp` -> decline) and again
   in the engine. The probe failed on `M=8 big-K` with S0 always right and Sfull
   garbage, which is exactly this signature.
   New gate: `examples/dpas_fold_probe` (device, no model) compares every
   (row,op) S0/Sfull against a host reference over M=1/3/4/8, tight and
   row-strided workspaces and big-K, plus 4 negative cases that must DECLINE and
   write nothing (decline detected via a sentinel prerequisite event, because
   comparing against a bare `sycl::event{}` is not reliable). PASS: 3907 pairs,
   0 violations. MUTATION-PROVEN non-vacuous twice: (1) weakening the stride
   guard back to `< nOp` -> rc=1; (2) corrupting `dst[nOp+o] = sf` to
   `dst[o] = sf` -> rc=1 with the value comparison firing on M=1. Both reverted.
   It is wrapped as `scripts/test_dpas_fold.sh` — the TENTH correctness gate,
   and the last build-artifact consumer that had no build guard. It calls
   `gate_require_build dpas_fold_probe` (own log
   `gate_test_dpas_fold_build.log`, derived automatically by the shared helper
   from BASH_SOURCE[1], so a new gate needs no extra wiring) and honors
   INTFER_BUILD_DIR. WRAPPER MUTATION-PROVEN: with static_assert(false) in
   src/gemm/quant_gemm.cpp the stale probe binary was confirmed STILL PRINTING
   PASS (rc=0) — the trap genuinely armed — and the gate returned rc=1. Reverted,
   green. This mattered concretely: a failed build twice left a stale probe
   reporting a misleading result earlier in this same session. FAMILY 3 IS NOW
   FULLY CLOSED: 10 guarded correctness gates, 8 measurement harnesses
   deliberately unguarded by decision, 10 model-prep/microbench scripts
   honoring their tree's override. Nothing is both unguarded and unintentional.
   `scripts/test_gpu_optin.sh` gained 4 arms, PASS at TP=1 AND TP=2:
   DPAS48+FOLD == DPAS48 (n=24), DPAS48+PREQ+FOLD == DPAS48 (n=24), WIDE
   DPAS48+FOLD == DPAS48 (~97-token prompt => one ~97-row batch, the arm that
   actually exercises the M>1 row stride), and FOLD-alone == base (inert
   without DPAS48, same nested-flag shape as PREQ).
   Measured, load-insensitive (0.8B TP=1, device-side total_kernel_us, NOT wall
   time): decode n=8 88028.5 -> 86590.3us (-1.6%) with launches 2232 -> 2736
   (+504); prefill chunk128 43533.5 -> 42795.0us (-1.7%), launches 1116 -> 1368
   (+252). Read honestly: it costs ONE EXTRA LAUNCH PER DPAS GEMM and 0.8B is
   the weak case (small N => few tiles to amortize against). 27B wide-N is where
   it should pay and CANNOT be measured here (1 GPU; 27B needs --tp 2). No
   tok/s claim. Handover report with blank RESULT lines:
   `reports/2026-09-30-dpas48-fold.md` (E3), cross-referenced from
   `reports/2026-09-30-pending-bench-backlog.md`, which now also states plainly
   that the queue is blocked on 2-GPU box state, not on missing experiments.
   STILL NOT DONE, deliberately: the remaining 2D-tiled geometry work proper
   (subgroup DPAS with a genuinely cooperative 2D block, so B operands are
   loaded once per block instead of per WI). This change removes the ARITHMETIC
   replication, which was the blocker removable without a new numerics contract;
   the load-replication half is untouched. Registered as **E4** in
   reports/2026-09-30-pending-bench-backlog.md with its design and its real
   blocker: a cooperative block changes the FP32 accumulation ORDER, so unlike
   E3 it is NOT a bit-exact hoist and needs a tolerance-based numerics contract
   agreed BEFORE any code is written. Do not start it on the assumption it will
   be exact. Also note INTFER_DPAS_ROW is the cautionary precedent: re-partitioning
   one-WI-per-row without real cooperation measured ~460x SLOWER.

   === AUDIT #10: AUDIT #8's OWN ENUMERATION WAS THE SAME NARROW PATTERN ===
   7efed06 fixed 5 harnesses and then claimed "All 17 artifact consumers now
   honor the override". That claim was FALSE, and the failure is precisely the
   one the process lesson above warns about: the enumeration was done by
   grepping for three ARTIFACT NAMES the previous fix already knew about
   (intfer_cli / intfer_gguf / ga64_recip_test), so it could only ever
   rediscover the instances already on the list. A family is closed by
   enumerating from the SOURCE OF TRUTH, not from remembered names.
   Redone from the CMake target list (all 19 `add_executable` targets across
   examples/*, plus the separate build_quant_real tree), 10 more instances
   existed in 3 build trees, all silently ignoring their documented override:
     * MAIN tree, MEASUREMENT harness — tab_micro_ab.sh (gemm_bench): the
       INT2 down-proj interleaved A/B, hardcoded "$ROOT/build/...". Same bug,
       same class, same consequence ("measured the wrong binary") as the 5
       fixed in 7efed06; it was simply not in the artifact-name grep.
     * MAIN tree, model prep — merge_27b.sh, merge_27b_all.sh,
       merge_27b_batched.sh, merge_27b_final.sh, merge_batched.sh
       (quant_merge): hardcoded "./build/..." RELATIVE, so they only ever
       honored an override of zero. Root-relative now.
     * build_quant_real tree — quant_27b_chunks_redo.sh, quant_27b_re.sh,
       run_27b_chunks.sh, run_27b_int2_chunks.sh (quant_model_real):
       hardcoded "./build_quant_real/...", ignoring the INTFER_QUANT_REAL_BIN
       override that scripts/build_quant_real.sh documents and defines.
   All 10 now use the SCRIPT_DIR/ROOT idiom and their tree's override.
   Verified WITHOUT running them (they launch 27B quantization jobs): a probe
   evals only the prefix up to the assignment line, run from a real file in
   scripts/ so ${BASH_SOURCE[0]} is populated — default resolves byte-identical
   to the old hardcoded path (no behavior change), override resolves into the
   override tree. 10/10 both directions. `bash -n` clean on all 10.
   GOTCHA WORTH KEEPING: an eval-from-stdin harness SILENTLY misreports this
   class of fix — with bash reading from stdin ${BASH_SOURCE[0]} is empty, so
   ROOT degrades to `/var/home/fra/dev` (or empty) and a correct script looks
   broken. Probe with a real file in place, or you will "fix" a non-bug.
   Second gotcha, same session: the bulk `edit` tool DROPPED a trailing `"`
   from new_string in 5 files at once (`MERGE="${...quant_merge`), which
   `bash -n` caught as a syntax error — git showed HEAD clean, so the edits
   were the cause, not a pre-existing break. Corollary: `bash -n` on every
   edited script is not optional, and a script that was clean at HEAD and
   dirty after an edit means the edit, not the repo.
   Box state re-checked independently while here: the box really is in a
   1-GPU state (both GPUs xe-bound in PCI at 04:00.0 / 86:00.0, class 030000;
   ALL FOUR DRM nodes present as char devices, major 226 — card0/card1,
   renderD128/129; xpu-smi and sycl-ls enumerate ONE adapter). So #9's
   CONCLUSION (box state, not a code regression) stands. Its MECHANISM does
   NOT — see AUDIT #11: the min(tp,devices) clamp is correct, so this is not
   sharding into a missing device. Do not read the two statements below as
   confirming #9's wild-pointer story; they only describe how to measure the
   box. Also worth knowing
   on this box: `lspci` is NOT INSTALLED (a `lspci | grep -c` returning 0 is a
   missing tool, not evidence about hardware) and `ls -la /dev/dri` is WRONG
   here — an `ls` wrapper (`rtk ls`) hides entries and reported only
   `by-path/` while all four nodes existed; use `find /dev/dri -maxdepth 1`
   or `stat` when the hardware state matters.
   Re-verified after the edits (correctness only, no wall-time claim):
   ga64-recip PASS (480645 slices / 30,753,153 elements bit-identical, discrim
   103/192000/11/0 — IDENTICAL to the recorded baseline, confirming no source
   drift) and cpu-parity PASS. No gate or engine source was touched.

   === AUDIT #3b: the DPAS48 x SSM_GPU gate was asserting TIE-LUCK ===
  Forcing the single reciprocal site changed the host oracle's own codes
  (5 differing bytes over 4.1M codes on a fixed dataset) and the GPU gate
  then FAILED on `DPAS48+SSM_GPU == DPAS48`, diverging at decode step 15
  (TP=1 and TP=2). Note what did NOT fail: `DPAS48+PREQ == DPAS48` and
  `DPAS48+PREQ+SSM_GPU == DPAS48+SSM_GPU` still held exactly — the pre-pass
  contract survived, so the only differing component between those two arms
  is WHERE THE SSM RECURRENCE RUNS.
  Root cause (measured, not inferred): gemm::ssm_forward is NOT bit-exact with
  the host loop, and never was — the "bit-identical math, same element order"
  comment at the INTFER_SSM_GPU gate site was wrong. (a) sycl::exp differs from
  std::expf on 694714/1048576 = 66.25% of FP32 inputs over [-8,0], so
  g_t=exp(g) and beta=sigmoid(b) differ and the state S diverges from step 1;
  (b) the kv/readout dot products accumulate in a register on device (FMA
  contraction) vs into memory on the host. The resulting ~1ulp drift is
  invisible on the FP16 GEMM path (FP16 resolves ~1e-3), which is why
  base-stream identity held and SSM_GPU was certified. INT8/GA64 act quant
  resolves ~1/127 per slice, so the same drift flips a code at a tie and the
  greedy stream diverges. The pre-single-site build "passed" that assertion
  only because its accidental reciprocal bits happened to put every tie on
  the same side — the assertion was luck, not a property.
  FIX: SSM_GPU now DECLINES when INTFER_DPAS48=1 (single gate helper
  ssm_gpu_allowed() at main.cpp; all 5 env read sites route through it, with a
  one-time stderr notice). Same idiom as L4/L5: a fast path that cannot hold
  the bit-contract falls back to the host. Every existing gate assertion is
  therefore unchanged and still meaningful, and no flag combination silently
  yields a different token stream. Both flags are default-off experiments, so
  nothing production changes. The REAL fix (port the host exp polynomial to
  the device, or kill contraction there, then re-certify SSM_GPU x DPAS48) is
  NOT attempted — scoped as future work.
  CONSEQUENCE FOR RECORDED STREAMS: the host GA64 act-quant codes changed
  slightly (ties only), so DPAS48-family streams differ from those recorded
  at 7f005fe. Base/FP16 streams are unaffected (gate: base n=24 unchanged).
  Do not treat pre-audit DPAS48 token lists as golden.

   === AUDIT #3c: a build break the gates could NOT see (987c554 -> 1fbf9a6) ===
  ssm_gpu_allowed() was defined INSIDE #ifdef INTFER_HAS_GPU_BACKEND while
  its step() call site (main.cpp:4985) sits OUTSIDE it, so the CLI failed to
  compile without -DINTFER_HAS_GPU_BACKEND=1. All four gates still passed,
  because the define is unconditional in examples/intfer_cli/CMakeLists.txt.
  Lesson (same species as the two vacuous-gate bugs already recorded here):
  a gate that cannot exercise a configuration cannot protect it. The check
  that caught it was a targeted `-fsyntax-only` recompile of the single TU
  with the macro stripped — cheap, and the only thing that saw the defect.
  Note for whoever tries to enable a real CPU-only build: it was ALREADY
  unsupported before this work (the same check on 7f005fe fails with 20
  errors — GpuBackend / SsmRow used outside guards). Helpers added near the
  GPU backend must be hoisted OUT of the guard unless every caller is inside
  it; check with the macro stripped, do not assume the gates cover it.

   === OTHER FIXES ===
  (a) PARTIAL-DECLINE RACE (nondeterminism): if device d's pre-pass
  launched but device d+1 declined, the already-queued d kernel still
  writes ws48_[d].act while the host fallback memcpy's the same buffer.
  Now every submitted preq_event is drained before falling through.
  (b) `if (xd != x.data())` compared a USM pointer against a host
  std::vector pointer — never legitimately equal, always copied. Now an
  unconditional memcpy with a comment.
  (c) profile_note_launch() ran BEFORE the decline check, so a declined
  call inflated nk. Moved after.
  (d) Indentation regression in quant_gemm_dpas4_row fixed.

   === FIXED (was "KNOWN, UNRESOLVED"): PREQ x SSM_GPU tie divergence ===
  Real and deterministic (step 14, TP=1 and TP=2, ~2 mismatched elements per
  ~1950 GEMM calls), but NOT a tie-rule bug: it was the reciprocal (AUDIT #2
  above) — the device was the correctly rounded one and the HOST differed
  from it. The earlier "host gives 48 where trunc(v+0.5) gives 47" evidence was
  an artifact of a reference using `_mm_cvtss_si32` (CVTSS2SI = round to
  NEAREST) instead of `_mm_cvttss_si32` (CVTTSS2SI = truncate); dashq
  correctly uses `_mm256_cvttps_epi32`. Writing a host reference for this
  rule with the rounding intrinsic makes the real, correct device path look
  wrong on ~53% of elements. PREQ now composes with SSM_GPU and the gate
  ASSERTS it. Correctness-gated: gpu-optin PASS (base arms + PREQ==DPAS48 +
  ROW+PREQ + PREQ+SSM_GPU + SSM_GPU + PREQ-alone inert, n=24, TP1/TP2), cpu
  parity PASS, http batch PASS, plus the new ga64-recip numeric gate (AUDIT #3
  above). Path fires: decode nk 1116->1488
  (+372 = 4tok x 93), nw unchanged 744.
  NO wall-time claim (never benchmarked here) — needs a quiet-box 27B
  INT4/G64 A/B before any perf claim.
  Orthogonally: device-act-quant INTO the fused DPAS kernel stays a dead
  end (replication), but the standalone pre-pass is a different shape and
  is now open. SYCL-ENVS GOTCHA: a plain non-sourced shell cannot
  enumerate ANY GPU ("No devices matching filter 'Arc'") because
  libumf.so.1 (at /opt/intel/oneapi/umf/1.1/lib) is off LD_LIBRARY_PATH,
  so every UR adapter fails to load. Export
  LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:... before any --gpu run;
  verify with sycl-ls. xpu-smi still shows the GPUs fine — do not debug
  device_manager.cpp for this warning. (memory: sycl_env_libumf.md)
  DPAS48 re-exam at M=361 (interleaved A/B, contended box, 2/2 rounds):
  OFF wins prefill 147.8/146.3 vs 132.7/131.0 (~11%) AND decode
  6.70/6.87 vs 5.13/5.06 (~30%); kernel 318->496ms (one-WI-per-16N-tile
  replicates O(K) fold work nT x), rebuild 500->964ms (DPAS repack),
  copy 12.6->48.4ms (host act-quant). Streams bit-identical. Default-OFF
  stands at ALL shapes (M=1/128/361).
  DPAS48 M=1 device-side breakdown (2026-09-19, -n8 TP1 decode buckets —
  kernel_us is device event time, LOAD-IMMUNE, box load ~7): legacy
  kernel 138.0 / submit 136.4 / waitblock 154.9 / copy 0.1 / readback 65.6
  nk=1680 vs DPAS48 kernel 203.5 (+48% DEVICE-SIDE — the M=1 DPAS kernel
  itself is slower, not just its host quant) / submit 235.6 (+73% at LOWER
  nk=1488: UR submit blocks when the device is saturated by the longer
  kernels) / copy 91.8 (host GA64) / readback 100.2 (nr 1056 vs 576, ~2x
  readbacks). DPAS48 loses on EVERY bucket at M=1. Consequence: a
  standalone device-side GA64 act-quant kernel (row-parallel pre-pass —
  feasible, no replication, HalfAway rounding verifiable like the RNE
  convert) canNOT save DPAS48 alone (would fix only the 92ms copy of a
  path already +65ms kernel / +100ms submit underwater). XMX prefill
  needs FIRST a rewritten 2D-tiled DPAS kernel (proper subgroup DPAS, no
  one-WI-per-16N-tile K-fold replication); device act-quant is the
  complement AFTER, not the lever. Neither is a session-scale build —
  scoped, not started. Do NOT retry device-quant-into-DPAS (dead end
  stands) and do NOT build standalone device-quant expecting a DPAS48
  win without the kernel rewrite.
- Prefill defaults flipped (2026-09-18, a52715f): chunked M-row prefill
  is now default-ON (chunk_M 8->128, batch_gemm default-ON; opt-outs
  INTFER_PREFILL_CHUNK=0 / INTFER_BATCH_GEMM=0 reproduce old path
  bit-identically). Defaults previously ran the token-by-token step loop
  (20-25 tok/s on 361tok); new defaults 148.2 tok/s (6x, contended box).
  chunk128 ~= 95% of chunk361 gain (quiet-box 149.2 vs 156.6).
  Decode -n64 unchanged (26.90->26.14, run variance). 27B/GPU joint-batch
  wide-chunk guard synced to default-ON semantics (would have disarmed).
  Gates: cpu parity + http batch PASS, GPU streams identical.
  Serve re-baseline on new defaults (2026-09-18, contended load ~9):
  sweep PASS (C=1 2.46 / C=4 3.59 / C=8 2.27 tok/s MT=8; parity held);
  MT=64 probe 7.66 tok/s over 338 ctoks — same-run pair with MT=8 C=8
  2.27 (short requests prefill-dominated, batch drains).
- 40 tok/s recipe (2026-09-18, per-token arithmetic @25.9 tok/s =
  38.6ms: kernel 9.6 / submit 11.7 / waitblock 9.5 / transfers 10.4ms,
  194 launches): device-act seams kill transfers (->35 tok/s), batched
  L0 multi-launch halves submit (->31), both = 22.2ms = 45 tok/s.
  Engine-invasive; needs quiet box (load<2) + gates. Not attempted
  under load. (memory: path_to_target_40_4000.md)
- OpenVINO evaluated, NOT adopted (2026-09-18 spike): OV 2026.3 sees
  both B580s, but FP16 matmul M=361 2.6ms (~our EU kernel, 1.0 TFLOPS
  vs 100+ peak — no XMX unlock) and M=1 1.2ms vs our ~50us (20x worse,
  framework overhead). Native SYCL stays; XMX win must come from our
  own DPAS kernels with device-resident act-quant.
- Wave shared-input convert-once (2026-09-18, f12f76b): run_linear_wave
  Phase 1 converts each distinct (pointer, bytes) input once, reuses
  staging bytes for duplicates (gate/up share rin; M-row batch gate/up
  share c_rin). Bit-identical, slots stay per-item, call-local scope.
  Dormant for distinct inputs (k/v, z/b/a) and fused path. Measured
  load-immune copy counts (NO_FUSE -n 6): 1116->972 converts (-13%);
  fused TP=1 unchanged (276=276). Streams identical TP1/NO_FUSE/TP2.
  Gates: cpu parity + http batch PASS. Attribution CLOSED (2026-09-19,
  INTFER_WAVE_DBG probe, since reverted): the "6 full-attn skip" note was
  wrong — layers 3,7,11,15,19,23 fire the gate/up dedup-hit like every
  other layer (168 hits = 24 prefill + 144 decode = 24 layers x 7 passes).
  Dedup fires only on 2-item shared-input gate/up waves; never on z/b/a
  (3 distinct rotated inputs) or k/v (2 distinct). -n6 wall +2.8% (run
  variance — convert is not the wall, submit is).
- Dead ends (measured, do not retry as-is): device-side act-quant
  (bit-exact probe passed but fused kernel slower: one-WI-per-16N-tile
  layout replicates O(K) quant work nT x), host-private staging of DPAS
  writes (neutral), staged readback (neutral), SYCL graphs on B580
  (19.7us vs 11.9us plain submit), LA-branch run-parallel OMP in
  step_batch (2026-09-16, parity-clean incl. segfault fix for TLS reads
  inside OMP workers, but zero gain: CPU C=8 batch WIP 10.5s vs HEAD
  10.8s @20T, 16.18s vs 16.34s @1T; GPU serve-load C=8 3.86 vs 3.89
  tok/s. LA host recurrence is a negligible share — GEMMs dominate.
  Revisit only if profiling shows recurrence >10% of a step. NOTE
  2026-09-18: the 8.16/15.33/17.98 (C=1/4/8) + 36.3 (MT=64 probe)
  numbers claimed 2026-09-17 are WITHDRAWN as unreproducible. Four
  independent C=1/MT=8 sweeps (incl. quiet-box start-load 0.69 and
  1.57) all land 3.27-3.53 tok/s; only docs commits separate the
  3.89 (09-16) and 8.16 (09-17) readings, i.e. same code. Bisect
  (single warm req, quiet box): fused ON 1.78s vs OFF 1.77s (neutral),
  joint 1.78s vs split 2.01s (joint helps, active), dispatcher 1.78s
  vs serial 2.07s (dispatcher active), chunk361 1.68s (chunking minor).
  The MT=64 batch-full probe (9.02 tok/s over 338 ctoks) proves the
  path batches — MT=8 is prefill-dominated, not contention. C=8
  scaling inversions under rising box load are a sweep-harness
  artifact (load 2.5→10+ mid-run). Quoting rule retired (C=1>=7 never
  fires): quote the MT=8 sweep AND the MT=64 probe from the same run.
  Dead-end verdict on OMP stands),
  device-side SSM recurrence M=1
  (2026-09-16, INTFER_SSM_GPU=1, parity-clean incl. step_batch +
  spec-rollback resync: -9% decode, 7.4->6.7 tok/s pinned 3x3
  interleaved. Per-LA-layer attribution: prep ~4us, H2D ~15-130us,
  kernel-submit ~25us, D2H+wait ~150-550us vs replaced host loop
  ~0.1-0.2ms — the blocking wait is structural: host norm needs
  core on-host same-layer. Revisit only with fused device-side
  SSM+norm feeding la_out GEMM from device buffers, or M>>1 where
  the single wait/layer amortizes; M=8 estimate still loses),
  native L0 immediate submit (2026-09-18, examples/l0_submit_probe spike,
  since removed: native immediate append+host-sync 27.6us vs UR
  submit+event.wait 25.2us per launch on a trivial kernel, correctness
  PASS — same floor, not a UR-adapter artifact. Revisit only with a batched
  multi-launch list amortizing one sync over many kernels),
  INTFER_NO_PROF queue without enable_profiling (2026-09-18: 19.21 vs
  18.75 tok/s TP1 decode under box contention — within run variance;
  per-event L0 timestamping is not the submit tax).
- PLAN.md triage (2026-09-19, grounded vs tree): preload_all/USM-pool /
  fused-MLP already DONE (preload log: 186 shards pinned, 534MiB dev0,
  no evict; pool_acquire routed, pool holds 0 bufs post-preload as
  designed — grow-only workspaces). DPAS-enforce + SYCL-graphs + L0-P2P
  are measured dead-ends above (do not retry as-is). INT8-KV
  (INTFER_INT8_KV / INTFER_KV_Q8) + SSM-chain (ssm_forward_chain via
  INTFER_SSM_GPU) exist as opt-in paths, certified stream-identical on
  GPU by scripts/test_gpu_optin.sh (10/10: base/KV/Q8/Q8+HAD/SSM_GPU x
  TP1/TP2, .intifer stream 279/6511/314/279) — wall-time comparison
  still needs quiet box (load<2). SCOPE CAVEAT (audit #3b above): the
  SSM_GPU identity holds only on the FP16 GEMM path; its ~1ulp device/host
  recurrence drift is invisible in FP16 but visible under INT8/GA64 act
  quant, so SSM_GPU now declines under INTFER_DPAS48=1. "Certified
  stream-identical" never meant "bit-exact SSM recurrence".
- First-touch migration (2026-09-21, measured on 27B TP=2 serve):
  preload_all builds 496 shards in ~49s and pins them (zero rebuilds on
  first request), but shard USM is malloc_shared so pages stay host-resident
  until first kernel touch: xpu-smi device-local at startup dev0 791MB /
  dev1 51MB, jumping to 11.27GB / 10.94GB only after the first request.
  Same-prompt back-to-back (contended box, load ~6): req1 9.28s vs req2/req3
  2.75s/2.75s — first request pays ~6.5s of migration stalls, steady state
  reached after exactly one request (streams identical). q.prefetch() over
  all shards was built, measured (18.7GB "migrated" in 59ms), and REVERTED:
  device counters unchanged, first-request jump identical — prefetch is a
  no-op hint for shared USM on this Level Zero driver (do not retry as-is;
  note in engine.cpp preload_all). Mitigation: one warmup request after
  /health goes live (scripts/serve_27b.sh does this, INTFER_NO_WARMUP=1
  skips). A startup touch-kernel over all 496 shards would only move the
  cost, not eliminate it — unscoped.
- Wave-submit + vectorized host math (2026-09-18, wave 7d7ca72, AVX2
  5285b47, extended a78e99f, quantified on quiet box loadavg<2): z/b/a +
  gate/up + full-attn k/v GEMMs convert+submit back-to-back, one wait
  per wave (scalar + M-row paths; INTFER_NO_WAVE=1 restores sequential).
  a78e99f adds: scalar full-attn k/v wave-2 (q stays sequential — its
  split/norm/rope consumers precede k/v input prep), M-row wave-batch
  for full-attn k/v + gate/up at all three batch sites. 8bce85c
  (2026-09-19) joins q into the k/v wave-3 and qkv into the z/b/a
  wave-4 in step() decode: both GEMMs' consumers (q split/norm/rope,
  qkv conv) run after, so the joins are pure regrouping, bit-identical
  by the wave contract. CORRECTION: the joins do NOT reduce kernel-wait
  count (nw 828 = 828 — wait_count increments per kernel event, and
  grouping changes no event count). What they halve is host round-trip
  phases: one wait-loop + one readback pass per wave instead of per
  GEMM (decode nr 828->432). M-row joins (2026-09-19, this session):
  the same q/qkv joins applied to the M-row paths — prefill-chunk
  full-attn q+k/v wave-3 and qkv+z/b/a wave-4, plus step_batch (serve)
  full-attn wave-3 and LA wave-4 (4 sites; q/qkv inputs staged into
  c_q_in/c_qkv_in/b_q_in/b_qkv_in copies since c_x is reused for k/v/z/b/a
  staging). INTFER_NO_WAVE=1 still restores sequential at all sites.
  Verified load-immune: prefill nr 138->72 (25-tok prompt), decode
  streams identical TP1/TP2/NOWAVE (279 6511 314 279 ...); cpu parity +
  http batch + gpu-optin PASS. Wall-time A/B deferred (box load >2).
  Each saved phase removes a blocking
  event-wait loop + host readback-loop setup (~24/layer-pass pairs per
  token). Measured load-immune (TP1 fused -n 6),
  streams identical TP1/TP2/NOWAVE; cpu parity + http batch + gpu-optin
  PASS. A/B CLOSED (2026-09-19, quiet box load~1.6, -n64 TP1 interleaved
  2 rounds + 3 JOIN-only): JOIN 25.97/25.99, PARENT 26.09/25.97 = NEUTRAL
  (-0.2%, run variance). Buckets: kernel/submit/waitblock/usm_write
  identical; readback +8-12ms (bigger wave-3/4 readback loops cost what
  the saved loop setups save); nr 4608 vs 6144 confirms grouping.
  Keep (fewer phases, same streams) but quote NO tok/s gain — submit
  (~745ms) + kernel (~645ms) dominate unchanged. Precision:
  wave does NOT reduce kernel-launch count (nk identical 1674/10tok
  both ways — engine still issues one kernel per item); it groups
  convert+submit back-to-back for queue depth and one wait per wave.
  AVX2/FMA rmsnorm/l2norm (INTFER_HOST_AVX2=0 restores scalar). Both oracle-certified: CPU
  parity + HTTP batch PASS, scalar-vs-AVX2 streams identical on CPU
  AND GPU paths. Wave: +0.6-0.7% decode, 3/3 interleaved rounds
  (TP1 24.61->24.78, 25.03->25.17, 24.94->25.09) — real but small:
  wave saves waits, submit (~830ms/65tok) is the wall. AVX2 norms:
  NEUTRAL (+0.7/-1.3/+0.4%, mean ~0) — host rmsnorm/l2norm is not on
  the decode critical path (the ~33% host share is GEMM act
  convert+memcpy, not norms). Keep both (correct, default-on, free),
  but quote no big win; remaining gap needs submit-side work.
- L4 design verdict (2026-09-18, investigated; MLP half IMPLEMENTED):
  PART A (done, fused MLP): the device-act GEMM seam now EXISTS —
  run_mlp_fused feeds the down GEMM straight from a device buffer
  (fused silu-mul output), skipping host convert + ws memcpy.
  PART B (not done): fused device-side SSM+norm feeding la_out GEMM.
  Data flow per LA layer today (18/24 layers on 0.8B): ssm_rows kernel
  -> D2H core (2048 floats) + blocking wait -> host ssm-norm+silu(z)
  -> H2D normed_core via run_linear_into -> la_out GEMM (engine.cpp
  converts FP32->FP16 host-side, memcpy to ws act, submit, wait, D2H).
  Fusion would kill per-LA-layer: 1 D2H + 1 wait + 1 H2D + 1 FP32->FP16
  host convert (la_out act 1024 FP32 = 4KB). BUT the old -9% stands and
  the reason is now precise: ssm_rows itself already pays H2D inputs
  (~15-130us/layer) + kernel-submit (~25us) + D2H+wait (~150-550us);
  fused norm+FP16-convert on device saves only the core round-trip and
  the host norm (~0.05ms), while the la_out engine seam
  (run_linear_tp: host FP32 in -> FP16 convert -> ws memcpy -> submit)
  has NO device-pointer entry — feeding it from a device buffer needs
  a new Engine method (device-act GEMM: skip convert+ws-write, submit
  with USM dep on the fused event). Net estimate: saves ~0.2-0.3ms of
  ~1.7ms/LA-layer (submit+waitblock dominate) over 18 layers =
  ~4-5ms/token of ~41ms = ~+10% IF the new seam costs zero extra
  submits. Verdict: viable but engine-invasive (new device-act GEMM
  entry + fused ssm_forward_norm_convert kernel + oracle gate on the
  fused FP16-convert rounding); do it only together with batched
  multi-launch L0 lists (same seam work amortizes both). Standalone
  INTFER_SSM_GPU=1 stays default-off. NOTE: the SAME device-act seam
  pattern was just proven by the fused MLP below (+13% decode) — the
  L4 SSM fusion is now a second application of a working technique,
  not a new invention; the remaining L4-specific risk is only the
  fused norm+silu(z) numerics.
  Data flow per LA layer today (18/24 layers on 0.8B): ssm_rows kernel
  -> D2H core (2048 floats) + blocking wait -> host ssm-norm+silu(z)
  -> H2D normed_core via run_linear_into -> la_out GEMM (engine.cpp
  converts FP32->FP16 host-side, memcpy to ws act, submit, wait, D2H).
  Fusion would kill per-LA-layer: 1 D2H + 1 wait + 1 H2D + 1 FP32->FP16
  host convert (la_out act 1024 FP32 = 4KB). BUT the old -9% stands and
  the reason is now precise: ssm_rows itself already pays H2D inputs
  (~15-130us/layer) + kernel-submit (~25us) + D2H+wait (~150-550us);
  fused norm+FP16-convert on device saves only the core round-trip and
  the host norm (~0.05ms), while the la_out engine seam
  (run_linear_tp: host FP32 in -> FP16 convert -> ws memcpy -> submit)
  has NO device-pointer entry — feeding it from a device buffer needs
  a new Engine method (device-act GEMM: skip convert+ws-write, submit
  with USM dep on the fused event). Net estimate: saves ~0.2-0.3ms of
  ~1.7ms/LA-layer (submit+waitblock dominate) over 18 layers =
  ~4-5ms/token of ~41ms = ~+10% IF the new seam costs zero extra
  submits. Verdict: viable but engine-invasive (new device-act GEMM
  entry + fused ssm_forward_norm_convert kernel + oracle gate on the
  fused FP16-convert rounding); do it only together with batched
  multi-launch L0 lists (same seam work amortizes both). Standalone
  INTFER_SSM_GPU=1 stays default-off.
- Fused MLP down-projection (2026-09-18, IMPLEMENTED, TP=1 only):
  Engine::run_mlp_fused chains gate/up wave-submit on-device through a
  fused silu-mul + RNE FP16-convert kernel (gemm::silumul_to_half) straight
  into the down GEMM: one host wait + one D2H per MLP instead of wave-wait
  + 2 D2H (gate/up) + host silu-mul + convert + H2D + down-wait.
  TP>1 / DPAS48 / trace (NaN/H) / bad shapes return false and the caller
  keeps the host path (mg/mu consumers are trace-gated). Kill switch:
  INTFER_NO_FUSE_SILUMUL=1 restores wave + host silu-mul (bisection).
  Bit-contract: FP32 op order matches the host loop; sycl::exp may differ
  ~1ulp on rare elements; dev_float_to_half is RNE bit-identical to
  dashq::float_to_half (verified exhaustively over e in [-10,0], 2M
  random + boundary mantissas, zero mismatches — the AGENTS.md "e == -10
  edge" note was wrong: dashq mutates mant with the hidden bit BEFORE
  the round test, so host and device test the same bit).
  (box contended loadavg 5-11, interleaved A/B 2 rounds, TP=1 decode
  -n 32): 28.06/27.81 vs 24.77/24.48 = +13.3%/+13.6% (~+3.2 tok/s);
  prefill chunk361: 158.3 vs 154.2 = +2.7%. GPU streams bit-identical
  ON vs OFF (279 6511 314 279 3046 13 198 760) and vs TP2; all gates
  PASS with fused default-ON (cpu parity, http batch, serve-load).
  Quiet-box confirmed 2026-09-18 (loadavg ~1.9, fused default-ON,
  -n 64): TP=1 decode 27.24 tok/s (+11.6% vs 24.4 pre-fusion quiet
  baseline — matches the +13.3/+13.6% contended A/B within run
  variance), TP=2 14.14 (unchanged, fused is TP=1 only), prefill
  chunk361 156.6 (+2.8% vs 152.2, matches contended +2.7%). New
  single-stream ceiling: ~27 tok/s TP1. The +13% relative gain is the
  durable claim (waits/round-trips removed, submit wall untouched).
  fused device-side SSM+norm feeding la_out GEMM from device buffers.
  Data flow per LA layer today (18/24 layers on 0.8B): ssm_rows kernel
  -> D2H core (2048 floats) + blocking wait -> host ssm-norm+silu(z)
  -> H2D normed_core via run_linear_into -> la_out GEMM (engine.cpp
  converts FP32->FP16 host-side, memcpy to ws act, submit, wait, D2H).
  Fusion would kill per-LA-layer: 1 D2H + 1 wait + 1 H2D + 1 FP32->FP16
  host convert (la_out act 1024 FP32 = 4KB). BUT the old -9% stands and
  the reason is now precise: ssm_rows itself already pays H2D inputs
  (~15-130us/layer) + kernel-submit (~25us) + D2H+wait (~150-550us);
  fused norm+FP16-convert on device saves only the core round-trip and
  the host norm (~0.05ms), while the la_out engine seam
  (run_linear_tp: host FP32 in -> FP16 convert -> ws memcpy -> submit)
  has NO device-pointer entry — feeding it from a device buffer needs
  a new Engine method (device-act GEMM: skip convert+ws-write, submit
  with USM dep on the fused event). Net estimate: saves ~0.2-0.3ms of
  ~1.7ms/LA-layer (submit+waitblock dominate) over 18 layers =
  ~4-5ms/token of ~41ms = ~+10% IF the new seam costs zero extra
  submits. Verdict: viable but engine-invasive (new device-act GEMM
  entry + fused ssm_forward_norm_convert kernel + oracle gate on the
  fused FP16-convert rounding); do it only together with batched
  multi-launch L0 lists (same seam work amortizes both). Standalone
  INTFER_SSM_GPU=1 stays default-off.
- lm_head audit (2026-09-19): step_batch/serve scoring is ALREADY M-row
  batched — run_lm_head_batch (main.cpp:1174, M>1) does one launch per
  device tile instead of M scalar calls; INTFER_LMHEAD_BATCH=0 restores
  per-row for bisection. Nothing to implement; todo item cancelled.
  Verified on HEAD: cpu parity + http batch + gpu-optin (10/10) PASS,
  WAVE vs NOWAVE GPU streams identical TP1/TP2 (279 6511 314 279).
- L4-lite IMPLEMENTED, default-off (2026-09-19, INTFER_L4_FUSE=1 opt-in):
  second application of the fused-MLP device-act pattern, scoped to TP=1
  scalar decode (the 40 tok/s target). Per LA layer it chains solo-run
  ssm_forward -> fused ssm_norm_silu_to_half kernel (rms-norm + silu(z) +
  RNE FP16 convert, same loop order as host) -> la_out GEMM via the new
  Engine::run_linear_tp_devact seam (device FP16 act, no host convert/H2D,
  dep-chained, single wait + single D2H). New files touched:
  quant_gemm.hpp/cpp (SsmNormParams + kernel), engine.hpp/cpp
  (run_linear_tp_devact: TP=1-only, FP16-act-only, no-mutation-on-false,
  ws_[0].out alias guard), main.cpp (gpu_ssm_resolve extracted from
  gpu_ssm_rows as a shared helper — pure move; GpuBackend L4 staging
  buffers + ti_of; gpu_la_out_fused; step() hook with trace_h forcing the
  host path; INTFER_L4_DBG=1 debug counter). Saves vs ssm+host path: 1
  D2H (core) + 1 H2D (normed) + host norm/convert + 1 wait per LA layer;
  costs 1 norm submit + 2 small H2D (z, norm-w). Correctness-gated NOW
  (load-insensitive): L4-vs-base streams IDENTICAL TP1 n=4/16/32 (279 6511
  314 ...), TP2 flag-vs-noflag identical (L4 silent no-op TP>1), SSM_GPU x
  L4 4-way identical, cpu parity + http batch + gpu-optin (10/10) PASS on
  the final binary; [l4] counter proves the path fires (72 calls/run).
  Wall-time A/B NOT run (box load 8.78, rising — quiet rule load1<2):
  run scripts/bench_quiet_ab.sh on a quiet box before any perf claim.
  L0 verdict stands: raw command lists measured same-floor (27.6 vs
  25.2us), so no L0 scaffolding was built — the multi-submit-single-wait
  chain IS the achievable amortization on in-order UR queues, and the
  devact seam is the prep a future batched launcher needs.
- L4 serve M==1 fast path IMPLEMENTED, default-off (2026-09-19, same
  INTFER_L4_FUSE=1 flag): single-row step_batch serve batches route each
  LA layer through the certified gpu_la_out_fused (conv loop advances
  conv_state first, matching scalar order; success skips the ssm_rows +
  host-norm + M-row la_out sections for that layer; any false falls
  through to the standard path with eligibility cleared for the layer).
  M>1 deliberately NOT fused: run_linear_batch(la_out) is already one
  M-row GEMM (single submit+wait), so a thin norm offload would only trade
  host convert for M norm submits — wash or loss; the M>1 win needs
  device-resident cores (same precondition as the L4 verdict above).
  Gates (all PASS, load-insensitive): serve C=1 L4-vs-base byte-identical
  ([l4] fires on the serve path, BASE counter 0), TP2 flag/no-flag
  identical (silent no-op), SSM_GPU x L4 4-way CLI identical, SSM_GPU+L4
  serve stream identical to base (shared SsmDev residency consistent when
  mixing),   cpu parity + http batch (with L4 ON: single/batch/SSE/serial
  all PASS, oracle Paris) + gpu-optin 10/10. Wall-time A/B NOT run (box
  load 5.58 — quiet rule load1<2): run scripts/bench_quiet_l4_ab.sh on a
  quiet box before any perf claim (new script; covers CLI decode TP1 -n64
  + chunk361 no-op guard).
- Profiler blind spot FIXED (2026-09-19, profile_note_launch): SSM/norm
  kernels submit directly through quant_gemm free functions, so their
  launches never reached the [GPU profile] nk counter (L4-vs-base decode
  showed nk=1680 both ways while submit dropped 91->64ms — invisible
  work). ssm_forward/chain + ssm_norm_silu_to_half now count one launch
  each via intfer::gemm::profile_note_launch (defined in engine.cpp,
  zero-cost when profiling is off; intfer_gemm gets INTFER_PROFILE_GPU in
  src/CMakeLists.txt). Re-run TP1 -n8: prefill nk=210 both (L4 no-op,
  exact), decode BASE nk=1680 vs L4 nk=1968 = +288 = 18 LA layers x 8
  tokens x (1 SSM + 1 norm), streams identical. The +288 proves the whole
  decode fires the fused chain; the -n64 quiet A/B will show whether the
  submit/usm_write savings (-30%/-12% in buckets, contention noise aside)
  convert to tok/s.
- nk accounting matrix VALIDATED (2026-09-19, TP1/TP2 x BASE/SSM/L4/BOTH,
  -n8 decode, counters only — load 4-5, no wall-time read): all 8 streams
  identical (6511 314 279 3046 13 198 760 6511). TP1: BASE 1680, SSM 1824
  (+144 = 18 LA x 8 tok x 1 SSM), L4 1968 (+288 = x2, SSM+norm), BOTH 1968
  (L4 subsumes the SSM launch — no double-SSM, shared SsmDev residency
  confirmed in counters). TP2: BASE 2976 (= 2 x 1488? — per-device GEMM
  launches counted per shard while SSM stays single-queue; do NOT read
  TP2 nk as 2x work), SSM 3120 (+144, same single-queue SSM delta),
  L4/BOTH 2976/3120 (L4 silent no-op TP>1 — zero fused launches, the
  TP=1-only guard proven in the counter, not just in streams). The
  counters now decompose every launch source on the decode path.
- L4 M-row serve fusion IMPLEMENTED, default-off (2026-09-19, fcbade9,
  same INTFER_L4_FUSE=1 flag, TP1 only): supersedes the "M>1 deliberately
  NOT fused" note above — step_batch LA layers with M>1 now route through
  gpu_la_out_fused_batch (one cross-key ssm_forward over M distinct
  (req,layer) states via a gathered pointer table + one M-row
  ssm_norm_silu_to_half_m kernel + one M-row devact la_out GEMM) instead
  of ssm_rows + host norm + M-row la_out. Shared-state rows (same-request
  prefill runs in joint batches) and TP>1/DPAS48/bad shapes decline
  inside the callee to the standard path. The M>1 precondition that made
  the old verdict ("needs device-resident cores") is met by staging a
  contiguous M-row core block + pointer table in GpuBackend L4 buffers
  (grow-only, freed in dtor). Cost per LA layer: 1 SSM submit + 1 norm
  submit + seam GEMM submits (vs ssm_rows solos + host norm/convert +
  H2D + la_out-wait); host work unchanged (M x O(kd) q/k prep + 2 small
  H2D). Gates (all PASS, load-insensitive): 3-way concurrent serve batch
  L4-vs-base byte-identical (M=2/M=3 fused-batch fired in [l4] counters),
  scalar streams identical TP1/TP2 (279 6511 314 279 3046 13 198 760),
  cpu parity + http batch (4.12 tok/s) + gpu-optin 10/10 on the final
  binary. Debug: INTFER_L4_DBG=1 prints per-path counters ([l4] fused
  la_out / fused batch la_out M=%u). Wall-time A/B NOT run (box load
  ~6 — quiet rule load1<2): scripts/bench_quiet_l4_ab.sh running in
  background, gated on quiet.
- L4 M>1 measured on M=3 sched batch (2026-09-19, 3/3 BASE/L4 pairs,
  near-quiet load 2-3.8, 27 toks/3 reqs, -n8): submit −24% systematic
  (110.8/112.6/112.0 -> 84.4/84.7/85.2ms, variance <2ms — real), nk
  1536->1824 (+288, fused chain fires), usm_write −5ms, BUT waitblock
  +21ms (597->618ms) eats nearly all of it: net wall −1.5% (946-948 ->
  929-935ms, consistent sign 3/3). Mechanism: at M=3 the device is
  saturated (kernel 597ms dominates the 940ms wall), so fewer host gaps
  just deepen the device queue — submit savings only convert to tok/s
  where the device has idle gaps (scalar M=1, kernel ~26% of decode).
  Implication for 40 tok/s: the M>1 L4 batch is correctness-certified
  infrastructure (streams identical, fewer host round-trips), NOT a
  wall-time lever under saturation. UPDATE 2026-09-19 (bench_quiet_l4_ab.sh
  3 rounds DONE, box never quiet — load 1.91/2.24/2.55/2.78/3.04/3.17):
  scalar A/B is a VARIANCE WASH, no promotion. -n64 TP1 decode L4 vs BASE
  = 17.19/19.52, 19.63/16.28, 15.32/26.45 (round-3 BASE 26.45 at load 3.17
  beats round-1 L4 17.19 at load 1.91 — the box, not the code, decides).
  Buckets: submit L4 wins 2/3 (716/955, 665/1186, 845/726) but
  kernel/waitblock flip randomly with load; nk=15744 vs 13440 every round
  (+2304 = 18 LA x 64 x 2, chain fires). Prefill guard clean (152-155 both,
  L4 no-op confirmed). L4 STAYS default-OFF opt-in (certified infra, no
  tok/s claim either way). Do NOT re-run hoping for a better number —
  re-run ONLY on a sustained-quiet box (load1<2). Both A/Bs point the same
  way: submit-side savings convert only where the device has idle gaps.
- L5 LA-front chain (2026-09-19 hooked 6c31136, stream-identical; 2026-09-20
  op-cut 7a6cee3): extends the device-act pattern FORWARD from la_out to the
  whole LA input path. Per LA layer scalar path today = host wave-4 (4 GEMM
  round-trips) + host conv + host prep + L4 tail. L5 replaces with: H2D
  normed (1x) -> fwht_rot_to_half kernel (4x FWHT rot + RNE convert)
  -> 4x devact GEMM (existing seam) -> la_conv_prep kernel (conv FIR+silu +
  qk-prep pack into sd->in) -> certified L4 tail (ssm_forward + norm +
  devact la_out). conv_state goes device-resident in SsmDev (dirty-tracked,
  host resyncs). Bit-contracts: FWHT/conv-FIR/packs exact host order;
  silu/exp/log1p ~1ulp (oracle arbitrates). Kill switch: INTFER_L5_FRONT=1
  (default OFF); any false keeps the L4/host path. Stream-gated TP1 -n 6:
  base == L5 == L5+HOSTPREP == L5-TP2-decline (279 6511 314 279 3046 13).
  SUBMIT-WALL VERDICT (2026-09-20, structural, load-immune): L5-front scalar
  decode is a DEAD END as built — do not chase wall-time on it. Buckets
  (TP1 -n 6): base nk=1260/nw=828/kernel=63.7ms vs L5 nk=2124/nw=828/
  kernel=61.3ms, waitblock 318 vs 62ms. Same device work, same wait COUNT
  (the single-wait seam works — devact + dep chains, no extra waits), but
  +144 launches/token = +8/LA layer (fwht, 4x half_to_float, prep, tail
  ssm+norm the host path never launches) and ~150 extra UR submits/token
  serialize the in-order queue: the device starves between host submits and
  the final wait per layer drains idle time, inflating waitblock 5x. 7a6cee3
  cut the resend fat (per-layer cw/dtb/alog/kinds const cache l5c_ +
  dropped the provably-redundant conv-writeback wait, which returned
  immediately) — nk/nw confirm the remainder is kernel-launch count, not
  waits or H2Ds. Fusing 4x h2f into 1 kernel would save only ~3 launches/
  layer (~4ms) for real risk — not worth it. L5 stays default-OFF; the
  kernels + stream-identity proof are retained for a future fused
  fwht->GEMM->prep kernel or M-row/serve amortization, where launch count
  amortizes. FIX 2026-09-21: L4 divergence was a scoping bug from the 6c31136
  L5-hook restructure, NOT a numeric issue — the final la_out dispatch guard
  was renamed if(!l4_done)->if(!l5_done) when the outer scope was added, but
  l4_done (init =l5_done) is the flag recording L4 success. L4-on/L5-off:
  fused path filled o_la on-device, then the host GEMM re-ran over stale
  normed_core and clobbered it (step-1 divergence). Fixed by hoisting l4_done
  to l5_done scope and guarding if(!l5_done && !l4_done) (conjunction needed:
  L5 success skips the block where l4_done=l5_done is assigned). Root cause
  excluded the RNE-converter changes (shared converters verified neutral —
  all 8 combos identical). Stream-gated TP1 -n6: base==L4==L5==HOSTPREP==
  SSM==SSMxL4==TP2-base==TP2-L4 (279 6511 314 279 3046 13); cpu parity +
  gpu-optin + http batch PASS.
- Path to target: single-stream is ~27 tok/s TP1 (fused MLP, was ~24.4
  pre-fusion, was ~26 ceiling pre-DPAS-off):
  ~169 launches/token (~80us submit each: submit ~30%, waitblock +
  transfers most of the rest). 40 tok/s must come from aggregate
  batching (--serve scheduler already M-row batches; cross-request HTTP
  batching is the highest-value lever, C=8/MT=64 probe 9.02 tok/s
  aggregate on quiet box 2026-09-18 (338 ctoks, proves batching; the
  old 36.3 figure is withdrawn with the 8.16 line above). Gate
  re-verified PASS on HEAD (wave+AVX2+fused serving-certified) — the
  C=1 3.27-3.53 tok/s across four independent sweeps is structural
  (MT=8 prefill-dominated), not contention; quoting rule retired.
  Structural options: batched
  multi-launch L0 lists amortizing one sync over many kernels (single
  immediate append+sync measured no faster than UR submit+wait, so the
  win must come from fewer syncs, not a faster submit),
  device-side SSM/attention (host O(pos) attention dominates long ctx).
- Prefix-cache TTFT verdict (2026-09-26, ec86808, 27B TP=2 serve,
  bench_chat.sh --turns 3 --max-tokens 30, parity in
  /var/home/fra/build_tmp/chat_quiet_on.txt; logs serve_27b_18411
  (ON) + serve_27b_18412 (OFF)): single-entry last-completed-request
  snapshot (grid K=16, lowest-4+highest-12, INTFER_NO_PREFIX_CACHE=1
  kill switch, INTFER_LOAD_ORACLE override for non-0.8B serve-load
  oracle). REPRODUCED 6 consecutive sessions. Quiet-box A/B (BOTH
  sides load ~1.1, 27B TP=2, logs serve_27b_18413 ON +
  serve_27b_18412 OFF, parity chat_quiet_on2.txt == chat_quiet_on.txt
  bit-identical): ON t0 TTFT 3.6s/E2E 8.0s, t1 3.9s/23.6s pchit=16
  (prompt=54), t2 4.8s/28.1s pchit=48 (prompt=100). OFF (zero pchit,
  all 3 turns parity OK, rc=0): t0 3.7s/8.1s, t1 4.8s/28.5s, t2
  8.0s/42.9s. Delta t1 -0.9s (-19%), t2 -3.2s (-40%); all 6
  sessions agree within ~0.1-0.2s. The earlier loaded ON run (load
  ~9.2: t0 3.7/8.1, t1 3.9/23.4, t2 4.8/28.0) matches the quiet ON
  to ~0.1s — load inflates absolutes far less than feared on the
  cache-hit path. Quiet-box debt PAID (both sides); numbers above
  are quotable. use=16/48 are
  K=16 grid multiples, use=48 on prompt=100 proves mid-stream snaps
  survive retention (full-stream-only would decline). Sr_old-4 chat
  divergence absorbed by largest-grid-pos<=hit. q.fill=use ->
  P=min(chunk_M,Sr-fill) structurally skips `use` prefill rows.
  Divergence-hunt CLOSED same session: claimed HIT/MISS divergence
  does NOT reproduce on clean tree (install() copies bit-exact,
  10/10 + 12/12 identical outputs; earlier "alternation" was a probe
  history bug). TRACE-PROOF 2026-09-26 (INTFER_PCACHE_DBG, CPU serve,
  identical 17-tok prompts): REQ1 miss records pos=16 + snapshot
  state_len=24 grid_keys=16,24; REQ2 match hit=17 (capped 16) ->
  install use=16 pchit=16 -> snapshot grid_keys=24; installed
  ssm[0]/ssm[-1]/conv[0]/conv[-1] == recorded values on all 18 LA
  layers (BIT-EXACT 18/18), completions byte-identical
  ("The capital of France is **Paris**."). Log: build_tmp/pc_trace3.log. Memory: ~159MB/snap x 16 grid-max ~= 2.5GB host max
  (+KV mirrors), CONSTANT single-entry bound (code-bound: harvest
  caps grid at 16 incl. completion-top; cap unstressed at <=100tok
  prompts; no multi-entry LRU). Serve-load cache-ON (27B, load ~8,
  MT=8 sweep + MT=64 probe): c1/c4/c8 completion + c8 parity + floor
  + bigmt all PASS (c1 1.84 -> c8 1.59 tok/s MT=8 prefill-dominated
  as expected; MT=64 probe 2.45 tok/s over 346 ctoks); solo.oracle
  gate now has INTFER_LOAD_ORACLE override (249d3f8) for non-0.8B
  models — c8.parity byte-identical stays the real determinism gate.
  Cache fires under concurrency (pchit=16 at admit with pq=8).


# guidelines

- remember to commit any meaningful change made such that in case of regressions everything can be bisected.
- remember to not litter /tmp with large tmp files, RAM is precious
- try to parallelize work as much as possible, spawn tasks (subagents) in different worktrees that implement a specific optimization experiment, then merge all the experiments in one single batch. Every optimization experiment is gated behind a flag in the cli. The user the runs the benchmarks and reports back the results.
- MANDATORY: every batch of optimization experiments handed to the user for testing MUST be written up as a report file in `reports/` (create the directory if missing; filename `reports/<YYYY-MM-DD>-<batch-slug>.md`, e.g. `reports/2026-09-30-batch-dpas-preq.md`). The user benchmarks from these reports, so a batch delivered without one is an unfinished batch. Each report MUST contain, per experiment: (1) the flag + kill switch and default state, (2) what it changes and why it should help, (3) the exact benchmark command to run (and the quiet-box rule load1 < 2), (4) the load-INSENSITIVE buckets I measured myself (kernel/submit/waitblock/copy/usm_write/readback, nk/nw/nr counters) with the caveat that they are counters, not wall time, (5) the expected direction and what counts as a win/neutral/loss, (6) which correctness gates already passed. NEVER put a tok/s or wall-time number I did not measure in a report — leave a blank `RESULT:` line for the user to fill from a quiet-box run. Update the report (not just AGENTS.md) when a follow-up session re-scopes or kills one of the experiments, so the user's list of pending tests stays accurate.
