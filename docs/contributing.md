# Contributing to the Snapshot Helper Framework

This document describes how to use the snapshot helper (`_gate_snapshot.sh`) and contribute to the mutation matrix framework.

## Overview

The snapshot helper provides a disciplined way to:
- Snapshot target files before mutation.
- Enforce dirty-tree opt‑in for uncommitted changes.
- Ensure atomic restore after matrix runs.
- Prevent stale lock and residue issues.
- Provide concurrency safety via per‑run lock files.

## Using the Snapshot Helper

### Basic workflow

```bash
# 1. Create a snapshot of the files you will mutate
. /path/to/_gate_snapshot.sh
gate_snap_init "file1.txt file2.txt"

# 2. Perform your mutation (edit files in place)
echo "new content" > file1.txt

# 3. Run the matrix or checker that uses the snapshot
bash scripts/matrices/mutate_fold_matrix.sh

# 4. The helper automatically restores the files on exit (via trap)
```

### Unified matrix API

For new matrix definitions, use the `gate_snap_matrix_begin` and `gate_snap_matrix_end` functions:

```bash
gate_snap_matrix_begin "matrix_name" "file1.txt file2.txt"
# ... run tests, mutate files, etc.
gate_snap_matrix_end "matrix_name" "file1.txt file2.txt"
```

- **`gate_snap_matrix_begin`**: reap stale snapshot directories, acquire a per‑run lock, snapshot targets.
- **`gate_snap_matrix_end`**: restore all targets, release the lock, clean up the snapshot directory, and verify no mutation markers remain.

### Common functions

| Function | Purpose |
|----------|---------|
| `gate_snap_init "files"` | Create snapshot directory and copy files. |
| `gate_snap_dirty_guard "files"` | Refuse to run unless `GATE_SNAP_ALLOW_DIRTY=1`. Prints `INPUT: UNCOMMITTED` lines. |
| `gate_snap_restore FILE` | Restore a single file from its snapshot (skips copy if already matching). |
| `gate_snap_restore_all "files"` | Restore multiple files. |
| `gate_snap_check_residue "files"` | Ensure no mutation markers (`/*M1*/` etc.) remain. |
| `gate_snap_acquire_lock NAME` | Acquire a per‑run lock (fails if another run holds it). |
| `gate_snap_release_lock NAME` | Release the lock. |
| `gate_snap_reap_stale MAX_AGE` | Remove snapshot directories older than `MAX_AGE` seconds. |

## Example: Adding a New Matrix

1. **Create a matrix script** (e.g., `scripts/matrices/mutate_roofline.sh`).
2. **Use the snapshot helper**:
   ```bash
   . "$SCRIPT_DIR/../lib/_gate_snapshot.sh"
   GATE_SNAP_ALLOW_DIRTY=0 gate_snap_dirty_guard "roofline.txt"
   gate_snap_init "roofline.txt"
   # mutate roofline.txt ...
   gate_snap_restore "roofline.txt"
   ```
3. **Add a regression test** in `scripts/tests/regression_matrix_tests.sh` to verify the matrix passes.

## Running Regression Tests

The regression suite lives in `scripts/tests/regression_matrix_tests.sh`. It runs:

- Fold matrix (7/7 states)
- Gate inventory matrix (9/9 mutations)
- Roofline matrix (emerging type, snapshot/restore + mode)

Run it locally:

```bash
bash scripts/tests/regression_matrix_tests.sh
```

All tests should pass (`PASS` messages). Failures indicate a regression in the snapshot helper or matrix logic.

## Contributing

- **Code style**: Follow the existing shell style (tabs, no trailing spaces).
- **Testing**: Every matrix script must have a corresponding regression test.
- **Documentation**: Update `docs/contributing.md` with usage examples for any new API or helper function.
- **CI integration**: Ensure the GitHub Actions workflow runs the regression suite as part of the matrix validation.

## Frequently Asked Questions

**Q: My matrix fails because the repo is dirty.**  
A: Ensure `GATE_SNAP_ALLOW_DIRTY=1` is set if you intentionally want to run over uncommitted changes. The snapshot helper will refuse otherwise.

**Q: The lock is still held after a run.**  
A: Verify that `gate_snap_release_lock` runs on all exit paths (including INT, TERM, EXIT traps). The helper already installs these traps.

**Q: I see “residue marker” errors.**  
A: Make sure your mutation scripts write the exact markers (`/*M1*/`, etc.) and that `gate_snap_check_residue` is called after restore.

---

*Maintainers*: Keep the helper lightweight, avoid hard‑coding paths, and always test with `GATE_SNAP_ALLOW_DIRTY=1` when needed.

