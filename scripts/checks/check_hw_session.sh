#!/usr/bin/env bash
# Hardware-gating predicate for the mutation matrices that need a OneAPI
# compiler, a SYCL runtime and at least one visible GPU. Intended to be run
# only for its exit status (0 = usable session, other = skip).
set +u
setvars="${INTEL_ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh"
[ -x "$setvars" ] || exit 1
# shellcheck disable=SC1090
. "$setvars" --force >/dev/null 2>&1 || exit 1
set -u
command -v sycl-ls >/dev/null 2>&1 || exit 1
count="$(sycl-ls 2>/dev/null | grep -Ec '\[[^]]+\]' || true)"
[ "${count:-0}" -gt 0 ] || exit 1
exit 0
