// include/intfer/gemm/tile_scheduler.hpp
//
// 1.3 SLM work-stealing tile scheduler for prefill tiles.
//
// The prefill tiled DPAS kernels (quant_gemm_dpas48_2d) launch one work-item
// per tile and dispatch a fixed tile->WG assignment at launch time. Under
// device saturation that assignment leaves gaps: a WG that finishes its tile
// early waits while another WG's tile is still running, and the submit cost
// (~600 us/launch at M=128) is paid per launch regardless.
//
// A per-device SLM atomic counter lets a WG that has drained its tile pull
// the NEXT unfinished tile from the shared pool instead of idling. Fewer,
// better-filled launches are the prefill lever (vllm/sgl TileScheduler).
//
// Per-device, so it is TP-compatible: device d owns only its slice of N, and
// the counter lives in that device's SLM.
//
// INTFER_TILE_SCHED=1 enables it (default OFF). It is a pure dispatch
// helper: it changes WHICH tiles run in WHICH order, never the arithmetic,
// so the tiled kernel output is bit-identical to the fixed-assignment path --
// each tile computes the same dot products and accumulates in the same order
// (row-major over the tile's K, ascending), only the order tiles land on the
// queue differs.
//
// B580 note: Xe2 has 192 KB SLM per Xe-core and 16 KB per sub-group. The
// counter is one uint32_t per tile, i.e. 4 B * ceil(N/16) -- a few KB at 27B
// prefill, well inside the SLM budget.
//
// This is NOT the CUTLASS TiledMMA / block_2d_copy / xe_split_barrier
// machinery (tens of kLOC, not in the tree). It is the minimal SLM-atomic
// scheduler on top of the existing tiled kernel's launch shape, and it is
// sufficient because the tiled kernel already owns its own SLM tile
// buffers (sA / sW local_accessors in quant_gemm.cpp).

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

namespace intfer::gemm::tile_scheduler {

// SLM tile-claim queue for the prefill tiled DPAS kernels.
//
// THE CONTEXT CONSTRAINT, stated first because it dictates the whole spelling.
// The tiled DPAS kernels are SYCL_ESIMD_KERNEL (they issue the Intel XMX DPAS
// instruction through ESIMD intrinsics), and sycl::atomic_ref is NOT permitted
// in an ESIMD context -- the compiler rejects it with "not supported in ESIMD
// context". So the SLM claim below is written with ESIMD's own SLM atomics
// (lsc_atomic_update), which are the supported primitive there.
//
// Per-device, so it is TP-compatible: device d owns only its slice of N, and
// the queue lives in that device's SLM (local memory).
//
// INTFER_TILE_SCHED=1 enables it (default OFF). It is a pure dispatch helper:
// it changes WHICH tiles run in WHICH order, never the arithmetic, so the
// tiled kernel output is bit-identical to the fixed-assignment path -- each
// tile computes the same dot products and accumulates in the same order
// (row-major over the tile's K, ascending), only the order tiles land on the
// queue differs.
//
// This is NOT the CUTLASS TiledMMA / block_2d_copy / xe_split_barrier
// machinery (tens of kLOC, not in the tree). It is the minimal SLM-atomic
// scheduler on top of the existing tiled kernel's launch shape, and it is
// sufficient because the tiled kernel already owns its own SLM tile buffers
// (sA / sW local_accessors).

// One uint32_t per tile in SLM; 0 = unclaimed, 1 = claimed.
// max_tiles caps the reserved SLM so a pathological N cannot overflow it.
constexpr int kMaxTiles = 4096;

}  // namespace intfer::gemm::tile_scheduler