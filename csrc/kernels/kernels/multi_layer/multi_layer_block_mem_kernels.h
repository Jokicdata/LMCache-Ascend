// SPDX-License-Identifier: Apache-2.0

#ifndef MULTI_LAYER_BLOCK_MEM_KERNELS_H
#define MULTI_LAYER_BLOCK_MEM_KERNELS_H

#include "block_transfer_layout.h"

namespace kvcache_ops {

// Host-side launcher for the block-level multi-layer KV transfer kernel.
// One launch moves one LMCache memory object
//
// Layout semantics per engine KV format (see block_transfer_contract.md):
//   2  (NL_X_NB_TWO_BS_NH_HS): one [NB, 2, BS, NH, HS] tensor per layer.
//       The pointer table is one base data_ptr() per layer. Plane p is
//       base + plane.engine_base_offset_bytes. The planes are axis-1 slabs
//       (semantic K/V, or the two halves of an opaque sub-paged page).
//       LMC layout matches format 16.
//   16 (NL_X_TWO_X_NB_BS_NH_HS): per-layer (K, V) equal-width NHD tensors
//       -> LMC [2, L, T, NH*HS] (2LTD: K slab first, V slab after); work
//       items are (layer, plane, block) since K/V live in disjoint slabs.
//   17 (NL_X_NP_X_NB_BS_ONE_HS): per-layer tuple of 1..4 single-head planes
//       (MLA/DSA/DSv4) -> LMC [L, T, sum(payload)] packed byte rows; work
//       items are (layer, block) and one AIV core moves every plane of the
//       page, so no two cores ever share a 32B LMC line.
//
// Pointer table layout:
//   format 2: ptrs[layer], plus PlaneLayout::engine_base_offset_bytes;
//   formats 16/17: ptrs[layer * num_planes + plane], each entry the plane
//   tensor view's data_ptr().
// block 级多层 KV 传输 kernel 的 host 侧启动器。每次 launch 搬运一个
// LMCache 内存对象。
//
// 各引擎 KV 格式的布局语义（见 block_transfer_contract.md）：
//   2（NL_X_NB_TWO_BS_NH_HS）：每层一个 [NB, 2, BS, NH, HS] Tensor。
//       指针表每层保存一个 base data_ptr()，plane p 地址为 base 加
//       plane.engine_base_offset_bytes。两个 plane 是第 1 轴上的 slab
//       （语义 K/V，或不透明子分页页的两半）。LMC 布局与格式 16 相同。
//   16（NL_X_TWO_X_NB_BS_NH_HS）：每层等宽的（K, V）NHD Tensor 映射到
//       LMC [2, L, T, NH*HS]（2LTD：K slab 在前，V slab 在后）；由于 K/V
//       位于不同 slab，工作项为（layer, plane, block）。
//   17（NL_X_NP_X_NB_BS_ONE_HS）：每层 1..4 个单头 plane 的元组
//       （MLA/DSA/DSv4）映射到 LMC [L, T, sum(payload)] packed 字节行；
//       工作项为（layer, block），一个 AIV 核搬运该页的所有 plane，避免
//       两个核共享同一个 32B LMC 行。
//
// 指针表布局：
//   格式 2：ptrs[layer]，再加 PlaneLayout::engine_base_offset_bytes；
//   格式 16/17：ptrs[layer * num_planes + plane]，每项为 plane Tensor
//   视图的 data_ptr()。
// host launcher 仅将 kernel 提交到给定 stream；返回不表示传输完成，调用方
// 必须同步 stream，并在完成前保持指针表、block_ids 和对象缓冲区有效。

void multi_layer_block_transfer_kernel(
    uint32_t block_dim,            // min(AIV cores, ACTIVE work items =
                                   //   nl * plane_slots * (blocks - skip))
                                   // 最小值（AIV 核数，有效工作项数 =
                                   // nl * plane_slots * (blocks - skip)）
    void* stream,                  // aclrtStream (current NPU stream)
                                   // aclrtStream（当前 NPU stream）
    uint8_t* paged_buffer_ptrs,    // interleaved pointer table (see above)
                                   // 交错指针表（见上文）
    uint8_t* lmcache_obj,          // LMC object device VA (maybe staged)
                                   // LMC 对象设备虚拟地址（可能是 staging）
    uint8_t* engine_block_ids,     // int64 array: logical -> engine block
                                   // int64 数组：逻辑块到引擎块的映射
    int32_t num_blocks_per_object, // == lmcache_chunk_size / bs (host-checked)
                                   // == lmcache_chunk_size / bs（host 已校验）
    int32_t skip_prefix_n_blocks,  // leading blocks to skip (prefix dedup);
                                   //   == blocks -> no launch at all
                                   // 要跳过的前缀块数（前缀去重）；等于 blocks
                                   // 时完全不启动 kernel
    int32_t nl,                    // layers in this group
                                   // 本组的层数
    int32_t nb,                    // engine pool block count (ID upper bound)
                                   // 引擎池块数（块 ID 上界）
    int32_t bs,                    // tokens/slots per block
                                   // 每块的 token/slot 数
    bool separate_plane,           // true: (layer, plane, block) work items
                                   // true：工作项为（layer, plane, block）
    BlockTransferLayout layout,    // full byte layout (block_transfer_layout.h)
                                   // 完整字节布局（block_transfer_layout.h）
    bool to_engine,                // H2D = LMC->engine (retrieve), D2H =
                                   // engine->LMC (store); device-to-device
                                   // H2D = LMC->引擎（retrieve），D2H =
                                   // 引擎->LMC（store）；kernel 内为设备间搬运
    int64_t ub_bytes = 0,          // usable UB staging budget, host-queried
                                   //   via GetCoreMemSize(CoreMemType::UB);
                                   //   0 = built-in default
                                   // 可用 UB staging 预算，由 host 通过
                                   // GetCoreMemSize(CoreMemType::UB) 查询；
                                   // 0 表示使用内置默认值
    bool one_pointer_per_layer = false); // format 2: one base pointer per layer
                                         // 格式 2：每层一个基指针
} // namespace kvcache_ops

#endif // MULTI_LAYER_BLOCK_MEM_KERNELS_H
