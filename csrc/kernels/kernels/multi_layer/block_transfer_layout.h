// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Host/device shared POD for the MP block-level KV transfer.
//
// The host parser (csrc/mp_mem_kernels.cpp, prepare_group) resolves a
// PageBufferShapeDesc plus EngineKVFormat into this flat byte layout; the
// device kernel consumes it verbatim and never re-interprets the format.
// All sizes/offsets are in bytes (int64): planes may mix dtypes, and
// engine pools may exceed 32-bit element counts.
//
// Layout contracts:
//   - Engine plane p: [NB, BS, NH, HS] with dense token rows, so the
//     engine-side token step equals payload_bytes. Only the block axis may
//     carry padding, expressed by engine_block_stride_bytes.
//   - LMC side (formats 13/17): one packed row per token. Format 13 has one
//     fused plane; format 17 has lmc_token_stride_bytes = sum(payload_bytes)
//     and plane p sits at its prefix-sum column offset inside the row.
//   - Format 16: two independent slabs of nl * lmc_layer_stride_bytes;
//     planes[1].lmc_base_offset_bytes == nl * lmc_layer_stride_bytes.
// MP block 级 KV 传输在 host 与 device 间共享的 POD。
//
// host 解析器（csrc/mp_mem_kernels.cpp 中的 prepare_group）将
// PageBufferShapeDesc 和 EngineKVFormat 解析为该扁平字节布局；设备 kernel
// 原样使用，不再解释格式。所有大小和偏移均为 int64 字节数：不同 plane 可
// 混用 dtype，引擎池的元素数量也可超过 32 位范围。
//
// 布局契约：
//   - 引擎 plane p 为 [NB, BS, NH, HS]，token 行紧凑，因此引擎侧 token 步长
//     等于 payload_bytes。仅 block 轴可有填充，由 engine_block_stride_bytes
//     表示。
//   - LMC 侧（格式 13/17）每个 token 对应一个 packed 行。格式 13 只有一个
//     融合 plane；格式 17 的 lmc_token_stride_bytes = sum(payload_bytes)，
//     plane p 位于行内相应的前缀和列偏移处。
//   - 格式 16 包含两个独立的 nl * lmc_layer_stride_bytes slab；
//     planes[1].lmc_base_offset_bytes == nl * lmc_layer_stride_bytes。

namespace kvcache_ops {

constexpr int kMaxPlanes = 4;

// Per-AIV UB staging budget, a conservative cross-SoC floor (950 has 248KB
// physical UB per vector core). Bounds one token row, not a whole block;
// full budget contract in block_transfer_contract.md section 3.
// 每个 AIV 的 UB staging 预算，采用跨 SoC 的保守下限（950 的每个向量核有
// 248KB 物理 UB）。它约束一个 token 行而非整个 block；完整预算契约见
// block_transfer_contract.md 第 3 节。
constexpr int64_t kBlockTransferUbBytes = 128 * 1024;

constexpr int32_t kBlockTransferQueueDepth = 1;

// DataCopyExtParams::blockCount hardware upper bound for the DataCopyPad row
// path ([1, 4095] per the AscendC API reference). The kernel clamps
// rows-per-segment to this value: a thin plane (payload <= 32B) gives
// fit = kBlockTransferUbBytes / 32B = 4096 rows, one past the limit.
// DataCopyPad 行路径中 DataCopyExtParams::blockCount 的硬件上限（根据
// AscendC API 参考为 [1, 4095]）。kernel 将每段行数限制到该值：窄 plane
// （payload <= 32B）会得到 fit = kBlockTransferUbBytes / 32B = 4096 行，
// 比上限多一行。
constexpr int32_t kMaxRowsPerDataCopyPad = 4095;

// 单个物理 plane 的字节几何，同时描述引擎页与 LMC 对象内的位置。
struct PlaneLayout {
  int64_t payload_bytes;             // valid bytes per token slot
                                     // 每个 token slot 的有效字节数
  int64_t engine_block_stride_bytes; // dim-0 block step on the engine side
                                     // 引擎侧第 0 维的 block 步长
  int64_t lmc_base_offset_bytes;     // plane start inside the LMC object
                                     // plane 在 LMC 对象内的起始偏移
  // Format 2 only: bytes from the layer base pointer to this plane.
  // Formats 16/17 pass one pointer per plane, so this stays 0.
  // 仅格式 2 使用：从层基指针到该 plane 的字节偏移。格式 16/17 按 plane
  // 各传一个指针，因此此值保持为 0。
  int64_t engine_base_offset_bytes = 0;
};

// 一次 block 传输的完整共享布局；指针表按 layer/plane 索引，LMC 地址则由
// 以下行、层和对象步长计算。
struct BlockTransferLayout {
  int32_t num_planes;             // 1..kMaxPlanes (physical planes/layer)
                                  // 1..kMaxPlanes（每层物理 plane 数）
  int64_t lmc_token_stride_bytes; // LMC packed row width (all planes)
                                  // LMC packed 行宽（包含所有 plane）
  int64_t lmc_layer_stride_bytes; // bytes per layer = slots_per_object * row
                                  // 每层字节数 = slots_per_object * 行宽
  int64_t lmc_object_bytes;       // total object size
                                  // 对象总大小
  PlaneLayout planes[kMaxPlanes];
};

} // namespace kvcache_ops
