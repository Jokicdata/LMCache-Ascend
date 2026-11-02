// SPDX-License-Identifier: Apache-2.0

// Block-level multi-layer KV transfer between the engine's paged buffers and
// LMCache contiguous memory objects, for the MP (multiprocess) mode.
// MP（多进程）模式下，在引擎分页缓冲区与 LMCache 连续内存对象之间进行
// block 级多层 KV 传输。

#include "multi_layer_block_mem_kernels.h"
#include "kernel_operator.h"

namespace {

// AscendC 设备侧搬运实现：Init 绑定 GM 地址并配置 UB 队列，Process 划分
// 有效工作项，ProcessPlaneBlock 按 UB 段完成单个（layer, plane, block）传输。
class MultiLayerBlockTransfer {
public:
  // 初始化指针表、LMC 对象、block_ids、skip 前缀和静态布局；block_ids 将逻辑
  // 块映射到引擎物理块，skip 只排除每个对象开头的无效工作项。
  __aicore__ inline void Init(GM_ADDR pointers, GM_ADDR object, GM_ADDR ids,
                              int32_t blocks, int32_t skip, int32_t nl,
                              int32_t nb, int32_t bs, bool separate_plane,
                              kvcache_ops::BlockTransferLayout layout,
                              int64_t ub_bytes, AscendC::TPipe* pipe,
                              bool one_pointer_per_layer) {
    paged_kv_ptrs_ = reinterpret_cast<__gm__ uint64_t*>(pointers);
    block_ids_ = reinterpret_cast<__gm__ int64_t*>(ids);
    lmcache_obj_base_ = reinterpret_cast<__gm__ uint8_t*>(object);
    lmcache_obj_.SetGlobalBuffer(lmcache_obj_base_, layout.lmc_object_bytes);
    blocks_per_obj_ = blocks;
    skip_blocks_ = skip;
    num_layers_ = nl;
    engine_block_count_ = nb;
    tokens_per_block_ = bs;
    separate_plane_ = separate_plane;
    layout_ = layout;
    one_pointer_per_layer_ = one_pointer_per_layer;

    // UB segmentation: every token row occupies AlignUp32(payload) in UB
    // (DataCopyPad rounds each row up to the 32B boundary). All planes share
    // one queue buffer, so the segment must fit the widest aligned row.
    // UB 分段：每个 token 行在 UB 中占用 AlignUp32(payload)（DataCopyPad
    // 将每行向上对齐到 32B 边界）。所有 plane 共用一个队列缓冲区，因此
    // 分段必须能容纳最宽的对齐行。
    const int64_t max_row = MaxAlignedRowBytes();
    // One in-flight segment per queue slot: host guarantees
    // align_up32(payload) <= ub_budget / kBlockTransferQueueDepth, so
    // fit >= 1. ub_bytes == 0 -> built-in conservative default (see
    // layout.h).
    // 每个队列槽容纳一个在途分段；host 保证 align_up32(payload) <=
    // ub_budget / kBlockTransferQueueDepth，因此 fit >= 1。ub_bytes == 0
    // 时使用内置保守默认值（见 layout.h）。
    const int64_t ub_budget =
        ub_bytes > 0 ? ub_bytes : kvcache_ops::kBlockTransferUbBytes;
    const int64_t fit =
        ub_budget / (kvcache_ops::kBlockTransferQueueDepth * max_row);
    int64_t rows = fit < bs ? fit : bs;
    // Keep blockCount inside the DataCopyPad range; see
    // kMaxRowsPerDataCopyPad in block_transfer_layout.h.
    // 将 blockCount 保持在 DataCopyPad 范围内；见 block_transfer_layout.h
    // 中的 kMaxRowsPerDataCopyPad。
    if (rows > kvcache_ops::kMaxRowsPerDataCopyPad) {
      rows = kvcache_ops::kMaxRowsPerDataCopyPad;
    }
    if (rows < 1) {
      rows = 1; // unreachable after host validation; keeps InitBuffer sane
                // host 校验后不可达；此处保证 InitBuffer 参数有效
    }
    tokens_per_segment_ = static_cast<int32_t>(rows);
    pipe->InitBuffer(queue_, kvcache_ops::kBlockTransferQueueDepth,
                     tokens_per_segment_ * max_row);
  }

  // 按 AIV 核遍历有效工作项，并将 skip 前缀折回逻辑块索引；ToEngine=true
  // 执行 LMC->引擎的 H2D retrieve，否则执行引擎->LMC 的 D2H store。
  template <bool ToEngine> __aicore__ inline void Process() {
    //   separate_plane_ true -> formats 2/16: disjoint LMC slabs,
    //            work item = (layer, plane, block).
    //            Format 16 has one engine pointer per physical plane.
    //            Format 2 (NL_X_NB_TWO_BS_NH_HS) uses one pointer per layer;
    //            each plane starts at the layer base plus
    //            engine_base_offset_bytes.
    //   separate_plane_ false -> format 13 (one fused plane) or format 17
    //            (packed multi-plane rows), work item = (layer, block).
    //            One core moves every plane of the page so no two cores
    //            share a 32B LMC line.
    //   separate_plane_ 为 true：格式 2/16，LMC slab 相互独立，工作项为
    //            （layer, plane, block）。格式 16 每个物理 plane 一个引擎指针；
    //            格式 2（NL_X_NB_TWO_BS_NH_HS）每层一个指针，各 plane 起始
    //            地址为层基址加 engine_base_offset_bytes。
    //   separate_plane_ 为 false：格式 13（单融合 plane）或格式 17（多 plane
    //            packed 行），工作项为（layer, block）。一个核搬运该页所有
    //            plane，避免两个核共享同一个 32B LMC 行。
    const int32_t plane_slots = separate_plane_ ? layout_.num_planes : 1;
    const int32_t active_blocks = blocks_per_obj_ - skip_blocks_;
    const int64_t work =
        static_cast<int64_t>(num_layers_) * plane_slots * active_blocks;
    for (int64_t w = AscendC::GetBlockIdx(); w < work;
         w += AscendC::GetBlockNum()) {
      const int32_t block =
          skip_blocks_ + static_cast<int32_t>(w % active_blocks);
      const int32_t plane =
          static_cast<int32_t>((w / active_blocks) % plane_slots);
      const int32_t layer = static_cast<int32_t>(
          w / (static_cast<int64_t>(active_blocks) * plane_slots));
      // Engine block id: data-dependent (lives in GM), so unlike skip it
      // cannot be folded into the launch. Producer contract: 0 <= id < nb.
      // 引擎块 ID 依赖 GM 中的数据，因此不能像 skip 一样折叠进 launch。
      // 生产者契约：0 <= id < nb。
      const int64_t engine_block = block_ids_[block];
      if (engine_block < 0 || engine_block >= engine_block_count_)
        continue;
      if (separate_plane_) {
        ProcessPlaneBlock<ToEngine>(layer, plane, block, engine_block);
      } else {
        for (int32_t p = 0; p < layout_.num_planes; ++p) {
          ProcessPlaneBlock<ToEngine>(layer, p, block, engine_block);
        }
      }
    }
  }

private:
  __aicore__ inline int64_t AlignUp32(int64_t bytes) const {
    return (bytes + 31) & ~int64_t(31);
  }

  __aicore__ inline int64_t MaxAlignedRowBytes() const {
    int64_t max_row = 0;
    for (int32_t p = 0; p < layout_.num_planes; ++p) {
      const int64_t row = AlignUp32(layout_.planes[p].payload_bytes);
      max_row = row > max_row ? row : max_row;
    }
    return max_row;
  }

  // one_pointer_per_layer_: ptrs[layer] plus
  // planes[p].engine_base_offset_bytes. Otherwise ptrs[layer * num_planes +
  // plane] (each view's storage_offset is already in the stored pointer).
  // one_pointer_per_layer_ 为 true 时使用 ptrs[layer] 加
  // planes[p].engine_base_offset_bytes；否则使用 ptrs[layer * num_planes +
  // plane]（每个视图的 storage_offset 已包含在存储的指针中）。
  // 该函数是设备侧指针表的唯一寻址入口。
  __aicore__ inline __gm__ uint8_t* ResolveEnginePlanePtr(int32_t layer,
                                                          int32_t plane) const {
    const int64_t index =
        one_pointer_per_layer_
            ? layer
            : static_cast<int64_t>(layer) * layout_.num_planes + plane;
    auto* base = reinterpret_cast<__gm__ uint8_t*>(paged_kv_ptrs_[index]);
    if (!one_pointer_per_layer_) {
      return base;
    }
    return base + layout_.planes[plane].engine_base_offset_bytes;
  }

  // LMC side: plane column offset + layer stride + in-object block offset.
  // The in-object block index (not the engine block id) addresses LMC
  // pages: object token order is contiguous regardless of engine placement.
  // LMC 侧地址由 plane 列偏移、层步长和对象内块偏移组成。LMC 页使用对象内
  // 块索引（而非引擎块 ID）寻址：无论引擎如何放置，对象内 token 顺序始终
  // 连续。
  __aicore__ inline int64_t
  LmcGlobalOffset(int32_t layer, int32_t block,
                  const kvcache_ops::PlaneLayout& plane) const {
    return plane.lmc_base_offset_bytes +
           static_cast<int64_t>(layer) * layout_.lmc_layer_stride_bytes +
           static_cast<int64_t>(block) * tokens_per_block_ *
               layout_.lmc_token_stride_bytes;
  }

  // GM -> UB strided rows. DataCopyExtParams: blockCount = rows,
  // blockLen = payload, srcStride = GM-side row gap in bytes,
  // dstStride = extra UB gap in 32B blocks (0: rows land at
  // AlignUp32(payload) pitch, matching the InitBuffer budget).
  // GM->UB 的跨步长行拷贝。DataCopyExtParams 中 blockCount = rows、
  // blockLen = payload、srcStride 为 GM 侧行间隔字节数，dstStride 为以 32B
  // 块计的额外 UB 间隔（0 表示行按 AlignUp32(payload) 间距落盘，与
  // InitBuffer 预算一致）。
  __aicore__ inline void CopyRowsToUb(AscendC::LocalTensor<uint8_t> ub,
                                      AscendC::GlobalTensor<uint8_t>& gm,
                                      int64_t offset, uint16_t rows,
                                      uint32_t bytes, int64_t row_stride) {
    AscendC::DataCopyExtParams copy{
        rows, bytes, static_cast<uint32_t>(row_stride - bytes), 0u, 0u};
    AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0u, 0u, 0u};
    AscendC::DataCopyPad(ub, gm[offset], copy, pad);
  }

  // UB -> GM strided rows: the stride roles swap. UB rows are read at
  // AlignUp32(payload) pitch (srcStride = 0); dstStride is the GM-side row
  // gap in bytes.
  // UB->GM 的跨步长行拷贝，步长角色互换。UB 行按 AlignUp32(payload) 间距
  // 读取（srcStride = 0）；dstStride 为 GM 侧行间隔字节数。
  __aicore__ inline void CopyUbToRows(AscendC::GlobalTensor<uint8_t>& gm,
                                      int64_t offset,
                                      AscendC::LocalTensor<uint8_t> ub,
                                      uint16_t rows, uint32_t bytes,
                                      int64_t row_stride) {
    AscendC::DataCopyExtParams copy{
        rows, bytes, 0u, static_cast<uint32_t>(row_stride - bytes), 0u};
    AscendC::DataCopyPad(gm[offset], ub, copy);
  }

  // Move one (layer, plane, block) between the engine page and the LMC
  // packed row, segmented to fit the UB budget.
  // 在引擎页与 LMC packed 行之间搬运一个（layer, plane, block），按 UB
  // 预算分段；ToEngine 决定 retrieve 或 store 的拷贝方向。
  template <bool ToEngine>
  __aicore__ inline void ProcessPlaneBlock(int32_t layer, int32_t plane_idx,
                                           int32_t block,
                                           int64_t engine_block) {
    const auto& plane = layout_.planes[plane_idx];
    // Engine side: dense token rows, so the token step equals the payload.
    // 引擎侧为紧凑 token 行，因此 token 步长等于 payload。
    auto* engine_ptr = ResolveEnginePlanePtr(layer, plane_idx) +
                       engine_block * plane.engine_block_stride_bytes;
    AscendC::GlobalTensor<uint8_t> engine;
    engine.SetGlobalBuffer(engine_ptr,
                           static_cast<uint64_t>(tokens_per_block_) *
                               plane.payload_bytes);
    const int64_t lmc_base = LmcGlobalOffset(layer, block, plane);

    // Fast path: the LMC row holds exactly this plane, the payload is a 32B
    // multiple and both GM bases are 32B aligned -> whole segment in one
    // DataCopy. Packed multi-plane rows always take the row path.
    // 快速路径：LMC 行恰好只包含此 plane，payload 是 32B 的整数倍且两个 GM
    // 基址均按 32B 对齐时，整个分段可只用一次 DataCopy。多 plane packed 行
    // 始终走行拷贝路径。
    const bool contiguous =
        layout_.lmc_token_stride_bytes == plane.payload_bytes &&
        plane.payload_bytes % 32 == 0 &&
        reinterpret_cast<uint64_t>(engine_ptr) % 32 == 0 &&
        reinterpret_cast<uint64_t>(lmcache_obj_base_ + lmc_base) % 32 == 0;

    for (int32_t start = 0; start < tokens_per_block_;
         start += tokens_per_segment_) {
      const int32_t remaining = tokens_per_block_ - start;
      const int32_t rows =
          remaining < tokens_per_segment_ ? remaining : tokens_per_segment_;
      const int64_t engine_offset =
          static_cast<int64_t>(start) * plane.payload_bytes;
      const int64_t lmc_offset = lmc_base + static_cast<int64_t>(start) *
                                                layout_.lmc_token_stride_bytes;

      auto ub = queue_.AllocTensor<uint8_t>();
      if (contiguous) {
        const uint32_t bytes =
            static_cast<uint32_t>(rows * plane.payload_bytes);
        if constexpr (ToEngine) {
          AscendC::DataCopy(ub, lmcache_obj_[lmc_offset], bytes);
        } else {
          AscendC::DataCopy(ub, engine[engine_offset], bytes);
        }
      } else {
        // Row path, GM -> UB: the source stride is the LMC packed row for
        // H2D (reading LMC), the dense payload for D2H (reading engine).
        // 行拷贝路径，GM->UB：H2D（读取 LMC）的源步长为 LMC packed 行宽，
        // D2H（读取引擎）的源步长为紧凑 payload。
        if constexpr (ToEngine) {
          CopyRowsToUb(ub, lmcache_obj_, lmc_offset, rows, plane.payload_bytes,
                       layout_.lmc_token_stride_bytes);
        } else {
          CopyRowsToUb(ub, engine, engine_offset, rows, plane.payload_bytes,
                       plane.payload_bytes);
        }
      }
      queue_.EnQue(ub);
      ub = queue_.DeQue<uint8_t>();
      if (contiguous) {
        const uint32_t bytes =
            static_cast<uint32_t>(rows * plane.payload_bytes);
        if constexpr (ToEngine) {
          AscendC::DataCopy(engine[engine_offset], ub, bytes);
        } else {
          AscendC::DataCopy(lmcache_obj_[lmc_offset], ub, bytes);
        }
      } else {
        // Row path, UB -> GM: same stride choice on the destination side.
        // 行拷贝路径，UB->GM：目标侧采用相同的步长选择。
        if constexpr (ToEngine) {
          CopyUbToRows(engine, engine_offset, ub, rows, plane.payload_bytes,
                       plane.payload_bytes);
        } else {
          CopyUbToRows(lmcache_obj_, lmc_offset, ub, rows, plane.payload_bytes,
                       layout_.lmc_token_stride_bytes);
        }
      }
      queue_.FreeTensor(ub);
    }
  }

  __gm__ uint64_t* paged_kv_ptrs_;   // see ResolveEnginePlanePtr
                                     // 见 ResolveEnginePlanePtr
  __gm__ int64_t* block_ids_;        // logical block -> engine physical block
                                     // 逻辑块到引擎物理块的映射
  __gm__ uint8_t* lmcache_obj_base_; // LMC object base (device VA, staged?)
                                     // LMC 对象基址（设备虚拟地址，可能为 staging）
  AscendC::GlobalTensor<uint8_t> lmcache_obj_;
  AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT,
                    kvcache_ops::kBlockTransferQueueDepth>
      queue_;
  kvcache_ops::BlockTransferLayout layout_;
  int32_t blocks_per_obj_, skip_blocks_, num_layers_, engine_block_count_,
      tokens_per_block_, tokens_per_segment_;
  bool separate_plane_;
  bool one_pointer_per_layer_;
};

} // namespace

extern "C" __global__ __aicore__ void multi_layer_block_transfer_kernel(
    GM_ADDR pointers, GM_ADDR object, GM_ADDR ids, int32_t blocks, int32_t skip,
    int32_t nl, int32_t nb, int32_t bs, bool separate_plane, bool to_engine,
    int32_t num_planes, int64_t lmc_row_bytes, int64_t lmc_layer_stride,
    int64_t lmc_object_bytes, int64_t p0_payload, int64_t p0_block_stride,
    int64_t p0_base, int64_t p1_payload, int64_t p1_block_stride,
    int64_t p1_base, int64_t p2_payload, int64_t p2_block_stride,
    int64_t p2_base, int64_t p3_payload, int64_t p3_block_stride,
    int64_t p3_base, int64_t ub_bytes, bool one_pointer_per_layer,
    // When one_pointer_per_layer, plane GM bases use engine_base_offset_bytes
    // on the layer entry; only the first two offsets are launch arguments.
    // Per-plane pointer tables keep engine_base_offset_bytes at zero.
    // one_pointer_per_layer 为 true 时，plane GM 基址使用层表项加
    // engine_base_offset_bytes；launch 参数只携带前两个偏移。按 plane 设置的
    // 指针表则将 engine_base_offset_bytes 保持为 0。
    int64_t p0_engine_base, int64_t p1_engine_base) {
  kvcache_ops::BlockTransferLayout layout = {};
  layout.num_planes = num_planes;
  layout.lmc_token_stride_bytes = lmc_row_bytes;
  layout.lmc_layer_stride_bytes = lmc_layer_stride;
  layout.lmc_object_bytes = lmc_object_bytes;
  const int64_t payloads[kvcache_ops::kMaxPlanes] = {p0_payload, p1_payload,
                                                     p2_payload, p3_payload};
  const int64_t block_strides[kvcache_ops::kMaxPlanes] = {
      p0_block_stride, p1_block_stride, p2_block_stride, p3_block_stride};
  const int64_t bases[kvcache_ops::kMaxPlanes] = {p0_base, p1_base, p2_base,
                                                  p3_base};
  for (int32_t p = 0; p < num_planes; ++p) {
    layout.planes[p] =
        kvcache_ops::PlaneLayout{payloads[p], block_strides[p], bases[p]};
  }
  layout.planes[0].engine_base_offset_bytes = p0_engine_base;
  layout.planes[1].engine_base_offset_bytes = p1_engine_base;
  AscendC::TPipe pipe;
  MultiLayerBlockTransfer op;
  op.Init(pointers, object, ids, blocks, skip, nl, nb, bs, separate_plane,
          layout, ub_bytes, &pipe, one_pointer_per_layer);
  if (to_engine) {
    op.Process<true>();
  } else {
    op.Process<false>();
  }
}

namespace kvcache_ops {

// host launcher 将 BlockTransferLayout 展开为设备 kernel 的固定 ABI 参数，
// 并在指定 stream 上启动；它只完成 native 入队，不等待设备执行结束。
void multi_layer_block_transfer_kernel(
    uint32_t block_dim, void* stream, uint8_t* paged_buffer_ptrs,
    uint8_t* lmcache_obj, uint8_t* engine_block_ids,
    int32_t num_blocks_per_object, int32_t skip_prefix_n_blocks, int32_t nl,
    int32_t nb, int32_t bs, bool separate_plane, BlockTransferLayout layout,
    bool to_engine, int64_t ub_bytes, bool one_pointer_per_layer) {
  ::multi_layer_block_transfer_kernel<<<block_dim, nullptr, stream>>>(
      paged_buffer_ptrs, lmcache_obj, engine_block_ids, num_blocks_per_object,
      skip_prefix_n_blocks, nl, nb, bs, separate_plane, to_engine,
      layout.num_planes, layout.lmc_token_stride_bytes,
      layout.lmc_layer_stride_bytes, layout.lmc_object_bytes,
      layout.planes[0].payload_bytes,
      layout.planes[0].engine_block_stride_bytes,
      layout.planes[0].lmc_base_offset_bytes, layout.planes[1].payload_bytes,
      layout.planes[1].engine_block_stride_bytes,
      layout.planes[1].lmc_base_offset_bytes, layout.planes[2].payload_bytes,
      layout.planes[2].engine_block_stride_bytes,
      layout.planes[2].lmc_base_offset_bytes, layout.planes[3].payload_bytes,
      layout.planes[3].engine_block_stride_bytes,
      layout.planes[3].lmc_base_offset_bytes, ub_bytes, one_pointer_per_layer,
      layout.planes[0].engine_base_offset_bytes,
      layout.planes[1].engine_base_offset_bytes);
}

} // namespace kvcache_ops
