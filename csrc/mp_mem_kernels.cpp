// SPDX-License-Identifier: Apache-2.0

// C++ host layer for the block-level MP-mode KV transfer (mirrors upstream
// LMCache csrc/mp_mem_kernels.cu).
//
// Design contract:
//   - prepare_group resolves a PageBufferShapeDesc + EngineKVFormat into the
//     flat BlockTransferLayout POD ONCE per group, with every static geometry
//     check. Both native entry points consume only PreparedGroup; the
//     device kernel receives the POD verbatim and never re-interprets the
//     format.
//   - validate_launch checks the per-transfer dynamic variables.
//   - The plan executor validates ALL groups and ALL launches before any
//     staging copy or kernel launch is enqueued: an unsupported group
//     fails with "nothing started", never half-enqueued.
//   - The kernel is a pure byte mover; there is no packed/two-plane special
//     case and no dtype dispatch left in this file.
// MP 模式 block 级 KV 传输的 C++ host 层（与上游 LMCache
// csrc/mp_mem_kernels.cu 一致）。
//
// 设计契约：
//   - prepare_group 将 PageBufferShapeDesc 和 EngineKVFormat 解析为扁平的
//     BlockTransferLayout POD，每组仅解析一次，并完成所有静态几何检查。
//     两个 native 入口都只使用 PreparedGroup；设备 kernel 原样接收 POD，
//     不再解释格式。
//   - validate_launch 校验每次传输的动态变量。
//   - 计划执行器在任何 staging 拷贝或 kernel 入队前校验所有 group 和
//     launch：不支持的 group 会在“尚未开始任何操作”时失败，不会只入队一半。
//   - kernel 只负责字节搬运；此文件不保留 packed/双 plane 特例或 dtype 分派。

#include "mp_mem_kernels.h"

#include "kernels/multi_layer/multi_layer_block_mem_kernels.h"
#include "utils.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

// DataCopy granularity on the GM side: segment base addresses and segment
// byte lengths must both be multiples of 32B.
// GM 侧 DataCopy 的粒度：段基地址和段字节长度都必须是 32B 的整数倍。
constexpr int64_t kGmAlignBytes = 32;
// Single UB segment budget: the depth-2 queue must hold two segments, so one
// segment may use at most half of the total budget.
// 单个 UB 段预算：深度为 2 的队列必须能容纳两个段，因此单段最多使用
// 总预算的一半。
constexpr int64_t kUbSegmentBytes =
    kvcache_ops::kBlockTransferUbBytes / kvcache_ops::kBlockTransferQueueDepth;
// DataCopyPad GM-side gap fields are uint32.
// DataCopyPad 的 GM 侧间隔字段为 uint32。
constexpr int64_t kCopyGmGapMax =
    static_cast<int64_t>(std::numeric_limits<uint32_t>::max());

int64_t align_up32(int64_t v) { return (v + 31) & ~int64_t(31); }

int64_t checked_mul(int64_t a, int64_t b, const char *what) {
  TORCH_CHECK(a >= 0 && b >= 0, what, " must be non-negative, got ", a, " * ",
              b);
  if (a != 0) {
    TORCH_CHECK(b <= std::numeric_limits<int64_t>::max() / a, what,
                " overflows int64: ", a, " * ", b);
  }
  return a * b;
}

int64_t checked_add(int64_t a, int64_t b, const char *what) {
  TORCH_CHECK(a >= 0 && b >= 0 && b <= std::numeric_limits<int64_t>::max() - a,
              what, " overflows int64: ", a, " + ", b);
  return a + b;
}

// Usable per-AIV UB staging budget, queried from the platform: GetCoreMemSize
// reports the PHYSICAL UB size. Returns 0 on failure -- callers then use
// the kernel's built-in 128KB default.
// 从平台查询每个 AIV 可用的 UB staging 预算：GetCoreMemSize 报告物理 UB
// 大小。查询失败时返回 0，调用方随后使用 kernel 内置的 128KB 默认值。
int64_t query_ub_budget_bytes() {
  auto *platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(aclrtGetSocName());
  if (platform == nullptr) {
    return 0;
  }
  uint64_t ub_size = 0;
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  return ub_size == 0 ? 0 : static_cast<int64_t>(ub_size);
}

// ---------------------------------------------------------------------------
// prepare_group: static geometry, resolved once per group.
// ---------------------------------------------------------------------------
// prepare_group：静态几何信息，每组只解析一次。

// 保存 host 端预解析结果，包括设备侧字节布局、指针表寻址方式和 UB 预算。
struct PreparedGroup {
  kvcache_ops::BlockTransferLayout layout{};
  bool separate_plane = false; // format 16: (layer, plane, block) work items
                               // 格式 16：工作项为（layer, plane, block）
  bool one_pointer_per_layer = false; // format 2: ptrs[layer] + plane offset
                                      // 格式 2：ptrs[layer] 加 plane 偏移
  int32_t nl = 0;
  int32_t nb = 0;
  int32_t bs = 0;
  int64_t slots_per_object = 0;
  int64_t ub_bytes = 0; // queried UB budget; 0 = kernel default
                        // 查询到的 UB 预算；0 表示使用 kernel 默认值
};

// 将形状描述和引擎格式归一化为设备布局，集中校验 plane、步长、地址范围及
// UB 分段约束；后续 launch 不再重复解释这些静态格式。
PreparedGroup prepare_group(const PageBufferShapeDesc &shape_desc,
                            EngineKVFormat engine_kv_format,
                            int64_t slots_per_object, int64_t ub_bytes) {
  PageBufferShapeDesc sd = shape_desc;
  // Format 2 is one [NB, 2, BS, NH, HS] tensor per layer. The pointer table
  // stays one base pointer per layer; plane addresses are base + offset.
  // Axis 1 is two equal slabs (semantic K/V, or opaque page halves).
  // Its LMC geometry matches format 16.
  // 格式 2 每层是一个 [NB, 2, BS, NH, HS] Tensor。指针表仍按每层一个基址，
  // plane 地址为基址加偏移。第 1 轴是两个等大的 slab（语义上的 K/V，或
  // 不透明页的两半）。其 LMC 几何布局与格式 16 相同。
  const bool fmt2 = engine_kv_format == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS;
  const bool fmt13 = engine_kv_format == EngineKVFormat::NL_X_NB_BS_NH_CS;
  const bool separate =
      engine_kv_format == EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS || fmt2;
  const bool fmt17 = engine_kv_format == EngineKVFormat::NL_X_NP_X_NB_BS_ONE_HS;
  TORCH_CHECK(fmt13 || separate || fmt17,
              "LMCache-Ascend block-level MP transfer currently supports "
              "NL_X_NB_TWO_BS_NH_HS (2), NL_X_NB_BS_NH_CS (13), "
              "NL_X_TWO_X_NB_BS_NH_HS (16), and "
              "NL_X_NP_X_NB_BS_ONE_HS (17), "
              "got ",
              static_cast<int>(engine_kv_format));
  TORCH_CHECK(sd.nl > 0 && sd.nb > 0 && sd.bs > 0 && sd.nh > 0 && sd.hs > 0,
              "shape descriptor dims must be positive, got nl=", sd.nl,
              " nb=", sd.nb, " bs=", sd.bs, " nh=", sd.nh, " hs=", sd.hs);
  TORCH_CHECK(sd.element_size == 1 || sd.element_size == 2 ||
                  sd.element_size == 4,
              "element_size must be 1, 2 or 4, got ", sd.element_size);
  TORCH_CHECK(sd.kv_size == (separate ? 2 : 1), "kv_size must be ",
              (separate ? 2 : 1), " for format ",
              static_cast<int>(engine_kv_format), ", got ", sd.kv_size);

  const int64_t scalar_row =
      checked_mul(checked_mul(sd.nh, sd.hs, "nh * hs"), sd.element_size,
                  "LMC scalar row bytes");
  if (fmt13 && sd.num_planes == 0) {
    // Format 13 is one fused [NB, BS, NH, CS] tensor per layer. Its plane
    // geometry is fully determined by the scalar descriptor fields, so no
    // per-plane tensor introspection is needed during registration.
    // 格式 13 每层是一个融合的 [NB, BS, NH, CS] Tensor。其 plane 几何完全
    // 由标量描述字段决定，因此注册时无需逐个 plane 检查 Tensor。
    sd.num_planes = 1;
    sd.plane_slot_bytes[0] = scalar_row;
    sd.plane_block_stride_bytes[0] =
        sd.block_stride_elems > 0
            ? checked_mul(sd.block_stride_elems, sd.element_size,
                          "format 13 block stride bytes")
            : checked_mul(sd.bs, scalar_row, "format 13 block stride bytes");
  }

  // Format 2 is one [NB, 2, BS, NH, HS] tensor. The scalar descriptor
  // fixes both plane addresses, so fill them here.
  // 格式 2 是一个 [NB, 2, BS, NH, HS] Tensor。标量描述符已确定两个 plane
  // 的地址，因此在此填充。
  if (fmt2) {
    const int64_t row =
        checked_mul(checked_mul(static_cast<int64_t>(sd.nh), sd.hs, "nh * hs"),
                    sd.element_size, "format 2 row bytes");
    const int64_t half = checked_mul(sd.bs, row, "format 2 half page");
    const int64_t tight_elems = checked_mul(
        int64_t{2},
        checked_mul(sd.bs,
                    checked_mul(static_cast<int64_t>(sd.nh), sd.hs, "nh * hs"),
                    "bs * nh * hs"),
        "format 2 tight block elems");
    const int64_t block_elems =
        sd.block_stride_elems > 0 ? sd.block_stride_elems : tight_elems;
    const int64_t block_bytes = checked_mul(block_elems, sd.element_size,
                                            "format 2 block stride bytes");
    sd.num_planes = 2;
    sd.plane_slot_bytes[0] = sd.plane_slot_bytes[1] = row;
    sd.plane_block_stride_bytes[0] = sd.plane_block_stride_bytes[1] =
        block_bytes;
    sd.plane_engine_base_offset_bytes[0] = 0;
    sd.plane_engine_base_offset_bytes[1] = half;
  }

  // Formats 16/17 must carry explicit plane geometry filled at registration;
  // format 13 derives it above when num_planes == 0.
  // 格式 16/17 必须携带注册时填充的显式 plane 几何；格式 13 在
  // num_planes == 0 时由上述逻辑推导。
  TORCH_CHECK(sd.num_planes >= 1 && sd.num_planes <= kvcache_ops::kMaxPlanes,
              "num_planes must be in [1, ", kvcache_ops::kMaxPlanes,
              "] after resolving plane geometry, got ", sd.num_planes);
  if (separate) {
    TORCH_CHECK(
        sd.num_planes == 2,
        "separate-plane formats require exactly 2 physical planes, got ",
        sd.num_planes);
  }
  if (fmt13) {
    TORCH_CHECK(sd.num_planes == 1,
                "format 13 requires exactly 1 physical plane, got ",
                sd.num_planes);
  }
  if (fmt17) {
    TORCH_CHECK(sd.nh == 1,
                "format 17 requires one head per plane, got nh=", sd.nh);
  }
  TORCH_CHECK(slots_per_object > 0, "slots per object must be positive, got ",
              slots_per_object);

  PreparedGroup group;
  group.layout.num_planes = sd.num_planes;
  // Per-segment budget: host-queried UB (query_ub_budget_bytes) or the
  // built-in default; the kernel receives the same ub_bytes.
  // 单段预算：使用 host 查询的 UB（query_ub_budget_bytes）或内置默认值；
  // kernel 接收相同的 ub_bytes。
  const int64_t ub_segment =
      ub_bytes > 0 ? ub_bytes / kvcache_ops::kBlockTransferQueueDepth
                   : kUbSegmentBytes;
  int64_t prefix = 0;
  for (int32_t p = 0; p < sd.num_planes; ++p) {
    const int64_t payload = sd.plane_slot_bytes[p];
    const int64_t block_stride = sd.plane_block_stride_bytes[p];
    TORCH_CHECK(payload > 0, "plane ", p, " payload must be positive, got ",
                payload);
    TORCH_CHECK(align_up32(payload) <= ub_segment, "plane ", p,
                " aligned row (", align_up32(payload),
                " bytes) exceeds the per-segment UB budget (", ub_segment, ")");
    const int64_t span = checked_mul(sd.bs, payload, "plane block span");
    TORCH_CHECK(block_stride >= span, "plane ", p, " block stride (",
                block_stride, ") is below the dense block span (", span,
                "); blocks would overlap");
    TORCH_CHECK(block_stride % kGmAlignBytes == 0, "plane ", p,
                " engine block stride (", block_stride,
                " bytes) must be a multiple of ", kGmAlignBytes,
                " for DataCopy alignment");
    // Address-range overflow probe: (nb - 1) * stride + span must stay
    // inside int64.
    // 地址范围溢出检查：(nb - 1) * stride + span 必须仍在 int64 范围内。
    checked_add(checked_mul(sd.nb - 1, block_stride, "plane address range"),
                span, "plane address range");
    if (!fmt17) {
      TORCH_CHECK(payload == scalar_row, "plane ", p, " payload (", payload,
                  ") must equal the full scalar row (", scalar_row,
                  ") for format ", static_cast<int>(engine_kv_format));
    }
    // LMC packed-row gap = bytes of the OTHER planes in the row; it feeds a
    // uint32 DataCopyPad field (engine-side gap is always 0: dense rows).
    // LMC packed 行间隔等于同一行中其他 plane 的字节数；该值写入 uint32
    // DataCopyPad 字段（引擎侧为紧凑行，间隔始终为 0）。
    TORCH_CHECK(scalar_row >= payload && scalar_row - payload <= kCopyGmGapMax,
                "plane ", p, " LMC row gap (", scalar_row - payload,
                ") exceeds the DataCopyPad GM gap range");
    const int64_t base_offset = sd.plane_engine_base_offset_bytes[p];
    TORCH_CHECK(base_offset >= 0, "plane ", p,
                " engine base offset must be non-negative, got ", base_offset);
    if (fmt2) {
      const int64_t reach =
          checked_add(base_offset, span, "format 2 plane reach");
      TORCH_CHECK(reach <= block_stride, "format 2 plane ", p,
                  " offset + block span (", reach,
                  ") exceeds the block stride (", block_stride, ")");
      // The probe above starts at offset 0. Format 2 also reaches base_offset.
      // 上述检查从偏移 0 开始；格式 2 的访问范围还会到达 base_offset。
      checked_add(checked_mul(sd.nb - 1, block_stride, "plane address range"),
                  reach, "plane address range");
    } else {
      TORCH_CHECK(base_offset == 0, "plane ", p,
                  " engine base offset is only valid for format 2, got ",
                  base_offset);
    }
    group.layout.planes[p] =
        kvcache_ops::PlaneLayout{payload, block_stride, fmt17 ? prefix : 0};
    group.layout.planes[p].engine_base_offset_bytes = base_offset;
    prefix = checked_add(prefix, payload, "LMC row prefix sum");
  }
  if (fmt2) {
    TORCH_CHECK(sd.plane_block_stride_bytes[0] ==
                    sd.plane_block_stride_bytes[1],
                "format 2 planes must share one block stride, got ",
                sd.plane_block_stride_bytes[0], " and ",
                sd.plane_block_stride_bytes[1]);
    const int64_t o0 = sd.plane_engine_base_offset_bytes[0];
    const int64_t o1 = sd.plane_engine_base_offset_bytes[1];
    const int64_t span0 =
        checked_mul(sd.bs, sd.plane_slot_bytes[0], "format 2 plane 0 span");
    const int64_t span1 =
        checked_mul(sd.bs, sd.plane_slot_bytes[1], "format 2 plane 1 span");
    const int64_t first = o0 <= o1 ? o0 : o1;
    const int64_t first_span = o0 <= o1 ? span0 : span1;
    const int64_t second = o0 <= o1 ? o1 : o0;
    TORCH_CHECK(second >= checked_add(first, first_span, "format 2 plane gap"),
                "format 2 K/V planes overlap: offsets ", o0, " and ", o1);
  }
  if (fmt17) {
    TORCH_CHECK(prefix == scalar_row, "sum of plane payloads (", prefix,
                ") must equal nh * hs * element_size (", scalar_row,
                ") for format 17");
  }

  group.layout.lmc_token_stride_bytes = scalar_row;
  group.layout.lmc_layer_stride_bytes =
      checked_mul(slots_per_object, scalar_row, "LMC layer stride bytes");
  const int64_t slab =
      checked_mul(sd.nl, group.layout.lmc_layer_stride_bytes, "LMC slab bytes");
  group.layout.lmc_object_bytes =
      checked_mul(separate ? 2 : 1, slab, "LMC object bytes");
  if (separate) {
    // 2LTD: K slab first, V slab after (planes[1] base = one full slab).
    // 2LTD：K slab 在前，V slab 在后（planes[1] 基址偏移为一个完整 slab）。
    group.layout.planes[1].lmc_base_offset_bytes = slab;
  }
  // LMC pages must not share 32B lines.
  // 不同 LMC 页不得共享同一个 32B 行。
  TORCH_CHECK(
      checked_mul(sd.bs, scalar_row, "LMC page bytes") % kGmAlignBytes == 0,
      "BS * LMC row bytes (", sd.bs * scalar_row, ") must be a multiple of ",
      kGmAlignBytes, "; otherwise adjacent LMC pages share a 32B line");

  group.separate_plane = separate;
  group.one_pointer_per_layer = fmt2;
  group.nl = sd.nl;
  group.nb = sd.nb;
  group.bs = sd.bs;
  group.slots_per_object = slots_per_object;
  group.ub_bytes = ub_bytes;
  return group;
}

// ---------------------------------------------------------------------------
// validate_launch: per-transfer dynamic variables.
// ---------------------------------------------------------------------------
// validate_launch：每次传输的动态变量。

// 保存已通过校验的 launch 动态参数，供后续计算有效工作项和 block_ids 切片。
struct CheckedLaunch {
  int32_t total_blocks = 0;
  int32_t num_objects = 0;
  int32_t blocks_per_object = 0;
  int32_t skip_prefix_n_blocks = 0;
};

// 校验对象数量、每对象块数、block_ids 切片容量及 skip 前缀范围；skip 表示
// 每个对象开头无需传输的块数，但不改变 block_ids 中逻辑块的位置。
CheckedLaunch validate_launch(const PreparedGroup &group, int64_t total_blocks,
                              int num_objects, int64_t block_ids_offset,
                              int64_t block_ids_capacity,
                              int skip_prefix_n_blocks) {
  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1-4 LMCache objects, got ", num_objects);
  TORCH_CHECK(total_blocks >= 0, "total_blocks must be non-negative, got ",
              total_blocks);
  TORCH_CHECK(total_blocks % num_objects == 0, "block_ids length (",
              total_blocks, ") must be divisible by num_objects (", num_objects,
              ")");
  const int64_t blocks = total_blocks / num_objects;
  TORCH_CHECK(checked_mul(blocks, group.bs, "blocks * bs") ==
                  group.slots_per_object,
              "blocks_per_object * block_size (", blocks * group.bs,
              ") must equal slots per object (", group.slots_per_object, ")");
  TORCH_CHECK(skip_prefix_n_blocks >= 0 && skip_prefix_n_blocks <= blocks,
              "skip_prefix_n_blocks (", skip_prefix_n_blocks,
              ") must be within [0, ", blocks, "]");
  // Subtraction form: offset + length itself may not overflow.
  // 使用减法形式校验，避免 offset + length 本身发生溢出。
  TORCH_CHECK(block_ids_offset >= 0 && block_ids_offset <= block_ids_capacity,
              "block_ids_offset (", block_ids_offset, ") must be within [0, ",
              block_ids_capacity, "]");
  TORCH_CHECK(total_blocks <= block_ids_capacity - block_ids_offset,
              "block_ids slice [", block_ids_offset, ", ",
              block_ids_offset + total_blocks, ") exceeds block_ids capacity ",
              block_ids_capacity);
  TORCH_CHECK(total_blocks <= std::numeric_limits<int32_t>::max() &&
                  blocks <= std::numeric_limits<int32_t>::max(),
              "block counts exceed the kernel's int32 launch arguments");

  CheckedLaunch out;
  out.total_blocks = static_cast<int32_t>(total_blocks);
  out.num_objects = num_objects;
  out.blocks_per_object = static_cast<int32_t>(blocks);
  out.skip_prefix_n_blocks = skip_prefix_n_blocks;
  return out;
}

// ---------------------------------------------------------------------------
// Launch: one kernel launch per object.
// ---------------------------------------------------------------------------
// Launch：每个对象启动一次 kernel。

// 按“层、plane（可选）、有效块”计算工作项，并逐对象提交 kernel；skip 前缀
// 由设备索引映射处理，launch 只覆盖实际需要搬运的块。
void launch_prepared_objects(uint32_t aiv_num, void *stream,
                             const PreparedGroup &group,
                             uint8_t *paged_buffer_ptrs,
                             const std::vector<int64_t> &obj_device_ptrs,
                             int64_t *block_ids_base,
                             const CheckedLaunch &launch, bool to_engine) {
  // blockDim matches the ACTIVE work-item count (the kernel folds skip into
  // its index map). skip == blocks -> nothing to transfer; return early
  // because <<<0>>> would be an invalid launch.
  // blockDim 与有效工作项数量匹配（kernel 将 skip 折叠进索引映射）。
  // skip == blocks 时没有数据需要传输，因此提前返回，避免无效的 <<<0>>>。
  const int32_t plane_slots =
      group.separate_plane ? group.layout.num_planes : 1;
  const int64_t active_blocks =
      launch.blocks_per_object - launch.skip_prefix_n_blocks;
  const int64_t work =
      static_cast<int64_t>(group.nl) * plane_slots * active_blocks;
  if (work == 0) {
    return;
  }
  const uint32_t blockDim =
      static_cast<uint32_t>(std::min<int64_t>(aiv_num, work));
  for (int32_t i = 0; i < launch.num_objects; ++i) {
    uint8_t *engine_block_ids = reinterpret_cast<uint8_t *>(
        block_ids_base + static_cast<int64_t>(i) * launch.blocks_per_object);
    kvcache_ops::multi_layer_block_transfer_kernel(
        blockDim, stream, paged_buffer_ptrs,
        reinterpret_cast<uint8_t *>(obj_device_ptrs[i]), engine_block_ids,
        launch.blocks_per_object, launch.skip_prefix_n_blocks, group.nl,
        group.nb, group.bs, group.separate_plane, group.layout, to_engine,
        group.ub_bytes, group.one_pointer_per_layer);
  }
}

// ---------------------------------------------------------------------------
// LMC object device-VA resolution and staging (direct entry).
// ---------------------------------------------------------------------------
// LMC 对象的设备虚拟地址解析与 staging（直接入口）。

// CUDA UVA lets the kernel consume host data_ptr()s. Ascend DataCopy needs a
// device VA: registered aclrtMallocHost (get_device_ptr) or a true NPU
// allocation. MixedMemoryAllocator SHM L1 is neither — stage on-device and
// aclrtMemcpy.
// CUDA UVA 允许 kernel 直接使用 host data_ptr()；Ascend DataCopy 则需要设备
// 虚拟地址：已注册 aclrtMallocHost 的映射地址（get_device_ptr）或真正的 NPU
// 分配地址。MixedMemoryAllocator 的 SHM L1 两者都不是，需要先在设备侧
// staging，再执行 aclrtMemcpy。
// 普通 CPU 对象由调用方持有，需复制到 NPU staging；已注册锁页对象同样由
// 调用方持有，但可直接使用其设备映射；NPU 对象直接使用原设备地址。未注册
// 的锁页 host 指针也走 staging，三类外部对象的所有权都不会转移给本层。
bool is_npu_memory_ptr(void *ptr) {
  aclrtPtrAttributes attributes{};
  const aclError ret = aclrtPointerGetAttributes(ptr, &attributes);
  if (ret != ACL_ERROR_NONE) {
    return false;
  }
  return attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE;
}

std::vector<int64_t> device_lmc_ptrs(const std::vector<int64_t> &ptrs) {
  std::vector<int64_t> out;
  out.reserve(ptrs.size());
  for (int64_t p : ptrs) {
    void *mapped =
        get_device_ptr(reinterpret_cast<void *>(static_cast<uintptr_t>(p)));
    out.push_back(mapped != nullptr ? reinterpret_cast<int64_t>(mapped) : p);
  }
  return out;
}

// NPU staging 的拥有者：buf 持有设备分配，host_ptr 只是外部 host 对象的
// 非拥有地址。stage 由 PreparedLmcPtrs 集中保存，异步生命周期遵循调用方的
// stream 同步契约。
struct HostStage {
  torch::Tensor buf;
  int64_t host_ptr = 0;
  size_t nbytes = 0;
};

// Resolve LMCache object pointers to device to device virtual addresses.
// SHM/host objects that are neither registered aclrtMallocHost nor NPU memory
// get an on-device staging tensor (HostStage); the caller memcpy-asyncs around
// the kernel. Allocations happen BEFORE any enqueue; the caller's cache context
// synchronizes the stream before releasing them, so the staging tensors stay
// valid for the async copies.
// 将 LMCache 对象指针解析为设备虚拟地址。既不是已注册 aclrtMallocHost、也不
// 是 NPU 内存的 SHM/host 对象会获得设备侧 staging Tensor（HostStage），调用方
// 在 kernel 前后安排异步 memcpy。所有分配都先于任何入队完成；调用方的 cache
// context 在释放前同步 stream，从而保证 staging Tensor 对异步拷贝保持有效。
// kernel_obj_ptrs 是供 kernel 使用的地址表，host_stages 则持有需要保活的
// staging 资源；普通 CPU、锁页 host 与 NPU 外部对象仍由调用方负责释放。
struct PreparedLmcPtrs {
  std::vector<int64_t> kernel_obj_ptrs;
  std::vector<HostStage> host_stages;
};

// 按“已注册锁页映射、NPU 原生地址、其他 host 地址”三类解析对象；仅第三类
// 创建 staging，并统一返回 kernel 地址表和 staging 所有权容器。
PreparedLmcPtrs prepare_lmc_ptrs(const std::vector<int64_t> &lmc_obj_ptrs,
                                 const torch::Device &device,
                                 const PreparedGroup &group) {
  PreparedLmcPtrs prepared;
  prepared.kernel_obj_ptrs.reserve(lmc_obj_ptrs.size());
  const auto staging_opts =
      torch::TensorOptions().dtype(torch::kUInt8).device(device);
  // One capacity formula for every layout.
  // 所有布局共用同一个对象容量计算公式。
  const int64_t object_bytes = group.layout.lmc_object_bytes;
  TORCH_CHECK(object_bytes > 0, "LMCache object byte size must be positive");
  for (int64_t p : lmc_obj_ptrs) {
    void *raw = reinterpret_cast<void *>(static_cast<uintptr_t>(p));
    void *mapped = get_device_ptr(raw);
    if (mapped != nullptr) {
      prepared.kernel_obj_ptrs.push_back(reinterpret_cast<int64_t>(mapped));
      continue;
    }
    if (is_npu_memory_ptr(raw)) {
      prepared.kernel_obj_ptrs.push_back(p);
      continue;
    }
    // A torch-pinned host pointer may not have a mapping in our registry.
    // torch 锁页 host 指针可能并未在我们的注册表中建立映射。
    HostStage stage;
    stage.buf = torch::empty({object_bytes}, staging_opts);
    stage.host_ptr = p;
    stage.nbytes = static_cast<size_t>(object_bytes);
    prepared.kernel_obj_ptrs.push_back(
        reinterpret_cast<int64_t>(stage.buf.data_ptr()));
    prepared.host_stages.push_back(std::move(stage));
  }
  return prepared;
}

// Enqueue HostStage memcpys + the block kernel(s) on ``stream``: H2D stages
// before the kernel, D2H stages after, and a partial D2H store first reads
// the host object into staging so the untouched prefix is preserved when
// the whole object is written back.
// 在 stream 上依次入队 HostStage memcpy 和 block kernel：H2D retrieve 在
// kernel 前将 host 数据搬入 staging，D2H store 在 kernel 后写回 host；带 skip
// 前缀的部分 D2H store 还会先读出完整 host 对象，确保整体写回时保留未触及
// 的前缀。本函数只负责 native 入队，成功返回不表示传输完成；调用方必须同步
// stream 后再访问结果或释放 host 对象与 staging 相关资源。
int enqueue_block_transfer(void *stream, uint32_t aiv_num,
                           uint8_t *paged_buffer_ptrs,
                           const PreparedGroup &group,
                           const PreparedLmcPtrs &prepared,
                           int64_t *block_ids_base, const CheckedLaunch &launch,
                           bool to_engine) {
  if (to_engine || launch.skip_prefix_n_blocks > 0) {
    for (const auto &stage : prepared.host_stages) {
      const aclError ret =
          aclrtMemcpyAsync(stage.buf.data_ptr(), stage.nbytes,
                           reinterpret_cast<const void *>(
                               static_cast<uintptr_t>(stage.host_ptr)),
                           stage.nbytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
      if (ret != ACL_ERROR_NONE) {
        return static_cast<int>(ret);
      }
    }
  }
  launch_prepared_objects(aiv_num, stream, group, paged_buffer_ptrs,
                          prepared.kernel_obj_ptrs, block_ids_base, launch,
                          to_engine);
  if (!to_engine) {
    for (const auto &stage : prepared.host_stages) {
      const aclError ret = aclrtMemcpyAsync(
          reinterpret_cast<void *>(static_cast<uintptr_t>(stage.host_ptr)),
          stage.nbytes, stage.buf.data_ptr(), stage.nbytes,
          ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        return static_cast<int>(ret);
      }
    }
  }
  return 0;
}

} // namespace

// 直接 native 入口：先完成静态/动态校验和 LMCache 地址准备，再在当前 NPU
// stream 上提交 staging 与 kernel；H2D 为 retrieve，D2H 为 store。
void multi_layer_block_kv_transfer(
    const torch::Tensor &paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor &block_ids,
    const torch::Device &device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  // Device guard first: the UB query below needs the current device's SoC.
  // 首先设置设备 guard：下面的 UB 查询需要当前设备的 SoC。
  const c10::OptionalDeviceGuard device_guard(device);
  // --- Static geometry (prepare_group) + dynamic variables (validate_launch)
  // --- 静态几何（prepare_group）与动态变量（validate_launch）
  const PreparedGroup group =
      prepare_group(shape_desc, engine_kv_format, lmcache_chunk_size,
                    query_ub_budget_bytes());
  const int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  const int64_t total_blocks = block_ids.size(0);
  const CheckedLaunch launch = validate_launch(
      group, total_blocks, num_objects,
      /*block_ids_offset=*/0,
      /*block_ids_capacity=*/total_blocks, skip_prefix_n_blocks);

  TORCH_CHECK(paged_buffer_ptrs_tensor.scalar_type() == at::kLong,
              "paged_buffer_ptrs_tensor must be int64");
  TORCH_CHECK(paged_buffer_ptrs_tensor.is_privateuseone(),
              "paged_buffer_ptrs_tensor must live on the NPU");
  TORCH_CHECK(paged_buffer_ptrs_tensor.dim() == 1,
              "paged_buffer_ptrs_tensor must be one-dimensional");
  TORCH_CHECK(paged_buffer_ptrs_tensor.is_contiguous(),
              "paged_buffer_ptrs_tensor must be contiguous");
  if (group.one_pointer_per_layer) {
    TORCH_CHECK(paged_buffer_ptrs_tensor.numel() == group.nl,
                "paged_buffer_ptrs_tensor must contain nl pointers: expected ",
                group.nl, ", got ", paged_buffer_ptrs_tensor.numel());
  } else {
    // Pointer table: one device pointer per (layer, physical plane).
    // 指针表：每个（layer, physical plane）组合对应一个设备指针。
    const int64_t expected_ptrs =
        static_cast<int64_t>(group.layout.num_planes) * group.nl;
    TORCH_CHECK(paged_buffer_ptrs_tensor.numel() == expected_ptrs,
                "paged_buffer_ptrs_tensor must contain num_planes * nl "
                "pointers: expected ",
                expected_ptrs, ", got ", paged_buffer_ptrs_tensor.numel());
  }
  // block_ids 是 NPU 上的 int64 逻辑块到引擎物理块映射，按对象连续分段；
  // 本层不拥有该 Tensor，调用方须使其在 stream 同步完成前保持有效。
  TORCH_CHECK(block_ids.is_privateuseone(), "block_ids must live on the NPU");
  TORCH_CHECK(block_ids.scalar_type() == at::kLong,
              "block_ids must have dtype int64");
  TORCH_CHECK(block_ids.dim() == 1, "block_ids must be one-dimensional");
  TORCH_CHECK(block_ids.is_contiguous(), "block_ids must be contiguous");

  const bool to_engine = (direction == TransferDirection::H2D);

  PreparedLmcPtrs prepared =
      prepare_lmc_ptrs(lmcache_objects_ptrs, device, group);

  uint8_t *paged_buffer_ptrs =
      static_cast<uint8_t *>(paged_buffer_ptrs_tensor.data_ptr());
  int64_t *block_ids_base = block_ids.data_ptr<int64_t>();

  // .stream() drains torch_npu's ACL task queue before returning the raw
  // stream. The drain is per-call by contract: each call must keep its direct
  // enqueues ordered after torch ops issued since the last drain. This op can
  // be invoked in a loop (per-block fallback paths in callers); the first
  // call flushes the queue and the following ones find it empty, so the
  // repeated drain is a cheap empty-queue check.
  // .stream() 在返回原始 stream 前排空 torch_npu 的 ACL 任务队列。按契约每次
  // 调用都要排空：本次直接入队的操作必须排在上次排空后提交的 torch op 之后。
  // 该操作可能被循环调用（调用方的逐 block 回退路径）；第一次调用刷新队列，
  // 后续调用发现队列为空，因此重复排空只是一次低开销的空队列检查。
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  // Direct launch (CUDA cudaLaunchKernel style): the kernel + staging memcpys
  // below are enqueued on the calling thread, no OpCommand / queue-thread hop.
  // 直接启动（CUDA cudaLaunchKernel 风格）：下面的 kernel 和 staging memcpy
  // 都在调用线程入队，不经过 OpCommand 或队列线程跳转。
  const char *socName = aclrtGetSocName();
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  const uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
  const int ret =
      enqueue_block_transfer(stream, aiv_num, paged_buffer_ptrs, group,
                             prepared, block_ids_base, launch, to_engine);
  TORCH_CHECK(ret == ACL_ERROR_NONE,
              "multi_layer_block_transfer staging failed, aclrtMemcpyAsync "
              "ret=",
              ret);
}

void lmcache_memcpy_async_on_stream(uintptr_t dest, uintptr_t src,
                                    size_t nbytes, TransferDirection direction,
                                    size_t host_buffer_offset,
                                    size_t host_buffer_alignments,
                                    aclrtStream stream) {
  TORCH_CHECK(host_buffer_alignments > 0 &&
                  (host_buffer_alignments & (host_buffer_alignments - 1)) == 0,
              "host_buffer_alignments must be a non-zero power of two");

  size_t offset = 0;
  const size_t mask = host_buffer_alignments - 1;
  const aclrtMemcpyKind kind = (direction == TransferDirection::H2D)
                                   ? ACL_MEMCPY_HOST_TO_DEVICE
                                   : ACL_MEMCPY_DEVICE_TO_HOST;

  while (offset < nbytes) {
    const size_t aligned_area_end =
        ((offset + host_buffer_offset) & ~mask) + host_buffer_alignments;
    const size_t real_end =
        std::min<size_t>(host_buffer_offset + nbytes, aligned_area_end);
    const size_t max_nbytes = real_end - offset - host_buffer_offset;

    const aclError ret = aclrtMemcpyAsync(
        reinterpret_cast<void *>(dest + offset), max_nbytes,
        reinterpret_cast<const void *>(src + offset), max_nbytes, kind, stream);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMemcpyAsync failed: ret=", ret);

    offset += max_nbytes;
  }
}

void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments) {
  lmcache_memcpy_async_on_stream(dest, src, nbytes, direction,
                                 host_buffer_offset, host_buffer_alignments,
                                 c10_npu::getCurrentNPUStream().stream());
}

// 执行对象组计划：先预校验全部 group、staging 和 launch，再按 BatchStep 顺序
// 提交。H2D retrieve 先 host->临时 NPU 缓冲区再执行 kernel，D2H store 先执行
// kernel 再写回 host；所有 native 操作均为异步入队，调用方负责最终同步。
void execute_object_group_transfer(
    TransferDirection direction, const torch::Device &device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec> &kernel_group_specs,
    const std::vector<BatchStep> &batch_steps) {
  // Set the device guard once for the whole plan so every staging copy and
  // kernel launch below is enqueued on this device's current stream, in
  // order (mirrors upstream execute_object_group_transfer).
  // 整个计划只设置一次设备 guard，使下面的 staging 拷贝和 kernel launch 按序
  // 入队到该设备的当前 stream（与上游 execute_object_group_transfer 一致）。
  const c10::OptionalDeviceGuard device_guard(device);
  const bool is_h2d = (direction == TransferDirection::H2D);
  TORCH_CHECK(device.is_privateuseone(), "device must be an NPU device");
  TORCH_CHECK(host_buffer_alignment > 0 &&
                  (host_buffer_alignment & (host_buffer_alignment - 1)) == 0,
              "host_buffer_alignment must be a non-zero power of two, got ",
              host_buffer_alignment);

  // --- Whole-plan pre-validation: every group is prepared and
  // every launch is validated before ANY staging copy or kernel launch is
  // enqueued. A failure here means "nothing started".
  // --- 完整计划预校验：任何 staging 拷贝或 kernel launch 入队前，先准备所有
  // group 并校验所有 launch。此处失败意味着“尚未开始任何操作”。
  const int64_t ub_bytes = query_ub_budget_bytes();
  std::vector<PreparedGroup> groups;
  groups.reserve(kernel_group_specs.size());
  for (const auto &spec : kernel_group_specs) {
    groups.push_back(prepare_group(spec.shape_desc, spec.engine_kv_format,
                                   spec.lmcache_chunk_size, ub_bytes));
  }

  struct PreparedLaunch {
    int32_t group_idx;
    CheckedLaunch checked;
    int64_t block_ids_offset;
  };
  std::vector<std::vector<PreparedLaunch>> prepared_steps;
  prepared_steps.reserve(batch_steps.size());
  for (const auto &step : batch_steps) {
    for (const auto &copy : step.staging) {
      // Raw external pointers carry no derivable allocation capacity; the
      // plan builder owns pointer/size validity. What can be
      // checked cheaply here is checked.
      // 原始外部指针无法推导分配容量，指针和大小的有效性由计划构建方负责；
      // 此处只执行开销较低的检查。
      TORCH_CHECK(copy.nbytes > 0, "StagingCopy nbytes must be positive");
      TORCH_CHECK(copy.dest != 0 && copy.src != 0,
                  "StagingCopy pointers must be non-null");
    }
    std::vector<PreparedLaunch> launches;
    launches.reserve(step.launches.size());
    for (const auto &launch : step.launches) {
      TORCH_CHECK(launch.group_idx >= 0 &&
                      launch.group_idx <
                          static_cast<int>(kernel_group_specs.size()),
                  "LaunchVar.group_idx out of range: ", launch.group_idx);
      const KernelGroupSpec &spec = kernel_group_specs[launch.group_idx];
      TORCH_CHECK(launch.num_objects <=
                      static_cast<int>(spec.lmcache_objects_ptrs.size()),
                  "LaunchVar.num_objects (", launch.num_objects,
                  ") exceeds available temp buffers (",
                  spec.lmcache_objects_ptrs.size(), ")");
      PreparedLaunch prepared;
      prepared.group_idx = launch.group_idx;
      prepared.checked =
          validate_launch(groups[launch.group_idx], launch.total_blocks,
                          launch.num_objects, launch.block_ids_offset,
                          spec.block_ids_capacity, launch.skip_prefix_n_blocks);
      prepared.block_ids_offset = launch.block_ids_offset;
      launches.push_back(prepared);
    }
    prepared_steps.push_back(std::move(launches));
  }

  const char *socName = aclrtGetSocName();
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  const uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  // Direct launch (CUDA cudaLaunchKernel style): .stream() above drains
  // torch_npu's ACL task queue first, so every staging copy and kernel launch
  // below is enqueued in Python-call order without a queue-thread hop. The
  // pointer table, block IDs, staging buffers and engine plane views are
  // owned by the caller's cache context, which synchronizes the stream before
  // releasing them, so the raw pointers below stay valid.
  // 直接启动（CUDA cudaLaunchKernel 风格）：上面的 .stream() 先排空 torch_npu
  // 的 ACL 任务队列，因此下面的 staging 拷贝和 kernel launch 都按 Python 调用
  // 顺序入队，不经过队列线程。指针表、block_ids、staging 缓冲区和引擎 plane
  // 视图均由调用方的 cache context 持有；它在释放前同步 stream，所以下面的
  // 原始指针在异步执行期间保持有效。入队本身不是完成通知。
  const auto do_staging = [&](const std::vector<StagingCopy> &staging) {
    for (const auto &copy : staging) {
      lmcache_memcpy_async_on_stream(copy.dest, copy.src, copy.nbytes,
                                     direction, copy.host_offset,
                                     host_buffer_alignment, stream);
    }
  };

  for (size_t step_idx = 0; step_idx < batch_steps.size(); ++step_idx) {
    // H2D stages CPU->NPU temp buffers before the kernel reads them; D2H
    // stages NPU->CPU after the kernel writes them. The per-step ordering
    // must be preserved because temp buffers are reused across steps.
    // H2D 在 kernel 读取前执行 CPU->NPU 临时缓冲区 staging；D2H 在 kernel
    // 写入后执行 NPU->CPU staging。临时缓冲区会跨 step 复用，因此必须保持
    // 每个 step 内的顺序。
    if (is_h2d) {
      do_staging(batch_steps[step_idx].staging);
    }
    for (const auto &launch : prepared_steps[step_idx]) {
      const KernelGroupSpec &spec = kernel_group_specs[launch.group_idx];
      const PreparedGroup &group = groups[launch.group_idx];
      std::vector<int64_t> obj_device_ptrs =
          device_lmc_ptrs(std::vector<int64_t>(
              spec.lmcache_objects_ptrs.begin(),
              spec.lmcache_objects_ptrs.begin() + launch.checked.num_objects));
      int64_t *block_ids_base = reinterpret_cast<int64_t *>(
          spec.block_ids_base +
          static_cast<uintptr_t>(launch.block_ids_offset) * sizeof(int64_t));
      launch_prepared_objects(
          aiv_num, stream, group,
          reinterpret_cast<uint8_t *>(spec.paged_buffer_ptrs), obj_device_ptrs,
          block_ids_base, launch.checked, is_h2d);
    }
    if (!is_h2d) {
      do_staging(batch_steps[step_idx].staging);
    }
  }
}
