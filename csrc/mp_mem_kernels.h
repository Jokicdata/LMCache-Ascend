// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

// Direction of an MP KV-cache transfer (mirrors upstream kv_transfer_types.h:
// H2D=0 LMCache->engine, D2H=1 engine->LMCache).
// MP KV-cache 的传输方向（与上游 kv_transfer_types.h 一致）：
// H2D=0 表示 LMCache->引擎（retrieve），D2H=1 表示引擎->LMCache（store）。
enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

// Engine KV-cache physical layout enum, value-compatible with the upstream
// csrc/engine_kv_format.h enum of the same name (Ascend dispatches
// NL_X_TWO_X_NB_BS_NH_HS first; the full set is retained so the Python
// side can bind the same symbols as upstream).
// 引擎 KV-cache 的物理布局枚举，其值与上游 csrc/engine_kv_format.h 中的
// 同名枚举兼容（Ascend 优先分派 NL_X_TWO_X_NB_BS_NH_HS；保留完整集合，
// 以便 Python 侧绑定与上游相同的符号）。
enum class EngineKVFormat : int {
  NB_NL_TWO_BS_NH_HS = 0,
  NL_X_TWO_NB_BS_NH_HS = 1,
  NL_X_NB_TWO_BS_NH_HS = 2,
  NL_X_NB_BS_HS = 3,
  TWO_X_NL_X_NBBS_NH_HS = 4,
  NL_X_NBBS_ONE_HS = 5,
  NL_X_TWO_NB_NH_BS_HS = 6,
  NL_X_NB_TWO_NH_BS_HS = 7,
  NB_NL_TWO_NH_BS_HS = 8,
  TWO_X_NL_X_NB_BS_NH_HS = 9,
  NL_X_NB_NH_BS_TWO_HS = 10,
  NL_X_NB_BS_NH_TWO_HS = 11,
  NL_X_NB_NH_BS_CS = 12,
  NL_X_NB_BS_NH_CS = 13,
  NL_X_NB_BSV_BSS = 14,
  NL_X_TWO_NB_NH_ONE_BS_HS = 15,
  NL_X_TWO_X_NB_BS_NH_HS = 16,
  NL_X_NP_X_NB_BS_ONE_HS = 17,
};

// Compile-time shape descriptor, field-for-field aligned with upstream
// PageBufferShapeDesc (LMCache csrc/mp_mem_kernels.cuh:11). block_stride_elems
// honours engine-side dim-0 padding when > 0, else falls back to the tight
// bs * nh * hs stride.
// 编译期形状描述符，各字段与上游 PageBufferShapeDesc
// （LMCache csrc/mp_mem_kernels.cuh:11）逐一对齐。block_stride_elems 大于 0
// 时采用引擎侧第 0 维的填充步长，否则回退为紧凑的 bs * nh * hs 步长。
struct PageBufferShapeDesc {
  int kv_size;
  int nl;
  int nb;
  int bs;
  int nh;
  int hs;
  int element_size;
  int block_stride_elems;
  // Ascend-only plane count. Formats 16/17 set it at registration.
  // prepare_group sets it to 2 for format 2. Unset 16/17 stays 0 and is
  // rejected at group prepare. For format 13, prepare_group derives 1 when
  // unset.
  // Ascend 专用的 plane 数量。格式 16/17 在注册时设置；prepare_group 为
  // 格式 2 设置为 2。格式 16/17 未设置时保持 0，并在准备 group 时被拒绝；
  // 格式 13 未设置时由 prepare_group 推导为 1。
  int32_t num_planes = 0;
  // Per-plane payload bytes per token slot (64-bit byte addressing). The
  // engine-side token step is derived from the payload (dense token rows).
  // 每个 plane 中每个 token slot 的有效载荷字节数（采用 64 位字节寻址）。
  // 引擎侧 token 步长由有效载荷推导（token 行是紧凑的）。
  int64_t plane_slot_bytes[4] = {0, 0, 0, 0};
  // Per-plane per-block dim-0 step in bytes (stride(0) * itemsize).
  // 每个 plane 沿 block 第 0 维的字节步长（stride(0) * itemsize）。
  int64_t plane_block_stride_bytes[4] = {0, 0, 0, 0};
  // Format 2 only: bytes from each layer's base data_ptr() to plane p.
  // Zero for formats whose pointer table already has one entry per plane.
  // 仅格式 2 使用：从每层 base data_ptr() 到 plane p 的字节偏移。
  // 指针表已按 plane 各存一个表项的格式将此值置为 0。
  int64_t plane_engine_base_offset_bytes[4] = {0, 0, 0, 0};

  template <typename ScalarType> inline size_t scalars_per_head() const {
    return hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType> inline size_t scalars_per_token() const {
    return nh * hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType> inline size_t scalars_per_block() const {
    const size_t elems = block_stride_elems > 0
                             ? static_cast<size_t>(block_stride_elems)
                             : static_cast<size_t>(bs) * nh * hs;
    return elems * element_size / sizeof(ScalarType);
  }
};

// Object-group transfer plan types (upstream mp_mem_kernels.cuh:80-116).
// 对象组传输计划的类型定义（上游 mp_mem_kernels.cuh:80-116）。
// 一条 staging 拷贝描述；dest/src 是外部原始地址，host_offset 用于按 host
// 对齐边界切分，该结构本身不拥有所指向的内存。
struct StagingCopy {
  uintptr_t dest;
  uintptr_t src;
  size_t nbytes;
  size_t host_offset;
};

// 一次对象组 launch 的动态参数：group_idx 选择静态组，block_ids_offset 和
// total_blocks 指定 NPU block_ids 切片，skip_prefix_n_blocks 跳过每个对象
// 前缀中无需重复传输的块。
struct LaunchVar {
  int group_idx;
  int64_t block_ids_offset;
  int total_blocks;
  int num_objects;
  int skip_prefix_n_blocks;
};

// 同一调度步骤中的 staging 拷贝与 kernel launch。H2D 先 staging 后 launch，
// D2H 顺序相反；临时缓冲区会跨 step 复用，因此该顺序不可交换。
struct BatchStep {
  std::vector<StagingCopy> staging;
  std::vector<LaunchVar> launches;
};

// 一组共享静态几何和指针资源的配置。paged_buffer_ptrs 指向设备指针表，
// block_ids_base/capacity 描述设备块号数组；所有指针均由调用方持有，
// 并须在异步操作完成前保持有效。
struct KernelGroupSpec {
  uintptr_t paged_buffer_ptrs;
  std::vector<int64_t> lmcache_objects_ptrs;
  PageBufferShapeDesc shape_desc;
  int lmcache_chunk_size;
  EngineKVFormat engine_kv_format;
  uintptr_t block_ids_base;
  int64_t block_ids_capacity;
};

// Plan executor: enqueues every staging copy and kernel launch described by
// batch_steps within a single GIL release (configured at the pybind layer).
// Mirrors upstream execute_object_group_transfer (mp_mem_kernels.cuh:133).
// 计划执行器：在同一次 GIL 释放期间（由 pybind 层配置），将 batch_steps
// 描述的所有 staging 拷贝和 kernel launch 入队。与上游
// execute_object_group_transfer（mp_mem_kernels.cuh:133）一致。
// 该接口先校验完整计划，再按 step 顺序提交 native 异步操作；返回不代表
// 传输完成，调用方必须同步 stream 后才能访问结果或释放、复用相关资源。
void execute_object_group_transfer(
    TransferDirection direction, const torch::Device &device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec> &kernel_group_specs,
    const std::vector<BatchStep> &batch_steps);

// Block-level multi-layer KV transfer between vLLM paged buffers and LMCache
// contiguous memory objects. Mirrors upstream multi_layer_block_kv_transfer
// (mp_mem_kernels.cuh:155). Phase 1 loops over the object batch inside this
// entry point (one kernel launch per object).
// 在 vLLM 分页缓冲区与 LMCache 连续内存对象之间进行 block 级多层 KV 传输。
// 与上游 multi_layer_block_kv_transfer（mp_mem_kernels.cuh:155）一致。
// 阶段 1 在此入口内遍历对象批次，每个对象启动一次 kernel。
// H2D 是 LMCache 到引擎的 retrieve，D2H 是引擎到 LMCache 的 store；普通
// CPU 对象可能需要 NPU staging，已注册锁页对象使用设备映射，NPU 对象直接
// 使用设备地址。三者均由调用方持有，且 native 入队后仍需由调用方同步。
void multi_layer_block_kv_transfer(
    const torch::Tensor &paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor &block_ids,
    const torch::Device &device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks);

// Async host<->device copy chunked at the host buffer's alignment boundary
// (port of upstream lmcache_memcpy_async, mem_kernels.cu). Used by the plan
// executor's staging step and by the Python fallback path.
// 在 host 缓冲区的对齐边界处切分异步 host<->device 拷贝（移植自上游
// lmcache_memcpy_async，mem_kernels.cu）。供计划执行器的 staging 步骤和
// Python 回退路径使用；函数返回仅表示已入队，调用方仍需同步 stream。
void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments);
