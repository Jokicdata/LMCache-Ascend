// SPDX-License-Identifier: Apache-2.0
//
// Ascend NPU host-callback buffer for stream-completion records.
// Ported from lmcache csrc/cuda/completion_recorder.h; the only device
// dependency is aclrtLaunchHostFunc (CANN >= 9.0), which mirrors
// cudaLaunchHostFunc. The callback runs on an ACL-internal thread without
// the GIL; Python drains the buffer and dispatches to a handler keyed by
// ``kind``.
// 用于流完成记录的昇腾 NPU 主机回调缓冲区。
// 移植自 lmcache csrc/cuda/completion_recorder.h；唯一的设备依赖是
// aclrtLaunchHostFunc（CANN >= 9.0），其作用对应 cudaLaunchHostFunc。
// 回调在 ACL 内部线程上运行且不持有 GIL；Python 取走缓冲区中的记录，
// 再按 ``kind`` 分发给对应的处理器。

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// Ascend CANN runtime
// 昇腾 CANN 运行时。
#include "acl/acl.h"

using lmcache_completion_stream_t = aclrtStream;

struct PendingCompletion {
  std::string kind;    // dispatch key, e.g. "finish_write"
  // 分发键，例如“finish_write”。
  std::string payload; // opaque encoded bytes (e.g. msgpack)
  // 不透明的编码字节（例如 msgpack）；std::string 按字节保存，
  // 可以包含 '\0'，本组件不会解释、解码或重新编码其内容。
};

class CompletionRecorder {
public:
  static CompletionRecorder &instance();
  // Takes ownership of the heap-allocated PendingCompletion.
  // 接管堆分配的 PendingCompletion 的所有权。
  void push(std::unique_ptr<PendingCompletion> completion);
  // 只取走当前已入缓冲区的完成记录，不等待流上的回调，也不是流同步。
  std::vector<std::unique_ptr<PendingCompletion>> drain();

private:
  CompletionRecorder() = default;
  std::mutex mutex_;
  std::vector<std::unique_ptr<PendingCompletion>> buffer_;
};

// Schedule a completion record. Called WITHOUT the GIL.
// 调度一条完成记录，调用时不持有 GIL。
// 非零 stream_ptr 通过 aclrtLaunchHostFunc 将回调排入流；stream_ptr == 0 时
// 不调用 ACL，直接内联执行回调，使记录立即入缓冲区。
// payload 按原始编码字节透传，包括其中可能存在的 '\0'。
void record_completion_on_stream(int64_t stream_ptr, const std::string &kind,
                                 std::string payload);

using CompletionDrainResult = std::vector<std::pair<std::string, std::string>>;

// 取走当前缓冲的 (kind, payload) 记录；不会同步流，也不保证未来回调已完成。
CompletionDrainResult drain_recorded_completions();
