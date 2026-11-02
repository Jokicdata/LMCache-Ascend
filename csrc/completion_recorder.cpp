// SPDX-License-Identifier: Apache-2.0

#include "completion_recorder.h"

#include <utility>

CompletionRecorder &CompletionRecorder::instance() {
  static CompletionRecorder recorder;
  return recorder;
}

// 在锁内接管完成记录并放入缓冲区；unique_ptr 始终负责对象生命周期。
void CompletionRecorder::push(std::unique_ptr<PendingCompletion> completion) {
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_.push_back(std::move(completion));
}

// 交换并返回当前缓冲区，不等待尚未触发的 ACL 回调，也不执行流同步。
std::vector<std::unique_ptr<PendingCompletion>> CompletionRecorder::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::unique_ptr<PendingCompletion>> result;
  result.swap(buffer_);
  return result;
}

// ACL host callback — runs on an ACL-internal thread, no GIL.
// Same restriction as CUDA host functions: never call any aclrt* API from
// this callback; std::chrono / mutex / vector only.
// ACL 主机回调——在 ACL 内部线程上运行，不持有 GIL。
// 与 CUDA 主机函数有相同限制：绝不能在此回调中调用任何 aclrt* API；
// 只使用 std::chrono / mutex / vector。
// 非零流由 aclrtLaunchHostFunc 调度到此回调；这里只接管对象并入缓冲区，
// 不调用 Python，也不获取 GIL。
static void completion_host_callback(void *data) {
  // Adopt the raw pointer back into a unique_ptr.
  // 将裸指针重新纳入 unique_ptr 管理。
  std::unique_ptr<PendingCompletion> completion(
      static_cast<PendingCompletion *>(data));
  CompletionRecorder::instance().push(std::move(completion));
}

// 创建完成记录并按 stream_ptr 分流：零值直接内联回调，使记录立即可 drain；
// 非零值通过 aclrtLaunchHostFunc 排入流，回调执行时才进入缓冲区。
// payload 是不透明的编码字节，按原值移动和保存，包括其中可能存在的 '\0'。
// 本函数由 Python 在不持有 GIL 的情况下调用；启动失败时回收裸指针。
void record_completion_on_stream(int64_t stream_ptr, const std::string &kind,
                                 std::string payload) {
  auto completion = std::make_unique<PendingCompletion>(
      PendingCompletion{kind, std::move(payload)});
  if (stream_ptr == 0) {
    // Null stream: no stream dependency — run the callback inline so the
    // record lands in the buffer immediately.
    // 空流：不存在流依赖，直接内联运行回调，使记录立即进入缓冲区。
    completion_host_callback(completion.release());
    return;
  }
  auto stream = reinterpret_cast<lmcache_completion_stream_t>(
      static_cast<uintptr_t>(stream_ptr));
  // Pass ownership through the runtime as a raw pointer; the host callback
  // re-adopts it. Reclaim if the launch itself fails.
  // 以裸指针形式通过运行时传递所有权，主机回调会重新接管它。
  // 如果启动本身失败，则回收该指针。
  PendingCompletion *raw = completion.release();
  auto err = aclrtLaunchHostFunc(stream, completion_host_callback, raw);
  if (err != ACL_SUCCESS) {
    delete raw;
  }
}

// 将当前已缓冲记录转换为 (kind, payload) 列表，payload 字节不作解释。
// 该操作只 drain 缓冲区，不同步流，也不会等待排队中的回调。
CompletionDrainResult drain_recorded_completions() {
  auto completions = CompletionRecorder::instance().drain();
  CompletionDrainResult result;
  result.reserve(completions.size());
  for (auto &c : completions) {
    result.emplace_back(std::move(c->kind), std::move(c->payload));
  }
  return result;
}
