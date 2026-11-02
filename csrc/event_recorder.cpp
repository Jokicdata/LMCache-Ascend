// SPDX-License-Identifier: Apache-2.0

#include "event_recorder.h"

#include <chrono>
#include <utility>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
// 辅助函数。
// ---------------------------------------------------------------------------

// Monotonic wall-clock reader.
//
// Why not just `system_clock::now()`?  `system_clock` maps to CLOCK_REALTIME,
// which NTP / chrony can slew backward by tens of microseconds to stay synced
// with upstream.  Host callbacks fire at arbitrary wall moments, so two
// consecutive callbacks (e.g. MP_STORE_START then MP_STORE_END on the same
// stream) can straddle a backward slew and land with end_ts < start_ts.
// Jaeger then renders span duration as unsigned 64-bit subtraction, producing
// the characteristic ~213503982d span duration (= 2^64 microseconds).
//
// Fix: anchor once to the (system_clock, steady_clock) pair at first call.
// Every subsequent timestamp is computed as
//   epoch_sys + (steady_now - epoch_steady)
// which is monotonic (steady_clock == CLOCK_MONOTONIC on Linux) while
// remaining expressed in Unix-epoch seconds so downstream consumers don't
// notice any change.
// 单调的墙上时钟读取器。
//
// 为什么不直接使用 `system_clock::now()`？`system_clock` 对应 CLOCK_REALTIME，
// NTP / chrony 为与上游保持同步，可能将其向后调整数十微秒。
// 主机回调可能在任意实际时刻触发，因此两个连续回调（例如同一流上的
// MP_STORE_START 和随后的 MP_STORE_END）可能跨越一次向后调整，
// 导致 end_ts < start_ts。Jaeger 随后用无符号 64 位减法计算 span 时长，
// 产生典型的约 213503982 天时长（即 2^64 微秒）。
//
// 修复方式：首次调用时记录一组 (system_clock, steady_clock) 基准。
// 后续每个时间戳都按以下公式计算：
//   epoch_sys + (steady_now - epoch_steady)
// 该结果具有单调性（Linux 上 steady_clock == CLOCK_MONOTONIC），
// 同时仍以 Unix 纪元秒表示，因此下游使用方无需改变。
static double wall_clock_time() {
  static const auto epoch_sys = std::chrono::system_clock::now();
  static const auto epoch_steady = std::chrono::steady_clock::now();
  auto now_steady = std::chrono::steady_clock::now();
  auto since_epoch = epoch_sys.time_since_epoch() + (now_steady - epoch_steady);
  return std::chrono::duration<double>(since_epoch).count();
}

// ---------------------------------------------------------------------------
// EventRecorder
// ---------------------------------------------------------------------------
// 事件记录器。
// ---------------------------------------------------------------------------

EventRecorder &EventRecorder::instance() {
  static EventRecorder recorder;
  return recorder;
}

// 由主机回调移交 event 的所有权；在锁内移动字段到缓冲区，锁外删除原对象。
void EventRecorder::push(PendingEvent *event) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.push_back(std::move(*event));
  }
  delete event;
}

// 在锁内交换缓冲区并立即返回当前已有事件；不等待 ACL 回调，不做流同步。
std::vector<PendingEvent> EventRecorder::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<PendingEvent> result;
  result.swap(buffer_);
  return result;
}

// ---------------------------------------------------------------------------
// ACL host callback — runs on an ACL-internal thread, no GIL.
// Same restriction as CUDA host functions: never call any aclrt* API from
// this callback; std::chrono / mutex / vector only.
// ---------------------------------------------------------------------------
// ACL 主机回调——在 ACL 内部线程上运行，不持有 GIL。
// 与 CUDA 主机函数有相同限制：绝不能在此回调中调用任何 aclrt* API；
// 只使用 std::chrono / mutex / vector。
// ---------------------------------------------------------------------------
// 非零流由 aclrtLaunchHostFunc 调度到此回调；回调只打时间戳并入缓冲区，
// 不执行 Python 代码，也不获取 GIL。

static void event_host_callback(void *data) {
  auto *event = static_cast<PendingEvent *>(data);
  event->timestamp = wall_clock_time();
  EventRecorder::instance().push(event);
}

// ---------------------------------------------------------------------------
// Free functions for pybind11
// ---------------------------------------------------------------------------
// 供 pybind11 使用的自由函数。
// ---------------------------------------------------------------------------

// 创建 PendingEvent 后按 stream_ptr 分流：零值直接内联回调并立即入缓冲区；
// 非零值通过 aclrtLaunchHostFunc 排入对应流，由 ACL 在流执行到该点时调用回调。
// 本函数由 Python 在不持有 GIL 的情况下调用；启动失败时回收 event。
void record_event_on_stream(
    int64_t stream_ptr, const std::string &event_type_name,
    const std::string &session_id,
    const std::unordered_map<std::string, std::string> &str_metadata,
    const std::unordered_map<std::string, int64_t> &int_metadata) {
  auto *event = new PendingEvent{
      event_type_name, session_id, 0.0, str_metadata, int_metadata,
  };
  if (stream_ptr == 0) {
    // Null stream: no stream dependency — run the callback inline so the
    // record lands in the buffer immediately.
    // 空流：不存在流依赖，直接内联运行回调，使记录立即进入缓冲区。
    event_host_callback(event);
    return;
  }

  auto stream =
      reinterpret_cast<lmcache_stream_t>(static_cast<uintptr_t>(stream_ptr));
  auto err = aclrtLaunchHostFunc(stream, event_host_callback, event);
  if (err != ACL_SUCCESS) {
    delete event;
  }
}

// 将当前缓冲区内容转换为 Python 绑定所需的元组列表。
// 仅 drain 已完成的回调记录，不同步流，也不保证排队中的事件已经可见。
DrainResult drain_recorded_events() {
  auto events = EventRecorder::instance().drain();
  DrainResult result;
  result.reserve(events.size());
  for (auto &e : events) {
    result.emplace_back(std::move(e.event_type_name), std::move(e.session_id),
                        e.timestamp, std::move(e.str_metadata),
                        std::move(e.int_metadata));
  }
  return result;
}
