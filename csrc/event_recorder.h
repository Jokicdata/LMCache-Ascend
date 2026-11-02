// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// Ascend CANN runtime (>= 9.0, provides aclrtLaunchHostFunc)
// 昇腾 CANN 运行时（>= 9.0，提供 aclrtLaunchHostFunc）。
#include "acl/acl.h"

using lmcache_stream_t = aclrtStream;

// ---------------------------------------------------------------------------
// PendingEvent — lightweight struct held in a lock-free-ish buffer.
// All fields are pure C++ (no Python objects) so the ACL host callback
// can write the timestamp without touching the GIL.
// ---------------------------------------------------------------------------
// PendingEvent——保存在近似无锁缓冲区中的轻量级结构。
// 所有字段都是纯 C++ 类型（不含 Python 对象），因此 ACL 主机回调
// 无需接触 GIL 即可写入时间戳。
// ---------------------------------------------------------------------------

struct PendingEvent {
  std::string event_type_name; // e.g. "mp.store.start"
  // 例如“mp.store.start”。
  std::string session_id;
  double timestamp; // wall-clock, set by host callback
  // 墙上时钟时间，由主机回调设置。
  std::unordered_map<std::string, std::string> str_metadata;
  std::unordered_map<std::string, int64_t> int_metadata;
};

// ---------------------------------------------------------------------------
// EventRecorder — global singleton that buffers events from ACL callbacks.
// ---------------------------------------------------------------------------
// EventRecorder——用于缓冲 ACL 回调事件的全局单例。
// ---------------------------------------------------------------------------

class EventRecorder {
public:
  static EventRecorder &instance();

  // Called from the ACL host callback (no GIL held).
  // Takes ownership of *event, moves it into the buffer, then deletes it.
  // 由 ACL 主机回调调用（不持有 GIL）。
  // 接管 *event 的所有权，将其内容移入缓冲区，然后删除该对象。
  void push(PendingEvent *event);

  // Called from Python (GIL held) to drain all buffered events.
  // 由 Python 调用（持有 GIL），取走当前已缓冲的全部事件。
  // 这里只交换并清空 C++ 缓冲区，不等待回调，也不是 NPU 流同步操作。
  std::vector<PendingEvent> drain();

private:
  EventRecorder() = default;
  std::mutex mutex_;
  std::vector<PendingEvent> buffer_;
};

// ---------------------------------------------------------------------------
// Free functions exposed via pybind11
// ---------------------------------------------------------------------------
// 通过 pybind11 暴露的自由函数。
// ---------------------------------------------------------------------------

// Schedule an event recording on an NPU stream.  The host callback stamps
// the wall-clock time and pushes to the global EventRecorder.
// Called WITHOUT the GIL (py::call_guard<py::gil_scoped_release>).
// 在 NPU 流上调度一次事件记录。主机回调写入墙上时钟时间，
// 并将事件推入全局 EventRecorder。
// 调用时不持有 GIL（py::call_guard<py::gil_scoped_release>）。
// 非零 stream_ptr 通过 aclrtLaunchHostFunc 排队；stream_ptr == 0 时不依赖流，
// 直接内联执行回调，使记录立即进入缓冲区。
void record_event_on_stream(
    int64_t stream_ptr, const std::string &event_type_name,
    const std::string &session_id,
    const std::unordered_map<std::string, std::string> &str_metadata,
    const std::unordered_map<std::string, int64_t> &int_metadata);

// Drain all buffered events.  Returns a list of tuples:
//   (event_type_name, session_id, timestamp, str_metadata, int_metadata)
// 取走当前全部缓冲事件，返回元组列表：
//   (event_type_name, session_id, timestamp, str_metadata, int_metadata)。
// 不会同步 NPU 流，因此尚未执行的流回调不会出现在本次结果中。
using DrainResult =
    std::vector<std::tuple<std::string, std::string, double,
                           std::unordered_map<std::string, std::string>,
                           std::unordered_map<std::string, int64_t>>>;

DrainResult drain_recorded_events();
