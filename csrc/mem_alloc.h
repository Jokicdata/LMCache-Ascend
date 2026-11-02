#pragma once
#include <cstdint>

/*
 * These following APIs are called directly in LMCache,
 * therefore we assume the ptrs management are done by the python program for
 * now.
 */
/*
 * 以下 API 由 LMCache 直接调用，
 * 因此目前假定这些指针的生命周期管理由 Python 程序负责。
 */

// 分配页锁定主机内存；非 310 设备还会经 register_ptr 建立主机到设备的映射。
// 返回原始主机指针，调用方须用 free_pinned_ptr 释放。
uintptr_t alloc_pinned_ptr(std::size_t size, unsigned int flags);

// 先通过 unregister_ptr 解除设备侧注册，再通过 aclrtFreeHost 释放主机内存。
// 必须由 alloc_pinned_ptr 的调用方触发，不能在注销前释放内存。
void free_pinned_ptr(uintptr_t ptr);

// 在指定 NUMA 节点上分配并注册主机内存；可能因 mbind 而较慢。
// 返回原始主机指针，释放时必须同时传回相同的指针和大小。
uintptr_t alloc_pinned_numa_ptr(std::size_t size, int node);

// 先注销设备映射，再按原大小 munmap；主机内存所有权由调用方负责收尾。
void free_pinned_numa_ptr(uintptr_t ptr, std::size_t size);