#include "mem_alloc.h"
#include "managed_mem.h"
#include <acl/acl.h>
#include <cstdlib> // for std::getenv
// 用于 std::getenv。
#include <cstring> // for strerror
// 用于 strerror。
#include <errno.h>
#include <numaif.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>

// 通过 aclrtMallocHost 分配页锁定主机内存，再按芯片类型决定是否调用
// register_ptr 建立设备映射。成功时返回主机指针；注册失败时先走
// free_pinned_ptr 的注销、释放路径，再抛出异常。
uintptr_t alloc_pinned_ptr(std::size_t size, unsigned int flags) {
  void *ptr = nullptr;
  // no flags
  // 不使用标志位。
  aclError err = aclrtMallocHost(&ptr, size);
  if (err != ACL_SUCCESS) {
    throw std::runtime_error("aclrtMallocHost failed: " + std::to_string(err));
  }

  const char *socVersion = aclrtGetSocName();

  // nullptr means that the chip version failed to be obtained. We cannot be
  // sure about the version of the device. Unless we are sure that we deal with
  // a 310 device, we try to register.
  // nullptr 表示获取芯片版本失败，因此无法确定设备版本。
  // 除非能确定当前是 310 设备，否则都会尝试注册。
  if (socVersion == nullptr ||
      std::string(socVersion).find("310") == std::string::npos) {
    // not 310p
    // 不是 310P。
    auto devPtr = register_ptr(ptr, size);
    if (devPtr == nullptr) {
      free_pinned_ptr(reinterpret_cast<uintptr_t>(ptr));
      throw std::runtime_error("register ptr failed");
    }
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

// 释放 alloc_pinned_ptr 返回的内存：先由 unregister_ptr 按驱动版本解除
// ACL/HAL 注册，再调用 aclrtFreeHost。主机内存在本函数返回前仍由该分配路径拥有，
// 调用方必须保证不在注销前提前释放，也不能重复释放。
void free_pinned_ptr(uintptr_t ptr) {
  unregister_ptr(reinterpret_cast<void *>(ptr));
  aclError err = aclrtFreeHost(reinterpret_cast<void *>(ptr));
  if (err != ACL_SUCCESS) {
    throw std::runtime_error("aclrtFreeHost failed: " + std::to_string(err));
  }
}

/*
 * This function is potentially slow for the mbind
 */
/*
 * 此函数可能因 mbind 而较慢。
 */
// 依次执行 mmap、NUMA mbind 和 register_ptr；登记设备指针供后续
// get_device_ptr 查询，但对外仍返回原始主机指针。失败路径负责解除已建立的资源。
uintptr_t alloc_pinned_numa_ptr(std::size_t size, int node) {
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
  }

  // Maximum of 64 numa nodes
  // 最多支持 64 个 NUMA 节点。
  unsigned long mask = 1UL << node;
  long maxnode = 8 * sizeof(mask);
  int err = mbind(ptr, size, MPOL_BIND, &mask, maxnode,
                  MPOL_MF_MOVE | MPOL_MF_STRICT);
  if (err != 0) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(errno));
  }

  memset(ptr, 0, size);

  // as before we need to actually save the dev ptr for later reuse,
  // because acl APIs do not allow retrieving register dev ptr
  // 与前面相同，需要保存设备指针供后续复用，
  // 因为 ACL API 不允许重新获取已注册的设备指针。
  auto devPtr = register_ptr(ptr, size);
  if (devPtr == nullptr) {
    munmap(ptr, size);
    aclError err = aclrtGetLastError(aclrtLastErrLevel::ACL_RT_THREAD_LEVEL);
    if (err != ACL_SUCCESS) {
      throw std::runtime_error(
          std::string("unable to register Pinned Numa HostPtr: ") +
          std::to_string(err));
    } else {
      throw std::runtime_error(
          std::string("unable to register Pinned Numa HostPtr."));
    }
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

// 释放 alloc_pinned_numa_ptr 返回的内存：先注销设备映射，再按分配时的
// size 调用 munmap。unregister_ptr 和 munmap 均不接管对方的资源职责；
// 调用方必须传入原始指针与原始大小，且只释放一次。
void free_pinned_numa_ptr(uintptr_t p, std::size_t size) {
  void *ptr = reinterpret_cast<void *>(p);

  auto unRegErr = unregister_ptr(ptr);
  auto unMapErr = munmap(ptr, size);
  if (unRegErr) {
    throw std::runtime_error("unregister_ptr failed: " +
                             std::to_string(unRegErr));
  }
  if (unMapErr) {
    throw std::runtime_error("munmap failed: " + std::to_string(unMapErr));
  }
}
