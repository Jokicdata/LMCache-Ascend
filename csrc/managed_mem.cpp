#include "managed_mem.h"
#include "driver/ascend_hal.h"
#include "driver/ascend_hal_define.h"
#include <acl/acl.h>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <sys/mman.h>

// Only required for old driver version (look at registerHostPtr)
// 仅旧驱动版本需要（参见 registerHostPtr）。
#ifdef PROF_ERROR
// You can add a pragma message to see this in your build log if you want:
// #pragma message("Undefining PROF_ERROR from ascend_hal.h before NPU headers")
// 如有需要，可以添加一条 pragma message，在构建日志中查看此信息：
// 上述示例用于提示“在包含 NPU 头文件之前取消 ascend_hal.h 中 PROF_ERROR 的定义”。
#undef PROF_ERROR
#endif
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "torch/extension.h"

namespace lmc {
// Signatures for internal helper functions
// 内部辅助函数的声明。

// Gets the current device offsetting on ASCEND_RT_VISIBLE_DEVICES when needed
// 获取当前设备，并在需要时根据 ASCEND_RT_VISIBLE_DEVICES 修正设备编号。
int get_device();

// Class implementations
// 类成员函数的实现。

HostRegisteredMemoryManager::HostRegisteredMemoryManager() {};

HostRegisteredMemoryManager::~HostRegisteredMemoryManager() {
  this->unregisterAll();
};

// 析构时遍历登记表并调用 aclrtHostUnregister，随后清空记录；
// 这里只处理注册关系，不释放 hostPtr 指向的主机内存。
void HostRegisteredMemoryManager::unregisterAll() {
  const std::unique_lock<std::shared_mutex> guard(this->mux);

  // Iterate through each key-value pair in the map.
  // 遍历映射表中的每个键值对。
  for (const auto &pair : this->allocatedMap) {
    void *hostPtr = pair.first;
    aclrtHostUnregister(hostPtr);
  }

  // After unregistering all pointers, clear the map completely.
  // 注销所有指针后，彻底清空映射表。
  this->allocatedMap.clear();
};

// Register a pointer through high level APIs (aclrt) return devPtr
// Returns the created RegisteredMemoryRecord
// 通过高层 API（aclrt）注册指针并返回 devPtr。
// 返回创建的 RegisteredMemoryRecord。
// register_ptr 在驱动主版本 >= 25 时调用本函数；aclrtHostRegister 建立映射，
// allocatedMap 保存主机地址、设备地址和大小，主机内存仍由调用方分配和释放。
RegisteredMemoryRecord *HostRegisteredMemoryManager::registerHostPtr(
    void *hostPtr, size_t bufferSize) { // torch::Tensor& tensor){

  TORCH_CHECK(
      !(hostPtr == nullptr || bufferSize == 0),
      "Error: hostPtr cannot be null and bufferSize must be greater than 0.");
  const std::unique_lock<std::shared_mutex> guard(this->mux);

  // Check if the host pointer is already registered
  // 检查主机指针是否已经注册。
  if (this->allocatedMap.count(hostPtr)) {
    return &this->allocatedMap[hostPtr];
  }

  void *devPtr;
  aclError err = aclrtHostRegister(hostPtr, static_cast<uint64_t>(bufferSize),
                                   ACL_HOST_REGISTER_MAPPED, (void **)&devPtr);

  if (err != ACL_SUCCESS) {
    std::cerr << "Unable to aclrtHostRegister, errcode: " << err << std::endl;
    return nullptr;
  }

  this->allocatedMap.emplace(
      hostPtr, RegisteredMemoryRecord{reinterpret_cast<uintptr_t>(hostPtr),
                                      reinterpret_cast<uintptr_t>(devPtr),
                                      bufferSize, -1});

  return &this->allocatedMap[hostPtr];
};

// Register an existing host-device pointer mapping to the memory manager
// Returns the created RegisteredMemoryRecord
// 将已有的主机—设备指针映射登记到内存管理器。
// 返回创建的 RegisteredMemoryRecord。
// 本函数只登记调用方提供的地址和大小，不分配内存，不调用 ACL/HAL 注册，
// 也不接管 hostPtr 或 devPtr 所代表资源的所有权。
RegisteredMemoryRecord *
HostRegisteredMemoryManager::registerMappedMem(void *hostPtr, void *devPtr,
                                               size_t bufferSize) {
  TORCH_CHECK(
      !(hostPtr == nullptr || devPtr == nullptr || bufferSize == 0),
      "Error: hostPtr and devPtr cannot be null and bufferSize must be greater "
      "than 0.");
  const std::unique_lock<std::shared_mutex> guard(this->mux);

  // Check if the host pointer is already registered
  // 检查主机指针是否已经注册。
  TORCH_CHECK(!(this->allocatedMap.count(hostPtr)),
              "Error: hostPtr already registered to host memory manager.");

  this->allocatedMap.emplace(
      hostPtr, RegisteredMemoryRecord{reinterpret_cast<uintptr_t>(hostPtr),
                                      reinterpret_cast<uintptr_t>(devPtr),
                                      bufferSize, -1});

  return &this->allocatedMap[hostPtr];
};

// Register a pointer through low level APIs (HAL). Allocates a new pinned host
// memory This should be used for driver versions, where cannot rely on
// aclrtHostRegister() Returns the created RegisteredMemoryRecord
// 通过底层 API（HAL）注册指针。分配新的页锁定主机内存。
// 适用于无法依赖 aclrtHostRegister() 的驱动版本，返回创建的记录。
// 当前实现实际接收调用方已分配的 hostPtr，不在此处分配主机内存；
// register_ptr 在旧驱动或版本无法确认时调用本函数，HAL 注册成功后再执行 mlock。
RegisteredMemoryRecord *
HostRegisteredMemoryManager::halRegisterHostPtr(void *hostPtr,
                                                size_t bufferSize) {
  // We allocate a new chunk of memory, register it, and replace the tensor.
  // Essentially, the halHostRegister function requires a ptr given by mmap.
  // 我们分配一块新内存、注册它，并替换 tensor。
  // 本质上，halHostRegister 函数要求传入由 mmap 返回的指针。
  // 上述是原有设计说明；当前代码直接注册传入的 hostPtr，并未替换 tensor。
  TORCH_CHECK((bufferSize >= 0), "Error: bufferSize must be greater than 0.");
  const std::unique_lock<std::shared_mutex> guard(this->mux);

  void *devPtr;
  int device = get_device();
  auto drvRet = halHostRegister(
      (void *)hostPtr, static_cast<UINT64>(bufferSize),
      HOST_MEM_MAP_DEV_PCIE_TH, (UINT32)device, (void **)&devPtr);
  if (drvRet != 0) {
    std::cerr << "Unable to halHostRegister: " << drvRet
              << " . Please ensure your driver version >= 24.1.0" << std::endl;
    return nullptr;
  }

  // Lock the memory and fail if impossible to lock
  // 锁定内存；如果无法锁定则判定失败。
  auto lockErr = mlock(reinterpret_cast<void *>(hostPtr), bufferSize);
  if (lockErr == -1) {
    // This can happen in non-privileged mode or not enough rlimit,
    // let's not proceed since we wanted to guarantee pinned
    // because we already alloced, let's free
    // 在非特权模式或 rlimit 不足时可能发生这种情况；
    // 由于需要保证内存已页锁定，因此不再继续；
    // 因为内存已经分配，接下来将其释放。
    auto ret = halHostUnregisterEx(reinterpret_cast<void *>(hostPtr),
                                   static_cast<UINT32>(device),
                                   HOST_MEM_MAP_DEV_PCIE_TH);
    TORCH_CHECK(
        ret == 0,
        "Unable to pin host memory, unable to unregister. Error code: " +
            std::to_string(ret))
    auto mret = munmap(reinterpret_cast<void *>(hostPtr), bufferSize);
    TORCH_CHECK(false, "Unable to pin host memory with error code: " +
                           std::to_string(lockErr))
  }

  this->allocatedMap.emplace(
      hostPtr,
      RegisteredMemoryRecord{reinterpret_cast<uintptr_t>(hostPtr),
                             reinterpret_cast<uintptr_t>(devPtr), bufferSize,
                             static_cast<int32_t>(device)});

  return &this->allocatedMap[hostPtr];
};

// unregister_ptr 在新驱动分支调用本函数：通过 ACL 解除注册并删除登记记录。
// 本函数不调用 aclrtFreeHost 或 munmap，主机内存由外层 free 路径随后释放。
int HostRegisteredMemoryManager::aclUnregisterHostPtr(void *hostPtr) {
  TORCH_CHECK(hostPtr != nullptr, "Error: hostPtr cannot be null.");

  // we don't actually mind if it doesn't unregister,
  // at context destroy it should be unregister anyway.
  // 实际上并不介意注销是否成功，
  // 因为上下文销毁时无论如何都应注销。
  const std::unique_lock<std::shared_mutex> guard(this->mux);
  if (this->allocatedMap.count(hostPtr) == 0) {
    // we probably did not register anyway
    // 可能原本就没有注册。
    return 0;
  }
  aclError err = aclrtHostUnregister(hostPtr);
  this->allocatedMap.erase(hostPtr);
  return static_cast<int>(err);
};

// unregister_ptr 在旧驱动分支调用本函数：使用记录中的设备号执行 HAL 注销。
// 不解引用或释放主机内存；调用方随后按原分配方式执行 aclrtFreeHost 或 munmap。
int HostRegisteredMemoryManager::halUnregisterHostPtr(void *hostPtr) {
  TORCH_CHECK(hostPtr != nullptr, "Error: hostPtr cannot be null.");
  const std::unique_lock<std::shared_mutex> guard(this->mux);
  if (this->allocatedMap.count(hostPtr) == 0) {
    // we probably did not register anyway
    // 可能原本就没有注册。
    return 0;
  }
  auto record = this->allocatedMap[hostPtr];
  auto err = halHostUnregisterEx(reinterpret_cast<void *>(hostPtr),
                                 static_cast<UINT32>(record.device),
                                 HOST_MEM_MAP_DEV_PCIE_TH);
  return static_cast<int>(err);
}

/*
 *    For now we only do a linear search as we probably won't have a long list
 * of ptrs we go through each record and check whether we are in range, if so we
 * calculate the offset from the host ptr and apply to the device ptr finally we
 * return the device ptr.
 */
/*
 * 目前只进行线性搜索，因为指针列表可能不会很长。
 * 遍历每条记录并检查地址是否位于区间内；若命中，则计算其相对主机指针的
 * 偏移，将该偏移应用到设备指针，最后返回得到的设备指针。
 */
// get_device_ptr 是本查询的入口。匹配范围为半开区间
// [record.ptr, record.ptr + record.buffSize)，所以区间内部地址会返回
// record.devptr + offset，而不要求传入登记时的主机基址。
void *HostRegisteredMemoryManager::getDevicePtr(void *hostPtr) {
  if (hostPtr == nullptr) {
    return nullptr;
  }
  const std::shared_lock<std::shared_mutex> guard(this->mux);

  const uintptr_t hostAddrPtr = reinterpret_cast<uintptr_t>(hostPtr);

  for (const auto &pair : this->allocatedMap) {
    const RegisteredMemoryRecord &record = pair.second;

    if (hostAddrPtr >= record.ptr &&
        hostAddrPtr < (record.ptr + record.buffSize)) {
      const size_t offset = hostAddrPtr - record.ptr;

      const uintptr_t deviceAddrPtr = record.devptr + offset;

      return reinterpret_cast<void *>(deviceAddrPtr);
    }
  }

  return nullptr;
};

// 使用与 getDevicePtr 相同的区间匹配方式，返回整条登记记录的大小；
// 供 HAL 注销辅助路径获取原始映射大小，不返回剩余区间长度。
size_t HostRegisteredMemoryManager::getRecordSize(void *hostPtr) {
  if (hostPtr == nullptr) {
    return 0;
  }
  const std::shared_lock<std::shared_mutex> guard(this->mux);

  const uintptr_t hostAddrPtr = reinterpret_cast<uintptr_t>(hostPtr);

  for (const auto &pair : this->allocatedMap) {
    const RegisteredMemoryRecord &record = pair.second;

    if (hostAddrPtr >= record.ptr &&
        hostAddrPtr < (record.ptr + record.buffSize)) {
      return record.buffSize;
    }
  }
  return 0;
};

// register_ptr 和 unregister_ptr 均通过本函数查询驱动版本。
// 运行时动态加载 libdrvdsmi_host.so 和 dsmi_get_version，避免编译期直接绑定。
std::string get_driver_version() {
  void *handle = nullptr;
  int (*dsmi_get_version)(int, char *, unsigned int, unsigned int *) = nullptr;
  std::string result;

  handle = dlopen("libdrvdsmi_host.so", RTLD_LAZY);
  if (!handle) {
    TORCH_CHECK(false,
                std::string("Error opening libdrvdsmi_host.so: ") + dlerror());
    return result;
  }
  dlerror();

  // Load the function
  // 加载函数。
  *(void **)(&dsmi_get_version) = dlsym(handle, "dsmi_get_version");
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    dlclose(handle);
    TORCH_CHECK(false,
                std::string("Error loading dsmi_get_version: ") + dlsym_error);
    return result;
  }

  // Call the function
  // 调用函数。
  int device_id = c10_npu::getCurrentNPUStream().device_index();
  const unsigned int buffer_size = 256;
  std::vector<char> version_buffer(buffer_size);
  unsigned int ret_len = 0;
  int ret =
      dsmi_get_version(device_id, version_buffer.data(), buffer_size, &ret_len);
  if (ret == 0) {
    if (ret_len > 0 && ret_len <= buffer_size) {
      version_buffer[ret_len] = '\0'; // Ensure null-termination
      // 确保字符串以空字符结尾。
      result = version_buffer.data();
    } else {
      TORCH_CHECK(false,
                  "Error: Invalid length returned: " + std::to_string(ret_len));
    }
  } else {
    TORCH_CHECK(false,
                "Error: dsmi_get_version returned " + std::to_string(ret));
  }

  dlclose(handle);

  return result;
}

// To be on the safe side, returns false in case of uncertainties
// 为稳妥起见，无法确定时返回 false。
// 从版本字符串开头解析主版本号；主版本 >= 25 时选择 ACL，否则选择 HAL。
bool is_version_at_least_25(const std::string &version_str) {
  if (version_str.empty()) {
    return false;
  }

  size_t num_end = 0;
  long major_version = 0;

  try {
    major_version = std::stol(version_str, &num_end);
  } catch (const std::invalid_argument &) {
    // No valid number at start
    // 开头没有有效数字。
    return false;
  } catch (const std::out_of_range &) {
    // Should never happen, here for robustness
    // 理论上不应发生，此处用于增强健壮性。
    return false;
  }
  return major_version >= 25;
}

// HAL 注册和独立注销辅助路径通过本函数取得设备号；
// 设置 ASCEND_RT_VISIBLE_DEVICES 时，将当前逻辑序号转换为列表中的设备号。
int get_device() {
  int device = c10_npu::getCurrentNPUStream().device_index();
  const char *env_visible_devices_p = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  // If we are using a custom list of visible devices, the index refers to that
  // 如果使用自定义可见设备列表，索引就指向该列表。
  if (env_visible_devices_p != nullptr) {
    std::string env_visible_devices = env_visible_devices_p;
    std::vector<uint32_t> list_visible_devices;
    std::stringstream ss(env_visible_devices);
    std::string item;
    while (std::getline(ss, item, ',')) {
      list_visible_devices.push_back(std::stoi(item));
    }
    std::sort(list_visible_devices.begin(), list_visible_devices.end());
    // Here two cases are possible:
    // 1. no hccl, we just use current_device, even though we have specify the
    // ASCEND_RT_VISIBLE_DEVICES
    // 2. hccl, and we use current_device that seems to be correct
    // for case 2, since the current_device would have been correct anyway,
    // obtaining from the list would be fine. for case 1, we have shifted the
    // device to the RT_VISIBLE_DEVICES, so it should be corrected.
    // 这里可能有两种情况：
    // 1. 不使用 HCCL，即使指定了 ASCEND_RT_VISIBLE_DEVICES，
    //    也只使用 current_device。
    // 2. 使用 HCCL，此时使用的 current_device 看起来是正确的。
    // 对情况 2，由于 current_device 本来就正确，从列表中取值也没有问题；
    // 对情况 1，设备号已映射到 RT_VISIBLE_DEVICES，因此需要修正。
    device = list_visible_devices[device];
  }
  return device;
}

// 独立的 HAL 释放辅助路径：先查询登记大小并调用 halHostUnregisterEx，
// 再按该大小 munmap。调用顺序仍是先注销、后释放；调用后主机指针失效。
void hal_host_unregister_ptr(void *ptr) {
  if (ptr) {
    int device = get_device();
    auto &hmm = HostRegisteredMemoryManager::GetInstance();
    size_t bufferSize = hmm.getRecordSize(ptr);
    auto ret = halHostUnregisterEx(reinterpret_cast<void *>(ptr),
                                   static_cast<UINT32>(device),
                                   HOST_MEM_MAP_DEV_PCIE_TH);
    if (ret != 0) {
      std::cout << "Unable to hal host unregister: " << ret << std::endl;
    }
    auto mret = munmap(reinterpret_cast<void *>(ptr), bufferSize);
    if (mret != 0) {
      std::cout << "Unable to unmap memory: " << ret << std::endl;
    }
  }
}

} // namespace lmc

/*
 * Caller should check whether this return nullptr for error handling.
 */
/*
 * 调用方应检查返回值是否为 nullptr，以进行错误处理。
 */
// 对外注册入口，由 alloc_pinned_ptr 和 alloc_pinned_numa_ptr 调用。
// 先查询驱动版本：主版本 >= 25 时调用 registerHostPtr，走 aclrtHostRegister；
// 否则调用 halRegisterHostPtr，走 halHostRegister 和 mlock。
// 返回登记记录中的设备指针，不分配主机内存，也不转移主机内存所有权。
void *register_ptr(void *ptr, size_t size) {
  // assumed this is a host ptr
  // 假定这是一个主机指针。
  TORCH_CHECK(ptr != nullptr, "ptr is a nullptr.");
  auto &hmm = lmc::HostRegisteredMemoryManager::GetInstance();
  std::string verString = lmc::get_driver_version();
  lmc::RegisteredMemoryRecord *record;

  if (lmc::is_version_at_least_25(
          verString)) { // New driver version, supports aclrtHostRegister()
    // 新驱动版本，支持 aclrtHostRegister()。
    record = hmm.registerHostPtr(ptr, size);
  } else {
    record = hmm.halRegisterHostPtr(ptr, size);
  }

  if (record == nullptr) {
    return nullptr;
  }

  return reinterpret_cast<void *>(record->devptr);
}

// 只登记调用方已经建立的 hostPtr/devPtr 映射，供 get_device_ptr 做区间查询。
// 不分配主机或设备内存，不调用 ACL/HAL 注册，也不接管两侧内存的所有权。
void *register_mapping(void *hostPtr, void *devPtr, size_t size) {
  TORCH_CHECK(hostPtr != nullptr, "hostPtr is a nullptr.");
  TORCH_CHECK(devPtr != nullptr, "devPtr is a nullptr.");
  auto &hmm = lmc::HostRegisteredMemoryManager::GetInstance();
  lmc::RegisteredMemoryRecord *record =
      hmm.registerMappedMem(hostPtr, devPtr, size);

  if (record == nullptr) {
    return nullptr;
  }

  return reinterpret_cast<void *>(record->devptr);
}

// 对外注销入口，由 free_pinned_ptr 和 free_pinned_numa_ptr 在释放前调用。
// 与 register_ptr 一样按驱动版本分流到 ACL/HAL 注销；本函数不释放主机内存，
// 外层 free 路径必须在注销返回后再执行 aclrtFreeHost 或 munmap。
int unregister_ptr(void *ptr) {
  TORCH_CHECK(ptr != nullptr, "ptr is a nullptr.");
  auto &hmm = lmc::HostRegisteredMemoryManager::GetInstance();
  std::string verString = lmc::get_driver_version();
  if (lmc::is_version_at_least_25(verString)) {
    return hmm.aclUnregisterHostPtr(ptr);
  } else {
    return hmm.halUnregisterHostPtr(ptr);
  }
}

// 将查询委托给 HostRegisteredMemoryManager::getDevicePtr；
// 支持主机登记区间内部的地址，并按相同偏移返回设备地址。
void *get_device_ptr(void *ptr) {
  auto &hmm = lmc::HostRegisteredMemoryManager::GetInstance();
  return hmm.getDevicePtr(ptr);
};
