#pragma once
#include <map>
#include <shared_mutex>
#include <string>

namespace lmc {

struct RegisteredMemoryRecord {
  uintptr_t ptr;
  uintptr_t devptr;
  size_t buffSize;
  int32_t device;
};

/*
 * We are not responsible for acl init and ctx initialization,
 * we assume the user responsible for ctx initialization
 */
/*
 * 本类不负责 ACL 初始化和上下文初始化，
 * 假定上下文初始化由用户负责。
 */
class HostRegisteredMemoryManager {
private:
  HostRegisteredMemoryManager();

  // Delete copy constructor and assignment operator
  // 删除拷贝构造函数和赋值运算符。
  HostRegisteredMemoryManager(const HostRegisteredMemoryManager &) = delete;
  HostRegisteredMemoryManager &
  operator=(const HostRegisteredMemoryManager &) = delete;
  HostRegisteredMemoryManager(HostRegisteredMemoryManager &&) = delete;
  HostRegisteredMemoryManager &
  operator=(HostRegisteredMemoryManager &&) = delete;

  std::map<void *, RegisteredMemoryRecord> allocatedMap;
  mutable std::shared_mutex mux;

public:
  static HostRegisteredMemoryManager &GetInstance() {
    static HostRegisteredMemoryManager instance;
    return instance;
  }
  ~HostRegisteredMemoryManager();

  // Register a pointer through high level APIs (aclrt) return devPtr
  // Returns an already existing RegisteredMemoryRecord or the newly created one
  // Inputs:
  // -hostPtr: host pointer of the allocated memory area to register on device
  // -bufferSize: size of the allocated memory area to register on device
  // 通过高层 API（aclrt）注册指针并返回 devPtr。
  // 返回已存在的 RegisteredMemoryRecord，或新建的记录。
  // 输入：
  // - hostPtr：要在设备上注册的已分配主机内存区域指针。
  // - bufferSize：要在设备上注册的已分配主机内存区域大小。
  // 由 register_ptr 在新驱动分支调用；只注册并登记，不分配主机内存。
  RegisteredMemoryRecord *
  registerHostPtr(void *hostPtr,
                  size_t bufferSize); // torch::Tensor& tensor); //
  // Register a pointer through low level APIs (hal)
  // This should be used for driver versions, where cannot rely on
  // aclrtHostRegister() Returns the created RegisteredMemoryRecord Inputs:
  // -hostPtr: host pointer of the allocated memory area to register on device
  // -bufferSize: size of the allocated memory area to register on device
  // 通过底层 API（HAL）注册指针。
  // 适用于无法依赖 aclrtHostRegister() 的驱动版本，返回创建的记录。
  // 输入：
  // - hostPtr：要在设备上注册的已分配主机内存区域指针。
  // - bufferSize：要在设备上注册的已分配主机内存区域大小。
  // 由 register_ptr 在旧驱动分支调用，并负责 HAL 注册与锁定调用方提供的内存。
  RegisteredMemoryRecord *halRegisterHostPtr(void *hostPtr, size_t bufferSize);
  // 只把调用方已有的 hostPtr/devPtr 映射和大小登记到管理器；
  // 不分配内存，也不调用 ACL/HAL 注册接口，不取得底层内存所有权。
  RegisteredMemoryRecord *registerMappedMem(void *hostPtr, void *devPtr,
                                            size_t bufferSize);
  // 分别通过 ACL 或 HAL 解除注册；主机内存本身的释放由调用方负责。
  int aclUnregisterHostPtr(void *hostPtr);
  int halUnregisterHostPtr(void *hostPtr);
  // 支持传入登记区间内部的地址：命中半开区间后返回 devptr 加对应偏移。
  void *getDevicePtr(void *hostPtr);
  // 查询地址所属登记区间的总大小；未命中或空指针返回 0。
  size_t getRecordSize(void *hostPtr);
  // 析构时批量解除登记并清空记录表，不负责释放主机内存。
  void unregisterAll();
};

// 动态查询当前设备的驱动版本，供注册和注销路径进行版本分流。
std::string get_driver_version();
bool is_version_at_least_25(const std::string &version_str);
// Uregisters the malloced hostPtr
// 注销已分配的主机指针。
// 该辅助函数还按登记大小解除 mmap 映射，调用后不得再使用或重复释放该指针。
void hal_host_unregister_ptr(void *ptr);

} // namespace lmc

// 注册调用方提供的主机内存并返回设备指针：驱动主版本 >= 25 走 ACL，
// 其他或无法确认版本的情况走 HAL；不在此分配主机内存。
void *register_ptr(void *ptr, size_t size);
// 按驱动版本选择 ACL/HAL 注销路径，只解除注册，不释放主机内存。
int unregister_ptr(void *ptr);
// 仅登记已经存在的映射，不分配或注册内存，也不接管 hostPtr/devPtr 的所有权。
void *register_mapping(void *hostPtr, void *devPtr, size_t size);

// Takes in input a host pointer, returns the corresponding device pointer
// 输入一个主机指针，返回对应的设备指针。
// 指针可位于已登记区间内部，返回设备基址加相同的区间内偏移；未登记则返回空指针。
void *get_device_ptr(void *ptr);
