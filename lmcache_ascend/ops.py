# SPDX-License-Identifier: Apache-2.0
"""Ops-only binding surface for ``NpuDeviceOps.ensure_native``.

Mirrors upstream's ``lmcache.cuda_ops`` contract: ops plus
``PageBufferShapeDesc`` only. The compiled ``c_ops`` additionally exports
the format enums and ``is_*`` predicates that upstream keeps on
``lmcache_native`` alone. Binding either through
``bind_native`` would violate the device-ops facade contract or shadow
the class-level NPU overrides, so both are filtered out here.

为 ``NpuDeviceOps.ensure_native`` 提供仅含操作的绑定界面。

本模块遵循上游 ``lmcache.cuda_ops`` 的约定：只提供操作函数和
``PageBufferShapeDesc``。编译出的 ``c_ops`` 还额外导出格式枚举和
``is_*`` 谓词，而上游只把这些名字放在 ``lmcache_native`` 上。若经
``bind_native`` 绑定它们，会破坏 device-ops 门面约定，或遮蔽类级 NPU
覆盖实现，因此这里会过滤这些名字。

导入本模块时会遍历 ``_c_ops``，把允许的公开函数和类写入模块全局命名
空间，供 ``NpuDeviceOps.ensure_native`` 的后续原生绑定使用；与
``_torch_ops`` 中相同的合并回退对象会被跳过。
"""

# Third Party
# 第三方依赖
from lmcache.v1.platform import torch_ops as _torch_ops

# First Party
# 本项目模块
import lmcache_ascend.c_ops as _c_ops

# Names upstream keeps on lmcache_native only; never on the ops facade.
# 这些名字在上游只属于 lmcache_native，绝不放入 ops 门面。
_FACADE_EXCLUDES = frozenset(
    (
        "TransferDirection",
        "EngineKVFormat",
        "GPUKVFormat",
        "is_cross_layer",
        "is_kv_list",
        "is_layer_list",
        "is_mla",
    )
)

for _name in dir(_c_ops):
    if _name.startswith("_") or _name in _FACADE_EXCLUDES:
        continue
    _symbol = getattr(_c_ops, _name)
    if not (callable(_symbol) or isinstance(_symbol, type)):
        continue
    if _symbol is getattr(_torch_ops, _name, None):
        continue  # merged torch fallback; class-level NPU overrides win
    # 上面的跳过条件：已合并的 torch 回退实现，让类级 NPU 覆盖优先。
    globals()[_name] = _symbol
