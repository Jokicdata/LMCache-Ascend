# SPDX-License-Identifier: Apache-2.0

"""Relay SGLang's two-list MHA layout onto the per-layer ``(K, V)`` format.

Upstream LMCache models the SGLang MP path as ``TWO_X_NL_X_NB_BS_NH_HS``
(format 9): ``[K_layers, V_layers]`` with all K layers first. The Ascend
block-level transfer kernel only accepts ``NL_X_TWO_X_NB_BS_NH_HS``
(format 16): one ``(K, V)`` tuple per layer, pointers interleaved
``K_0, V_0, K_1, V_1, ...``. SGLang's own detector has no branch that
yields format 16 -- only the vLLM detector does -- so on NPU every store
of a chunk-aligned SGLang MHA request dies with::

    RuntimeError: LMCache-Ascend block-level MP transfer currently
    supports ... (16) ..., got 9

The two layouts hold the same tensors; only the Python nesting differs::

    format 9 : [[K_0, ..., K_n], [V_0, ..., V_n]]
    format 16: [(K_0, V_0), ..., (K_n, V_n)]

This module wraps ``normalize_and_discover_per_layer_formats`` as bound in
the NPU cache-context module (the function is imported there by name, so
rebinding the module attribute intercepts the server-side detection). The
upstream result is kept unless it is exactly the two-equal-list 4-D MHA
shape; then the lists are transposed in place into per-layer tuples and the
format is rewritten to 16. No tensor is copied or recreated: the IPC-
rebuilt tensor objects are re-nested as-is.

This runs inside the ``lmcache server`` process (the plugin activates on
``import lmcache``), where nesting never crosses a process boundary.

把 SGLang 的"两个列表"MHA 布局桥接到逐层 ``(K, V)`` 格式。

上游 LMCache 把 SGLang MP 路径建模为 ``TWO_X_NL_X_NB_BS_NH_HS``
（格式 9）：``[K层, V层]``，所有 K 层在前。昇腾块级传输内核只接受
``NL_X_TWO_X_NB_BS_NH_HS``（格式 16）：每层一个 ``(K, V)`` 元组，指针
按 ``K_0, V_0, K_1, V_1, ...`` 交错。SGLang 自己的检测器没有任何分支
会给出格式 16——只有 vLLM 检测器会——因此在 NPU 上，每个按 chunk 对齐的
SGLang MHA 请求在 store 时都会失败：

    RuntimeError: LMCache-Ascend block-level MP transfer currently
    supports ... (16) ..., got 9

两种布局持有相同的张量，只是 Python 嵌套方式不同：

    格式 9 ：[[K_0, ..., K_n], [V_0, ..., V_n]]
    格式 16：[(K_0, V_0), ..., (K_n, V_n)]

本模块包装 NPU 缓存上下文模块中绑定的
``normalize_and_discover_per_layer_formats``（该函数在那里按名字导入，
因此重新绑定模块属性即可拦截 server 端的检测）。除非上游结果恰好是
"两个等长列表 + 4 维"的 MHA 形态，否则原样保留上游结果；满足时把两个
列表就地转置为逐层元组，并把格式改写为 16。不拷贝、不重建任何张量：
IPC 重建出的张量对象原样重新嵌套。

本补丁运行在 ``lmcache server`` 进程内（``import lmcache`` 时激活插件），
嵌套结构不跨进程边界。
"""

# Standard
# 标准库
from typing import Any

# Third Party
# 第三方依赖
import torch

# First Party
# 本项目模块
import lmcache.lmcache_native as lmcache_native
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Attribute stamped on the wrapper to make installation idempotent.
# 打在包装函数上的标记属性，保证安装幂等。
_PATCHED_FLAG = "_lmcache_ascend_two_list_relay"

# The two-list layout this relay converts from, and the per-layer tuple
# layout the Ascend block-level transfer kernel requires.
# 本桥接的来源（两个列表布局）与昇腾块级传输内核要求的目标（逐层元组布局）。
_SOURCE_FORMAT = lmcache_native.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS
_TARGET_FORMAT = lmcache_native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS


def _is_two_list_mha(
    discovered: Any, formats: Any
) -> bool:
    """Return True when ``discovered`` is exactly ``[K_layers, V_layers]``.

    Requires: the upstream format is the SGLang two-list format, the
    normalized structure is a 2-element list of two equal-length lists,
    and every entry is a 4-D tensor of one shared shape and dtype. The
    strictness matters: anything that merely looks similar (an SWA plus
    FULL pair, a Mamba recurrent group, a DSV4 pool) must fall through
    untouched so upstream keeps producing its own clean error.

    当 ``discovered`` 恰好为 ``[K层, V层]`` 时返回 True。

    要求：上游格式是 SGLang 的两列表格式；归一化后的结构是含两个等长
    列表的 2 元素列表；每个条目都是形状、dtype 一致的 4 维张量。严格性
    很重要：任何只是"长得像"的结构（SWA 加 FULL 的一对、Mamba 递归组、
    DSV4 池）都必须原样放行，让上游继续给出它自己的清晰报错。
    """
    if not formats or any(fmt is not _SOURCE_FORMAT for fmt in formats):
        return False
    if not isinstance(discovered, list) or len(discovered) != 2:
        return False
    k_layers, v_layers = discovered
    if not isinstance(k_layers, list) or not isinstance(v_layers, list):
        return False
    if not k_layers or len(k_layers) != len(v_layers):
        return False
    if not all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in k_layers):
        return False
    if not all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in v_layers):
        return False
    k0 = k_layers[0]
    return all(
        t.shape == k0.shape and t.dtype == k0.dtype
        for t in (*k_layers, *v_layers)
    )


def relay_two_list_to_per_layer(
    discovered: Any, formats: Any
) -> "tuple[Any, Any]":
    """Transpose ``[K_layers, V_layers]`` into per-layer ``(K, V)`` tuples.

    Returns ``(discovered, formats)`` unchanged when the input is not the
    two-list MHA shape. Otherwise returns the transposed structure with
    every per-layer format rewritten to the per-layer tuple format.

    把 ``[K层, V层]`` 转置为逐层 ``(K, V)`` 元组。

    输入不是两列表 MHA 形态时，原样返回 ``(discovered, formats)``。
    否则返回转置后的结构，并把每个逐层格式改写为逐层元组格式。
    """
    if not _is_two_list_mha(discovered, formats):
        return discovered, formats
    k_layers, v_layers = discovered
    per_layer = list(zip(k_layers, v_layers))
    num_layers = len(per_layer)
    relayed_formats = [_TARGET_FORMAT] * len(formats)
    logger.info(
        "LMCache-Ascend relayed SGLang two-list MHA onto per-layer (K, V) "
        "tuples for the NPU transfer kernel: %d layers, format %s -> %s, "
        "layer shape %s dtype %s",
        num_layers,
        _SOURCE_FORMAT.name,
        _TARGET_FORMAT.name,
        tuple(k_layers[0].shape),
        k_layers[0].dtype,
    )
    return per_layer, relayed_formats


def _bind_relay_once() -> None:
    """Rebind the detection function on an already-imported cache context.

    Called only after ``lmcache.v1.platform.devices.npu.cache_context`` has
    been imported, so the module attribute exists and no import happens
    here. Safe to call repeatedly; the second call is a no-op.

    在已导入的缓存上下文模块上重新绑定检测函数。

    仅在 ``lmcache.v1.platform.devices.npu.cache_context`` 已被导入之后
    调用，因此模块属性一定存在，这里不触发任何导入。可重复调用，第二次
    起为空操作。
    """
    # Third Party
    # 第三方依赖
    import lmcache.v1.platform.devices.npu.cache_context as npu_cache_context

    current = npu_cache_context.normalize_and_discover_per_layer_formats
    if getattr(current, _PATCHED_FLAG, False):
        return
    original = current

    def _relay_detect(*args: Any, **kwargs: Any) -> Any:
        discovered, formats = original(*args, **kwargs)
        return relay_two_list_to_per_layer(discovered, formats)

    _relay_detect.__name__ = "normalize_and_discover_per_layer_formats"
    setattr(_relay_detect, _PATCHED_FLAG, True)
    npu_cache_context.normalize_and_discover_per_layer_formats = _relay_detect
    logger.info(
        "Installed SGLang two-list MHA -> per-layer (K, V) relay for the "
        "NPU transfer kernel"
    )


def install_two_list_relay() -> None:
    """Wrap ``NpuDeviceSpec.create_cache_context`` to install the relay lazily.

    ``NpuCacheContext._initialize`` calls
    ``normalize_and_discover_per_layer_formats`` by its module-level name, so
    the interception point is that binding. But ``cache_context`` is imported
    *lazily* inside ``create_cache_context``, and the plugin activates in the
    middle of ``import lmcache`` -- importing that module eagerly from here
    would recurse into ``lmcache`` submodules that are still initialising.
    Wrapping the already-loaded ``create_cache_context`` method defers the
    binding to the first cache-context construction, by which time
    ``import lmcache`` has long finished.

    包装 ``NpuDeviceSpec.create_cache_context`` 以延迟安装桥接。

    ``NpuCacheContext._initialize`` 按模块级名字调用
    ``normalize_and_discover_per_layer_formats``，因此拦截点就是该绑定。
    但 ``cache_context`` 是在 ``create_cache_context`` 内部*懒导入*的，
    而插件在 ``import lmcache`` 中途激活——从这里立即导入该模块会递归
    进入仍在初始化中的 ``lmcache`` 子模块。包装已加载的
    ``create_cache_context`` 方法，把绑定推迟到第一次构造缓存上下文时，
    那时 ``import lmcache`` 早已结束。
    """
    # Third Party
    # 第三方依赖
    from lmcache.v1.platform.devices.npu import NpuDeviceSpec

    if getattr(NpuDeviceSpec.create_cache_context, _PATCHED_FLAG, False):
        return
    original_create = NpuDeviceSpec.create_cache_context

    def _create_cache_context_with_relay(self, *args: Any, **kwargs: Any) -> Any:
        _bind_relay_once()
        return original_create(self, *args, **kwargs)

    _create_cache_context_with_relay.__name__ = "create_cache_context"
    setattr(_create_cache_context_with_relay, _PATCHED_FLAG, True)
    NpuDeviceSpec.create_cache_context = _create_cache_context_with_relay
    logger.info(
        "SGLang two-list MHA -> per-layer (K, V) relay armed for the NPU "
        "transfer kernel (binds on first cache-context construction)"
    )
