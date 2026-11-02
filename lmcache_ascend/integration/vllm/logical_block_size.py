# SPDX-License-Identifier: Apache-2.0
"""Wrap LMCache ``get_tokens_per_block`` when Ascend ``block_size`` is physical.

Pre-#13242 ``spec.block_size`` is physical slots; the logical span is
``block_size * compress_ratio``. Post-#13242 that field is already logical, so
LMCache core (``spec.block_size``) is correct and this wrap is skipped.

Detection is class-level at patch time: pool math
``max_memory_usage_bytes`` still names ``compress_ratio`` iff ``block_size`` is
physical. Do not ``getattr`` ``storage_block_size`` (GPU inherited property).

``merge()`` is always wrapped to restore ``compress_ratio`` / ``model_version``
dropped by some Ascend spec merges.

当 Ascend 的 ``block_size`` 表示物理大小时，包装 LMCache 的
``get_tokens_per_block``。

#13242 之前，``spec.block_size`` 是物理槽数，一个 block id 对应的逻辑
token 跨度是 ``block_size * compress_ratio``。#13242 之后该字段已经表示
逻辑 token 数，LMCache 核心直接读取 ``spec.block_size`` 就是正确的，
因此跳过这项包装。可以把它理解为：物理 block 描述缓存里实际占用多少
槽位，逻辑 block 描述对上层模型而言覆盖多少 token；压缩缓存中两者
可能不同。

补丁安装时通过类级实现检测版本语义：当且仅当 ``block_size`` 是物理
槽数时，池容量计算 ``max_memory_usage_bytes`` 仍引用
``compress_ratio``。不要读取 ``storage_block_size``，因为它可能是从 GPU
规格继承的属性，不能代表 Ascend 的语义。

无论是否包装 token 数计算，都会包装 ``merge()``，恢复某些 Ascend
规格合并时丢失的 ``compress_ratio`` 和 ``model_version``。本模块入口是
``install_overrides``，由插件在 vLLM 运行时的导入激活路径调用。
"""

# Future
# 来自 __future__ 的特性
from __future__ import annotations

# Standard
# 标准库
from collections.abc import Iterable
from types import ModuleType
from typing import Any
import sys

_INSTALLED = False
_GET_TOKENS_WRAPPED = False
_UNPATCHED_GET_TOKENS: Any = None

_ASCEND_LEAF_NAMES = frozenset(
    {
        "AscendMLAAttentionSpec",
        "AscendSlidingWindowMLASpec",
        "AscendSFAIndexerCacheSpec",
        "AscendIndexerKPoolStateSpec",
    }
)


def _is_ascend_leaf(spec: Any) -> bool:
    """判断 ``spec`` 是否属于需要物理到逻辑换算的 Ascend 叶子规格。

    沿类型的 MRO 检查类名以同时识别这些规格及其子类；由
    ``tokens_per_block_id`` 调用，返回布尔值。
    """
    return any(cls.__name__ in _ASCEND_LEAF_NAMES for cls in type(spec).__mro__)


def _block_size_is_physical(cls: Any | None) -> bool:
    """True when ``cls.max_memory_usage_bytes`` still uses ``compress_ratio``.

    当 ``cls.max_memory_usage_bytes`` 仍引用 ``compress_ratio`` 时返回
    ``True``，表示 ``block_size`` 是物理槽数。``cls`` 是待检测的规格类；
    类或函数字节码不可用时也保守地返回 ``True``。
    """
    if cls is None:
        return True
    fn = getattr(cls, "max_memory_usage_bytes", None)
    code = getattr(fn, "__code__", None)
    if code is None:
        return True
    return "compress_ratio" in code.co_names


def _ascend_block_size_is_physical() -> bool:
    """True on pre-#13242 vLLM-Ascend (physical ``block_size``).

    在 #13242 之前的 vLLM-Ascend 上返回 ``True``，即 ``block_size`` 表示
    物理槽数。由 ``install_overrides`` 调用；无法导入规格类时沿用旧语义。
    """
    try:
        # Third Party
        # 第三方依赖
        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
    except ImportError:
        return True
    return _block_size_is_physical(AscendMLAAttentionSpec)


def tokens_per_block_id(spec: Any) -> int:
    """Logical tokens covered by one block id of ``spec`` (no DCP).

    返回 ``spec`` 的一个 block id 覆盖的逻辑 token 数，不考虑 DCP。
    对包含 ``kv_cache_specs`` 的组合规格，递归计算各叶子并取最大值；
    普通叶子直接返回 ``block_size``，Ascend 压缩叶子则返回
    ``block_size * compress_ratio``（或相应的 ``tokens_per_state``）。
    本函数由下方包装后的 ``get_tokens_per_block`` 调用。
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        return max(tokens_per_block_id(leaf) for leaf in inner.values())
    block = int(spec.block_size)
    if not _is_ascend_leaf(spec):
        return block
    ratio = int(
        getattr(spec, "compress_ratio", None)
        or getattr(spec, "tokens_per_state", 1)
        or 1
    )
    return block * ratio


def _restore_dropped_merge_fields(merged: Any, specs: list[Any]) -> Any:
    """把首个输入规格中的关键字段恢复到合并结果，并返回该结果。

    ``merged`` 是上游 ``merge`` 的返回值，``specs`` 是参与合并的规格列表。
    仅处理 ``compress_ratio`` 和 ``model_version``；由下方 ``merge`` 包装器
    调用，不重新创建合并对象。
    """
    first = specs[0]
    for name in ("compress_ratio", "model_version"):
        expected = getattr(first, name, None)
        if expected is None:
            continue
        if getattr(merged, name, None) != expected:
            object.__setattr__(merged, name, expected)
    return merged


def _wrap_merge(cls: Any) -> None:
    """在 ``cls.merge`` 外层补上字段恢复，并用标记避免重复包装。"""
    orig = cls.merge
    orig_fn = getattr(orig, "__func__", orig)
    if getattr(orig_fn, "_lmcache_ascend_logical_block_size", False):
        return

    def merge(cls_: type, specs: list[Any]) -> Any:
        """调用原合并实现，再恢复其遗漏字段；参数和返回值与原接口一致。"""
        merged = orig_fn(cls_, specs)
        return _restore_dropped_merge_fields(merged, specs)

    merge._lmcache_ascend_logical_block_size = True  # type: ignore[attr-defined]
    cls.merge = classmethod(merge)


def install_on(
    leaf_classes: Iterable[type],
    uniform_type_cls: type | None = None,
) -> None:
    """Wrap ``merge()`` on the given spec classes (idempotent).

    在给定的规格类上幂等地包装 ``merge()``。``leaf_classes`` 是待处理的
    叶子类；``uniform_type_cls`` 仅为兼容调用接口而保留，不会被使用。
    无返回值，由 ``install_logical_block_size`` 调用。
    """
    del uniform_type_cls
    for cls in leaf_classes:
        if hasattr(cls, "merge"):
            _wrap_merge(cls)


def install_logical_block_size() -> bool:
    """Wrap real vLLM-Ascend spec ``merge()`` if the classes are importable.

    如果真实的 vLLM-Ascend 规格类可以导入，就包装其 ``merge()``。
    成功安装返回 ``True``；依赖不可导入时返回 ``False``。
    """
    try:
        # Third Party
        # 第三方依赖
        from vllm_ascend.core.kv_cache_interface import (
            AscendMLAAttentionSpec,
            AscendSlidingWindowMLASpec,
        )
    except ImportError:
        return False
    install_on((AscendMLAAttentionSpec, AscendSlidingWindowMLASpec))
    return True


def _rebind_get_tokens_per_block(orig: Any, wrapped: Any) -> None:
    """Rebind existing module globals named ``get_tokens_per_block``.

    Only bindings whose value is ``orig`` are replaced. Renamed imports
    such as ``from module import get_tokens_per_block as foo``, existing
    local bindings, and captured references are not updated.

    Access module dictionaries directly to avoid invoking lazy lookup
    or assignment hooks.

    重新绑定各模块现有的、名为 ``get_tokens_per_block`` 的全局变量。

    只替换值恰好为 ``orig`` 的绑定。重命名导入（例如
    ``from module import get_tokens_per_block as foo``）、已有局部绑定以及
    已捕获的引用都不会更新。``orig`` 是上游原函数，``wrapped`` 是支持
    Ascend 逻辑跨度的替代函数；由 ``_wrap_get_tokens_per_block`` 调用。

    直接访问模块字典，以避免触发懒查找或赋值钩子。
    """
    for mod in list(sys.modules.values()):
        if not isinstance(mod, ModuleType):
            continue
        # Attribute probes can invoke lazy imports; only rebind existing aliases.
        # 属性探测可能触发懒导入，因此只重新绑定已经存在的别名。
        namespace = ModuleType.__getattribute__(mod, "__dict__")
        if namespace.get("get_tokens_per_block") is orig:
            namespace["get_tokens_per_block"] = wrapped


def _wrap_get_tokens_per_block() -> None:
    """Replace ``get_tokens_per_block`` with the Ascend-aware span (idempotent).

    用能识别 Ascend 逻辑跨度的实现替换 ``get_tokens_per_block``，并保证
    重复调用不会重复包装。由 ``install_overrides`` 在检测到物理
    ``block_size`` 时调用；内部再调用 ``_rebind_get_tokens_per_block``
    更新已加载模块中的原有别名。
    """
    global _GET_TOKENS_WRAPPED, _UNPATCHED_GET_TOKENS
    if _GET_TOKENS_WRAPPED:
        return
    try:
        # Third Party
        # 第三方依赖
        import lmcache.integration.vllm.kv_cache_groups as kg
    except ImportError:
        return
    orig = kg.get_tokens_per_block
    if getattr(orig, "_lmcache_ascend_logical_block_size", False):
        _GET_TOKENS_WRAPPED = True
        return

    def get_tokens_per_block(kv_cache_spec: Any, dcp_size: int) -> int:
        """返回 KV 组中一个 block id 的逻辑 token 数。

        ``kv_cache_spec`` 是叶子或组合 KV 缓存规格，``dcp_size`` 是 DCP
        并行度。先通过 ``tokens_per_block_id`` 得到未计 DCP 的跨度；
        仅注意力规格在 ``dcp_size > 1`` 时再乘以该并行度。LMCache 的
        KV 组调用方经重新绑定的全局函数进入这里。
        """
        install_logical_block_size()
        block = tokens_per_block_id(kv_cache_spec)
        if dcp_size <= 1:
            return block
        if kg._is_attention_spec(kv_cache_spec):
            return block * dcp_size
        return block

    get_tokens_per_block._lmcache_ascend_logical_block_size = True  # type: ignore[attr-defined]
    _UNPATCHED_GET_TOKENS = orig
    _rebind_get_tokens_per_block(orig, get_tokens_per_block)
    _GET_TOKENS_WRAPPED = True


def install_overrides(*, scale_physical: bool | None = None) -> None:
    """Wrap spec ``merge()``; wrap ``get_tokens_per_block`` only if physical.

    ``scale_physical`` overrides the patch-time detector (tests). ``None``
    means detect from vLLM-Ascend. Idempotent: the first decision sticks.

    始终包装规格的 ``merge()``；只有 ``block_size`` 是物理槽数时，才包装
    ``get_tokens_per_block``。``scale_physical`` 可在测试中覆盖安装时的
    自动检测：``True`` 强制启用跨度换算，``False`` 跳过，``None`` 则从
    vLLM-Ascend 的实际类检测。该入口幂等，第一次作出的决定会保留，无返回值。

    调用链为：导入 ``lmcache_ascend`` -> ``_patch_logical_block_size`` ->
    本函数 -> ``install_logical_block_size``，并按需进入
    ``_wrap_get_tokens_per_block``。
    """
    global _INSTALLED, _GET_TOKENS_WRAPPED
    install_logical_block_size()
    if not _GET_TOKENS_WRAPPED:
        if scale_physical is None:
            scale_physical = _ascend_block_size_is_physical()
        if scale_physical:
            _wrap_get_tokens_per_block()
        else:
            _GET_TOKENS_WRAPPED = True
    _INSTALLED = True
