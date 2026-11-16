# SPDX-License-Identifier: Apache-2.0

"""Divert SGLang's LMCRadixCache onto the unified path for non-standard pools.

SGLang's ``--enable-lmcache`` branch constructs ``LMCRadixCache``, whose
upstream implementation only understands standard MHA pools (per-layer
``k_buffer`` / ``v_buffer`` lists). Hybrid (Qwen3.8 / GLM-5.3), MLA and
DeepSeek-V4 pools crash there.

This module installs a ``RadixCache.__init_subclass__`` hook while
``lmc_radix_cache`` is still being imported (the plugin activates in the
middle of that import, before the ``class LMCRadixCache(RadixCache):``
statement executes). The hook wraps ``LMCRadixCache.__init__`` with a
dispatcher:

* standard pool  -> the original constructor (legacy connector path,
  already verified with Qwen2.5-7B; the stacked-KV compat patch from
  ``mp_connector_compat`` keeps applying underneath);
* hybrid/MLA/DSV4 pool -> ``LMCacheAscendUnifiedCache.convert`` (see
  ``unified_linker.py``), which re-initializes the instance as a
  ``UnifiedRadixCache`` driving LMCache's unified MP connector.

Heavy SGLang/LMCache imports are deferred to construction time so the
fragile mid-import activation window never touches them.

把 SGLang 的 LMCRadixCache 对非标准池分流到统一路径。

SGLang 的 ``--enable-lmcache`` 分支构造 ``LMCRadixCache``，其上游实现
只认识标准 MHA 池（逐层 ``k_buffer`` / ``v_buffer`` 列表）。hybrid
（Qwen3.8 / GLM-5.3）、MLA、DeepSeek-V4 池在那里会崩溃。

本模块在 ``lmc_radix_cache`` 尚在导入期间安装一个
``RadixCache.__init_subclass__`` 钩子（插件恰在该 import 中途激活，
早于 ``class LMCRadixCache(RadixCache):`` 语句执行）。钩子把
``LMCRadixCache.__init__`` 包装为 dispatcher：

* 标准池 -> 原构造函数（旧连接器路径，Qwen2.5-7B 已验证；
  ``mp_connector_compat`` 的堆叠 KV 兼容补丁在下面继续生效）；
* hybrid/MLA/DSV4 池 -> ``LMCacheAscendUnifiedCache.convert``（见
  ``unified_linker.py``），把实例重新初始化为驱动 LMCache 统一 MP
  连接器的 ``UnifiedRadixCache``。

重量级 SGLang/LMCache 导入全部推迟到构造时，脆弱的 import 中途
激活窗口不碰它们。
"""

# Standard
# 标准库
from typing import Any, Optional

# First Party
# 本项目模块
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Stamp attributes for idempotency / wrap detection.
# 幂等 / 包装检测用的标记属性。
_DIVERSION_FLAG = "_lmcache_ascend_unified_diversion"
_INIT_WRAP_FLAG = "_lmcache_ascend_unified_dispatch"


def install_unified_diversion() -> None:
    """Hook ``RadixCache.__init_subclass__`` (idempotent, light imports).

    Called from the plugin activation block while SGLang is loaded but
    ``lmc_radix_cache`` is still mid-import.
    """
    # Third Party
    # 第三方依赖
    from sglang.srt.mem_cache.radix_cache import RadixCache

    if getattr(RadixCache, _DIVERSION_FLAG, False):
        return

    # Resolve the currently-effective __init_subclass__ (defined here or
    # inherited, e.g. object's) and chain to it so default semantics --
    # including TypeError on stray class kwargs -- are preserved.
    # 解析当前生效的 __init_subclass__（本类定义或继承自如 object），
    # 链式调用以保留默认语义——包括对多余 class kwargs 的 TypeError。
    prev = RadixCache.__init_subclass__.__func__

    def _subclass_hook(cls: type, **kwargs: Any) -> None:
        prev(cls, **kwargs)
        if cls.__name__ == "LMCRadixCache" and not getattr(
            cls.__dict__.get("__init__", None), _INIT_WRAP_FLAG, False
        ):
            _wrap_lmcradix_init(cls)

    RadixCache.__init_subclass__ = classmethod(_subclass_hook)  # type: ignore[assignment]
    setattr(RadixCache, _DIVERSION_FLAG, True)
    logger.info("Installed LMCRadixCache unified-path diversion hook")


def _wrap_lmcradix_init(cls: type) -> None:
    """Replace ``cls.__init__`` with the standard/unified dispatcher."""
    original_init = cls.__init__

    def _dispatch_init(
        self,
        params,
        model_config=None,
        tp_size: int = 1,
        rank: int = 0,
        tp_group=None,
        **kwargs,
    ):
        if _needs_unified(params):
            # Heavy imports deferred to construction time (runtime).
            # 重导入推迟到构造时（运行时）。
            # First Party
            # 本项目模块
            from lmcache_ascend.integration.sglang.unified_linker import (
                LMCacheAscendUnifiedCache,
            )

            LMCacheAscendUnifiedCache.convert(
                self, params, model_config, tp_size, rank, tp_group
            )
            return None
        return original_init(
            self,
            params,
            model_config=model_config,
            tp_size=tp_size,
            rank=rank,
            tp_group=tp_group,
            **kwargs,
        )

    _dispatch_init.__name__ = "LMCRadixCache.__init__"
    setattr(_dispatch_init, _INIT_WRAP_FLAG, True)
    cls.__init__ = _dispatch_init  # type: ignore[assignment]
    logger.info("LMCRadixCache.__init__ wrapped with unified-path dispatcher")


def _get_kvcache(params) -> Optional[Any]:
    """Best-effort KV-pool extraction from the allocator."""
    allocator = getattr(params, "token_to_kv_pool_allocator", None)
    if allocator is None:
        return None
    getter = getattr(allocator, "get_kvcache", None)
    if getter is not None:
        try:
            return getter()
        except Exception:  # noqa: BLE001
            pass
    return getattr(allocator, "_kvcache", None)


def _needs_unified(params) -> bool:
    """Return True for pools the legacy LMCRadixCache cannot serve.

    Detection is attribute/name based (no imports of pool classes), so it
    stays safe inside the mid-import activation window:

    * ``full_kv_pool`` / ``mamba_pool``      -> HybridLinearKVPool family
      (Qwen3.8, GLM-5.3 KDA/DSA hybrids);
    * non-empty ``kv_buffer``                -> MLA-family single latent
      buffer (DeepSeek V2/V3/V4.1 MLA planes);
    * class name prefix ``DeepSeekV4``       -> DSV4 compressed pools.

    Standard MHA pools (NPU stacked ``k_buffer`` included) return False
    and keep the verified legacy path.
    """
    kv_pool = _get_kvcache(params)
    if kv_pool is None:
        return False
    if hasattr(kv_pool, "full_kv_pool") or hasattr(kv_pool, "mamba_pool"):
        return True
    if getattr(kv_pool, "kv_buffer", None):
        return True
    if type(kv_pool).__name__.startswith("DeepSeekV4"):
        return True
    return False


def _pool_family_name(params) -> str:
    """Human-readable pool family for logs."""
    kv_pool = _get_kvcache(params)
    return type(kv_pool).__name__ if kv_pool is not None else "<none>"
