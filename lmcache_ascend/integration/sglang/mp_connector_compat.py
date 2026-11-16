# SPDX-License-Identifier: Apache-2.0

"""Runtime compatibility patch for the SGLang MP connector's KV pool layout.

SGLang's 0.5.2x dev mem_cache refactor turned ``MHATokenToKVPool``'s
``k_buffer`` / ``v_buffer`` into single stacked tensors shaped
``[num_layers, tokens, kv_heads, head_dim]``. LMCache's
``LMCacheMPConnector`` (``lmcache/integration/sglang/multi_process_adapter.py``)
still expects per-layer ``list[torch.Tensor]``, so connector construction
dies in ``_validate_sglang_kv_pools`` with::

    RuntimeError: Boolean value of Tensor with more than one value is ambiguous

(``if not k_pool`` cannot be evaluated on a multi-element tensor.)

This module wraps ``LMCacheMPConnector.__init__`` and splits stacked
buffers into per-layer views before calling the original constructor.
Each view shares the underlying device storage (no copy), so the
downstream ``wrap_one_kv_cache`` -> NPU IPC registration behaves exactly
as it does with native per-layer pools.

新版 SGLang（0.5.2x dev）的 mem_cache 重构把 ``MHATokenToKVPool`` 的
``k_buffer`` / ``v_buffer`` 变成了单张堆叠张量
``[层数, tokens, kv_heads, head_dim]``，而 LMCache 的
``LMCacheMPConnector`` 仍期望逐层的 ``list[torch.Tensor]``，导致连接器
构造在 ``_validate_sglang_kv_pools`` 处抛出
``RuntimeError: Boolean value of Tensor ... is ambiguous``。

本模块包装 ``LMCacheMPConnector.__init__``，在调用原构造函数之前把
堆叠张量按层拆成视图列表。每个视图共享底层设备存储（零拷贝），下游
``wrap_one_kv_cache`` -> NPU IPC 注册的行为与原生逐层池完全一致。
"""

# Standard
# 标准库

# Third Party
# 第三方依赖
import torch

# First Party
# 本项目模块
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Attribute stamped on the wrapped __init__ to make installation idempotent.
# 打在包装后 __init__ 上的标记属性，保证安装幂等。
_INIT_PATCHED_FLAG = "_lmcache_ascend_mp_compat"


def _split_stacked_pool(pool: object, name: str) -> object:
    """Return ``pool`` as the per-layer list the connector expects.

    Stacked SGLang buffers (``[L, tokens, H, D]``) are split along dim 0
    into one contiguous view per layer. Anything that is not a multi-dim
    tensor (older SGLang lists, tuples, ...) passes through unchanged so
    the upstream validator still produces its own clean error for
    genuinely invalid pools.

    将堆叠式 SGLang 缓冲（``[层数, tokens, H, D]``）沿第 0 维拆成每层
    一个连续视图。非多维张量的输入（旧版 SGLang 的列表、元组等）原样
    返回，真正的非法输入仍由上游校验函数报出干净的错误。
    """
    if not isinstance(pool, torch.Tensor) or pool.dim() < 2:
        return pool
    layers = [pool[i] for i in range(pool.shape[0])]
    logger.info(
        "SGLang stacked %s buffer shape=%s dtype=%s split into %d per-layer views",
        name,
        tuple(pool.shape),
        pool.dtype,
        len(layers),
    )
    return layers


def install_mp_connector_compat() -> None:
    """Wrap ``LMCacheMPConnector.__init__`` (idempotent, kwargs-based).

    The interception targets the keyword arguments ``k_pool`` / ``v_pool``
    because SGLang's ``lmc_radix_cache.py`` is the only construction site
    and passes both via ``**connector_kwargs``.

    包装 ``LMCacheMPConnector.__init__``（幂等，基于关键字参数）。
    拦截对象是关键字参数 ``k_pool`` / ``v_pool``——SGLang 的
    ``lmc_radix_cache.py`` 是唯一构造点，且两者都通过
    ``**connector_kwargs`` 传入。
    """
    # Third Party
    # 第三方依赖
    import lmcache.integration.sglang.multi_process_adapter as mpa

    connector_cls = getattr(mpa, "LMCacheMPConnector", None)
    if connector_cls is None:
        return
    if getattr(connector_cls.__init__, _INIT_PATCHED_FLAG, False):
        return

    original_init = connector_cls.__init__

    def _compat_init(self, *args, **kwargs):
        # Standard
        # 标准库
        if "k_pool" in kwargs:
            kwargs["k_pool"] = _split_stacked_pool(kwargs["k_pool"], "k_pool")
        if "v_pool" in kwargs:
            kwargs["v_pool"] = _split_stacked_pool(kwargs["v_pool"], "v_pool")
        return original_init(self, *args, **kwargs)

    _compat_init.__name__ = "LMCacheMPConnector.__init__"
    setattr(_compat_init, _INIT_PATCHED_FLAG, True)
    connector_cls.__init__ = _compat_init
    logger.info("Installed SGLang MP connector stacked-KV compat patch")

    # The unified path hits the same stacked-buffer assumption one layer
    # deeper: ``SGLangUnifiedKVAdapter._resolve_pool_tensors`` reads the
    # hybrid pool's ``full_kv_pool.k_buffer`` / ``.v_buffer`` -- on NPU
    # these are single stacked tensors ``[L, pages, page, H, D]``, and its
    # ``if not k_buffer`` list assumption raises the same
    # Boolean-value-ambiguous RuntimeError. Wrap it with the same
    # zero-copy per-layer split so both the legacy connector and the
    # unified adapter accept NPU pools.
    # 统一路径在更深一层撞上同一个堆叠假设：
    # ``SGLangUnifiedKVAdapter._resolve_pool_tensors`` 读 hybrid 池的
    # ``full_kv_pool.k_buffer`` / ``.v_buffer``——NPU 上它们是单张堆叠
    # 张量 ``[L, pages, page, H, D]``，其 ``if not k_buffer`` 的列表
    # 假设抛出同款 Boolean 歧义 RuntimeError。用同样的零拷贝逐层
    # 拆分包装它，使旧连接器与统一 adapter 都接受 NPU 池。
    try:
        # Third Party
        # 第三方依赖
        import lmcache.integration.sglang.unified_kv_adapter as uka

        adapter_cls = getattr(uka, "SGLangUnifiedKVAdapter", None)
        if adapter_cls is not None and not getattr(
            adapter_cls._resolve_pool_tensors, _INIT_PATCHED_FLAG, False
        ):
            original_resolve = adapter_cls._resolve_pool_tensors

            def _compat_resolve_pool_tensors(self, kv_pool):
                k_buffer = getattr(kv_pool, "k_buffer", None)
                v_buffer = getattr(kv_pool, "v_buffer", None)
                # Split stacked tensors on the *pool object* before the
                # original validator runs, so every downstream consumer
                # in the original method sees per-layer lists.
                # 在原校验器运行前于 *池对象* 上拆分堆叠张量，
                # 使原方法内的所有下游消费者都看到逐层列表。
                if isinstance(k_buffer, torch.Tensor) and k_buffer.dim() >= 2:
                    kv_pool.k_buffer = _split_stacked_pool(k_buffer, "k_buffer")
                if isinstance(v_buffer, torch.Tensor) and v_buffer.dim() >= 2:
                    kv_pool.v_buffer = _split_stacked_pool(v_buffer, "v_buffer")
                try:
                    return original_resolve(self, kv_pool)
                finally:
                    # The pool keeps the split lists (they are views of
                    # the same storage), which is also what SGLang's own
                    # runtime expects for the FIA layout; no restore.
                    # 池保留拆分后的列表（同一存储的视图）——这也与
                    # SGLang 运行时对 FIA 布局的期望一致；无需还原。
                    pass

            _compat_resolve_pool_tensors.__name__ = (
                "SGLangUnifiedKVAdapter._resolve_pool_tensors"
            )
            setattr(_compat_resolve_pool_tensors, _INIT_PATCHED_FLAG, True)
            adapter_cls._resolve_pool_tensors = _compat_resolve_pool_tensors
            logger.info(
                "Installed unified KV adapter stacked-KV compat patch"
            )
    except ImportError:
        # Older lmcache builds without the unified adapter stay on the
        # legacy path; nothing to wrap.
        # 没有统一 adapter 的旧版 lmcache 保持旧路径；无需包装。
        logger.debug("unified_kv_adapter not present; skip adapter wrap")
