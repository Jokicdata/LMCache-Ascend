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


def _fold_page_native_pool(kv_pool: object) -> bool:
    """Fold a page-native NPU pool's per-layer buffers to slot-native rows.

    SGLang's NPU memory pools allocate each layer as
    ``[pages, page_size, heads, dim]`` where dim 0 counts *pages* (plus
    one padding sentinel page), while LMCache's unified connector treats
    dim 0 as *token slots* and requires ``rows % slots_per_block == 0``
    with ``slots_per_block == page_size``.

    Each already-split per-layer buffer is re-viewed as
    ``[pages * page_size, heads, dim]`` -- a zero-copy reshape of the
    same storage that satisfies the connector's divisibility contract
    (every page contributes exactly ``page_size`` rows, so the sentinel
    page needs no truncation). The connector later re-folds these rows
    back into blocks via ``_to_wire_block_tensor(slots_per_block=
    page_size)``, and page 0 keeps SGLang's padding-sink semantics that
    the store path already handles (``allow_dummy_page``). Buffers that
    are not 4-D page-native (already slot-native, or mamba state) pass
    through.

    Returns True when at least one buffer was re-viewed, so callers can
    restore the engine-native layout afterwards (the fold is for the
    connector's eyes only -- SGLang's own attention reads
    ``pool.k_buffer[layer_id]`` every forward and must keep seeing the
    page-native objects the engine created).

    把 page-native NPU 池的逐层缓冲折叠为 slot-native 行。

    SGLang 的 NPU 内存池把每层分配为 ``[页数, page_size, 头数, 维度]``，
    第 0 维计数的是"页"；而 LMCache 统一连接器把第 0 维当"token 槽"，
    并要求 ``行数 % slots_per_block == 0`` 且 ``slots_per_block ==
    page_size``。

    对已拆分的每个逐层缓冲重视图为 ``[页数 * page_size, 头数, 维度]``
    ——同一存储的零拷贝 reshape，满足整除契约（每页恰好贡献
    ``page_size`` 行，哨兵页无需截断）。连接器随后经
    ``_to_wire_block_tensor(slots_per_block=page_size)`` 把这些行重折回
    block，页 0 保持 SGLang 的 padding-sink 语义（存储路径已通过
    ``allow_dummy_page`` 处理）。非 4-D page-native 的缓冲（已是
    slot-native、或 mamba 状态）原样通过。

    至少一个缓冲被重视图时返回 True，供调用方随后还原引擎原生布局
    （折叠只给连接器看——SGLang 自己的注意力每个 forward 都读
    ``pool.k_buffer[layer_id]``，必须继续看到引擎创建的 page-native
    原对象）。
    """
    # Third Party
    # 第三方依赖
    import torch as _torch

    page_size = getattr(kv_pool, "page_size", None)
    if not page_size or page_size <= 1:
        return False
    folded_any = False
    for attr in ("k_buffer", "v_buffer"):
        buffer = getattr(kv_pool, attr, None)
        if not isinstance(buffer, list):
            continue
        folded = []
        for layer in buffer:
            if (
                isinstance(layer, _torch.Tensor)
                and layer.dim() == 4
                and layer.shape[1] == page_size
            ):
                rows = layer.shape[0] * page_size
                folded.append(layer.reshape(rows, layer.shape[2], layer.shape[3]))
                folded_any = True
            else:
                folded.append(layer)
        setattr(kv_pool, attr, folded)
    if folded_any:
        logger.info(
            "SGLang page-native pool folded to slot-native rows (page_size=%d)",
            page_size,
        )
    return folded_any


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
                # Snapshot the engine-native buffer objects. The pool
                # attributes are mutated below ONLY so the connector's
                # validator/registration sees the per-layer slot-native
                # layout it demands; SGLang's own attention re-reads
                # ``pool.k_buffer[layer_id]`` on every forward through
                # ``get_key_buffer``, so the pool must keep exposing the
                # exact page-native objects the engine created. Leaving
                # the folded views installed corrupts every attention
                # call (garbage output under low load, PagedAttention
                # acl failures under load).
                # 先快照引擎原生的缓冲对象。下面仅在池属性上临时改装，
                # 让连接器的校验/注册看到它要求的逐层 slot-native 布局；
                # 而 SGLang 自己的注意力每个 forward 都经
                # ``get_key_buffer`` 重读 ``pool.k_buffer[layer_id]``，
                # 池必须继续暴露引擎创建的 page-native 原对象。若把
                # 折叠视图留在池上，每次注意力调用都会读到错误布局
                # （低负载输出乱码、高负载 PagedAttention acl 失败）。
                snapshot = (k_buffer, v_buffer)
                mutated = False
                if isinstance(k_buffer, torch.Tensor) and k_buffer.dim() >= 2:
                    kv_pool.k_buffer = _split_stacked_pool(k_buffer, "k_buffer")
                    mutated = True
                if isinstance(v_buffer, torch.Tensor) and v_buffer.dim() >= 2:
                    kv_pool.v_buffer = _split_stacked_pool(v_buffer, "v_buffer")
                    mutated = True
                # Page-native -> slot-native fold (connector-side view).
                # Page-native -> slot-native 折叠（连接器侧视图）。
                if _fold_page_native_pool(kv_pool):
                    mutated = True
                try:
                    return original_resolve(self, kv_pool)
                finally:
                    if mutated:
                        # Restore the engine-native layout. The connector
                        # captured its own tensor references during
                        # ``original_resolve`` (``self._kv_caches``), and
                        # those zero-copy views share the same storage,
                        # so transfers stay correct while the engine
                        # keeps reading its native layout.
                        # 还原引擎原生布局。连接器在 ``original_resolve``
                        # 期间已捕获自己的张量引用（``self._kv_caches``），
                        # 这些零拷贝视图与原对象共享同一存储，因此传输
                        # 不受影响，而引擎继续使用原生布局。
                        kv_pool.k_buffer, kv_pool.v_buffer = snapshot
                        logger.info(
                            "Restored engine-native KV pool layout after "
                            "connector registration (connector keeps its "
                            "own zero-copy views)"
                        )

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
