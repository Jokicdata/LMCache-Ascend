# SPDX-License-Identifier: Apache-2.0

"""Connector driver for the unified SGLang path (hybrid / MLA / DSV4 models).

Bridges SGLang's ``UnifiedRadixCache`` hook points to LMCache's async
``UnifiedLMCacheMPConnector`` (submit/poll/complete) with blocking
semantics, mirroring the legacy ``LMCacheMPConnector`` integration shape
that the standard-model path already uses.

Design notes (why not the ``UnifiedCacheLinker`` ABC):

1. ``MambaComponent.build_external_linker_transfer`` raises
   "does not support external linker mode, will support soon" -- the
   linker-ABC route is blocked for hybrid models at the SGLang side.
2. ``PoolTransfer`` carries storage *hashes*, while the connector keys
   on raw ``token_ids``; our hook points (match_prefix /
   cache_finished_req) hold the real tokens, so no hash<->token
   translation is needed.
3. The connector natively understands per-group device indices
   (``device_indices_by_group``), including the Mamba checkpoint slot,
   so the tree-side bookkeeping stays in ``UnifiedRadixCache`` and only
   transport lives here.

Every public method is fail-open: on transport errors we log, degrade
to "no cache" for that request, and keep serving.

面向统一 SGLang 路径（hybrid / MLA / DSV4 模型）的连接器驱动。

把 SGLang ``UnifiedRadixCache`` 的钩子点桥接到 LMCache 异步
``UnifiedLMCacheMPConnector``（submit/poll/complete），采用阻塞语义——
与标准模型路径已在用的旧 ``LMCacheMPConnector`` 集成形状一致。

设计说明（为什么不用 ``UnifiedCacheLinker`` ABC）：

1. ``MambaComponent.build_external_linker_transfer`` 直接抛
   "does not support external linker mode, will support soon"——
   linker-ABC 路线在 SGLang 侧对 hybrid 模型被封死。
2. ``PoolTransfer`` 携带的是存储*哈希*，而连接器以原始 ``token_ids``
   为键；我们的钩子点（match_prefix / cache_finished_req）手握真实
   tokens，无需哈希<->token 互转。
3. 连接器原生理解逐组设备索引（``device_indices_by_group``），
   含 Mamba checkpoint 槽位，因此树侧簿记留在 ``UnifiedRadixCache``，
   这里只做传输。

所有公开方法均为失败开放：传输出错时记日志、对该请求降级为
"无缓存"，服务不中断。
"""

# Standard
# 标准库
import time
from dataclasses import dataclass
from typing import Any, Optional

# Third Party
# 第三方依赖
import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

# First Party
# 本项目模块
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class _PendingLookup:
    """State carried from match_prefix to init_load_back for one request.

    从 match_prefix 携带到 init_load_back 的单请求状态。
    """

    key: Any  # RadixKey snapshot; RadixKey 切片即前缀键
    operation: Any  # LMCacheLookupOperation
    total_hit_tokens: int
    device_hit_tokens: int
    cache_salt: str


class LMCacheMPDriver:
    """Blocking driver around ``UnifiedLMCacheMPConnector``.

    Constructed once per scheduler process by the unified-path diversion
    (see ``unified_diversion.py``); owned by ``LMCacheAscendUnifiedCache``.

    ``UnifiedLMCacheMPConnector`` 的阻塞式驱动。

    由统一路径分流（见 ``unified_diversion.py``）在每个 scheduler 进程
    构造一次；持有者为 ``LMCacheAscendUnifiedCache``。
    """

    def __init__(
        self,
        cache: Any,
        *,
        config_file: Optional[str],
        model_config: Any,
        tp_size: int,
        tp_rank: int,
        tp_group: Any,
        sliding_window_size: Optional[int],
    ) -> None:
        # Third Party
        # 第三方依赖
        from lmcache.integration.sglang.unified_lmcache_mp_connector import (
            UnifiedLMCacheMPConnector,
        )

        self._cache = cache
        self.enabled = False
        # rid -> pending lookup; rid -> 待处理 lookup。
        self._pending: dict[str, _PendingLookup] = {}

        self.connector = UnifiedLMCacheMPConnector(
            config_file=config_file,
            model_config=model_config,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            page_size=cache.page_size,
            token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
            req_to_token_pool=cache.req_to_token_pool,
            tree_components=cache.tree_components,
            mamba_component=cache.components.get(
                _component_type_mamba(cache),
            ),
            sliding_window_size=sliding_window_size,
        )
        self.enabled = True
        logger.info(
            "LMCache-Ascend unified driver ready (components=%s, chunk=%d)",
            [c.name for c in cache.tree_components],
            self.connector.chunk_size,
        )

    # ------------------------------------------------------------------
    # lookup path (match_prefix hook)
    # 查询路径（match_prefix 钩子）
    # ------------------------------------------------------------------

    def lookup(
        self,
        rid: str,
        tokens: list[int],
        *,
        local_hit_tokens: int,
        cache_salt: str,
        key: Any,
    ) -> int:
        """Blocking prefix lookup; returns total cached tokens (>= local).

        Fail-open: any error logs and returns ``local_hit_tokens``
        (i.e. nothing new beyond the device cache).

        阻塞式前缀查询；返回总命中 token 数（>= local）。
        失败开放：任何错误记日志并返回 ``local_hit_tokens``
        （即设备缓存之外无新增命中）。
        """
        if not self.enabled:
            return local_hit_tokens
        try:
            operation = self.connector.submit_lookup(
                rid,
                tokens,
                local_hit_tokens=local_hit_tokens,
                cache_salt=cache_salt,
            )
        except Exception:
            logger.exception("LMCache unified lookup submit failed for %s", rid)
            return local_hit_tokens

        total = self._poll_lookup(rid, operation)
        if total is None or total <= local_hit_tokens:
            # Nothing restorable beyond the device tree; drop the pending
            # entry so init_load_back takes the plain path.
            # 设备树之外无可恢复内容；丢弃 pending，init_load_back 走原路。
            self._pending.pop(rid, None)
            return local_hit_tokens

        self._pending[rid] = _PendingLookup(
            key=key,
            operation=operation,
            total_hit_tokens=total,
            device_hit_tokens=local_hit_tokens,
            cache_salt=cache_salt,
        )
        return total

    def _poll_lookup(self, rid: str, operation: Any) -> Optional[int]:
        """Poll until the lookup resolves or times out (leader-broadcast).

        Every ``poll_lookup`` call performs one cross-rank collective, and
        all scheduler ranks run ``match_prefix`` in lockstep, so spinning
        here keeps the ranks synchronized the same way the legacy
        connector's blocking lookup did.

        轮询直至 lookup 完成或超时（leader 广播）。

        每次 ``poll_lookup`` 调用包含一次跨 rank 集合通信，而所有
        scheduler rank 以锁步执行 ``match_prefix``，因此在此自旋与
        旧连接器的阻塞式 lookup 一样保持 rank 同步。
        """
        deadline = time.monotonic() + float(self.connector.operation_timeout)
        while True:
            result = self.connector.poll_lookup(operation)
            if result is not None:
                return int(result)
            if time.monotonic() > deadline:
                logger.error("LMCache unified lookup timed out for %s", rid)
                return None
            time.sleep(0.0005)

    def take_pending(self, rid: str) -> Optional[_PendingLookup]:
        """Pop the pending lookup state for ``rid`` (consumed by load)."""
        return self._pending.pop(rid, None)

    # ------------------------------------------------------------------
    # load path (init_load_back hook)
    # 加载路径（init_load_back 钩子）
    # ------------------------------------------------------------------

    def submit_load(
        self,
        pending: _PendingLookup,
        group_device_indices: list[torch.Tensor],
    ) -> Any:
        """Submit the retrieve into pre-allocated per-group device slots."""
        return self.connector.submit_load(
            pending.operation,
            group_device_indices,
            local_hit_tokens=pending.device_hit_tokens,
        )

    def complete_load(self, operation: Any) -> bool:
        """Wait for the retrieve; fail-open logs and returns False."""
        try:
            return bool(self.connector.complete_load(operation, synchronize=True))
        except Exception:
            logger.exception("LMCache unified load failed for %s", operation.request_id)
            return False

    # ------------------------------------------------------------------
    # store path (cache_finished_req hook)
    # 存储路径（cache_finished_req 钩子）
    # ------------------------------------------------------------------

    def store(
        self,
        rid: str,
        tokens: list[int],
        full_device_indices: torch.Tensor,
        *,
        mamba_value: Optional[torch.Tensor],
        cache_salt: str,
    ) -> None:
        """Persist the finished request's prefix (fire, complete, end).

        Called before the tree insert so the request's own device slots
        (incl. its live Mamba state slot) are still valid for the DMA.
        Fail-open: errors never propagate into the serving loop.

        持久化已完成请求的前缀（提交、完成、结束会话）。

        在树插入之前调用，使请求自身的设备槽位（含其活跃 Mamba 状态
        槽）在 DMA 时仍然有效。失败开放：错误不会传入服务循环。
        """
        if not self.enabled:
            return
        try:
            group_indices = self.connector.device_indices_by_group(
                full_device_indices, mamba_value=mamba_value
            )
            operation = self.connector.submit_store(
                rid,
                tokens,
                group_indices,
                device_indices_start=0,
                cache_salt=cache_salt,
            )
            if operation is not None:
                self.connector.complete_store(operation, synchronize=True)
        except Exception:
            logger.exception("LMCache unified store failed for %s", rid)
        finally:
            self.end_session(rid)

    # ------------------------------------------------------------------
    # session lifecycle
    # 会话生命周期
    # ------------------------------------------------------------------

    def end_session(self, rid: str) -> None:
        """Release server-side locks/session for ``rid`` (best effort)."""
        if not self.enabled:
            return
        try:
            self.connector.end_session(rid)
        except Exception:
            logger.exception("LMCache unified end_session failed for %s", rid)

    def clear(self) -> None:
        """Server-side clear on cache reset (best effort)."""
        if not self.enabled:
            return
        try:
            self.connector.clear()
        except Exception:
            logger.exception("LMCache unified clear failed")
        self._pending.clear()

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self.connector.close()
        except Exception:
            logger.exception("LMCache unified close failed")
        self.enabled = False


def _component_type_mamba(cache: Any) -> Any:
    """Return SGLang's ``ComponentType.MAMBA`` (imported lazily)."""
    # Third Party
    # 第三方依赖
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    return ComponentType.MAMBA


def _global_cache_salt() -> str:
    """Best-effort global cache salt (empty when unset)."""
    try:
        # Third Party
        # 第三方依赖
        from sglang.srt.runtime_context import get_memory

        return str(getattr(get_memory(), "cache_salt", "") or "")
    except Exception:  # noqa: BLE001
        return ""


class LMCacheAscendUnifiedCache(UnifiedRadixCache):
    """UnifiedRadixCache driving LMCache's unified MP connector.

    This is the SGLang-side integration that upstream references as
    ``LMCacheUnifiedRadixCache`` but never shipped. Instances are created
    by re-initializing a half-built ``LMCRadixCache`` (see
    ``unified_diversion.py``): ``convert()`` runs ``UnifiedRadixCache.__init__``
    and swaps ``__class__``.

    Hook points (same shape as the legacy LMCRadixCache integration):

    * ``match_prefix``       -> blocking ``submit_lookup``/``poll_lookup``;
    * ``init_load_back``     -> allocate device slots + Mamba state slot,
       ``submit_load``/``complete_load``, insert the restored tail into
       the tree (mirrors ``UnifiedCacheLinkerWrapper.load_back``);
    * ``cache_finished_req`` -> ``submit_store`` before the tree insert
       (the request's own slots are still valid for the DMA);
    * ``cache_unfinished_req``/``reset`` -> session cleanup / clear.

    驱动 LMCache 统一 MP 连接器的 UnifiedRadixCache。

    即上游报错文案里引用却从未实现的 ``LMCacheUnifiedRadixCache``。
    实例通过对半构造的 ``LMCRadixCache`` 重新初始化而来（见
    ``unified_diversion.py``）：``convert()`` 执行
    ``UnifiedRadixCache.__init__`` 并切换 ``__class__``。
    """

    # ---- construction (class swap) ----

    @classmethod
    def convert(
        cls,
        instance: Any,
        params: Any,
        model_config: Any,
        tp_size: int,
        rank: int,
        tp_group: Any,
    ) -> None:
        """Re-init ``instance`` (a half-built LMCRadixCache) as unified.

        ``instance`` must not have run RadixCache.__init__ yet (the
        dispatcher guarantees this by not calling the original for
        unified pools).
        """
        # Third Party
        # 第三方依赖
        from sglang.srt.mem_cache.unified_cache.components import ComponentType
        from sglang.srt.runtime_context import get_memory

        components = [ComponentType.FULL]
        has_mamba = hasattr(params.req_to_token_pool, "mamba_pool")
        if has_mamba:
            components.append(ComponentType.MAMBA)
        # NOTE(v1): SWA / C128 sidecar components are not wired yet;
        # models needing them fall back to FULL-only caching.
        # 注意(v1)：SWA / C128 边车组件尚未接线；需要它们的模型
        # 暂时退化为仅 FULL 缓存。
        params.tree_components = tuple(components)

        UnifiedRadixCache.__init__(instance, params)
        instance.__class__ = cls
        instance._lmc_has_mamba = has_mamba
        instance._lmc_driver = LMCacheMPDriver(
            instance,
            config_file=(get_memory().lmcache_config_file or ""),
            model_config=model_config,
            tp_size=tp_size,
            tp_rank=rank,
            tp_group=tp_group,
            sliding_window_size=params.sliding_window_size,
        )
        logger.info(
            "LMCache-Ascend unified path engaged (components=%s, mamba=%s)",
            [c.name for c in components],
            has_mamba,
        )

    # ---- lookup: match_prefix hook ----

    def match_prefix(self, params: Any) -> Any:
        result = super().match_prefix(params)
        driver = getattr(self, "_lmc_driver", None)
        if driver is None or not driver.enabled:
            return result
        key = getattr(params, "key", None)
        req = getattr(params, "req", None)
        if key is None or req is None:
            return result
        try:
            device_hit = int(result.device_indices.numel())
            if device_hit >= len(key):
                return result
            tokens = list(key.token_ids)
            salt = str(key.cache_salt or "") or _global_cache_salt()
            total = driver.lookup(
                req.rid,
                tokens,
                local_hit_tokens=device_hit,
                cache_salt=salt,
                key=key,
            )
            if total <= device_hit:
                return result
            updates = dict(
                last_host_node=result.best_match_node,
                host_hit_length=total - device_hit,
            )
            if self._lmc_has_mamba:
                # One restorable state slot per hit boundary (matches
                # UnifiedCacheLinkerWrapper semantics).
                # 每个命中边界一个可恢复状态槽（与
                # UnifiedCacheLinkerWrapper 语义一致）。
                updates["mamba_host_hit_length"] = max(
                    result.mamba_host_hit_length, 1
                )
            return result._replace(**updates)
        except Exception:  # noqa: BLE001
            logger.exception("LMCache unified match_prefix failed")
            return result

    # ---- load: init_load_back hook ----

    def init_load_back(self, params: Any):
        driver = getattr(self, "_lmc_driver", None)
        req = getattr(params, "req", None)
        pending = (
            driver.take_pending(req.rid)
            if driver is not None and driver.enabled and req is not None
            else None
        )
        if pending is None:
            return super().init_load_back(params)
        try:
            return self._load_back_via_lmcache(req, pending)
        except Exception:  # noqa: BLE001
            logger.exception(
                "LMCache unified init_load_back failed for %s",
                getattr(req, "rid", None),
            )
            driver.end_session(req.rid)
            return (
                self.tree_core.empty_match_result.device_indices,
                req.last_node,
            )

    def _full_allocator(self) -> Any:
        allocator = self.token_to_kv_pool_allocator
        # SWA-enabled composites expose full_attn_allocator; plain and
        # hybrid allocators are used directly.
        # SWA 复合分配器暴露 full_attn_allocator；普通与 hybrid
        # 分配器直接使用其本身。
        if getattr(self, "is_swa_enabled", False):
            return getattr(allocator, "full_attn_allocator", allocator)
        return allocator

    def _load_back_via_lmcache(self, req: Any, pending: _PendingLookup):
        driver = self._lmc_driver
        total = pending.total_hit_tokens
        device_hit = pending.device_hit_tokens
        num_tokens = total - device_hit

        full_alloc = self._full_allocator()
        available = getattr(full_alloc, "available_size", None)
        if available is not None:
            shortfall = num_tokens - int(available())
            if shortfall > 0:
                self.evict(EvictParams(num_tokens=shortfall))
        full_indices = full_alloc.alloc(num_tokens)
        if full_indices is None:
            logger.warning(
                "LMCache unified load: FULL allocation failed (%d tokens)",
                num_tokens,
            )
            driver.end_session(req.rid)
            return (
                self.tree_core.empty_match_result.device_indices,
                req.last_node,
            )
        full_indices = full_indices.to(torch.int64)

        mamba_value = None
        group_indices = [full_indices]
        if self._lmc_has_mamba:
            token_pool = self.req_to_token_pool
            mamba_value = token_pool.mamba_allocator.alloc(1)
            if mamba_value is None:
                full_alloc.free(full_indices)
                logger.warning("LMCache unified load: mamba slot alloc failed")
                driver.end_session(req.rid)
                return (
                    self.tree_core.empty_match_result.device_indices,
                    req.last_node,
                )
            group_indices.append(
                token_pool.translate_mamba_indices(mamba_value.view(-1))
            )

        operation = driver.submit_load(pending, group_indices)
        if not driver.complete_load(operation):
            full_alloc.free(full_indices)
            if mamba_value is not None:
                self.req_to_token_pool.mamba_allocator.free(mamba_value)
            return (
                self.tree_core.empty_match_result.device_indices,
                req.last_node,
            )

        # Insert the restored tail into the tree so later requests (and
        # the tree's own eviction accounting) see it, mirroring
        # UnifiedCacheLinkerWrapper.load_back.
        # 把恢复的尾段插入树中，使后续请求（及树自身的淘汰记账）可见，
        # 镜像 UnifiedCacheLinkerWrapper.load_back。
        prefix_key = pending.key[:total]
        value = torch.cat([req.prefix_indices.to(torch.int64), full_indices])
        insert_result = self.insert(
            InsertParams(
                key=prefix_key,
                value=value,
                mamba_value=(
                    mamba_value[:1] if mamba_value is not None else None
                ),
                prev_prefix_len=device_hit,
                swa_evicted_seqlen=(
                    req.kv.swa_evicted_seqlen if req.kv is not None else 0
                ),
                chunked=True,
                priority=getattr(req, "priority", 0) or 0,
                track_adopted_ranges=True,
            )
        )
        if mamba_value is not None and insert_result.mamba_exist:
            self.req_to_token_pool.mamba_allocator.free(mamba_value[:1])

        canonical = self.tree_core.collect_full_device_indices(
            insert_result.last_device_node, req.last_node
        )
        if int(canonical.numel()) != num_tokens:
            logger.debug(
                "LMCache unified load: canonical %d != allocated %d tokens",
                int(canonical.numel()),
                num_tokens,
            )
        return canonical, insert_result.last_device_node

    # ---- store: cache_finished_req hook ----

    def cache_finished_req(self, req: Any, **kwargs) -> Any:
        driver = getattr(self, "_lmc_driver", None)
        if driver is not None and driver.enabled:
            try:
                self._store_via_lmcache(driver, req)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "LMCache unified store failed for %s", req.rid
                )
        return super().cache_finished_req(req, **kwargs)

    def _store_via_lmcache(self, driver: LMCacheMPDriver, req: Any) -> None:
        tokens = list(getattr(req, "origin_input_ids", []) or [])
        full_indices = getattr(req, "prefix_indices", None)
        if not tokens or full_indices is None or int(full_indices.numel()) == 0:
            return
        mamba_value = None
        if self._lmc_has_mamba and req.kv is not None:
            slot = getattr(req.kv, "mamba_pool_idx", None)
            if slot is not None:
                mamba_value = slot.reshape(-1)
        driver.store(
            req.rid,
            tokens,
            full_indices.to(torch.int64),
            mamba_value=mamba_value,
            cache_salt=_global_cache_salt(),
        )

    # ---- lifecycle ----

    def cache_unfinished_req(self, req: Any, chunked: bool = False, **kwargs) -> Any:
        driver = getattr(self, "_lmc_driver", None)
        if driver is not None and driver.enabled and not chunked:
            # Request aborted/evicted before finishing: release the
            # server-side session (and any held lookup locks). Chunked
            # prefill boundaries keep the session alive.
            # 请求在完成前被中止/逐出：释放 server 侧会话（及已持有的
            # lookup 锁）。chunked prefill 边界保留会话。
            driver.take_pending(req.rid)
            driver.end_session(req.rid)
        return super().cache_unfinished_req(req, chunked=chunked, **kwargs)

    def reset(self) -> None:
        driver = getattr(self, "_lmc_driver", None)
        if driver is not None:
            driver.clear()
        super().reset()
