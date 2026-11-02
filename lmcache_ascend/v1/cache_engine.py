# SPDX-License-Identifier: Apache-2.0
"""
LMCacheEngine for Ascend NPU.

"""

# Standard
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import queue
import threading
import time

# Third Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    convert_tokens_to_list,
)
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.token_database import TokenDatabase
import torch

logger = init_logger(__name__)

ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]


class ThreadSafeEventList:
    """queue.Queue-backed, list-compatible thread-safe buffer for
    ``CacheStoreEvent`` objects.

    Upstream ``LMCacheEngine`` treats ``self.kv_events`` as a list that
    callers ``.append(...)`` to and ``get_kv_events`` snapshots-and-resets.
    When ``store_async`` is active, appends happen on the background
    worker thread while drains happen on the main thread — a data race
    on a plain ``list``.  This wrapper funnels all appends through a
    ``queue.Queue`` so we inherit its producer/consumer thread safety,
    while preserving ``.append(...)`` / truthiness semantics that
    upstream code paths rely on.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[CacheStoreEvent]" = queue.Queue()

    def append(self, event: CacheStoreEvent) -> None:
        self._q.put(event)

    def __bool__(self) -> bool:
        return not self._q.empty()

    def __len__(self) -> int:
        return self._q.qsize()

    def drain(self) -> List[CacheStoreEvent]:
        out: List[CacheStoreEvent] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out


class AscendLMCacheEngine(LMCacheEngine):
    """Ascend NPU variant of ``LMCacheEngine`` with an async store path."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        super().__init__(
            config,
            metadata,
            token_database,
            gpu_connector,
            broadcast_fn,
            broadcast_object_fn,
        )
        self.is_store_async = self.config.store_async
        self._store_queue_maxsize = max(0, int(self.config.store_async_max_queue_size))
        if self.is_store_async:
            self._store_queue: Optional[queue.Queue] = None
            self._store_worker_thread: Optional[threading.Thread] = None
            self._store_lock = threading.Lock()
            self._store_cv = threading.Condition(self._store_lock)

            # req_id -> number of in-flight background stores.  Entry
            # removed when count hits 0.
            self._pending_store_reqs: Dict[str, int] = {}
            # req_ids whose generation finished while stores were still
            # draining.  Re-checked on each ``get_finished_stores`` call.
            self._deferred_finished_req_ids: set = set()
            # req_ids already reported as finished_sending, to prevent
            # duplicate reports after the scheduler frees blocks.
            self._reported_finished_store_ids: set = set()

        # Serialize between store and lookup.
        self._engine_state_lock = threading.RLock()

        self._device_id: Optional[int] = None

        # None means auto-estimate in post_init based on available NPU memory.
        self._broadcast_shard_size = self.config.get_extra_config_value(
            "broadcast_shard_size", None
        )

        if self.kv_events_enabled and self.is_store_async:
            self.kv_events = ThreadSafeEventList()

    def _ensure_store_worker(self) -> None:
        if self._store_queue is not None:
            return
        self._store_queue = queue.Queue(maxsize=self._store_queue_maxsize)
        self._store_worker_thread = threading.Thread(
            target=self._store_worker_loop,
            daemon=True,
            name="lmcache-ascend-store-worker",
        )
        self._store_worker_thread.start()

    def post_init(self, **kwargs) -> None:
        """中文导读：输入为上游初始化所需参数（通常含 ``kvcaches``），无显式返回值。
        调用方向为 vLLM/SGLang 连接器构造 → 本函数 → 上游 ``post_init``；
        随后按配置启动异步 store 工作线程，并准备首 rank 广播所需的 NPU stream。
        这里只完成运行期资源接线，实际读写仍由 retrieve/store 入口发起。
        """
        super().post_init(**kwargs)
        if self.is_store_async:
            self._device_id = torch.npu.current_device()
            self._ensure_store_worker()
            queue_mode = "unbounded" if self._store_queue_maxsize == 0 else "bounded"
            logger.info(
                "Ascend async store queue initialized: mode=%s maxsize=%d",
                queue_mode,
                self._store_queue_maxsize,
            )

        # Override upstream broadcast_stream with a dedicated NPU stream
        # so broadcast and to_gpu can execute on separate streams.
        # 中文：用独立 NPU stream 覆盖上游 broadcast_stream，
        # 使广播和 to_gpu 能在不同 stream 上执行。
        if self.save_only_first_rank:
            if not hasattr(self.gpu_connector, "load_stream"):
                raise RuntimeError(
                    "gpu_connector must have 'load_stream' when "
                    "save_only_first_rank is True"
                )
            # Auto-estimate or validate shard_size now that model weights
            # are loaded and NPU memory state is stable.
            # 中文：模型权重已加载且 NPU 显存状态稳定后，自动估算或校验分片大小。
            if self._broadcast_shard_size is None:
                self._broadcast_shard_size = self._estimate_shard_size()
            else:
                estimated = self._estimate_shard_size()
                if self._broadcast_shard_size > estimated:
                    logger.warning(
                        "broadcast_shard_size=%d (user-set) may cause NPU OOM "
                        "(estimated safe max=%d). Suggestions: "
                        "(1) reduce broadcast_shard_size to <= %d; "
                        "(2) or increase --gpu-memory-utilization appropriately "
                        "(note: KV cache also uses free memory, do not set too high)",
                        self._broadcast_shard_size,
                        estimated,
                        estimated,
                    )
                    self._broadcast_shard_size = estimated

            self.broadcast_stream = torch.npu.Stream()
            logger.info(
                "Ascend broadcast stream initialized: shard_size=%d",
                self._broadcast_shard_size,
            )

    def _estimate_shard_size(self) -> int:
        """Estimate a safe ``broadcast_shard_size`` from available NPU memory.

        Uses 1/4 of the free memory as the pool budget so the remainder
        is reserved for KV-cache growth and temporary buffers.

        Returns the estimated shard size, clamped to [1, 16].
        """
        chunk_size = self.metadata.chunk_size
        shapes = self.metadata.get_shapes(chunk_size)
        dtypes = self.metadata.get_dtypes()
        per_chunk_bytes = get_size_bytes(shapes, dtypes)

        device = self.metadata.worker_id
        props = torch.npu.get_device_properties(device)
        total_mem = props.total_memory
        allocated = torch.npu.memory_allocated(device)
        available = total_mem - allocated
        pool_budget = available // 4

        if per_chunk_bytes <= 0:
            raise RuntimeError(
                f"Invalid per-chunk size {per_chunk_bytes} bytes; "
                "cannot estimate broadcast_shard_size. "
                "Check model config (shapes/dtypes)."
            )
        max_shard = int(pool_budget // (2 * per_chunk_bytes))
        if max_shard <= 0:
            raise RuntimeError(
                f"NPU free memory insufficient for broadcast pool "
                f"(available={available / 1024**3:.2f} GB, "
                f"pool_budget={pool_budget / 1024**3:.2f} GB, "
                f"per_chunk={per_chunk_bytes / 1024**2:.1f} MB; "
                f"need at least {2 * per_chunk_bytes / 1024**2:.1f} MB "
                f"pool budget for shard_size=1). "
                "Consider reducing --gpu-memory-utilization or chunk_size."
            )
        recommended = min(max_shard, 16)

        logger.info(
            "Estimated broadcast_shard_size=%d "
            "(per_chunk=%.1f MB, available=%.2f GB, "
            "pool_budget=%.2f GB)",
            recommended,
            per_chunk_bytes / 1024**2,
            available / 1024**3,
            pool_budget / 1024**3,
        )
        return recommended

    def _build_shard_plan(
        self,
        total: int,
        shard_size: int,
    ) -> List[Tuple[int, int]]:
        """Split ``range(total)`` into contiguous shards.

        Returns a list of ``(offset, count)`` tuples such that
        ``offset + count <= total`` and the shards tile the full range
        with at most ``shard_size`` elements each.  The last shard may
        be smaller than ``shard_size`` when ``total`` is not evenly
        divisible.

        :param total: Number of chunks to tile.
        :param shard_size: Maximum width of each shard.
        :return: List of ``(offset, count)`` pairs in increasing offset.
        """
        if total <= 0:
            return []
        step = max(1, shard_size)
        return [(i, min(step, total - i)) for i in range(0, total, step)]

    def _try_release_pending(
        self,
        pending: List[Tuple[List[MemoryObj], torch.npu.Event]],
    ) -> None:
        """Release ref-counts of shards whose ``to_gpu`` Event has fired."""
        remaining: List[Tuple[List[MemoryObj], torch.npu.Event]] = []
        for objs, ev in pending:
            if ev.query():
                for obj in objs:
                    obj.ref_count_down()
            else:
                remaining.append((objs, ev))
        pending[:] = remaining

    def _submit_togpu(
        self,
        ctx: Tuple[List[MemoryObj], List[int], List[int], "torch.npu.Event", int, int],
        load_stream: "torch.npu.Stream",
        pending: List[Tuple[List[MemoryObj], "torch.npu.Event"]],
        **kwargs,
    ) -> None:
        """Enqueue ctx's to_gpu on load_stream and release its slot."""
        objs, starts, ends, ev_bc, idx, slot = ctx
        load_stream.wait_event(ev_bc)
        with torch.npu.stream(load_stream):
            for obj, s, e in zip(objs, starts, ends, strict=False):
                self.gpu_connector.to_gpu(obj, s, e, **kwargs)
            ev_togpu = torch.npu.Event()
            ev_togpu.record()
        self._pool_scatter_ev[slot].record(load_stream)
        pending.append((objs, ev_togpu))
        self._try_release_pending(pending)
        logger.debug(
            "rank=%d shard[%d] cnt=%d submitted",
            self.metadata.worker_id,
            idx,
            len(objs),
        )

    # Ping-pong merged-buffer pool for batched broadcast.  Two slots
    # suffice: while slot A's to_gpu runs on load_stream, slot B
    # receives the next shard's broadcast on broadcast_stream.
    _merged_pool: List["torch.Tensor"] = []
    _pool_scatter_ev: List["torch.npu.Event"] = []
    _pool_bytes: int = 0

    def _ensure_merged_pool(self, max_bytes: int, device: str) -> bool:
        """Allocate the 2-slot ping-pong pool if not yet created or if a
        larger shard demands it.  Returns False on OOM."""
        if self._merged_pool and self._pool_bytes >= max_bytes:
            return True
        try:
            self._merged_pool = [
                torch.empty(max_bytes, dtype=torch.uint8, device=device),
                torch.empty(max_bytes, dtype=torch.uint8, device=device),
            ]
        except RuntimeError as e:
            logger.warning(
                "rank=%d NPU OOM allocating merged-buffer pool "
                "(%d bytes/slot); batched broadcast disabled: %s",
                self.metadata.worker_id,
                max_bytes,
                e,
            )
            self._merged_pool = []
            self._pool_bytes = 0
            return False
        self._pool_scatter_ev = [torch.npu.Event(), torch.npu.Event()]
        self._pool_bytes = max_bytes
        return True

    def _broadcast_metadata_table(
        self,
        reordered_chunks: list,
        shard_size: int,
        first_rank: int,
    ) -> Optional[Dict[str, Any]]:
        """Exchange the chunk metadata table + shard plan in one collective.

        Rank 0 builds the plan; other ranks receive it.  The plan
        contains ``total``, ``shard_plan``, ``meta`` (per-chunk
        metadata), ``shard_layouts`` (per-chunk byte offsets within
        each shard's merged buffer), and ``max_shard_bytes``.
        """
        if self.metadata.is_first_rank():
            total = len(reordered_chunks)
            meta_table: List[Tuple[int, int, Dict[str, Any]]] = [
                (
                    start,
                    end_pos,
                    mem_obj.metadata.to_dict(),
                )
                for (_, mem_obj, start, end_pos) in reordered_chunks
            ]
            shard_plan = self._build_shard_plan(total, shard_size)

            # Pre-compute per-chunk byte offsets within each shard's
            # merged buffer.
            #
            # ``shard_layouts`` parallels ``shard_plan``: each entry is
            # a list of (chunk_global_idx, byte_offset, byte_size)
            # tuples describing where the chunk lives inside the
            # shard's merged buffer.
            shard_layouts: List[List[Tuple[int, int, int]]] = []
            max_shard_bytes = 0
            for offset, count in shard_plan:
                layout: List[Tuple[int, int, int]] = []
                byte_off = 0
                for i in range(offset, offset + count):
                    # Use the logical data size (not phy_size, which
                    # may include alignment padding) so that the
                    # receiver can faithfully reconstruct the chunk
                    # tensor via dtype/shape views.
                    chunk_meta = MemoryObjMetadata.from_dict(meta_table[i][2])
                    chunk_bytes = chunk_meta.get_size()
                    layout.append((i, byte_off, chunk_bytes))
                    byte_off += chunk_bytes
                shard_layouts.append(layout)
                if byte_off > max_shard_bytes:
                    max_shard_bytes = byte_off

            plan: Optional[Dict[str, Any]] = {
                "total": total,
                "shard_plan": shard_plan,
                "meta": meta_table,
                "shard_layouts": shard_layouts,
                "max_shard_bytes": max_shard_bytes,
            }
        else:
            plan = None
        return self.broadcast_object_fn(plan, first_rank)

    def _pipelined_sharded_broadcast_and_load(
        self,
        reordered_chunks: list,
        ret_mask: torch.Tensor,
        **kwargs,
    ) -> None:
        """Shard-level pipelined broadcast + ``to_gpu`` on two NPU streams.

        Phase 1: rank 0 broadcasts the metadata table + shard plan in a
        single collective.  Phase 2: per-shard loop batch-broadcasts on
        ``broadcast_stream`` and defers ``to_gpu`` to ``load_stream``
        one shard behind, so the two overlap on the NPU.
        """
        shard_size = self._broadcast_shard_size
        first_rank = self.metadata.first_rank
        load_stream = self.gpu_connector.load_stream

        plan = self._broadcast_metadata_table(reordered_chunks, shard_size, first_rank)
        if plan is None or plan.get("total", 0) == 0:
            # Sender's CPU mem_objs won't enter the pipeline; release
            # them now to avoid leaking.
            if self.metadata.is_first_rank():
                for _, mem_obj, _, _ in reordered_chunks:
                    mem_obj.ref_count_down()
            return

        if self.metadata.is_first_rank():
            # sender
            self._pipeline_broadcast_and_load(
                plan,
                load_stream,
                reordered_chunks=reordered_chunks,
                **kwargs,
            )
        else:
            # receiver
            self._pipeline_broadcast_and_load(
                plan,
                load_stream,
                ret_mask=ret_mask,
                **kwargs,
            )

    def _fill_shard_sender(
        self,
        merged: "torch.Tensor",
        layout: List[Tuple[int, int, int]],
        meta_table: List[Tuple[int, int, Dict[str, Any]]],
        reordered_chunks: list,
    ) -> Tuple[List[TensorMemoryObj], List[int], List[int]]:
        """Sender: H2D-copy each chunk into ``merged`` and build views.

        Returns ``(objs, starts, ends)``.  Broadcast is issued by the
        caller after this returns.
        """
        objs, starts, ends = [], [], []
        for ci, byte_off, byte_size in layout:
            _, mem_obj, _, _ = reordered_chunks[ci]
            start, end_pos, _ = meta_table[ci]
            raw = mem_obj.raw_tensor
            if raw is None:
                raise ValueError(f"rank=0 chunk [{start}:{end_pos}] raw_tensor is None")

            dst = merged[byte_off : byte_off + byte_size]
            dst.copy_(
                raw.contiguous().view(torch.uint8).flatten(),
                non_blocking=True,
            )

            meta = mem_obj.metadata
            objs.append(
                TensorMemoryObj(
                    raw_data=dst.view(meta.dtype).view(meta.shape),
                    metadata=MemoryObjMetadata(
                        shape=meta.shape,
                        dtype=meta.dtype,
                        address=meta.address,
                        phy_size=meta.phy_size,
                        ref_count=1,
                        fmt=meta.fmt,
                        shapes=meta.shapes,
                        dtypes=meta.dtypes,
                    ),
                    parent_allocator=None,
                )
            )
            starts.append(start)
            ends.append(end_pos)
        return objs, starts, ends

    def _fill_shard_receiver(
        self,
        merged: "torch.Tensor",
        layout: List[Tuple[int, int, int]],
        meta_table: List[Tuple[int, int, Dict[str, Any]]],
        ret_mask: torch.Tensor,
    ) -> Tuple[List[TensorMemoryObj], List[int], List[int]]:
        """Receiver: split ``merged`` into per-chunk views.

        Broadcast must have already been issued into ``merged``.
        Returns ``(objs, starts, ends)`` and updates ``ret_mask``.
        """
        objs, starts, ends = [], [], []
        for ci, byte_off, byte_size in layout:
            start, end_pos, meta_dict = meta_table[ci]
            metadata = MemoryObjMetadata.from_dict(meta_dict)
            dst = merged[byte_off : byte_off + byte_size]
            objs.append(
                TensorMemoryObj(
                    raw_data=dst.view(metadata.dtype).view(metadata.shape),
                    metadata=metadata,
                    parent_allocator=None,
                )
            )
            starts.append(start)
            ends.append(end_pos)
            ret_mask[start:end_pos] = True
        return objs, starts, ends

    def _pipeline_broadcast_and_load(
        self,
        plan: Dict[str, Any],
        load_stream: "torch.npu.Stream",
        *,
        reordered_chunks: Optional[list] = None,
        ret_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Unified sharded broadcast + to_gpu pipeline for both ranks.

        For each shard, on ``broadcast_stream``: sender H2D-copies
        chunks into a ping-pong ``merged_buffer`` slot then broadcasts;
        receiver broadcasts into the slot then splits it back.  In both
        cases ``to_gpu[N]`` is deferred until ``broadcast[N+1]`` is
        enqueued so the two overlap on the NPU.
        """
        is_sender = reordered_chunks is not None
        meta_table = plan["meta"]
        shard_plan = plan["shard_plan"]
        shard_layouts = plan["shard_layouts"]
        device = f"npu:{self.metadata.worker_id}"
        if not self._ensure_merged_pool(plan["max_shard_bytes"], device):
            raise RuntimeError(
                f"Failed to allocate merged broadcast pool on {device} "
                f"({plan['max_shard_bytes']} bytes/slot). "
                "Consider reducing broadcast_shard_size."
            )

        pending: List[Tuple[List[MemoryObj], torch.npu.Event]] = []

        prev_ctx = None
        try:
            for shard_idx, _ in enumerate(shard_plan):
                layout = shard_layouts[shard_idx]
                slot = shard_idx % 2

                # Slot was last used by shard_idx-2; wait for its
                # to_gpu before overwriting.
                if shard_idx >= 2:
                    self.broadcast_stream.wait_event(self._pool_scatter_ev[slot])

                merged = self._merged_pool[slot]
                shard_bytes = sum(s for _, _, s in layout)

                with torch.npu.stream(self.broadcast_stream):
                    if is_sender:
                        objs, starts, ends = self._fill_shard_sender(
                            merged,
                            layout,
                            meta_table,
                            reordered_chunks,
                        )
                        self.broadcast_fn(
                            merged[:shard_bytes], self.metadata.first_rank
                        )
                    else:
                        self.broadcast_fn(
                            merged[:shard_bytes], self.metadata.first_rank
                        )
                        objs, starts, ends = self._fill_shard_receiver(
                            merged,
                            layout,
                            meta_table,
                            ret_mask,
                        )

                    ev_bc = torch.npu.Event()
                    ev_bc.record()

                # Now that broadcast[N+1] is enqueued, submit to_gpu[N].
                if prev_ctx is not None:
                    self._submit_togpu(prev_ctx, load_stream, pending, **kwargs)
                prev_ctx = (objs, starts, ends, ev_bc, shard_idx, slot)

            if prev_ctx is not None:
                self._submit_togpu(prev_ctx, load_stream, pending, **kwargs)

        finally:
            load_stream.synchronize()
            for objs, _ in pending:
                for obj in objs:
                    try:
                        obj.ref_count_down()
                    except Exception:
                        pass
            pending.clear()

            # Sender-only: release the original CPU mem_objs whose
            # raw_tensor was H2D-copied into the merged buffer.
            if is_sender:
                for _, mem_obj, _, _ in reordered_chunks:
                    try:
                        mem_obj.ref_count_down()
                    except Exception:
                        pass

    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        Overrides upstream to use sharded broadcast when
        ``broadcast_shard_size != 0``, reducing peak NPU memory usage by
        interleaving broadcast receive and ``batched_to_gpu`` per shard.

        :param tokens: The tokens of the corresponding KV caches.
        :param mask: The mask for the tokens. Should have the same length as
            tokens. The mask should ALWAYS be like FFFFFTTTTTTT, where True
            means the tokens needs to be matched, and the Falses will ALWAYS
            be at the PREFIX of the tensor.
        :param **kwargs: Forwarded to ``batched_to_gpu``. Should include KV
            cache specific information (e.g., paged KV buffer and the page
            tables).
        :return: Boolean mask indicating which tokens are retrieved. Same
            length as *tokens*. On CPU.
        :raises ValueError: If the number of Falses in the mask is not a
            multiple of the chunk size.

        中文导读：输入为待取回的 token、匹配掩码及 KV 池/槽位等 ``kwargs``，
        返回与 token 等长的 CPU 布尔张量，标记哪些 token 已成功取回。
        调用方向为连接器 load/start_load_kv → 本函数 → token 处理与存储查询 →
        NPU connector；数据方向为 CPU/远端缓存内存对象 → ``batched_to_gpu`` →
        NPU 分页 KV。取回完成后按所有权调用 ``ref_count_down`` 释放读引用，
        首 rank 分片广播已接管的对象不能在这里重复释放。
        """
        # Health check: block operation if LMCache is unhealthy
        # 中文：健康检查；LMCache 状态异常时跳过 retrieve。
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve operation")
            return torch.zeros(len(tokens), dtype=torch.bool)

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )

        # Get req_id for logging
        # 中文：取得 req_id，供日志关联本次请求。
        req_id = self._get_req_id(kwargs)

        tot_kv_size = 0

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        # KVCache Check logging
        # 中文：记录本次 KV 缓存操作，供运行检查与诊断。
        self._log_kvcache_for_check(
            operation="retrieve",
            kwargs=kwargs,
            token_count=num_required_tokens,
            require_req_id=True,
        )

        retrieve_stats = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[ProcessedChunk] = []
        if not self._is_passive():
            with retrieve_stats.profile_process_tokens():
                if self.async_loading:
                    reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )
                else:
                    reordered_chunks, tot_kv_size = self._process_tokens_internal(
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )

        # NOTE(niming) --- Sharded broadcast ---
        # The sharded pipeline handles both broadcast and to_gpu internally,
        # so we do not invoke batched_to_gpu below for either rank.
        # Rank 0's to_gpu reuses the NPU tensors produced by the broadcast
        # and non-rank-0's runs on the broadcast receive buffers, avoiding
        # a second CPU->NPU PCIe transfer.
        # 中文：分片广播流水线内部同时完成 broadcast 和 to_gpu，因此各 rank
        # 不再执行下面的 batched_to_gpu。rank 0 复用广播产生的 NPU 张量，
        # 非 rank 0 在广播接收缓冲上执行 to_gpu，避免第二次 CPU→NPU PCIe 传输。
        if self.save_only_first_rank:
            with retrieve_stats.profile_broadcast():
                self._pipelined_sharded_broadcast_and_load(
                    reordered_chunks, ret_mask, **kwargs
                )
        elif len(reordered_chunks) > 0:
            with retrieve_stats.profile_to_gpu():
                _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
                self.gpu_connector.batched_to_gpu(
                    list(memory_objs), list(starts), list(ends), **kwargs
                )

        # --- Cleanup ---
        # When save_only_first_rank is set, the sharded-broadcast pipeline
        # takes ownership of the sender's CPU mem_objs (it releases them in
        # its finally block after the H2D copies complete).  So we skip the
        # ref_count_down here for the sender to avoid double-free.
        # 中文：清理读引用。save_only_first_rank 下，分片广播流水线接管发送端
        # CPU mem_objs，并在 H2D 完成后的 finally 中释放；因此发送端这里跳过
        # ref_count_down，避免重复释放，其余路径仍按引用计数正常回收。
        skip_refcnt = self.save_only_first_rank and self.metadata.is_first_rank()
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                if self.storage_manager is None:
                    raise ValueError("storage_manager is required for remove")
                self.storage_manager.remove(key, self.retrieve_locations)
                if self._is_sync_pd_backend() and not skip_refcnt:
                    memory_obj.ref_count_down()
            elif not self.async_loading and not skip_refcnt:
                memory_obj.ref_count_down()

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(
            retrieve_stats,
            retrieved_tokens,
        )
        onload_time = retrieve_stats.time_to_retrieve()
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d required tokens "
                "(from %d total tokens). size: %.4f gb, "
                "cost %.4f ms, throughput: %.4f GB/s;",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
                tot_kv_size / 1024**3,
                onload_time * 1000,
                tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
            )
        return ret_mask

    def _store_worker_loop(self) -> None:
        """中文导读：无参数、无显式返回值，是异步 store 的后台消费者循环。
        调用方向为 ``post_init``/``store`` 启动线程 → 本循环 → 从队列取任务 →
        ``_run_store_pipeline``；每个任务结束后递减该请求的 in-flight 计数并
        通知等待者，使 ``get_finished_stores`` 能判断何时允许释放请求 KV 块。
        """
        if not self.is_store_async:
            return
        if self._device_id is not None:
            torch.npu.set_device(self._device_id)
        while True:
            work = self._store_queue.get()
            if work is None:  # poison pill
                # 中文：None 是停止后台循环的“毒丸”信号。
                self._store_queue.task_done()
                break

            (
                req_id,
                tokens,
                hashes,
                offsets,
                mask,
                num_to_store_tokens,
                kwargs,
            ) = work
            try:
                self._run_store_pipeline(
                    req_id, tokens, hashes, offsets, mask, num_to_store_tokens, kwargs
                )
            except Exception:
                logger.exception("Background store failed for req %s", req_id)
            finally:
                with self._store_lock:
                    cnt = self._pending_store_reqs.get(req_id, 1) - 1
                    if cnt <= 0:
                        self._pending_store_reqs.pop(req_id, None)
                    else:
                        self._pending_store_reqs[req_id] = cnt
                    logger.debug(
                        "Async store done for req %s; remaining=%d",
                        req_id,
                        max(cnt, 0),
                    )
                    self._store_cv.notify_all()
                self._store_queue.task_done()

    @torch.inference_mode()
    def _run_store_pipeline(
        self,
        req_id: str,
        tokens: Optional[Union[torch.Tensor, list]],
        hashes: Optional[List[int]],
        offsets: Optional[List[int]],
        mask: Optional[torch.Tensor],
        num_to_store_tokens: int,
        kwargs: dict,
    ) -> None:
        """Shared implementation for sync and async store.
        From upstream store function.

        中文导读：输入为请求 ID、token 或预计算哈希、分块偏移、掩码、待存数量
        及 KV 传输参数，无显式返回值；同步由 ``store`` 直接调用，异步由后台
        ``_store_worker_loop`` 调用。流程先用 ``process_tokens`` 生成键和范围，
        再通过 ``storage_manager.allocate`` 分配 CPU ``memory_objs``；
        数据经 ``batched_from_gpu`` 按 NPU 分页 KV → CPU 内存对象方向写入，
        最后由 ``batched_put`` 提交后端并接管引用计数。若提交前失败，本函数
        逐个 ``ref_count_down``，避免已分配对象泄漏。
        """
        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        # KVCache Check logging
        # 中文：记录本次 KV 缓存操作，供运行检查与诊断。
        self._log_kvcache_for_check(
            operation="Store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=False,
        )

        # Check if freeze mode is enabled
        # 中文：检查是否处于冻结模式；冻结时跳过保存。
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store operation for %d tokens",
                num_to_store_tokens,
            )
            return

        starts: List[int] = []
        ends: List[int] = []
        keys: List[CacheEngineKey] = []
        memory_objs: List[MemoryObj] = []

        tot_kv_size = 0
        tot_token_num = 0

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        with store_stats.profile_process_tokens():
            prev_key = 0
            for start, end, key in self.token_database.process_tokens(
                tokens,
                hashes,
                offsets,
                mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)
                # Allocate the memory object
                # 中文：为当前分块分配 CPU 内存对象，作为 NPU→CPU 保存传输的目标。
                num_tokens = end - start
                kv_shapes = self.metadata.get_shapes(num_tokens)
                kv_dtypes = self.metadata.get_dtypes()

                # TODO (Jiayi): should be batched in the future
                # 中文：待优化项——未来应将这里的逐个分配改为批量分配。
                memory_obj = self.storage_manager.allocate(
                    kv_shapes,
                    kv_dtypes,
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                    fmt=self.fmt,
                )
                if memory_obj is None:
                    logger.warning(
                        "Local cpu memory under pressure so"
                        " choosing to store only "
                        " %d total chunks of KV cache.",
                        len(memory_objs),
                    )
                    break

                starts.append(start)
                ends.append(end)
                keys.append(key)
                memory_objs.append(memory_obj)
                tot_kv_size += memory_obj.get_size()
                tot_token_num += num_tokens

                # Create KV event
                # 中文：如启用事件上报，为当前缓存块创建 KV 保存事件。
                if self.kv_events_enabled:
                    stored_event = CacheStoreEvent(
                        block_hashes=[key.chunk_hash],
                        parent_block_hash=None if start == 0 else prev_key,
                        token_ids=[],
                        block_size=num_tokens,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    if tokens is not None:
                        stored_event.token_ids = convert_tokens_to_list(
                            tokens,
                            start,
                            end,
                        )
                        if isinstance(tokens, torch.Tensor):
                            stored_event.medium = tokens.device
                    elif hashes is not None:
                        stored_event.token_ids = hashes[start : end + 1]
                    logger.debug(
                        (
                            "Added kv cache event '%s' to kv cache events queue"
                            % stored_event
                        )
                    )
                    self.kv_events.append(stored_event)
                    prev_key = key.chunk_hash

        # memory_objs might be empty, directly return to avoid sending tokens
        # 中文：memory_objs 可能为空；此时直接返回，避免提交空传输和空写入。
        if not memory_objs:
            return

        put_submitted = False
        try:
            result = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )
            store_stats.from_gpu_time = result if isinstance(result, float) else 0.0

            with self._engine_state_lock:
                with store_stats.profile_put():
                    transfer_spec = kwargs.get("transfer_spec", None)
                    # TODO: we implicitly rely on batched_put to call ref_count_down
                    # this management should be done in a cleaner way
                    # 中文：这里依赖 batched_put 在接管对象后调用 ref_count_down；
                    # 所以成功提交后不能再次释放，未来应以更清晰的方式管理所有权。
                    self.storage_manager.batched_put(
                        keys,
                        memory_objs,
                        transfer_spec=transfer_spec,
                        location=self.store_location,
                    )
                    put_submitted = True

                self.stats_monitor.on_store_finished(
                    store_stats,
                    tot_token_num,
                )
        except Exception:
            if not put_submitted:
                for mem_obj in memory_objs:
                    mem_obj.ref_count_down()
            raise

        tot_time = store_stats.time_to_store()

        # NOTE(#233): `from_gpu_time` includes a `wait_for_forward` stall since
        # PR #221 (slot_mapping/ordering async copy), so it is NOT pure device
        # copy time. Report the breakdown explicitly to avoid misleading
        # "offload_time" semantics; a dedicated device-copy metric should be
        # added at the connector layer (see Issue #233 fix plan).
        # 中文：自 PR #221 引入 slot_mapping/排序异步复制后，from_gpu_time
        # 包含 wait_for_forward 等待，并非纯设备复制耗时。日志明确拆分各项，
        # 避免误解 offload_time；后续应在 connector 层增加独立复制指标。
        logger.info(
            "[req_id=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_total_time: %.4f ms "
            "(process_tokens: %.4f ms, from_gpu: %.4f ms), "
            "put_time: %.4f ms",
            req_id,
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
            store_stats.process_tokens_time * 1000,
            store_stats.from_gpu_time * 1000,
            store_stats.put_time * 1000,
        )

    def get_finished_stores(self, finished_req_ids: set) -> set:
        """中文导读：输入为生成已结束、等待确认保存状态的请求 ID 集合，
        返回本轮可安全上报为 finished_sending 的 ID 集合；同步模式返回 ``None``。
        调用方向为 vLLM ``get_finished`` → 本函数；仍存在于
        ``_pending_store_reqs`` 的请求继续延后，后台保存计数清零后才可上报，
        并通过去重避免调度器重复释放同一请求的块。
        """
        if not self.is_store_async:
            return None
        result: set = set()
        with self._store_lock:
            # Forget req_ids the scheduler no longer asks about.
            # This bounds the set to at most |finished_req_ids|.
            # 中文：移除调度器不再询问的 req_id，使已上报集合的规模不超过
            # 当前 finished_req_ids 的大小。
            self._reported_finished_store_ids &= finished_req_ids

            for req_id in list(self._deferred_finished_req_ids):
                if req_id not in self._pending_store_reqs:
                    result.add(req_id)
                    self._deferred_finished_req_ids.discard(req_id)

            for req_id in finished_req_ids:
                if req_id in self._reported_finished_store_ids:
                    # Already reported — skip to avoid scheduler seeing
                    # a duplicate finished_sending for blocks it has
                    # already freed.
                    # 中文：已经上报过的请求直接跳过，避免调度器对其已释放的块
                    # 再收到一次重复的 finished_sending。
                    continue
                if req_id in self._pending_store_reqs:
                    self._deferred_finished_req_ids.add(req_id)
                else:
                    result.add(req_id)

            self._reported_finished_store_ids.update(result)
        return result

    def wait_for_pending_stores(self, req_ids: Iterable[str]) -> set[str]:
        """Wait until async stores for the given requests have drained.

        vLLM reports preempted request ids to workers before the next forward can
        overwrite their freed KV blocks.  If one of those ids still has a
        background store reading paged KV, drain it here to avoid store-after-free.
        """
        if not self.is_store_async:
            return set()

        req_id_set = set(req_ids)
        if not req_id_set:
            return set()

        with self._store_cv:
            pending_at_start = {
                req_id for req_id in req_id_set if req_id in self._pending_store_reqs
            }
            if not pending_at_start:
                return set()

            pending_counts = {
                req_id: self._pending_store_reqs[req_id] for req_id in pending_at_start
            }
            logger.info(
                "Waiting for pending async stores before preemption: "
                "req_ids=%s pending_counts=%s",
                sorted(pending_at_start),
                pending_counts,
            )
            start_time = time.monotonic()
            self._store_cv.wait_for(
                lambda: (
                    not any(req_id in self._pending_store_reqs for req_id in req_id_set)
                )
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Pending async stores drained before preemption: req_ids=%s "
            "elapsed=%.4f ms",
            sorted(pending_at_start),
            elapsed_ms,
        )
        return pending_at_start

    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.kv_events_enabled and self.kv_events:
            if self.is_store_async:
                return self.kv_events.drain()
            events = list(self.kv_events)
            self.kv_events.clear()
            return events
        return []

    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        """中文导读：输入为 token 或哈希/偏移、查询后端范围、lookup_id 和 pin
        选项，返回连续命中的 token 数；不直接搬运 KV。
        调用方向为连接器加载前查询 → 本函数 → 上游 ``lookup``；``pin=True``
        会为 lookup_id 固定命中对象，供后续 retrieve 使用，完成后必须调用
        ``lookup_unpin``。引擎锁用于与异步 store 的状态变更互斥。
        """
        # Serialize against the store-worker thread's
        # 中文：通过引擎状态锁与 store 后台线程串行化相关状态操作。
        with self._engine_state_lock:
            return super().lookup(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                search_range=search_range,
                lookup_id=lookup_id,
                pin=pin,
                request_configs=request_configs,
            )

    def lookup_unpin(self, lookup_id: str) -> None:
        """中文导读：输入为此前 ``lookup(pin=True)`` 使用的标识，无显式返回值。
        调用方向为请求加载/保存收尾或中止 → 本函数 → 上游 ``lookup_unpin``，
        释放该请求的查询固定引用；它与 retrieve 后的 ``ref_count_down`` 不同，
        前者解除 lookup 固定，后者归还具体内存对象的引用。
        """
        with self._engine_state_lock:
            super().lookup_unpin(lookup_id)

    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.

        中文导读：输入为待保存的 token 或哈希/偏移、保存掩码及 KV 池/槽位等
        ``kwargs``，无显式返回值。调用方向为连接器 wait_for_save/store_kv →
        本函数 → ``_run_store_pipeline``；数据方向始终是 NPU 分页 KV →
        ``batched_from_gpu`` → ``allocate`` 得到的 CPU 内存对象 →
        ``batched_put`` 存储后端。同步模式直接执行，异步模式只排队并登记
        in-flight 计数，由后台线程完成保存。
        """
        # Health check: block operation if LMCache is unhealthy
        # 中文：健康检查；LMCache 状态异常时跳过 store。
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store operation")
            return

        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )

        if self._is_passive():
            logger.debug("rank=%d ignore store", self.metadata.worker_id)
            return

        assert self.storage_manager is not None

        # Get req_id for logging
        # 中文：取得 req_id，供保存日志和异步任务跟踪使用。
        req_id = self._get_req_id(kwargs)

        # Initialize num_to_store_tokens to avoid reference before assignment
        # 中文：先初始化待保存 token 数，避免后续引用尚未赋值的变量。
        num_to_store_tokens = 0

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device="npu"
            )

        # lmcache-ascend start ---------------------
        if not self.is_store_async:
            self._run_store_pipeline(
                req_id, tokens, hashes, offsets, mask, num_to_store_tokens, kwargs
            )
        else:
            self._ensure_store_worker()
            with self._store_lock:
                self._pending_store_reqs[req_id] = (
                    self._pending_store_reqs.get(req_id, 0) + 1
                )
                pending = self._pending_store_reqs[req_id]

            enqueued = False
            try:
                self._store_queue.put(
                    (
                        req_id,
                        tokens,
                        hashes,
                        offsets,
                        mask,
                        num_to_store_tokens,
                        kwargs,
                    )
                )
                enqueued = True
            finally:
                if not enqueued:
                    with self._store_lock:
                        cnt = self._pending_store_reqs.get(req_id, 1) - 1
                        if cnt <= 0:
                            self._pending_store_reqs.pop(req_id, None)
                        else:
                            self._pending_store_reqs[req_id] = cnt
                        self._store_cv.notify_all()

            logger.debug(
                "Enqueued async store for req %s; pending=%d tokens=%d",
                req_id,
                pending,
                num_to_store_tokens,
            )
        # lmcache-ascend end ---------------------

    def close(self) -> None:
        """Stop the bg worker gracefully, then close the base engine."""
        # Push poison pill first so any in-flight work drains before
        # ``storage_manager.close()`` runs inside ``super().close()``.
        if self._store_queue is not None:
            try:
                # Bounded queues can be full here; avoid blocking forever
                # while still preferring graceful drain.
                while True:
                    try:
                        self._store_queue.put_nowait(None)
                        break
                    except queue.Full:
                        if (
                            self._store_worker_thread is None
                            or not self._store_worker_thread.is_alive()
                        ):
                            logger.warning(
                                "Ascend store queue is full and worker is not alive; "
                                "cannot enqueue shutdown signal cleanly."
                            )
                            break
                        time.sleep(0.01)
                if self._store_worker_thread is not None:
                    self._store_worker_thread.join(timeout=10)
                    if self._store_worker_thread.is_alive():
                        logger.warning(
                            "Ascend store worker did not stop within 10s; "
                            "proceeding with engine shutdown."
                        )
            except Exception:
                logger.exception("Error stopping Ascend store worker")

        super().close()
