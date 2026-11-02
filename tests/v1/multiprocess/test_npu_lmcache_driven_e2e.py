# SPDX-License-Identifier: Apache-2.0
"""E2E: LMCacheDriven store/retrieve across processes on Ascend NPU.

The worker process owns paged MLA-tuple KV planes, exports them through the
plane-aggregating NpuIPCWrapper (one wrapper per layer) plus interprocess
events; the parent drives the real server-side LMCacheDrivenTransferModule
(fake storage bus, real cache context, real transfers, real events).

端到端测试：在 Ascend NPU 上跨进程执行 LMCacheDriven 的 store/retrieve。

worker 进程持有分页的 MLA 元组 KV plane，通过聚合 plane 的
``NpuIPCWrapper``（每层一个 wrapper）和进程间事件导出它们；父进程驱动
真正的服务端 ``LMCacheDrivenTransferModule``。存储总线使用测试替身，
但缓存上下文、传输和事件都是真实实现。

整体调用链：worker 分配并填充 NPU KV pool 的 plane -> 导出 IPC 张量句柄
和生产者事件 -> 父进程解码并导入句柄，创建真实 cache context -> 测试用的
``_FakeStorageManager`` 为对象分配独立的 CPU 暂存空间 -> ``store`` 将
NPU plane 按层 gather 到主机端 -> 同步传输流后校验并修改主机字节 ->
``retrieve`` 将主机字节 scatter 回相同 NPU block -> 再次同步并校验。
这里的 CPU 空间仅用于测试对象读写，不是生产 KV pool 分配器。
"""

# Standard
# 标准库
from contextlib import closing
from types import SimpleNamespace
from typing import Any
import math
import multiprocessing as mp

# Third Party
# 第三方依赖
import pytest
import torch
import torch_npu  # noqa: F401

# First Party
# 本项目测试辅助模块
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
# 第三方依赖
from lmcache.utils import EngineType  # noqa: E402
from lmcache.v1.distributed.api import ObjectKey  # noqa: E402
from lmcache.v1.gpu_connector.utils import LayoutHints  # noqa: E402
from lmcache.v1.multiprocess.custom_types import KVCache  # noqa: E402
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer  # noqa: E402
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (  # noqa: E402
    LMCacheDrivenTransferContext,
    create_transfer_context,
)
from lmcache.v1.platform.devices.npu.ipc_wrapper import NpuIPCWrapper  # noqa: E402
import lmcache.lmcache_native as lmcache_native  # noqa: E402

# First Party
# 本项目测试辅助模块
from tests.v1.multiprocess.test_custom_types import (  # noqa: E402
    get_customized_decoder,
    get_customized_encoder,
)
import lmcache_ascend  # noqa: F401, E402  (applies plugin patches)
# 导入 lmcache_ascend 以应用插件补丁。

NL = 4
NB = 16
BS = 16
CHUNK = 256  # BS * NB
# 一个 chunk 覆盖 NB 个 block、每块 BS 个 token，共 256 个 token。
CHUNK_KEY = ObjectKey(chunk_hash=b"e2e0", model_name="mla-e2e", kv_rank=0)
W_LATENT = 128
W_ROPE = 16
HIDDEN = W_LATENT + W_ROPE

requires_npu = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU hardware is required"
)


def _worker(device_index: int, conn) -> None:
    """在子进程分配 KV plane，导出 IPC 句柄，并保持映射直到测试结束。

    ``device_index`` 选择 worker 使用的 NPU；``conn`` 是与父进程通信的
    Pipe 一端。本函数由测试通过 ``multiprocessing.Process`` 启动，先为每层
    直接分配 latent/rope 两个测试 KV plane，再调用 ``NpuIPCWrapper.wrap``
    聚合一层的 plane。编码后的 wrapper 和两个生产者事件句柄经 ``conn``
    发给父进程，分别供后续 store 与 retrieve 做跨进程同步。
    """
    torch.npu.set_device(device_index)
    device = f"npu:{device_index}"
    wrappers: list[NpuIPCWrapper] = []
    planes: list[torch.Tensor] = []
    for layer in range(NL):
        latent = torch.zeros(NB, BS, 1, W_LATENT, device=device)
        rope = torch.zeros(NB, BS, 1, W_ROPE, device=device)
        for block in range(NB):
            latent[block].fill_(float(layer * 1000 + block) % 251.0)
            rope[block].fill_(float((layer * 7 + block) % 13.0))
        planes.extend([latent, rope])
        # One plane-aggregating wrapper per layer, exactly what
        # wrap_kv_caches produces for vLLM-Ascend's tuple registration.
        # 每层使用一个聚合 plane 的 wrapper，与 vLLM-Ascend 注册元组时
        # wrap_kv_caches 生成的结构完全一致。
        wrappers.append(NpuIPCWrapper.wrap((latent, rope)))
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for plane in planes:
            plane.mul_(1.0)  # materialize writes on the worker stream
            # 在 worker 流上实际执行写入，确保随后记录的事件覆盖这些写入。
    # The parent imports these handles from another process; this CANN
    # build rejects same-process ``from_ipc_handle`` re-imports, so the
    # worker exports one producer event per consumer call.
    # 父进程会从另一个进程导入这些句柄；当前 CANN 构建拒绝同进程再次
    # 调用 ``from_ipc_handle`` 导入，因此 worker 为每次消费方调用各导出
    # 一个生产者事件。
    producer_a = torch.npu.Event(enable_timing=False, interprocess=True)
    producer_a.record(stream)
    producer_b = torch.npu.Event(enable_timing=False, interprocess=True)
    producer_b.record(stream)
    encoder = get_customized_encoder(type=list[NpuIPCWrapper])
    conn.send(
        {
            "wrappers": encoder.encode(wrappers),
            "producer_a": producer_a.ipc_handle(),
            "producer_b": producer_b.ipc_handle(),
            "device_index": device_index,
        }
    )
    conn.recv()  # hold the mappings until the parent finishes
    # 阻塞等待父进程完成，避免 worker 提前退出并释放 IPC 映射。


class _FakeMemoryObj:
    """MemoryObj stand-in hitting the plain-tensor memcpy branch.

    ``MemoryObj`` 的测试替身，用于进入普通张量的 memcpy 分支。
    ``raw_tensor`` 是实际承载字节的 CPU 张量；其余属性只满足传输模块
    读取地址、大小和对象归属所需的接口。
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self.raw_tensor = tensor
        # Object-group staging reads ``meta.address`` as the host-pool offset.
        # A dedicated tensor is the whole allocation, so the offset is 0.
        # 对象组暂存逻辑把 ``meta.address`` 当作主机池内偏移读取。
        # 这里每个独立张量本身就是完整分配，因此偏移为 0。
        self.meta = SimpleNamespace(address=0)
        self.metadata = self.meta

    def get_size(self) -> int:
        return self.raw_tensor.numel() * self.raw_tensor.element_size()

    def parent(self) -> None:
        return None

    @property
    def data_ptr(self) -> int:
        return self.raw_tensor.data_ptr()

    def get_l1_manager(self) -> int:
        return 0


class _FakeReadWindow:
    """模拟读取结果窗口；进入上下文时返回预设对象列表，退出时不做事。"""

    def __init__(self, objects: list[_FakeMemoryObj]) -> None:
        self._objects = objects

    def __enter__(self) -> list[_FakeMemoryObj]:
        return self._objects

    def __exit__(self, *args: object) -> None:
        pass


class _FakeStorageManager:
    """仅供测试的存储接口替身，不是生产环境的内存或 KV pool 分配器。

    真实的 ``LMCacheDrivenTransferModule`` 会调用本对象的读写方法。
    ``reserve_write`` 只是为每个 key 新建独立的 CPU ``uint8`` 张量并记录到
    ``objects`` 字典，用作 store/retrieve 之间的主机暂存；这里不实现生产
    分配器的池化、容量管理、淘汰或并发控制语义。
    """

    def __init__(self) -> None:
        self.objects: dict[Any, _FakeMemoryObj] = {}

    def _bytes(self, layout_desc: Any) -> int:
        """按布局中各层形状和 dtype 计算一个对象需要的总字节数。"""
        return sum(
            math.prod(shape) * dtype.itemsize
            for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
        )

    def reserve_write(
        self, keys: list[Any], layout_desc: Any
    ) -> dict[Any, _FakeMemoryObj]:
        """为 store 的每个 key 分配独立 CPU 暂存张量并返回 key 到对象的映射。

        ``layout_desc`` 只用于计算字节数；该分配发生在测试替身中，随后由
        真实传输路径把 NPU KV 内容写入这些张量。
        """
        reserved = {}
        for key in keys:
            tensor = torch.empty(
                self._bytes(layout_desc), dtype=torch.uint8, device="cpu"
            )
            obj = _FakeMemoryObj(tensor)
            reserved[key] = obj
            self.objects[key] = obj
        return reserved

    def finish_write(self, keys: list[Any], read_locks: int = 0) -> None:
        pass

    def finish_read_prefetched(self, keys: list[Any], read_locks: int = 0) -> None:
        pass

    def finish_write_by_owner(self, completion: Any) -> None:
        pass

    def finish_read_by_owner(self, completion: Any) -> None:
        pass

    def prepare_write_completion(
        self, objects: dict[Any, _FakeMemoryObj]
    ) -> list[tuple[int, list[Any]]]:
        groups: dict[int, list[Any]] = {}
        for key, obj in objects.items():
            groups.setdefault(obj.get_l1_manager(), []).append(key)
        return list(groups.items())

    def prepare_read_completion(
        self, keys: list[Any], l1_owners: Any = None
    ) -> list[Any]:
        return []

    def unsafe_read(
        self, keys: list[Any], l1_owners: Any = None
    ) -> tuple[list[Any], list[_FakeMemoryObj]]:
        """返回字典中已存在的 key 及其对象，供 retrieve 找到主机暂存字节。"""
        found = [key for key in keys if key in self.objects]
        return found, [self.objects[key] for key in found]

    def read_prefetched_results(self, keys: list[Any]) -> "_FakeReadWindow":
        """把已保存对象包装成读取窗口，模拟传输模块的消费接口。"""
        return _FakeReadWindow([self.objects[k] for k in keys])


def _expected_staging() -> torch.Tensor:
    """Rank-3 [L, NB*BS, W] expectation: latent then rope plane per layer.

    构造三维 ``[L, NB*BS, W]`` 的期望主机暂存数据；每层先放 latent
    plane，再放 rope plane。返回 CPU 张量，供 store 和 retrieve 后校验。
    """
    expected = torch.empty(NL, NB * BS, HIDDEN, dtype=torch.float32)
    for layer in range(NL):
        for block in range(NB):
            expected[layer, block * BS : (block + 1) * BS, :W_LATENT] = (
                float(layer * 1000 + block) % 251.0
            )
            expected[layer, block * BS : (block + 1) * BS, W_LATENT:] = float(
                (layer * 7 + block) % 13.0
            )
    return expected


@requires_npu
def test_forced_mode_selects_lmcache_driven_context() -> None:
    """Explicit mode routes NPU tensors to the handle-transfer context.

    显式指定模式时，应把 NPU 张量路由到句柄传输上下文。
    本测试调用 ``create_transfer_context``，断言返回
    ``LMCacheDrivenTransferContext``，然后关闭上下文。
    """
    tensors = {
        f"layer-{i}": torch.zeros(NB, BS, 1, W_LATENT, device="npu:0")
        for i in range(NL)
    }
    tensors.update(
        {
            f"layer-rope-{i}": torch.zeros(NB, BS, 1, W_ROPE, device="npu:0")
            for i in range(NL)
        }
    )
    context = create_transfer_context(
        tensors,
        instance_id=0,
        req_client=object(),
        mode="lmcache_driven",
    )
    assert isinstance(context, LMCacheDrivenTransferContext)
    context.close()


@requires_npu
@pytest.mark.parametrize("device_index", [0, 1])
def test_lmcache_driven_store_and_retrieve_roundtrip(
    device_index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证真实跨进程传输模块完成 NPU -> CPU -> NPU 的对象往返。

    ``device_index`` 选择测试设备，``monkeypatch`` 只替换模块的上下文查询，
    使其返回本测试创建的真实 cache context。流程如下：

    1. 以 spawn 启动 ``_worker``，由它分配并填充 KV plane，再通过 Pipe
       导出每层 wrapper 的 IPC 编码和两个生产者事件句柄。
    2. 父进程解码 wrapper，创建真实 ``NpuDeviceSpec`` cache context，
       由其识别 MLA 元组布局并接入 worker 的 NPU 张量。
    3. ``module.store`` 使用 ``producer_a`` 同步生产者，把指定 block 的
       全部层 gather 到 ``_FakeStorageManager`` 提供的 CPU 对象中。
    4. 同步传输流后校验主机字节并整体加一；``module.retrieve`` 再使用
       ``producer_b``，把修改后的对象 scatter 回相同 block。
    5. 再次同步传输流，逐层、逐 block 校验 NPU plane 的加一结果；最后
       通知 worker 退出。两个返回的非空字节句柄也会一并检查。

    存储管理器和事件总线使用测试替身；IPC、缓存布局、gather/scatter、
    NPU 流同步和事件句柄均走真实实现。
    """
    if device_index >= torch.npu.device_count():
        pytest.skip("single-device box")

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=_worker, args=(device_index, child_conn))
    process.start()
    try:
        message = parent_conn.recv()
        decoder = get_customized_decoder(type=list[NpuIPCWrapper])
        # One wrapper per layer; to_tensor() yields the (latent, rope)
        # plane tuple, so discovery classifies the MLA tuple layout from
        # the in-band structure alone -- no regroup hint involved.
        # 每层一个 wrapper；``to_tensor()`` 产生 ``(latent, rope)`` plane
        # 元组，因此布局发现仅凭传输数据内的结构即可识别 MLA 元组布局，
        # 不需要额外的重分组提示。
        kv_caches: KVCache = list(decoder.decode(message["wrappers"]))

        # Third Party
        # 第三方依赖
        from lmcache.v1.platform.devices.npu import NpuDeviceSpec
        from lmcache.v1.platform.devices.npu.event_ipc import NpuEventIPCBackend

        cache_context = NpuDeviceSpec().create_cache_context(
            kv_caches,
            CHUNK,
            layout_hints=LayoutHints(kv_layout="NHD"),
            engine_group_infos=(),
            engine_type=EngineType.VLLM,
        )
        with closing(cache_context):
            assert (
                cache_context.get_engine_kv_format(0)
                == lmcache_native.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS
            )
            event_backend = NpuEventIPCBackend()
            storage_manager = _FakeStorageManager()

            module = lmcache_driven_transfer.LMCacheDrivenTransferModule(
                SimpleNamespace(
                    chunk_size=CHUNK,
                    null_block_id=0,
                    storage_manager=storage_manager,
                    event_bus=SimpleNamespace(
                        publish=lambda event: None,
                        publish_on_stream=lambda stream, event: None,
                        has_subscribers=lambda event_type: False,
                    ),
                    resolve_obj_keys=lambda key, group_ids: [[CHUNK_KEY]],
                    get_read_owners=lambda request_id: None,
                )
            )
            with closing(module):
                entry = lmcache_driven_transfer.ContextEntry(
                    cache_context=cache_context,
                    model_name="mla-e2e",
                    world_size=1,
                    event_backend=event_backend,
                )
                monkeypatch.setattr(
                    module, "get_and_touch_context_entry", lambda iid: entry
                )
                key = SimpleNamespace(
                    request_id="e2e",
                    cache_salt="",
                    worker_id=0,
                    token_ids=list(range(CHUNK)),
                    start=0,
                    end=CHUNK,
                )

                block_ids = [list(range(NB))]
                # The producer events were exported by the worker process; the
                # module imports them cross-process (the supported path on CANN).
                # 生产者事件由 worker 进程导出；模块跨进程导入它们，这是
                # CANN 支持的路径。store 借此等待 worker 的 plane 写入完成。
                store_handle, stored = module.store(
                    key, 1, block_ids, message["producer_a"]
                )
                assert stored is True
                assert isinstance(store_handle, bytes) and len(store_handle) > 0

                # The module returns after the completion submit; drain the
                # transfer stream before reading the staged host bytes.
                # 模块提交完成操作后就返回；读取主机暂存字节前必须等待传输
                # 流排空，否则 gather 可能尚未完成。
                cache_context.stream.synchronize()

                # Stored bytes equal the worker's plane contents in [L, tokens, W]
                # staging order (latent plane then rope plane per layer).
                # 保存的字节按 ``[L, tokens, W]`` 暂存顺序等于 worker 的
                # plane 内容（每层先 latent plane，再 rope plane）。
                stored_obj = storage_manager.objects[CHUNK_KEY]
                host = stored_obj.raw_tensor.view(torch.float32)
                assert torch.allclose(
                    host.view(NL, NB * BS, HIDDEN), _expected_staging()
                )

                # Retrieve: mutate host bytes, scatter back into the same blocks.
                # Retrieve：先修改主机字节，再 scatter 回相同的 block。
                stored_obj.raw_tensor.view(torch.float32).add_(1.0)
                retrieve_handle, retrieved = module.retrieve(
                    key, 1, block_ids, message["producer_b"]
                )
                assert retrieved is True
                assert isinstance(retrieve_handle, bytes) and len(retrieve_handle) > 0
                cache_context.stream.synchronize()
                # 在读取 NPU plane 前同步，确保 retrieve 的 scatter 已完成。

                # Every imported plane element now carries the +1 mutation. The
                # context's own views are the tensors the scatter wrote into.
                # 现在每个导入的 plane 元素都应带有 +1 修改；cache context
                # 自己持有的视图正是 scatter 写入的张量。
                expected = _expected_staging()
                kv_tensors = cache_context.kv_tensors
                for layer in range(NL):
                    latent = kv_tensors[layer][0]
                    rope = kv_tensors[layer][1]
                    for block in range(NB):
                        exp_latent = (
                            expected[layer, block * BS : (block + 1) * BS, :W_LATENT]
                            + 1.0
                        )
                        exp_rope = (
                            expected[layer, block * BS : (block + 1) * BS, W_LATENT:]
                            + 1.0
                        )
                        assert torch.allclose(latent[block, :, 0, :].cpu(), exp_latent)
                        assert torch.allclose(rope[block, :, 0, :].cpu(), exp_rope)
    finally:
        parent_conn.send("done")
        process.join(timeout=60)
    # Producer-side CANN aborts (SIGABRT) when imported NPU IPC storages
    # are released after the worker; the roundtrip above is the contract.
    # 如果导入的 NPU IPC 存储在 worker 之后释放，生产端 CANN 会以
    # SIGABRT 中止；上面的完整往返流程才是本测试验证的契约。
    if process.exitcode not in (0, -6):
        pytest.fail(f"worker exitcode {process.exitcode}")
