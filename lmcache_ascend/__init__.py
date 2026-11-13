# SPDX-License-Identifier: Apache-2.0

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
# version.py 应当是独立的库，并且始终最先导入版本库。
# 某些定制化功能依赖这一假设。
from ._version import __version__ as __version__  # noqa: F401  # isort:skip
from ._version import __version_tuple__ as __version_tuple__  # noqa: F401  # isort:skip

# Standard
# 标准库
from typing import Any
import sys

# NOTE: Must be manually edited per each version and
# is also used by the test infrastructure.
# 注意：每个版本都必须手动修改此值，测试基础设施也会使用它。
LMCACHE_UPSTREAM_TAG = "v0.5.6"
LMCACHE_ASCEND_PATCHED = False


def _is_vllm_runtime():
    """判断当前进程是否处于 vLLM 运行时。

    由本文件底部的插件激活逻辑调用；已加载 ``vllm`` 模块，或命令行参数
    包含 ``vllm`` 时返回 ``True``，据此决定是否安装逻辑 block 大小补丁。
    """
    return "vllm" in sys.modules or any("vllm" in arg for arg in sys.argv)


def _is_sglang_runtime():
    """判断当前进程是否处于 SGLang 运行时。

    由本文件底部的插件激活逻辑调用；已加载 ``sglang`` 模块，或命令行
    参数包含 ``sglang`` 时返回 ``True``，据此决定是否安装 SGLang MP
    连接器兼容补丁。SGLang 的 scheduler 进程在导入
    ``lmcache.integration.sglang.multi_process_adapter`` 时会先触发
    ``import lmcache``（进而激活本插件），此时 ``sglang`` 早已完整加载。
    """
    return "sglang" in sys.modules or any("sglang" in arg for arg in sys.argv)


def _seed_partial_lmcache_device_ops() -> None:
    """Bridge ``lmcache.device_ops`` while ``import lmcache`` is still running.

    On the serving path the plugin is first pulled in by lmcache's own
    platform init (``NpuDeviceOps.ensure_native`` -> ``import
    lmcache_ascend.c_ops``), i.e. in the middle of ``import lmcache`` --
    before ``lmcache/__init__.py`` binds its ``device_ops`` singleton (which
    lives after the platform import that invoked us). Core modules imported
    during the patch activation (``lmcache.v1.kv_layer_groups``, ...) do
    ``from lmcache import device_ops`` at module level and would raise
    ``ImportError`` on the half-initialized package, aborting the whole
    activation -- lmcache then soft-fails to "plugin not found" and every MP
    transfer stays on the torch baseline (~8x slower).

    Seed a lazy proxy for exactly that window: ``lmcache/__init__.py``
    overwrites the attribute with the real singleton right after platform
    init returns, and any proxy holder bound during the window transparently
    delegates to it on every access. No-op whenever ``device_ops`` already
    exists (plugin imported after lmcache finished, or plugin-first).

    在 ``import lmcache`` 尚未执行完时，桥接 ``lmcache.device_ops``。

    服务路径中，LMCache 自身的平台初始化会先加载本插件
    （``NpuDeviceOps.ensure_native`` -> ``import lmcache_ascend.c_ops``）。
    此时仍在 ``import lmcache`` 中途，``lmcache/__init__.py`` 尚未执行到
    平台导入之后的 ``device_ops`` 单例绑定。补丁激活期间导入的核心模块
    （如 ``lmcache.v1.kv_layer_groups``）会在模块级执行
    ``from lmcache import device_ops``；半初始化包上没有该属性就会抛出
    ``ImportError``，使激活中止。随后 LMCache 将其软降级为“找不到插件”，
    所有多进程传输都会留在 torch 基线路径（约慢 8 倍）。

    因此只在这个时间窗口预置懒代理：平台初始化返回后，
    ``lmcache/__init__.py`` 会立即用真正的单例覆盖该属性；此前拿到代理的
    调用方也会在每次访问时透明委托给真正的单例。如果 ``device_ops``
    已存在（LMCache 导入完成后才加载插件，或插件先导入），本函数不做事。

    调用方是本文件底部的模块导入激活逻辑；被调方包括 LMCache 包、
    Ascend ``c_ops`` 扩展和 torch 基线 ``DeviceOps``。
    """
    # Third Party
    # 第三方依赖
    import lmcache

    if hasattr(lmcache, "device_ops"):
        return

    class _LazyDeviceOpsProxy:
        """Resolve the real ``lmcache.device_ops`` singleton per access.

        Accesses before lmcache binds its singleton (module-level grabs such
        as ``system_detection``'s ``get_gpu_pci_bus_id = device_ops.
        get_gpu_pci_bus_id``) are served from the ascend extension directly
        -- the same function objects ``bind_native`` later puts on the real
        instance -- and anything the extension does not export falls back to
        the torch-baseline method, so no caller degrades to a broken binding.

        每次访问属性时都解析真正的 ``lmcache.device_ops`` 单例。

        在 LMCache 绑定单例之前发生的访问（例如 ``system_detection``
        在模块级保存 ``get_gpu_pci_bus_id = device_ops.get_gpu_pci_bus_id``）
        会直接由 Ascend 扩展响应；这些正是 ``bind_native`` 稍后放到真实
        实例上的同一批函数对象。扩展未导出的属性回退到 torch 基线方法，
        因而不会让调用方拿到失效绑定。代理本身不缓存真实单例，所以即使
        模块级代码一直持有代理，也能在 LMCache 初始化完成后转到真实对象。
        """

        def __getattr__(self, name: str) -> Any:
            """按名称解析设备操作；优先真实单例，再用 Ascend，最后回退基线。

            ``name`` 是调用方读取的属性名，返回对应函数或属性。本方法由
            Python 的属性访问机制触发，是懒代理连接调用方与上述实现的关键。
            """
            # Standard
            # 标准库
            import sys

            real = getattr(sys.modules.get("lmcache"), "device_ops", None)
            if real is not None and not isinstance(real, _LazyDeviceOpsProxy):
                return getattr(real, name)
            # First Party
            # 本项目模块
            import lmcache_ascend.c_ops as ascend_c_ops

            sym = getattr(ascend_c_ops, name, None)
            if sym is not None:
                return sym
            # Third Party
            # 第三方依赖
            from lmcache.v1.platform.base.device_ops import DeviceOps

            return getattr(DeviceOps(), name)

    # Overwritten with the real singleton by lmcache's own __init__ as soon
    # as the platform import currently in progress returns.
    # 当前正在进行的平台导入一返回，lmcache 自身的 __init__ 就会用真正的
    # 单例覆盖这里预置的代理。
    lmcache.device_ops = _LazyDeviceOpsProxy()  # type: ignore[assignment]

    # Third Party
    # 第三方依赖
    from lmcache.logging import init_logger

    init_logger(__name__).debug("Seeded lazy lmcache.device_ops proxy (partial import)")


def _patch_logical_block_size():
    """Wrap LMCache ``get_tokens_per_block`` when Ascend ``block_size`` is physical.

    Pre-#13242 pool math still uses ``block_size * compress_ratio``; the wrap
    reports that span for Ascend leaves. Post-#13242 ``block_size`` is already
    logical, so LMCache core is left unpatched. Never mutate ``spec.block_size``.

    当 Ascend 的 ``block_size`` 表示物理大小时，包装 LMCache 的
    ``get_tokens_per_block``。#13242 之前的池容量计算仍使用
    ``block_size * compress_ratio``，包装器会为 Ascend 叶子规格报告这一
    逻辑 token 跨度；#13242 之后 ``block_size`` 已是逻辑大小，不再修改
    LMCache 核心。任何情况下都不直接改写 ``spec.block_size``。

    本函数由模块导入激活逻辑在 vLLM 运行时调用，并调用
    ``logical_block_size.install_overrides`` 完成检测与安装。
    """
    # First Party
    # 本项目模块
    from lmcache_ascend.integration.vllm.logical_block_size import install_overrides

    install_overrides()


def _patch_make_page_buffer_shape_desc():
    """Attach per-plane byte geometry to upstream's desc factory (fmt 16/17).

    为上游描述符工厂附加逐 plane 的字节几何信息（格式 16/17）。

    本函数由模块导入激活逻辑调用，并委托
    ``shape_desc.install_plane_geometry_fill`` 包装上游工厂。
    """
    # First Party
    # 本项目模块
    from lmcache_ascend.v1.shape_desc import install_plane_geometry_fill

    install_plane_geometry_fill()


def _patch_sglang_mp_connector_compat():
    """Adapt the SGLang MP connector to SGLang's stacked k/v pool buffers.

    SGLang's 0.5.2x dev mem_cache refactor exposes ``k_buffer`` /
    ``v_buffer`` as single stacked tensors; LMCache's
    ``LMCacheMPConnector`` still expects per-layer ``list[Tensor]`` and
    dies with "Boolean value of Tensor ... is ambiguous" at construction.

    将 SGLang MP 连接器适配新版 SGLang 的堆叠式 k/v 池缓冲。

    本函数由模块导入激活逻辑在 SGLang 运行时调用，并委托
    ``mp_connector_compat.install_mp_connector_compat`` 包装连接器构造，
    把堆叠张量按层拆成视图列表后转交原构造函数（零拷贝）。
    """
    # First Party
    # 本项目模块
    from lmcache_ascend.integration.sglang.mp_connector_compat import (
        install_mp_connector_compat,
    )

    install_mp_connector_compat()


def _patch_sglang_unified_diversion():
    """Divert LMCRadixCache onto the unified path for hybrid/MLA/DSV4 pools.

    Installs a ``RadixCache.__init_subclass__`` hook while SGLang's
    ``lmc_radix_cache`` module is still being imported, so the
    ``class LMCRadixCache(RadixCache):`` statement gets a dispatching
    ``__init__``: standard pools keep the verified legacy path, while
    hybrid (Qwen3.8 / GLM-5.3), MLA and DeepSeek-V4 pools re-initialize
    as ``LMCacheAscendUnifiedCache`` driving LMCache's unified MP
    connector.

    对 hybrid/MLA/DSV4 池把 LMCRadixCache 分流到统一路径。

    在 SGLang 的 ``lmc_radix_cache`` 模块仍在导入期间安装
    ``RadixCache.__init_subclass__`` 钩子，使 ``class LMCRadixCache
    (RadixCache):`` 语句获得一个分派式 ``__init__``：标准池保持已验证
    的旧路径；hybrid（Qwen3.8 / GLM-5.3）、MLA、DeepSeek-V4 池则重新
    初始化为驱动 LMCache 统一 MP 连接器的
    ``LMCacheAscendUnifiedCache``。
    """
    # First Party
    # 本项目模块
    from lmcache_ascend.integration.sglang.unified_diversion import (
        install_unified_diversion,
    )

    install_unified_diversion()


# Check if we've already patched to avoid redundant work
# 检查是否已打补丁，避免重复工作。
if not LMCACHE_ASCEND_PATCHED:
    # Must run before any patch: the activation imports core modules that
    # bind ``lmcache.device_ops`` at module level, which only exists once
    # ``import lmcache`` finished -- and we are typically invoked from inside
    # it (see the docstring).
    # 必须在所有补丁之前运行：激活过程会导入在模块级绑定
    # ``lmcache.device_ops`` 的核心模块，而该属性通常要等
    # ``import lmcache`` 完成后才存在；我们通常正是在这次导入内部被调用
    # （详见上面的 docstring）。
    _seed_partial_lmcache_device_ops()

    if _is_vllm_runtime():
        _patch_logical_block_size()

    if _is_sglang_runtime():
        # Import failures here (e.g. partial-init ordering of lmcache's own
        # sglang integration module) must never abort the whole activation:
        # the connector simply keeps its upstream behavior and the mismatch
        # resurfaces as the original construction error. Each patch is
        # guarded separately so one failing never disables the other.
        # 此处的导入失败（如 lmcache 自身 sglang 集成模块的半初始化顺序
        # 问题）绝不能中断整个激活流程：连接器仅保持上游行为，形状不
        # 匹配会以原来的构造错误形式再次暴露。两个补丁分别守卫，
        # 一个失败不拖累另一个。
        # Third Party
        # 第三方依赖
        from lmcache.logging import init_logger

        for _patch in (
            _patch_sglang_mp_connector_compat,
            _patch_sglang_unified_diversion,
        ):
            try:
                _patch()
            except Exception as exc:  # noqa: BLE001
                init_logger(__name__).warning(
                    "%s not installed: %s", _patch.__name__, exc
                )

    _patch_make_page_buffer_shape_desc()

    LMCACHE_ASCEND_PATCHED = True
