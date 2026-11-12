# SPDX-License-Identifier: Apache-2.0
"""Ascend helpers for the plugin PageBufferShapeDesc plane extras.

CUDA's compiled struct is unchanged. After bind_native, factory/grouping
construct ``device_ops.PageBufferShapeDesc()`` (``c_ops``). NPU kernels
take that class by value. Also hosts the wrap that attaches per-plane
byte geometry to upstream's desc factory (:func:`install_plane_geometry_fill`).

为插件的 ``PageBufferShapeDesc`` plane 扩展提供 Ascend 辅助函数。

CUDA 的编译结构体保持不变。``bind_native`` 之后，工厂和分组逻辑构造
``device_ops.PageBufferShapeDesc()``（来自 ``c_ops``），NPU 内核按值接收
这个类。本模块还负责包装上游描述符工厂，为其附加逐 plane 的字节几何
信息（见 :func:`install_plane_geometry_fill`）。

这里的 plane 指同一层中分开存放、但按 token/block 对齐的张量，例如
MLA 的 latent 与 rope，或普通注意力的 K 与 V。除了总形状，NPU 内核还需
知道每个 plane 每 token 占多少字节、相邻 block 相距多少字节。
"""

# Future
# 来自 __future__ 的特性
from __future__ import annotations

# Standard
# 标准库
from collections.abc import Sequence
from typing import Any

_MAX_PLANES = 4


def attach_tuple_planes(
    desc: Any,
    plane_slot_bytes: Sequence[int],
) -> None:
    """Record per-token plane widths (bytes) on ``desc``.

    ``plane_slot_bytes[i]`` is ``numel * itemsize // (nb * bs)`` for plane
    ``i``. Caps at ``_MAX_PLANES``; the packed kernel only accepts NP==2.

    在 ``desc`` 上记录每个 plane 的每 token 宽度（字节）。

    ``desc`` 是待补充的形状描述符；``plane_slot_bytes[i]`` 是 plane ``i``
    的 ``numel * itemsize // (nb * bs)``，即一个 token 在该 plane 中占用的
    字节数。函数直接写入 ``num_planes`` 和 ``plane_slot_bytes``，无返回值。
    plane 数最多为 ``_MAX_PLANES``，而 packed 内核只接受 ``NP == 2``。
    """
    slots = tuple(int(b) for b in plane_slot_bytes)
    if len(slots) > _MAX_PLANES:
        raise ValueError(
            f"attach_tuple_planes: {len(slots)} planes exceeds max {_MAX_PLANES}"
        )
    desc.num_planes = len(slots)
    desc.plane_slot_bytes = slots


def attach_tuple_block_strides(
    desc: Any,
    plane_block_stride_bytes: Sequence[int],
) -> None:
    """Record per-block dim-0 byte strides on ``desc``.

    ``plane_block_stride_bytes[i]`` is ``stride(0) * itemsize`` for plane
    ``i``. Packed native uses plane 0/1; 0 means fall back to the shared
    ``block_stride_elems``.

    在 ``desc`` 上记录每个 plane 沿第 0 维逐 block 的字节步长。

    ``desc`` 是待补充的形状描述符；``plane_block_stride_bytes[i]`` 是
    plane ``i`` 的 ``stride(0) * itemsize``，表示从一个 block 的起点走到
    下一个 block 的起点需要跳过多少字节。函数直接写入
    ``plane_block_stride_bytes``，无返回值。packed 原生实现使用 plane
    0/1；步长为 0 表示回退到共享的 ``block_stride_elems``。
    """
    strides = tuple(int(b) for b in plane_block_stride_bytes)
    if len(strides) > _MAX_PLANES:
        raise ValueError(
            "attach_tuple_block_strides: "
            f"{len(strides)} planes exceeds max {_MAX_PLANES}"
        )
    desc.plane_block_stride_bytes = strides


def install_plane_geometry_fill() -> None:
    """Wrap upstream ``make_page_buffer_shape_desc`` with plane geometry.

    Upstream's factory fills only the scalar fields; its compiled struct
    carries no plane arrays (the pybind class is ``dynamic_attr``), so the
    per-plane byte geometry both Ascend formats require is attached here,
    at the same construction point:

    - fmt 17 (``NL_X_NP_X_NB_BS_ONE_HS``): MLA/DSA planes
      ``[NB, BS, 1, HS]``;
    - fmt 16 (``NL_X_TWO_X_NB_BS_NH_HS``): per-layer ``(K, V)`` planes
      ``[NB, BS, NH, HS]``.

    Both share the per-layer plane-tuple structure: the per-token row is
    the product of the trailing two dims (``shape[2] * shape[3] *
    element_size``; ``shape[2] == 1`` for fmt 17, reducing to its plane
    width) and the per-block stride is ``stride(0) * element_size``.

    Upstream call sites (``kv_layer_groups`` and the MP transfer-context
    gather/scatter utilities) import the function lazily inside the call,
    so rebinding the module attribute takes effect regardless of import
    order.

    为上游 ``make_page_buffer_shape_desc`` 包装逐 plane 几何信息。

    上游工厂只填写标量字段，其编译结构体没有 plane 数组（pybind 类启用了
    ``dynamic_attr``），因此在同一个构造位置为两种 Ascend 格式附加所需的
    逐 plane 字节几何：

    - 格式 17（``NL_X_NP_X_NB_BS_ONE_HS``）：MLA/DSA 的 plane 形状为
      ``[NB, BS, 1, HS]``；
    - 格式 16（``NL_X_TWO_X_NB_BS_NH_HS``）：每层的 ``(K, V)`` plane
      形状为 ``[NB, BS, NH, HS]``。

    两者都按层组织为 plane 元组。每 token 一行占用的字节数是末尾两维
    与元素大小的乘积（``shape[2] * shape[3] * element_size``；格式 17 的
    ``shape[2] == 1``，所以就是该 plane 的宽度）；每 block 的字节步长为
    ``stride(0) * element_size``。其中 NB 是 block 数，BS 是每 block 的
    token 槽数，NH/HS 分别是注意力头数和每头宽度。

    上游调用点（``kv_layer_groups`` 以及多进程传输上下文的
    gather/scatter 工具）会在调用时懒导入该函数，所以无论导入顺序如何，
    重新绑定模块属性都能生效。本函数由插件导入时的
    ``_patch_make_page_buffer_shape_desc`` 调用，无返回值。
    """
    # Third Party
    # 第三方依赖
    import lmcache.lmcache_native as lmcache_native
    import lmcache.v1.common.utils as common_utils

    original = common_utils.make_page_buffer_shape_desc
    plane_tuple_fmts = (
        lmcache_native.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS,
        lmcache_native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS,
    )

    def make_page_buffer_shape_desc(
        kv_caches,
        engine_kv_format,
        layer_idx: int,
        num_layers_in_group: int,
        num_blocks: int,
        block_size: int,
        block_stride_elems=None,
    ):
        """先调用上游工厂，再为 Ascend 的 plane 元组格式补齐几何字段。

        ``kv_caches`` 是按层组织的 KV 张量；``engine_kv_format`` 指定内存
        布局；``layer_idx`` 和 ``num_layers_in_group`` 指明本组从哪层开始、
        包含几层；``num_blocks`` 与 ``block_size`` 给出 block 数和每块
        槽数；``block_stride_elems`` 是可选的共享元素步长。

        返回值是上游创建的 ``PageBufferShapeDesc``。格式 16/17 会额外带有
        ``num_planes``、``plane_slot_bytes`` 和
        ``plane_block_stride_bytes``，供后续 NPU gather/scatter 内核寻址。
        """
        desc = original(
            kv_caches,
            engine_kv_format,
            layer_idx,
            num_layers_in_group,
            num_blocks,
            block_size,
            block_stride_elems,
        )
        if engine_kv_format in plane_tuple_fmts:
            planes = kv_caches[layer_idx]
            desc.num_planes = len(planes)
            desc.plane_slot_bytes = tuple(
                int(t.shape[2]) * int(t.shape[3]) * int(t.element_size())
                for t in planes
            )
            desc.plane_block_stride_bytes = tuple(
                int(t.stride(0)) * int(t.element_size()) for t in planes
            )
        return desc

    common_utils.make_page_buffer_shape_desc = make_page_buffer_shape_desc
