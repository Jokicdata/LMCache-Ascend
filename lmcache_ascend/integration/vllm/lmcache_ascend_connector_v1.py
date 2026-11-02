# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger

# First Party
import lmcache_ascend  # noqa: F401

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole) -> None:
        """中文导读：输入为 vLLM 配置和连接器角色，无显式返回值。
        调用方向为 vLLM 创建动态连接器 → 本构造函数 → 上游构造函数；
        这里沿用上游实现，让 Ascend 的适配代码进入旧版 in-process 调用链。
        """
        super().__init__(vllm_config=vllm_config, role=role)
