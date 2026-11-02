# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Iterable, List, Optional, Tuple, Union

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
import torch

logger = init_logger(__name__)

# Type alias for process_tokens return value
# (start_index, end_index, cache_engine_key｜hash)
# 中文：process_tokens 的返回值类型，即（起始下标、结束下标、缓存引擎键或哈希）。
ProcessTokensResult = Tuple[int, int, Union[CacheEngineKey, int]]


def TokenDatabase_process_tokens(
    self,
    tokens: Optional[Union[torch.Tensor, List[int]]] = None,
    hashes: Optional[List[int]] = None,
    offsets: Optional[List[int]] = None,
    mask: Optional[torch.Tensor] = None,
    make_key: bool = True,
    request_configs: Optional[dict] = None,
) -> Iterable[ProcessTokensResult]:
    """Process the tokens and return the corresponding cache engine keys.

    :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

    :param Optional[List[int]] hashes: The hashes to process. If provided,
        it will be used instead of tokens to generate cache engine keys.

    :param Optional[List[int]] offsets: The number of tokens in each chunk.

    :param Optional[torch.Tensor] mask: The mask for the tokens. Should
        have the same length as tokens. And the mask should ALWAYS be like
        FFFFFTTTTTTT, where True means the tokens needs to be matched,
        and the Falses will ALWAYS be at the PREFIX of the tensor.

    :param bool make_key: Whether to make the cache engine key or not.
        If False, the hash value will be returned instead.

    :param Optional[dict] request_configs: The configs of the request.

    :returns: A iterable of tuples with three elements. The first element
        is the start index of the tokens for the key. The second element
        is the end index of the tokens for the key. The third element is
        the cache engine key for the tokens.

    中文导读：输入可以是原始 ``tokens``，也可以是预先计算的 ``hashes`` 及
    各块长度 ``offsets``；``mask`` 指定实际参与匹配的后缀，``make_key``
    决定第三项返回缓存键还是整数哈希。输出是可迭代的三元组
    ``(start, end, CacheEngineKey/hash)``，不直接搬运 KV 数据。
    调用方向为引擎的 store/retrieve 处理流程 → 本函数；token 路径先经
    ``_fast_split_by_subtensor`` 分块，再由 ``_hash_tokens`` 计算分块哈希，
    需要键时由 ``_make_key_by_hash`` 把哈希与请求配置封装成缓存引擎键。
    """

    if tokens is not None:
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long, device="cpu")
        else:
            tokens = tokens.to(device="cpu", dtype=torch.long)

        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0

        # NOTE(niming): Boundary case - return gracefully without raising exceptions
        # 中文：边界情况——空输入直接结束迭代，不抛出异常。
        if len(tokens) == 0:
            logger.warning(
                "Process aborted: 'tokens' is empty. (num_falses=%s)",
                num_falses,
            )
            return

        if num_falses == len(tokens):
            logger.warning(
                "Full mask detected: All %s tokens are masked as False. "
                "Nothing to process for this request.",
                len(tokens),
            )
            return

        assert num_falses < len(tokens), (
            "The number of Falses in the mask shouldn't "
            "be less than the length of tokens."
        )

        token_chunks = self._fast_split_by_subtensor(tokens)
        start_idx = 0
        for idx, token_chunk in enumerate(token_chunks):
            token_chunk_len = len(token_chunk)
            end_idx = start_idx + token_chunk_len
            if idx > 0:
                start_idx += self.sep_len
                end_idx += self.sep_len
            # NOTE: upstream SegmentTokenDatabase.process_tokens does not skip
            # empty split chunks. _fast_split_by_subtensor can yield empty ranges
            # for leading/trailing/consecutive separators (e.g. prompts ending
            # with the special-token separator). Downstream batched_allocate then
            # crashes with ZeroDivisionError when num_tokens == 0 produces a
            # zero-sized aligned_size. Filter here defensively. Also worth
            # reporting upstream — see PROGRESS_2026-05-29.md PR-review items.
            # 中文：上游 SegmentTokenDatabase.process_tokens 不会跳过空分块；
            # 分隔符位于开头、结尾或连续出现时，_fast_split_by_subtensor 可能产生
            # 空区间。下游 batched_allocate 遇到 num_tokens == 0 和零长度
            # aligned_size 会触发 ZeroDivisionError，因此这里防御性过滤；
            # 该问题也应反馈上游，详见 PROGRESS_2026-05-29.md 的评审事项。
            if token_chunk_len == 0 or end_idx <= start_idx:
                start_idx = end_idx
                continue
            if start_idx >= num_falses:
                if make_key:
                    yield (
                        start_idx,
                        end_idx,
                        self._make_key_by_hash(
                            self._hash_tokens(token_chunk), request_configs
                        ),
                    )
                else:
                    yield start_idx, end_idx, self._hash_tokens(token_chunk)
            start_idx = end_idx
    elif hashes is not None:
        assert offsets is not None, (
            "If hashes are provided, offsets must also be provided."
        )
        start_idx = 0
        for hash_val, offset in zip(hashes, offsets, strict=False):
            end_idx = start_idx + offset
            if make_key:
                yield (
                    start_idx,
                    end_idx,
                    self._make_key_by_hash(hash_val, request_configs),
                )
            else:
                yield start_idx, end_idx, hash_val
            start_idx = end_idx
    else:
        raise ValueError("Either tokens or hashes must be provided.")
