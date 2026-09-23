# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Lossless LWD expert-ID transport and request-position bookkeeping."""

import torch


def hash_layer_count(config) -> int:
    """返回 DeepSeek V4 的 Hash MoE 层数，其他模型返回零。"""
    return config.num_hash_layers if getattr(config, "model_type", None) == "deepseek_v4" else 0


def hash_payload_numel(num_tokens: int, hidden_size: int, num_layers: int, top_k: int) -> int:
    """计算传输载荷的元素数，包含 embedding 和按四字节编码的专家 ID。"""
    # Each int32 expert ID is encoded as four exactly representable BF16 bytes.
    return num_tokens * (hidden_size + 4 * num_layers * top_k)


def pack_hash_payload(embeds: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """将 embedding 与 [tokens, hash_layers, top_k] 专家 ID 打包为一维 BF16 载荷。"""
    if embeds.dtype != torch.bfloat16:
        raise ValueError("LWD V4 expert transport requires bfloat16 embeddings")
    if expert_ids.ndim != 3 or expert_ids.shape[0] != embeds.shape[0]:
        raise ValueError("LWD expert IDs must have shape [tokens, hash_layers, top_k]")
    # Numeric BF16 conversion of IDs would round values above 256 (e.g. Pro).
    # Encode byte values, rather than interpreting integer bits as BF16 NaNs.
    encoded = expert_ids.to(torch.int32).contiguous().view(torch.uint8).flatten().to(embeds.dtype)
    return torch.cat((embeds.flatten(), encoded.to(embeds.device)))


def unpack_hash_payload(payload, num_tokens, hidden_size, num_layers, top_k):
    """校验并拆解载荷，返回原设备上的 embedding 和 CPU 上的 int32 专家 ID。"""
    expected = hash_payload_numel(num_tokens, hidden_size, num_layers, top_k)
    if payload.numel() != expected:
        raise ValueError(f"LWD V4 payload size mismatch: got {payload.numel()}, expected {expected}")
    flat = payload.flatten()
    embed_end = num_tokens * hidden_size
    embeds = flat[:embed_end].view(num_tokens, hidden_size)
    # Request-position bookkeeping lives on CPU; decode the wire bytes there.
    expert_ids = flat[embed_end:].cpu().to(torch.uint8).contiguous().view(torch.int32)
    return embeds, expert_ids.view(num_tokens, num_layers, top_k)


def merge_hash_expert_ids(tid2eid, token_ids, prompt_experts, prompt_mask):
    """Use remote IDs for prompt rows and checkpoint lookup for decode rows."""
    # The token buffer's prompt slots are intentionally unspecified. Never use
    # them as table indices, even in the unselected branch of torch.where.
    safe_token_ids = torch.where(prompt_mask | (token_ids == -1), 0, token_ids).to(torch.int64)
    decode_experts = tid2eid[safe_token_ids]
    return torch.where(prompt_mask.unsqueeze(-1), prompt_experts, decode_experts).to(torch.int32)


class LwdHashRoutingState:
    """CPU prompt tables keyed by request ID, with explicit received-row masks."""

    def __init__(self, num_layers: int, top_k: int, num_experts: int):
        """保存 Hash 路由维度，并初始化按请求管理的 prompt 专家 ID 缓存。"""
        self.num_layers = num_layers
        self.top_k = top_k
        self.num_experts = num_experts
        self.requests: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def discard(self, req_id: str) -> None:
        """释放指定请求的专家 ID 和有效位置标记；请求不存在时无需处理。"""
        self.requests.pop(req_id, None)

    def add_chunk(self, req_id, prompt_len, start, expert_ids):
        """校验远端专家 ID 分块，将其写入请求的绝对 prompt 位置并标记为有效。"""
        ids = expert_ids.to(device="cpu", dtype=torch.int32)
        if ids.ndim != 3 or tuple(ids.shape[1:]) != (self.num_layers, self.top_k):
            raise ValueError(f"Invalid LWD expert shape for {req_id}: {tuple(ids.shape)}")
        end = start + ids.shape[0]
        if not 0 <= start <= end <= prompt_len:
            raise ValueError(f"Invalid LWD expert chunk for {req_id}: [{start}, {end})/{prompt_len}")
        if torch.any(ids < 0) or torch.any(ids >= self.num_experts):
            raise ValueError(f"LWD expert IDs out of range for {req_id}")
        entry = self.requests.get(req_id)
        if entry is None or entry[0].shape[0] != prompt_len:
            entry = (
                torch.zeros(prompt_len, self.num_layers, self.top_k, dtype=torch.int32),
                torch.zeros(prompt_len, dtype=torch.bool),
            )
            self.requests[req_id] = entry
        entry[0][start:end].copy_(ids)
        entry[1][start:end] = True

    def build_batch(self, req_ids, counts, computed, prompt_lengths):
        """按请求顺序和已计算位置组装 CPU 专家 ID 及 prompt 掩码，缺失 prompt 数据时报错。"""
        if not len(req_ids) == len(counts) == len(computed) == len(prompt_lengths):
            raise ValueError("LWD Hash routing batch metadata lengths do not match")
        total = sum(int(n) for n in counts)
        batch = torch.zeros(total, self.num_layers, self.top_k, dtype=torch.int32)
        mask = torch.zeros(total, dtype=torch.bool)
        row = 0
        for req_id, count, start, prompt_len in zip(req_ids, counts, computed, prompt_lengths):
            count, start, prompt_len = int(count), int(start), int(prompt_len)
            prompt_count = max(0, min(count, prompt_len - start))
            if prompt_count:
                entry = self.requests.get(req_id)
                end = start + prompt_count
                if entry is None or end > entry[0].shape[0] or not entry[1][start:end].all():
                    raise ValueError(f"Missing LWD Hash MoE expert IDs for {req_id} positions [{start}, {end})")
                batch[row : row + prompt_count].copy_(entry[0][start:end])
                mask[row : row + prompt_count] = True
            row += count
        return batch, mask
