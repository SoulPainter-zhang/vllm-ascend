# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Load the edge's Hash MoE lookup tables without constructing MoE layers."""

import logging
import re
from collections.abc import Sequence

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)


def load_tid2eid_tables(
    weight_files: Sequence[str],
    num_hash_layers: int,
    vocab_size: int,
    experts_per_token: int,
    num_experts: int,
) -> list[torch.Tensor]:
    """Read only tid2eid tensors from safetensors shards into CPU memory.

    Keep the small integer tables on CPU, outside the NPU sleep allocator.
    No expert weights are materialized, and no synthetic tables are allowed.
    """
    if num_hash_layers < 0:
        raise ValueError("num_hash_layers must be nonnegative")
    if num_hash_layers == 0:
        return []
    pattern = re.compile(r"(?:model\.)?layers\.(\d+)\.(?:ffn|mlp)\.gate\.tid2eid")
    tables: dict[int, torch.Tensor] = {}
    for filename in sorted(weight_files):
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            names = checkpoint.keys()
            for name in names:
                match = pattern.fullmatch(name)
                if match is None or int(match[1]) >= num_hash_layers:
                    continue
                layer = int(match[1])
                if layer in tables:
                    raise ValueError(f"Duplicate tid2eid table for layer {layer}: {filename}:{name}")
                table = checkpoint.get_tensor(name)
                if tuple(table.shape) != (vocab_size, experts_per_token):
                    raise ValueError(f"Invalid tid2eid shape for {name}: {tuple(table.shape)}")
                if table.dtype not in (torch.int32, torch.int64):
                    raise ValueError(f"Invalid tid2eid dtype for {name}: {table.dtype}")
                minimum, maximum = table.min().item(), table.max().item()
                if minimum < 0 or maximum >= num_experts:
                    raise ValueError(f"Invalid tid2eid expert IDs for {name}: [{minimum}, {maximum}]")
                # Copy even int32 tensors so no checkpoint mmap remains owned.
                tables[layer] = table.to(dtype=torch.int32, copy=True).contiguous()
                logger.info(
                    "[lwd-edge] tid2eid loaded layer=%d key=%s file=%s shape=%s "
                    "dtype=%s device=%s expert_range=[%d,%d]",
                    layer,
                    name,
                    filename,
                    tuple(table.shape),
                    tables[layer].dtype,
                    tables[layer].device,
                    minimum,
                    maximum,
                )
    missing = sorted(set(range(num_hash_layers)) - tables.keys())
    if missing:
        raise ValueError(f"Missing tid2eid tables for Hash MoE layers {missing}")
    result = [tables[layer] for layer in range(num_hash_layers)]
    logger.info(
        "[lwd-edge] tid2eid loading complete layers=%d bytes=%d device=cpu",
        len(result),
        sum(table.numel() * table.element_size() for table in result),
    )
    return result
