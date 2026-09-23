# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


class LwdMTPProposer(AscendEagleProposer):
    """Adapt externally supplied LWD embeddings to the MTP SP layout."""

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        """Shard external embeddings in the active draft context before forward."""
        context = get_forward_context()
        if inputs_embeds is not None and context.flash_comm_v1_enabled and not self.is_multimodal_model:
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                if inputs_embeds.shape[0] != num_input_tokens:
                    raise ValueError("LWD MTP SP expects full-token inputs_embeds before slicing")
                # Embeddings are already replicated; reduce-scatter would sum
                # their values again, so only pad and slice the token rows.
                if context.pad_size:
                    inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, context.pad_size))
                inputs_embeds = inputs_embeds.chunk(tp_group.world_size, dim=0)[tp_group.rank_in_group]
        return super()._run_merged_draft(
            num_input_tokens,
            batch_size,
            token_indices_to_sample,
            target_positions,
            inputs_embeds,
            multi_steps_attn_metadata,
            num_tokens,
            is_prefill,
        )
