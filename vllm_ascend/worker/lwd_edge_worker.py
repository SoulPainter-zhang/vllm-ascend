#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""LWD (layerwise disaggregated) prefill_only mode edge worker."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from vllm.logger import logger

from vllm.v1.core.sched.output import (
    LwdBatchType,
    LwdEmbedBatch,
    LwdUnembedBatch,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm_ascend import envs
from vllm_ascend.batch_invariant import init_batch_invariance
from vllm_ascend.distributed.lwd_comm.lwd_parallel_init import (
    init_lwd_ascend_model_parallel,
)
from vllm_ascend.distributed.lwd_comm.service import get_lwd_comm_service
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest
from vllm_ascend.worker.worker import NPUWorker

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheSpec


# ---- token recovery / logit-rank lookup ----
def select_token_batch(
    logits: torch.Tensor, ranks: list[int]
) -> list[int]:
    """Batched rank→token lookup: one sort, one D2H sync for all rows.

    logits [R,V] rows sorted together (NPU sorts rows in parallel);
    sorted_idx[i, ranks[i]] is the token id at rank ranks[i] for row i.
    """
    _, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    row = torch.arange(len(ranks), device=logits.device)
    return sorted_idx[row, torch.tensor(ranks, device=logits.device)].tolist()


def compute_top_id_th(logits: torch.Tensor, token_id: int) -> int:
    """Look up the descending-logits ordinal of ``token_id`` (cloud side)."""
    logits = logits.reshape(-1)
    order = torch.argsort(logits, descending=True)
    return int((order == token_id).nonzero()[0].item())


# ---- edge worker ----
class LwdEdgeWorker(NPUWorker):
    """LWD edge worker: embed (prefill) + unembed (token recovery) only."""

    def _init_worker_distributed_environment(self) -> None:
        """覆写原生入口(worker.py):ascend 侧并行组按 Lwd 布局构建。

        vllm 侧分组由 parallel_state.initialize_model_parallel 的 Lwd
        分支完成;ascend 侧原生 init_ascend_model_parallel 按 (dp, pp,
        pcp, tp) 均匀网格切分,表达不了非对称边云拓扑,故以
        init_lwd_ascend_model_parallel 替代(构建 MC2 等组后注入)。
        其余步骤与原生 worker.py 保持一致。"""
        init_batch_invariance()
        init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            "hccl",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.prefill_context_parallel_size,
            self.parallel_config.decode_context_parallel_size,
        )
        init_lwd_ascend_model_parallel(self.parallel_config)
        ensure_ec_transfer_initialized(self.vllm_config)

    def init_device(self):
        # The duplex channels MUST be built here, never in ``__init__``.
        # ``init_lwd_duplex_channels`` creates the two HCCL process groups
        # (``dist.new_group``) and then warms them up with a two-sided P2P
        # exchange (world barrier plus the edge/cloud isend/irecv pair), so it
        # needs the distributed environment -- world group and PP group --
        # which ``NPUWorker._init_worker_distributed_environment`` only builds
        # inside ``init_device``.  The executor constructs the worker first and
        # calls ``init_device`` afterwards, so running it from ``__init__``
        # would trip the "group is not initialized" assertions; the two-sided
        # warmup additionally requires both peers to reach it in the same fixed
        # order, which only holds once every rank is past its distributed init.
        super().init_device()
        self.comm_service = get_lwd_comm_service()
        # 多模态请求级缓存:req_id -> mm_features(带 data,首 chunk 随
        # scheduled_new_reqs 到达时登记) / req_id -> [(mm_position,
        # encoder_embeds)](视觉塔输出,首用即算)。chunk 窗口越过全部
        # mm 段后摘除(见 _execute_lwd_embed),finished_req_ids 兜底。
        self._lwd_mm_features_dict: dict[str, list] = {}
        self._lwd_mm_embeds_dict: dict[str, list] = {}

        from vllm_ascend.distributed import lwd_wire
        lwd_wire.init_lwd_duplex_channels()
        logger.info(
            "[lwd-edge] worker ready: duplex channels (UP/DOWN) initialized "
            "on global rank=%d",
            self.rank,
        )

    def compile_or_warm_up_model(self):
        # LWD 边侧运行期 forward 被 LWD 流程劫持(execute_model 只走
        # embed/unembed 直调),模型级 warmup 与 cudagraph 捕获均用不到;
        # 且 0 层拓扑下 full-forward 会在 final norm 解包失败。整体跳过。
        from vllm.v1.worker.worker_base import CompilationTimes

        logger.info("[lwd-edge] skip model warmup/capture (forward is hijacked by LWD)")
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def get_kv_cache_spec(self) -> dict[str, "KVCacheSpec"]:
        """The LWD edge runs no attention/transformer, so it needs no KV cache."""
        return {}

    def execute_model(self, scheduler_output: "SchedulerOutput"):
        lwd_batch = scheduler_output.lwd_batch
        # 多模态缓存生命周期:请求完结(含 abort)即摘除其 mm 特征/
        # encoder 输出缓存,防跨请求泄漏(正常路径在 chunk 越过全部
        # mm 窗口后已提前摘除,此为兜底)。
        if scheduler_output.finished_req_ids:
            for req_id in scheduler_output.finished_req_ids:
                self._lwd_mm_features_dict.pop(req_id, None)
                self._lwd_mm_embeds_dict.pop(req_id, None)
        if lwd_batch is None:
            logger.debug("[lwd-edge] step carries no LWD batch; nothing to do")
            return None

        batch_meta = lwd_batch.batch_meta
        if lwd_batch.batch_type == LwdBatchType.LWD_EMBED:
            logger.info(
                "[Lwd][edge-worker] EMBED seqno=%d reqs=%d tokens=%d "
                "has_mrope=%s",
                lwd_batch.seqno,
                len(batch_meta.req_ids),
                sum(len(token_ids) for token_ids in batch_meta.token_ids),
                batch_meta.has_mrope,
            )
            return self._execute_lwd_embed(
                lwd_batch.seqno, batch_meta, scheduler_output
            )
        if lwd_batch.batch_type == LwdBatchType.LWD_UNEMBED:
            logger.info(
                "[Lwd][edge-worker] UNEMBED seqno=%d reqs=%d accepted=%d "
                "num_elements=%d",
                lwd_batch.seqno,
                len(batch_meta.req_ids),
                sum(batch_meta.num_accept_tokens),
                batch_meta.recv_num_elements,
            )
            return self._execute_lwd_unembed(lwd_batch.seqno, batch_meta)

        logger.debug(
            "[lwd-edge] unknown LWD batch type %r; nothing to do",
            lwd_batch.batch_type,
        )
        return None

    def _lwd_embed_ids_h2d(self, flat_token_ids: list[int]) -> torch.Tensor:
        """token ids 异步上卡:pinned 环(4 槽,与派发队列深度同阶)+
        non_blocking,替代 torch.tensor(list, device=...) 的 pageable
        同步拷贝——后者在主流上等设备排空(含上一块 chunk 的 UP send
        配对等待),把跨侧锁步传导进边 worker 主线程。同流序保证 embed
        kernel 读到的一定是本 chunk 的值(H2D 先于 embed 入队)。"""
        n = len(flat_token_ids)
        ring = getattr(self, "_lwd_ids_ring", None)
        if ring is None or ring[0][0].numel() < n:
            m = max(n, 8192)
            ring = [
                (
                    torch.empty(m, dtype=torch.int64, pin_memory=True),
                    torch.empty(
                        m, dtype=torch.int64, device=self.model_runner.device
                    ),
                )
                for _ in range(4)
            ]
            self._lwd_ids_ring = ring
            self._lwd_ids_ring_idx = 0
        idx = self._lwd_ids_ring_idx
        self._lwd_ids_ring_idx = (idx + 1) % len(ring)
        stage, dev = ring[idx]
        stage.numpy()[:n] = flat_token_ids
        dev[:n].copy_(stage[:n], non_blocking=True)
        return dev[:n]

    def _execute_lwd_embed(
        self,
        seqno: int,
        batch_meta: LwdEmbedBatch,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        model = self.model_runner.get_model()
        if not batch_meta.token_ids:
            return

        # 登记新到请求的 mm 特征(首 chunk 随 scheduled_new_reqs 携带
        # data;后续 chunk 走缓存)。注意登记不过滤 data:None 的项留到
        # encoder 执行点 fail-fast(见 _lwd_mm_encoder_outputs)——静默
        # 跳过会让图像 embeds 缺失、占位 token 按文本嵌入,即错算。
        for req_data in scheduler_output.scheduled_new_reqs:
            feats = [
                f
                for f in req_data.mm_features
                if f.modality != "prompt_embeds"
            ]
            if feats:
                self._lwd_mm_features_dict[req_data.req_id] = feats

        # Flatten all requests' prompt tokens into one batch (order = req_ids).
        flat_token_ids = [tid for token_ids in batch_meta.token_ids for tid in token_ids]
        logger.debug(
            "[Lwd][edge-worker] embed token_ids=%s", flat_token_ids
        )
        token_ids_tensor = self._lwd_embed_ids_h2d(flat_token_ids)
        _t0 = time.monotonic()
        # 多模态 merge:批内任一请求带 mm 数据即按本 chunk 窗口收集
        # 视觉塔 embeds + 构造 is_multimodal 掩码,走模型原生
        # embed_input_ids 的 merge 分支;纯文本批走原直达路径。
        mm_embeds, is_mm = self._lwd_gather_chunk_mm(batch_meta)
        if mm_embeds:
            embeds = model.embed_input_ids(
                token_ids_tensor,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm,
            )
        else:
            embeds = model.embed_input_ids(token_ids_tensor)  # (total_N, H)
        _t_fwd = time.monotonic()

        aux_tensor = None
        if batch_meta.has_mrope:
            aux_tensor = torch.tensor(
                batch_meta.mrope_positions, dtype=torch.int64,
                device=self.model_runner.device,
            )
            assert aux_tensor.shape == (len(flat_token_ids), 3), (
                f"mrope rows {aux_tensor.shape[0]} != chunk tokens "
                f"{len(flat_token_ids)} (seqno={seqno})"
            )

        request = LwdCommRequest(
            channel=LwdChannelType.UP,
            op="send",
            num_elements=embeds.numel(),                     # total_N * H
            tensor=embeds,
            seqno=seqno,
            aux_tensor=aux_tensor,
            aux_num_elements=(
                aux_tensor.numel() if aux_tensor is not None else 0
            ),
        )
        self.comm_service.submit_send(request)
        # [Lwd][perf] TTFT 探针:forward=embed 前向;submit_send=快照clone+
        # 广播提交(含 bridge 的 handle.wait)——若 UP 世界广播在等云侧
        # 入队,此段会显著变大(chunk 级锁步的直接证据)
        logger.info(
            "[Lwd][perf] embed seqno=%s forward=%.2f submit_send=%.2f "
            "total=%.2fms tokens=%d",
            seqno, (_t_fwd - _t0) * 1000,
            (time.monotonic() - _t_fwd) * 1000,
            (time.monotonic() - _t0) * 1000,
            len(flat_token_ids),
        )

    # ------------------------------------------------------------------ #
    # Multimodal (image) merge helpers                                     #
    # ------------------------------------------------------------------ #

    def _lwd_mm_encoder_outputs(self, req_id: str) -> list:
        """视觉塔输出(请求级惰性计算+缓存):[(mm_position, embeds)]。

        embeds 按 mm_hash 语义本应跨请求共享,这里按请求缓存(单请求
        组批下重复图极少);输出常驻至该请求 chunk 流越过全部 mm 窗口。"""
        cached = self._lwd_mm_embeds_dict.get(req_id)
        if cached is not None:
            return cached
        features = self._lwd_mm_features_dict.get(req_id)
        if not features:
            return []
        missing = [f.identifier for f in features if f.data is None]
        if missing:
            # data 缺失即无法计算图像 embeds——静默跳过 = 图像 token 按
            # 文本嵌入错算,fail-fast。
            raise RuntimeError(
                f"[Lwd] mm feature data missing for req={req_id} "
                f"(mm_hashes={missing}): cannot compute image embeddings; "
                "check mm processor/receiver cache chain"
            )
        model = self.model_runner.get_model()
        from vllm.multimodal.utils import group_and_batch_mm_kwargs

        mm_kwargs = [(f.modality, f.data) for f in features]
        outputs_by_modality: dict[str, list] = {}
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs,
            device=self.model_runner.device,
            pin_memory=getattr(self.model_runner, "pin_memory", False),
        ):
            batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
            assert len(batch_outputs) == num_items, (
                f"encoder outputs {len(batch_outputs)} != items "
                f"{num_items} (req={req_id}, modality={modality})"
            )
            outputs_by_modality.setdefault(modality, []).extend(batch_outputs)
        result = []
        for feature in features:
            result.append(
                (feature.mm_position, outputs_by_modality[feature.modality].pop(0))
            )
        self._lwd_mm_embeds_dict[req_id] = result
        logger.info(
            "[Lwd][edge-worker] mm encoder done: req=%s items=%d",
            req_id, len(result),
        )
        return result

    def _lwd_gather_chunk_mm(self, batch_meta: LwdEmbedBatch):
        """按本 chunk 窗口收集 mm embeds 行 + 构造 is_multimodal 掩码。

        窗口语义逐字对齐原生 _gather_mm_embeddings:feature 的占位区间
        [offset, offset+length) 与 chunk 窗口 [p_offset, p_offset+n) 求交,
        相交段切 embeds 行、置掩码位;图像跨 chunk 时逐段收集。同时完成
        越过全部 mm 窗口请求的缓存摘除。"""
        total = sum(len(t) for t in batch_meta.token_ids)
        is_mm = torch.zeros(total, dtype=torch.bool)
        mm_embeds: list = []
        if len(batch_meta.prompt_offsets) != len(batch_meta.req_ids):
            return mm_embeds, is_mm
        req_start = 0
        for req_id, token_ids, p_offset in zip(
            batch_meta.req_ids, batch_meta.token_ids, batch_meta.prompt_offsets
        ):
            n = len(token_ids)
            cached = self._lwd_mm_encoder_outputs(req_id)
            max_end = 0
            for pos_info, item_embeds in cached:
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length
                max_end = max(max_end, start_pos + num_encoder_tokens)
                start_idx = max(p_offset - start_pos, 0)
                end_idx = min(
                    p_offset - start_pos + n, num_encoder_tokens
                )
                if start_idx >= end_idx:
                    continue
                req_start_pos = req_start + start_pos - p_offset
                if (is_embed := pos_info.is_embed) is not None:
                    s, e = pos_info.get_embeds_indices_in_range(
                        start_idx, end_idx
                    )
                    if s == e:
                        continue
                    rows = item_embeds[s:e]
                    is_mm[req_start_pos + start_idx : req_start_pos + end_idx] |= (
                        is_embed[start_idx:end_idx]
                    )
                else:
                    rows = item_embeds[start_idx:end_idx]
                    is_mm[req_start_pos + start_idx : req_start_pos + end_idx] = True
                mm_embeds.append(rows)
            if cached and p_offset + n >= max_end:
                self._lwd_mm_features_dict.pop(req_id, None)
                self._lwd_mm_embeds_dict.pop(req_id, None)
            req_start += n
        return mm_embeds, is_mm

    def _execute_lwd_unembed(
        self, seqno: int, batch_meta: LwdUnembedBatch
    ) -> ModelRunnerOutput:
        model = self.model_runner.get_model()
        if not batch_meta.req_ids:
            return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])

        _t0 = time.monotonic()
        recv_future = self.comm_service.submit_recv(
            LwdCommRequest(
                channel=LwdChannelType.DOWN,
                op="recv",
                num_elements=batch_meta.recv_num_elements,  # int = rows_total * hidden_size
                seqno=seqno,
            )
        )
        _t_post = time.monotonic()
        # wait_for_comm:device 序等待,主机不阻塞
        recv_future.wait_for_comm()
        result = recv_future.result()
        _t_tensor = time.monotonic()
        hidden_size = self.model_config.get_hidden_size()
        hidden_states = result.tensor.view(-1, hidden_size)  # (rows_total, H)
        # 诊断旁路(VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE=1):DOWN 张量照收
        # (传输成本保留在测量内),但跳过 lm_head + rank-replay,直接交付
        # 云侧 c2e 通告捎带的 token ids——性能恢复即瓶颈在边侧 unembed。
        if envs.VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE and batch_meta.token_ids:
            assert len(batch_meta.token_ids) == len(batch_meta.req_ids)
            sampled_token_ids: list[list[int]] = [
                list(t) for t in batch_meta.token_ids
            ]
            _t_lm = time.monotonic()
        else:
            # lm_head 不允许直接调用(ParallelLMHead.forward 强制经 sampler);
            # compute_logits 是标准接口,包装层/单体模型都有。
            logits = model.compute_logits(hidden_states)   # (rows_total, V)
            _t_lm = time.monotonic()
            # L5 对拍:首行 top-20,与云侧 sampler 入口的 [layer-trace]
            # top20 逐位对照——排名换位即重放漂移的直接视图。
            try:
                from vllm_ascend.worker.lwd_layer_trace import (
                    lwd_layer_trace_enabled,
                )

                if lwd_layer_trace_enabled() and logits.dim() == 2:
                    row = logits[0].detach().float()
                    vals, tids = torch.topk(row, min(20, row.numel()))
                    logger.info(
                        "[layer-trace] edge lm_head logits shape=%s row0 l2=%.4f "
                        "top20=%s",
                        tuple(logits.shape), row.norm().item(),
                        list(zip(tids.tolist(), [round(v, 3) for v in vals.tolist()])),
                    )
            except Exception:  # noqa: BLE001
                pass

            flat_ids = select_token_batch(
                logits, [r for ths in batch_meta.top_id_ths for r in ths]
            )
            sampled_token_ids = []
            off = 0
            for accept, ths in zip(
                batch_meta.num_accept_tokens, batch_meta.top_id_ths
            ):
                sampled_token_ids.append(flat_ids[off : off + accept])
                off += len(ths)
        _t_sel = time.monotonic()
        # [Lwd][perf] 临时探针:单请求慢的归因分段——
        # post_recv=挂 irecv;wait_tensor=等张量落卡(网络);lm_head=词表
        # 前向(权重带宽);select=argsort 取名(.item 同步);total=边尾段全长
        logger.info(
            "[Lwd][perf] unembed seqno=%s post_recv=%.2f wait_tensor=%.2f "
            "lm_head=%.2f select=%.2f total=%.2fms reqs=%d",
            seqno,
            (_t_post - _t0) * 1000,
            (_t_tensor - _t_post) * 1000,
            (_t_lm - _t_tensor) * 1000,
            (_t_sel - _t_lm) * 1000,
            (_t_sel - _t0) * 1000,
            len(batch_meta.req_ids),
        )
        req_id_to_index = {
            req_id: index for index, req_id in enumerate(batch_meta.req_ids)
        }
        return ModelRunnerOutput(
            req_ids=batch_meta.req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
        )
