# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only LWD model runner subclass (cloud side).

All LWD runner-side logic lives here (moved out of
``model_runner_v1.py``): the per-step cloud-side collection of the
DOWN payload (hidden-only tensor) plus the step metadata (global ranks
/ num_accepted / req_ids) that rides back to the scheduler on
``ModelRunnerOutput.lwd_c2e_meta``.

Selected via ``LwdCloudWorker`` (worker_cls), which swaps the model
runner class at init_device when ``lwd_config`` enables prefill_only.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class LwdCloudModelRunner(NPUModelRunner):
    """NPUModelRunner + prefill_only LWD cloud-side runner logic."""

    def __init__(self, vllm_config, device, worker=None):
        super().__init__(vllm_config, device)
        self.worker = worker  # LwdCloudWorker ref (UP recv futures live there)
        self._lwd_pending_down_payload = None

    # ------------------------------------------------------------------ #
    # Remote embeds injection (cloud input has NO token ids — the prompt   #
    # embeddings arrive over the UP channel and must drive forward)        #
    # ------------------------------------------------------------------ #

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        """UP 收割与 mrope 写入在 super() 之前,embeds 覆写在之后。

        mrope 时序契约(多模态迁移复核发现的源分支缺陷修复):super()
        内部的 _calc_mrope_positions 当步即把 req_state.mrope_positions
        的窗口行拷上 GPU(model_runner_v1.py:1033-1045)——mrope 写入
        若晚于 super()(源实现形态),当步 chunk 的 mrope 行永远不被
        自己的前向看到(所有 chunk 用 ids=None 直通初始化的 arange
        值,图像行必错)。故收割(端点 host 门 + TP 组内现场广播)与
        mrope/delta 写入提到 super() 之前。

        embeds 保持在 super() 之后:基类填充循环 + copy_to_gpu 只看到
        零缓冲,随后把收到的 UP embeds 直写 inputs_embeds.gpu 当步调度
        窗口(后写为权威),经 future 完成事件在流上有序——非阻塞,
        recv 等待另有 [Lwd][perf] up-recv-wait 计时可见。
        """
        harvested = None
        if self._lwd_enabled():
            harvested = self._lwd_harvest_up_batches(num_scheduled_tokens)
        out = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        if self._lwd_enabled():
            self._lwd_rebuild_is_token_ids_mask(num_scheduled_tokens)
            self._lwd_release_consumed_prompt_embeds()
            self._lwd_inject_remote_embeds(harvested, num_scheduled_tokens)
        return out

    def _lwd_rebuild_is_token_ids_mask(self, num_scheduled_tokens) -> None:
        """重建 prompt-embeds 分支的扁平 is_token_ids 掩码。

        根因(async spec decode 乐观值):runner 的 num_computed_tokens_cpu
        假设上步草稿全被接受(比真实提交位置多 r),而原生扁平化
        (model_runner_v1.py 的 token_indices = positions + idx*M)用乐观
        positions 从二维掩码取值——decode 行整体右移 r 位,每段末尾
        r 行越出 True 标记区读到 False/陈旧值,被 prompt-embeds 分支
        当作 embeds 行而不做本地嵌入,保留 inputs_embeds.gpu 的陈旧
        垃圾;这些行恰是被采样行(logits_indices 取段末 1+k 行),
        垃圾 logits 直接污染已提交 token。原生/phase/单请求不受影响
        (掩码只在 prompt-embeds 分支被消费,那些形态走 MM 分支)。

        LWD 语义里只有注入行(prompt 段)应是 embeds 行(False),
        decode/草稿行恒为 token id(True)。此处按调度状态确定性重建,
        不依赖任何乐观值修正:decode 请求的 computed 乐观只会偏大,
        prefill_rows 恒为 0,天然安全。GPU 侧掩码全仓只写不读,只需
        修 CPU 视图(.np 与 .cpu 同存储)。
        """
        if not self.input_batch.req_prompt_embeds:
            # 批内无 embeds 缓冲(MM/text 分支,掩码无人读)
            return
        mask = self.is_token_ids.np
        computed = self.input_batch.num_computed_tokens_cpu
        num_prompt = self.input_batch.num_prompt_tokens
        n_reqs = len(num_scheduled_tokens)
        total = int(np.sum(num_scheduled_tokens[:n_reqs]))
        mask[:total] = True
        off = 0
        for i in range(n_reqs):
            n = int(num_scheduled_tokens[i])
            prefill_rows = min(
                n, max(int(num_prompt[i]) - int(computed[i]), 0)
            )
            if prefill_rows > 0:
                mask[off : off + prefill_rows] = False
            off += n

    def _lwd_enabled(self) -> bool:
        cfg = getattr(self.vllm_config, "lwd_config", None)
        return bool(cfg is not None and cfg.enabled)

    def _lwd_release_consumed_prompt_embeds(self) -> None:
        """Release assembly buffers whose prompt was fully consumed.

        A buffer becomes releasable once the request's computed tokens
        cover the whole prompt: the last chunk's rows were consumed by
        the previous step's fill loop, and the draft proposer read its
        window during that step's sampling (which precedes this call).
        Decode steps never read prompt embeds, so dropping the entry
        here cannot corrupt anything.  Entries follow native index
        compaction (condense/swap), so keying by idx stays correct.

        Also drops orphan entries whose slot no longer holds a live
        request (preemption removes the request without finishing it, so
        the flush hook never fires): a stale buffer at a recycled index
        would otherwise be mistaken for the new occupant's embeds.
        """
        embeds_map = self.input_batch.req_prompt_embeds
        if not embeds_map:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        slot_req_ids = self.input_batch.req_ids
        for idx in list(embeds_map.keys()):
            if idx >= len(computed):
                embeds_map.pop(idx)
                continue
            if idx >= len(slot_req_ids) or slot_req_ids[idx] is None:
                embeds_map.pop(idx)
                continue
            if computed[idx] >= num_prompt[idx]:
                embeds_map.pop(idx)

    def _lwd_harvest_up_batches(self, num_scheduled_tokens) -> list:
        """UP 批收割(super()._prepare_inputs 之前调用,时序契约见
        _prepare_inputs):端点过 host 门 + TP 组内现场广播收主帧
        embeds 与 aux(mrope)帧,完成窗口核验与 mrope 行写入
        req_state(末 chunk 自推 delta),返回 [(seqno, meta, up_flat)]
        供 super() 后 embeds 覆写消费。

        The recv uses the host gate + consume-time TP broadcast (each
        frame its own launch+wait atomic pair on the current stream —
        the prefill_only_demo_br discipline; 跨流排序不可依赖是此前
        广播段交付残缺的根因)。

        pdmix 混批下 has_mrope 是逐请求列表:aux 帧只含标记请求的
        [n,3] 行(Σ n_i×3,与边侧组帧同源同序),mrope 行游标独立于
        embeds 行游标推进。"""
        worker = self.worker
        if worker is None:
            return []
        posted = getattr(worker, "_lwd_up_recv_futures", None)
        if not posted:
            return []
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        hidden_size = self.model_config.get_hidden_size()

        harvested = []
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_tp_group

        _tp = get_tp_group()
        _lwd_cfg = worker.parallel_config.lwd_config
        _is_endpoint = worker.rank == _lwd_cfg.edge_npu_count
        for batch_seqno in list(posted.keys()):
            item = worker._lwd_up_recv_futures.pop(batch_seqno, None)
            if item is None:
                continue
            future, meta = item
            _t_wait = time.monotonic()
            numel = sum(len(t) for t in meta.token_ids) * hidden_size
            aux_numel = (
                sum(
                    len(t)
                    for t, hm in zip(meta.token_ids, meta.has_mrope)
                    if hm
                )
                * 3
            )
            if _is_endpoint:
                # 端点:先过 host 就绪门(demo 的 wait gate):done_event
                # 轮询通过即 P2P 数据已完整落 buffer;超时显式报错而非
                # 静默乱码。门通过后再广播,广播读到的必然是完整数据。
                res = future.wait(timeout=60.0)
                up_flat = res.tensor
                if up_flat is None:
                    continue
                # aux 帧与主帧同一逻辑请求:端点从 future 结果直接取
                # 第二载荷。
                aux_flat = res.aux_tensor
            else:
                # 非端点:现场分配接收 buffer,等端点广播转发。
                up_flat = torch.empty(
                    numel, dtype=torch.bfloat16, device="npu"
                )
                aux_flat = None
            if _tp.world_size > 1:
                # demo 同款:直接用框架 TP 通信域(建组顺序由框架保证,
                # PD 分离路径已验证),不再使用自建 _LWD_EMBED_BCAST_GROUP。
                work = dist.broadcast(
                    up_flat,
                    src=_tp.ranks[0],
                    group=_tp.device_group,
                    async_op=True,
                )
                work.wait()  # 桥接广播完成到当前(计算)流,后续 copy 有序
            mrope_flat = None
            if aux_numel > 0:
                # aux 帧的第二段广播:独立 launch+wait 原子对(与主广播
                # 同计算流背靠背,不共用句柄);非端点现场分配精确尺寸
                # int64 缓冲。
                if not _is_endpoint:
                    aux_flat = torch.empty(
                        aux_numel, dtype=torch.int64, device="npu"
                    )
                if _tp.world_size > 1:
                    work_aux = dist.broadcast(
                        aux_flat,
                        src=_tp.ranks[0],
                        group=_tp.device_group,
                        async_op=True,
                    )
                    work_aux.wait()
                mrope_flat = aux_flat.view(-1, 3)
            logger.info(
                "[Lwd][perf] cloud up-recv-wait seqno=%s dur=%.2fms",
                batch_seqno, (time.monotonic() - _t_wait) * 1000,
            )
            logger.info(
                "[Lwd][cloud-runner] UP embeds harvested seqno=%s rows=%d "
                "reqs=%s mrope_rows=%s",
                batch_seqno, numel // hidden_size, meta.req_ids,
                mrope_flat.shape[0] if mrope_flat is not None else 0,
            )
            # 窗口核验 + mrope 行写入(当步生效的前提:super() 内的
            # _calc_mrope_positions 随即按窗读 req_state)
            mrope_row = 0
            for k, (req_id, token_ids) in enumerate(
                zip(meta.req_ids, meta.token_ids)
            ):
                n = len(token_ids)
                idx = self.input_batch.req_id_to_index.get(req_id)
                if idx is None:
                    # fail-fast:embeds 已收齐但请求不在本步 batch——继续
                    # 走下去该请求将用未注入的脏 embeds 解码(静默乱码),
                    # 必须当场暴露而非丢弃。出现即调度/数据面时序契约
                    # 被破坏,需要修的是上游而不是这里。
                    raise RuntimeError(
                        f"[Lwd][cloud-runner] INJECT req={req_id} seqno="
                        f"{batch_seqno} rows={n}: req not in input_batch "
                        f"(batch={self.input_batch.req_ids})"
                    )
                if n > 0:
                    # fail-fast:本 chunk 行数必须恰等于该请求本步调度 token
                    # 数。不等(调度窗口 ≠ 边侧 chunk,如预算挤压截断)时
                    # 注入会越窗踩邻请求行/留下未注入尾巴——静默乱码源,
                    # 直接暴露。
                    scheduled = (
                        int(num_scheduled_tokens[idx])
                        if idx < len(num_scheduled_tokens) else 0
                    )
                    if n != scheduled:
                        raise RuntimeError(
                            f"[Lwd][cloud-runner] INJECT window mismatch "
                            f"req={req_id} seqno={batch_seqno}: chunk rows="
                            f"{n} != scheduled={scheduled} (computed="
                            f"{int(computed[idx])} prompt="
                            f"{int(num_prompt[idx])})"
                        )
                    if meta.has_mrope[k]:
                        # mrope 行与 embeds 行同窗(边侧同批切片);aux
                        # 帧只含标记请求,游标独立推进
                        self._lwd_inject_mrope_positions(
                            req_id, int(computed[idx]), n,
                            mrope_flat[mrope_row : mrope_row + n],
                            int(num_prompt[idx]),
                        )
                        mrope_row += n
            harvested.append((batch_seqno, meta, up_flat))
            # aux 帧跨流生命周期登记(端点 recv buffer;非端点本地分配,
            # 登记无害)——aux 的消费(注入内 D2H)在本函数内完成
            if aux_flat is not None:
                aux_flat.record_stream(torch.npu.current_stream())
        return harvested

    def _lwd_inject_remote_embeds(self, harvested, num_scheduled_tokens) -> None:
        """Device-side inject(super() 之后):把收割到的 UP embeds 直写
        ``inputs_embeds.gpu`` 各请求的当步调度窗口。

        The flattened output offsets reproduce the native fill loop's
        accumulation (per-request scheduled segment start), so rows land
        exactly where the prompt-embeds branch expects them.  The CPU
        assembly buffer is also filled via an on-stream D2H copy — its
        only downstream consumer (draft first-pass provider) reads it
        with stream-ordered H2D copies, so no host sync is needed there
        either.  The base fill loop may have copied stale buffer content
        earlier; our device write happens after it and is authoritative.
        """
        if not harvested:
            return
        embeds_map = self.input_batch.req_prompt_embeds
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        hidden_size = self.model_config.get_hidden_size()
        gpu_embeds = self.inputs_embeds.gpu

        # Flattened output offset per request (native fill loop 同款累计):
        # 每个请求的调度段在扁平 token 序列中的起点。
        out_offset: dict[str, int] = {}
        off = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            out_offset[req_id] = off
            off += int(num_scheduled_tokens[i]) if i < len(num_scheduled_tokens) else 0

        for batch_seqno, meta, up_flat in harvested:
            embeds = up_flat.view(-1, hidden_size)
            logger.info(
                "[Lwd][cloud-runner] UP embeds injected seqno=%s rows=%s "
                "reqs=%s",
                batch_seqno, embeds.shape[0], meta.req_ids,
            )
            row = 0
            for req_id, token_ids in zip(meta.req_ids, meta.token_ids):
                n = len(token_ids)
                idx = self.input_batch.req_id_to_index.get(req_id)
                if idx is None:
                    # 与收割期同款防线(双阶段各自独立核验)
                    raise RuntimeError(
                        f"[Lwd][cloud-runner] INJECT req={req_id} seqno="
                        f"{batch_seqno} rows={n}: req not in input_batch "
                        f"(batch={self.input_batch.req_ids})"
                    )
                if n > 0:
                    start = int(computed[idx])
                    out = out_offset.get(req_id, 0)
                    # stream 上直接写 inputs_embeds.gpu 的调度窗口
                    gpu_embeds[out : out + n].copy_(
                        embeds[row : row + n], non_blocking=True
                    )
                    # CPU 组装缓冲同 stream D2H(供 draft provider 用)
                    prompt_len = int(num_prompt[idx])
                    buf = embeds_map.get(idx)
                    if buf is not None and buf.shape[0] == prompt_len:
                        buf[start : start + n].copy_(
                            embeds[row : row + n], non_blocking=True
                        )
                    self.input_batch.is_token_ids[idx, start : start + n] = False
                row += n
            # 跨流生命周期登记:端点 recv buffer 由通道流分配与复用
            # (同尺寸 chunk 下分配器几乎总给同一块),本步 copy 在计算流。
            # 不登记 record_stream,分配器可在本批 copy 尚未执行时把块交给
            # 下一 seqno 的 irecv 覆写——深异步队列下表现为跨 seqno 串
            # 数据(多请求乱码)。登记后复用即被正确排序。
            up_flat.record_stream(torch.npu.current_stream())

    # ------------------------------------------------------------------ #
    # Collect probe (measurement only; output discarded, protocol        #
    # still returns token_ids directly)                                   #
    # ------------------------------------------------------------------ #

    def _lwd_inject_mrope_positions(
        self,
        req_id: str,
        start: int,
        n: int,
        chunk_positions: torch.Tensor,
        prompt_len: int,
    ) -> None:
        """写入一个 chunk 的线 mrope positions([n,3] int64,NPU)到请求
        缓存的 [3, prompt] 窗口;末 chunk 落地时自推 delta(=
        positions.max()+1-prompt_len,与边侧逐值一致)供 decode 期
        原生 _calc_mrope_positions 现算 completion 段位置。

        req_state.mrope_positions 由 ids=None 直通初始化(arange 纯文
        本位置)——多模态请求的窗口被逐 chunk 覆盖,纯文本请求永不走
        本路径(边侧不给它发 mrope 帧)。"""
        req_state = self.requests.get(req_id)
        assert chunk_positions.shape == (n, 3), (
            f"mrope frame shape {tuple(chunk_positions.shape)} != "
            f"({n}, 3) (req={req_id})"
        )
        req_state.mrope_positions[:, start : start + n] = (
            chunk_positions.t().cpu()
        )
        if start + n >= prompt_len:
            positions = req_state.mrope_positions[:, :prompt_len]
            req_state.mrope_position_delta = (
                int(positions.max().item()) + 1 - prompt_len
            )
            logger.info(
                "[Lwd][cloud-runner] mrope complete: req=%s prompt=%d "
                "delta=%d",
                req_id, prompt_len, req_state.mrope_position_delta,
            )

    def _sample(self, logits, spec_decode_metadata):
        """Capture the sampler output for the collect probe
        (the base sample_tokens clears execute_model_state on return)."""
        sampler_output = super()._sample(logits, spec_decode_metadata)
        self._lwd_captured_sampler_output = sampler_output
        return sampler_output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
        from vllm.distributed.parallel_state import get_tp_group

        # 只有云 TP 组首卡(= DOWN 通道端点 rank)采集/发送;
        # 其余 rank 无通道 peer,采集即弃也一并省掉(8 卡冗余)。
        wire_endpoint = get_tp_group().is_first_rank
        captured = None
        if wire_endpoint and self.execute_model_state is not None:
            # ExecuteModelState layout (see model_runner_v1.sample_tokens):
            # (scheduler_output, logits, spec_decode_metadata,
            #  spec_decode_common_attn_metadata, hidden_states,
            #  sample_hidden_states, ...)
            state = self.execute_model_state
            # 批序/掩码也一并快照:batch queue 重叠时,下一步的
            # _prepare_inputs 可能在 collect 前重排 input_batch,
            # 现读会拿到批序B去切批序A的张量(高并发错位根因)
            captured = (
                state[5], state[1], state[2], state[0],
                list(self.input_batch.req_ids),
                self.discard_request_mask.np.copy(),
            )
        self._lwd_captured_sampler_output = None
        output = super().sample_tokens(grammar_output)
        # VLLM_ASCEND_LWD_DISABLE_DOWN=1 诊断:不做 collect(云引擎改走
        # token_ids 直通通告),payload 为 None 时 worker 自然跳过 DOWN send
        if (captured is not None
                and self._lwd_captured_sampler_output is not None
                and not envs.VLLM_ASCEND_LWD_DISABLE_DOWN):
            self._lwd_pending_down_payload = self._lwd_collect_down_payload(
                captured[0], captured[1], captured[2], captured[3],
                captured[4], captured[5], self._lwd_captured_sampler_output,
            )
        return output

    def take_lwd_pending_down_payload(self):
        """worker 层取走本步 DOWN payload(单槽覆盖写,每步必被取走)。"""
        payload = getattr(self, "_lwd_pending_down_payload", None)
        self._lwd_pending_down_payload = None
        return payload

    @torch.inference_mode()
    def _lwd_collect_down_payload(
        self, sample_hidden_states, logits, spec_decode_metadata,
        scheduler_output, batch_req_ids, discard_mask_np, sampler_output,
    ):
        """生产级 DOWN 采集(rank-replay):hidden 组包 + 全局秩 + num_accepted。

        廉价计算:bf16 直比(logits 原生 bf16,与 cast 后逐位等价)、
        批全覆盖时整行直接算(不做高级索引取材)、ranks/counts/seg_lens
        拼单个设备张量;meta 经 pinned(4 轮换)在主流末尾异步拷贝,
        worker 属性上紧随记录就绪事件,由响应入队处 synchronize 后
        引擎侧解码——计算关键路径零新增同步。批序/掩码取捕获时刻
        快照(防 batch queue 重叠重排)。返回:
          (hidden_packet, pinned_view, None, req_ids, None)
        或 None(本步无在途请求)。
        """
        _t0 = time.monotonic()
        sampled = sampler_output.sampled_token_ids
        if sampled is None or sampled.dim() != 2 or logits is None:
            return None
        # batch_req_ids/discard_mask 来自捕获时刻快照(与 logits/hidden
        # 同批序),不读活的 input_batch
        valid = ~discard_mask_np[: len(batch_req_ids)]
        is_spec = spec_decode_metadata is not None

        rows_list, ranks_list, accepted = [], [], []
        hidden_packet: torch.Tensor | None = None
        lg = logits  # bf16 原生,直接比较(cast 无精度增益)
        full_cover = all(valid[: len(batch_req_ids)])
        if not is_spec:
            idx = [i for i in range(len(batch_req_ids)) if valid[i]]
            if not idx:
                return None
            lg_sel = lg if full_cover else lg[idx]
            sm_sel = sampled[:, 0] if full_cover else sampled[idx][:, 0]
            thresh = lg_sel.gather(1, sm_sel.long().unsqueeze(1))
            ranks_list.append((lg_sel > thresh).sum(dim=1).to(torch.int32))
            rows_list.extend(sample_hidden_states[i : i + 1] for i in idx)
            accepted.extend([1] * len(idx))
        else:
            # 全零同步 spec 路径(向量化):
            # 段长按 host 侧 scheduled_spec_decode_tokens 推(1+draft_len),
            # 按完整段打包(含被拒行,边侧按 num_accepted 取有效前缀);
            # 段长取 spec_decode_metadata.num_draft_tokens(host list,
            # 运行期真实布局,与 sample_hidden_states 段结构一致);
            # 不用 scheduler_output.scheduled_spec_decode_tokens
            # (调度输入,可能与实际运行不一致)
            seg_lens = [
                d + 1 for d in spec_decode_metadata.num_draft_tokens
            ]
            # counts 从同一个 sampled 张量 device 推导(与秩/行同源,
            # 天然按批位对齐);seg_lens/counts 只收 valid 请求,
            # 保证 [ranks|counts|seg_lens] 三段长度一致
            counts_all = (sampled != -1).sum(dim=1)
            # host 一趟建索引,替代逐请求 device 循环(原实现每请求
            # ~6 次小 kernel 发射,rank 段随并发涨到 20+ms;向量化后
            # 发射数与 N 无关)。索引语义与原循环逐位等价:
            # vidx/vlen = 有效请求批位/实际行数(按总段长封顶,
            # = 原 rows_i);req_idx/pos_idx = 每行的请求位/行内位
            # (原 sampled[i, :rows_i]);row_sel = 有效行全局行号。
            seg_lens_np = np.asarray(seg_lens, dtype=np.int32)
            seg_starts = np.cumsum(seg_lens_np) - seg_lens_np
            total_rows = sample_hidden_states.shape[0]
            rows_per_req = np.minimum(
                seg_lens_np, np.maximum(total_rows - seg_starts, 0)
            )
            vidx_np = np.nonzero(
                valid[: len(batch_req_ids)] & (rows_per_req >= 1)
            )[0].astype(np.int64)
            if vidx_np.size == 0:
                return None
            vlen_np = rows_per_req[vidx_np].astype(np.int64)
            starts_v = seg_starts[vidx_np].astype(np.int64)
            n_rows = int(vlen_np.sum())
            req_idx_np = np.repeat(vidx_np, vlen_np)
            pos_idx_np = np.arange(n_rows, dtype=np.int64) - np.repeat(
                np.cumsum(vlen_np) - vlen_np, vlen_np
            )
            row_sel_np = np.repeat(starts_v, vlen_np) + pos_idx_np
            # 全部索引一次 pinned 异步 H2D(pageable 同步 H2D 会在
            # 主流上等设备排空,见 seg_lens 同款教训)
            buf = self._lwd_h2d_stage_i64(
                np.concatenate(
                    [req_idx_np, pos_idx_np, row_sel_np, vidx_np, vlen_np]
                )
            )
            n_valid = vidx_np.size
            req_d, pos_d = buf[:n_rows], buf[n_rows : 2 * n_rows]
            sel_d = buf[2 * n_rows : 3 * n_rows]
            vidx_d = buf[3 * n_rows : 3 * n_rows + n_valid]
            vlen_d = buf[3 * n_rows + n_valid :]
            sampled_rows = sampled[req_d, pos_d]
            if n_rows == total_rows:
                # 全 valid 无封顶:行即连续前缀,免 gather
                lg_rows = lg[:n_rows]
                hidden_packet = sample_hidden_states[:n_rows]
            else:
                lg_rows = lg[sel_d]
                hidden_packet = sample_hidden_states[sel_d]
            thresh = lg_rows.gather(1, sampled_rows.long().unsqueeze(1))
            ranks_list.append((lg_rows > thresh).sum(dim=1).to(torch.int32))
            # 过期 sampled 位(spec 未运行的请求可能残留上个 spec 步的
            # token):accepted/秩/段长一律按实际 hidden 行数封顶——
            # 行数才是真实采样位置的真相
            counts_dev = torch.minimum(counts_all[vidx_d], vlen_d).to(
                torch.int32
            )
            seg_lens_list = vlen_np.tolist()
        if hidden_packet is None:
            if not rows_list:
                return None
            hidden_packet = torch.cat(rows_list)
        n_req = sum(1 for i in range(len(batch_req_ids)) if valid[i])
        seg_lens_host: list[int] | None = None
        if is_spec:
            meta_dev = torch.cat(ranks_list + [counts_dev])
            # seg_lens 不上设备(host 直写 pinned 尾段):与 D2H 前缀
            # 区域不相交,且引擎读 pinned 发生在 ready_event 同步
            # (更晚)之后,无竞态。
            seg_lens_host = seg_lens_list
        else:
            counts_dev = torch.ones(n_req, dtype=torch.int32,
                                    device=logits.device)
            seg_lens_dev = counts_dev
            meta_dev = torch.cat(ranks_list + [counts_dev, seg_lens_dev])
        # pinned 拷贝排主流末尾(异步),紧随记录就绪事件;
        # 由 worker 响应入队处在发送前 synchronize——
        # "响应发出 ⟹ pinned 就绪"成为硬保证(同步在输出线程,
        # 不在计算关键路径)。
        _t_pack = time.monotonic()
        n_dev = meta_dev.numel()
        n_meta = n_dev + (len(seg_lens_host) if seg_lens_host is not None else 0)
        pinned = self._lwd_meta_pinned(n_meta)
        pinned[:n_dev].copy_(meta_dev, non_blocking=True)
        if seg_lens_host is not None:
            pinned.numpy()[n_dev:n_meta] = seg_lens_host
        self.worker._lwd_meta_ready_event = torch.npu.Event()
        self.worker._lwd_meta_ready_event.record()
        # [Lwd][perf] 采集分段:rank=秩+行收集;pinned=meta 拼包+拷贝
        logger.info(
            "[Lwd][perf] collect rows=%d rank=%.2f pinned=%.2f total=%.2fms",
            hidden_packet.shape[0],
            (_t_pack - _t0) * 1000,
            (time.monotonic() - _t_pack) * 1000,
            (time.monotonic() - _t0) * 1000,
        )
        return (
            hidden_packet,
            pinned[:n_meta],
            None,
            [r for i, r in enumerate(batch_req_ids) if valid[i]],
            None,
        )

    def _lwd_meta_pinned(self, n: int):
        """轮换 pinned 缓冲(深度 4 > batch_queue 深度 2 + 引擎滞后 1):
        避免下一/N+2 步的边流拷贝覆盖引擎尚未读完的上一步 meta。"""
        ring = getattr(self, "_lwd_pinned_ring", None)
        if ring is None or ring[0].numel() < n:
            ring = [
                torch.empty(max(n, 4096), dtype=torch.int32, pin_memory=True)
                for _ in range(4)
            ]
            self._lwd_pinned_ring = ring
            self._lwd_pinned_ring_idx = 0
        idx = self._lwd_pinned_ring_idx
        self._lwd_pinned_ring_idx = (idx + 1) % len(ring)
        return ring[idx]

    def _lwd_h2d_stage_i64(self, arr: "np.ndarray") -> torch.Tensor:
        """int64 host 索引数组的异步 H2D 暂存(pinned 4 槽轮换 +
        non_blocking):替代 torch.tensor(arr, device=...) 的 pageable
        同步拷贝——后者在主流上等设备排空(MTP 下即 draft propose
        尾巴)。环深与 meta ring 同理:覆盖 batch_queue 深度 + 引擎滞后,
        防止下步 host 覆写时上步拷贝尚未执行。"""
        n = arr.size
        ring = getattr(self, "_lwd_h2d_ring", None)
        if ring is None or ring[0][0].numel() < n:
            m = max(n, 1024)
            ring = [
                (
                    torch.empty(m, dtype=torch.int64, pin_memory=True),
                    torch.empty(m, dtype=torch.int64, device=self.device),
                )
                for _ in range(4)
            ]
            self._lwd_h2d_ring = ring
            self._lwd_h2d_ring_idx = 0
        idx = self._lwd_h2d_ring_idx
        self._lwd_h2d_ring_idx = (idx + 1) % len(ring)
        stage, dev = ring[idx]
        stage.numpy()[:n] = arr
        dev[:n].copy_(stage[:n], non_blocking=True)
        return dev[:n]

