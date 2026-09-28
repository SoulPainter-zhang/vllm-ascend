# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only LWD (edge-cloud) worker subclass.

All LWD worker-side logic lives here (moved out of ``worker.py``):
init bring-up of the duplex channels + recv managers, the EDGE_EMBED
dispatch hook, the cloud finished-flush, and the
DOWN send (draining the runner's per-step hidden payload + attaching
``lwd_c2e_meta`` to the ModelRunnerOutput).

Selected via ``parallel_config.worker_cls`` (see platform.py:
``vllm_ascend.worker.lwd_cloud_worker.LwdCloudWorker``) when
``lwd_config`` enables prefill_only.
"""

from __future__ import annotations

import threading
import time

import torch
from vllm.logger import logger
from vllm.v1.core.sched.output import GrammarOutput  # noqa: F401  (type)
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized

from vllm_ascend.batch_invariant import init_batch_invariance
from vllm_ascend.distributed.lwd_comm.future import LwdCommFuture
from vllm_ascend.distributed.lwd_comm.lwd_parallel_init import (
    init_lwd_ascend_model_parallel,
)
from vllm_ascend.distributed.lwd_comm.service import get_lwd_comm_service
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest

# ops 须先于 model_runner 链初始化，否则 device_op 与 ops 包循环导入
#（对齐 worker.py 的导入顺序）
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.worker.lwd_cloud.lwd_cloud_model_runner import LwdCloudModelRunner
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import hash_layer_count, hash_payload_numel
from vllm_ascend.worker.worker import NPUWorker


class LwdCloudWorker(NPUWorker):
    """NPUWorker + LWD data-plane wiring (cloud side only).

    This subclass is selected by ``platform.py`` only when the LWD
    deployment enables LWD on a cloud process, so no role/mode checks
    are needed here — ``self.enable_lwd`` (set by NPUWorker.__init__
    from vllm_config.lwd_config) is the single switch."""

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
        super().init_device()
        # channel-global DOWN seqno counter (worker layer, send-time alloc)
        self._lwd_down_next_seqno = 0
        # req_id -> posted UP recv futures (consumed by the runner's device-side inject)
        self._lwd_up_recv_futures: dict[str, list] = {}
        # [Lwd][pre-recv] 提前收:端点 rank(= 唯一持跨机 P2P 通道的云 rank,
        # 与 lwd_wire 同款约定)才提前挂真实 irecv 并上报收完水位;其余 rank
        # 的数据在注入时刻由 TP 组内广播取到,既不挂 recv 也不上报。
        self._lwd_up_hidden_size = 0
        self._lwd_early_recv_endpoint = False
        self._init_lwd_early_recv_state()
        if not self.enable_lwd:
            return
        if self.use_v2_model_runner:
            # The LWD hooks live on the V1 model runner; the V2 runner
            # (separate class) lacks them entirely — fail fast at bring-up
            # instead of AttributeError at first request.
            raise RuntimeError(
                "prefill_only (LWD) data plane requires the V1 model runner; "
                "use_v2_model_runner is not supported"
            )
        # Swap in the LWD model runner BEFORE any model load/usage.
        # The cloud feeds on edge-computed prompt embeddings (no token
        # ids on its input), so the native prompt-embeds path must be
        # enabled for the runner to allocate inputs_embeds buffers.
        self.model_config.enable_prompt_embeds = True
        self.model_runner = LwdCloudModelRunner(
            self.vllm_config, self.device, worker=self
        )
        from vllm_ascend.distributed import lwd_wire

        lwd_wire.init_lwd_duplex_channels()
        # 端点 rank 才有跨机通道:把它的提前收通信线程起起来(hint MQ 已由
        # 执行器在建 worker 之前建好并写进环境变量;done MQ 由本处建 writer,
        # handle 随 READY 握手回传引擎)。其余 rank 只登记"非端点"。
        self._lwd_early_recv_endpoint = (
            self.rank == self.parallel_config.lwd_config.edge_npu_count
        )
        self._lwd_up_hidden_size = self.model_config.get_hidden_size()
        # 每个云 rank 都打一行:谁该起通信线程一目了然
        logger.info(
            "[Lwd][pre-recv] worker rank=%d endpoint=%s hidden=%d",
            self.rank, self._lwd_early_recv_endpoint, self._lwd_up_hidden_size,
        )
        if self._lwd_early_recv_endpoint:
            self._start_lwd_early_recv_thread()
        self._register_lwd_prompt_embeds_provider()

    def load_model(self):
        """加载后打实际切片:层数/首末层名/pp 切层参数,启动期即可裁决
        半模型嫌疑(全量且从 layer 0 起 = 正常;减半/起始非 0 = pp 切错)。"""
        super().load_model()
        model = self.model_runner.get_model()
        backbone = getattr(model, "model", model)
        layers = getattr(backbone, "layers", None) or getattr(
            backbone, "decoder_layers", None
        )
        pc = self.vllm_config.parallel_config
        layer_names = [
            n for n, _ in model.named_modules()
            if n.count("layers.") == 1 and n.endswith(tuple("0123456789"))
        ]
        logger.info(
            "[Lwd][cloud-model] loaded layers=%s first=%s last=%s "
            "(config pp=%d tp=%d, my rank=%d) — 全量应覆盖 layer 0 起的全部层",
            len(layers) if layers is not None else "?",
            layer_names[0] if layer_names else "?",
            layer_names[-1] if layer_names else "?",
            pc.pipeline_parallel_size, pc.tensor_parallel_size, self.rank,
        )

    # ------------------------------------------------------------------ #
    # Engine step wiring                                                  #
    # ------------------------------------------------------------------ #

    def execute_model(self, scheduler_output):
        # [Lwd][perf] worker 开工时刻(与引擎 rpc-enqueue ts 对减 =
        # 下达-开工延迟;各卡开工离散 = 集合对齐成本)
        logger.info(
            "[Lwd][perf] worker-exec-start rank=%d ts=%.3f",
            self.rank, time.monotonic(),
        )
        if self.enable_lwd:
            # cloud: post the exact-size UP irecv for every incoming
            # LWD_EMBED batch (control info rides scheduler_output.lwd_batch).
            self._lwd_up_post_recvs(scheduler_output)

        # cloud: finished requests trigger local cleanup (collector
        # bookkeeping + UP chunk table); finished_req_ids is the
        # engine's own liveness signal.
        if scheduler_output.finished_req_ids:
            self._lwd_cloud_flush_finished(scheduler_output.finished_req_ids)

        get_lwd_comm_service().poll_completions()  # lazy keepalive reap
        _t0 = time.monotonic()
        output = super().execute_model(scheduler_output)
        # [Lwd][perf] 云侧每步 exec_model 计时:total=前向全程
        # (forward 主段,sample/collect 在 sample_tokens 另有分项)
        logger.info(
            "[Lwd][perf] cloud-exec total=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return output

    def _lwd_up_post_recvs(self, scheduler_output) -> None:
        """Bind the UP recv for an incoming LWD_EMBED batch.

        UP 两段通信按 prefill_only_demo_br 范式拆到两个时刻:
          * post 时刻(本函数):仅端点 rank 经通道 P2P 挂 recv;非端点只登记
            meta 占位;
          * 消费时刻(runner 注入侧):各卡在*自己的计算流上*现场发起 TP 组内
            broadcast 并紧随 work.wait()——torch_npu 下两个 HCCL op 之间的跨流
            排序不可依赖(实测 post 时刻发起的广播抢跑 irecv、交付残缺),
            launch+wait 同流上下文原子对才是 demo 验证过的可靠形态。

        提前收(early recv):端点那条 irecv **已由通信线程按引擎提示提前挂好**
        (RangeNotify 到达即挂,不等本 SO),本处只是取用那条 future;闸门
        (``_lwd_pre_recv_ready``)保证放行时它一定已挂出。提示万一没到,本处用
        同一份信息现场补挂(见 ``get_or_post_early_recv``),不会漏挂。

        All control info rides the SchedulerOutput (edge -> cloud
        control plane fills it in):batch.seqno 为边侧派发号(跨机段
        配对键),recv 尺寸包含 embedding 及可选的 Hash 专家 ID。"""
        from vllm.v1.core.sched.output import LwdBatchType

        batch = getattr(scheduler_output, "lwd_batch", None)
        if batch is None or batch.batch_type is not LwdBatchType.LWD_EMBED:
            return
        meta = batch.batch_meta
        if meta is None or not meta.req_ids:
            return
        num_tokens = sum(len(t) for t in meta.token_ids)
        if num_tokens <= 0:
            return
        # aux 帧(mrope positions)行数:仅带 mrope 的请求算,与边侧组帧同源同序
        aux_rows = sum(
            len(t) for t, hm in zip(meta.token_ids, meta.has_mrope) if hm
        )

        if self._lwd_early_recv_endpoint:
            future = self.get_or_post_early_recv(
                batch.seqno, num_tokens, aux_rows
            )
        else:
            # 非端点:不碰跨机通道;广播接收推迟到注入时刻现场发起,
            # None 即"消费时现场广播"标记。
            future = None
        self._lwd_up_recv_futures[batch.seqno] = (future, meta)
        if future is None:
            # 每批 7 行 INFO 是纯噪音,降到 DEBUG(VLLM_LOGGING_LEVEL=DEBUG
            # 时仍能看到各 rank 进前向的时刻/离散)
            logger.debug(
                "[Lwd][pre-recv] POST seqno=%d reqs=%d tokens=%d aux_rows=%d "
                "endpoint=False ts=%.3f",
                batch.seqno, len(meta.req_ids), num_tokens, aux_rows,
                time.monotonic(),
            )
            return
        # 每批一条显存读数(仅端点 rank 有 recv buffer):峰值一路爬 = 有东西
        # 没释放;mem_free 逼近 0 = OOM 前兆。
        free_bytes, _total = torch.npu.mem_get_info()
        logger.info(
            "[Lwd][pre-recv] POST seqno=%d reqs=%d tokens=%d aux_rows=%d "
            "endpoint=True ts=%.3f mem_alloc=%.2fGiB mem_peak=%.2fGiB "
            "mem_free=%.2fGiB",
            batch.seqno, len(meta.req_ids), num_tokens, aux_rows,
            time.monotonic(),
            torch.npu.memory_allocated() / 2 ** 30,
            torch.npu.max_memory_allocated() / 2 ** 30,
            free_bytes / 2 ** 30,
        )

    # ------------------------------------------------------------------ #
    # 提前收:通信线程(提示下发 -> 挂 recv -> 上报水位)                     #
    # ------------------------------------------------------------------ #

    def _init_lwd_early_recv_state(self) -> None:
        """提前收收单(与引擎下发的提示、上报的水位同一套键)。

        key = (channel, seqno)。条目生命周期:提示到达即 submit_recv 并登记;
        消费点 ``get_or_post_early_recv`` 只是**取用并标记已消费**(不 pop:
        上报线程还要继续轮询它推进水位);"既已上报又已消费"才移除。

        **在飞上限 = 3**:一条已挂 recv 占一块整批大小的接收 buffer(含 aux 帧),
        而这个 3 与边侧的 pending 闸门同源 —— ``LwdEdgeScheduler``
        ``_LWD_MAX_PENDING_CHUNKS = 3`` 按"未消费 chunk(seqno)"限制边侧领先量,
        所以云侧最多也就需要这 3 条;上限之外的提示先排队(``_early_recv_pending``,
        按到达序,只存字段不占显存),前面的批随 SO 被消费腾出名额后立刻补挂。
        归还点必须是"随 SO 被消费"(=已下发)而不是"收完":闸门放行第 N 批的
        条件是水位 ≥ N,放行后 execute_model 立刻要取这条 future;名额若拖到
        收完才还,N+1 的 recv 可能还没挂上,闸门就会跑到 recv 前面。
        """
        self._early_recv_max_inflight = 3
        self._early_recv_futures: dict[tuple[LwdChannelType, int], LwdCommFuture] = {}
        self._early_recv_lock = threading.Lock()
        # 已被消费点取走的 key:防止通信线程在提示迟到时再挂一条重复 recv
        self._early_recv_consumed: set[tuple[LwdChannelType, int]] = set()
        # 已上报完成的 key
        self._early_recv_reported: set[tuple[LwdChannelType, int]] = set()
        # 名额已满时排队的提示(key -> hint,插入序即到达序)
        self._early_recv_pending: dict[tuple[LwdChannelType, int], dict] = {}
        # 队列卡住诊断的限频时间戳
        self._early_recv_stuck_log_at = 0.0
        # 下行提示 MQ(reader)/ 上行完成上报 MQ(writer)
        self._irecv_hint_mq = None
        self._irecv_done_mq = None
        # 公开别名:WorkerProc 在 worker 构造后取它,把 handle 随 READY 回传,
        # 引擎据此挂 done reader(见 vllm/v1/executor/multiproc_executor.py)。
        self.irecv_done_mq = None
        self._lwd_early_recv_thread: threading.Thread | None = None

    def _lwd_build_up_recv_request(
        self, seqno: int, num_tokens: int, aux_rows: int
    ) -> LwdCommRequest:
        """UP recv 请求的唯一构造点(提示路径与消费点补挂共用同一形状)。

        HCCL P2P 要求两端 numel 严格相等:主帧 = hash 载荷公式(embeds + 可选
        专家 ID 字节),aux 帧 = mrope rows*3(int64,[n,3])。尺寸公式只在本处
        算一次,两侧路径不会各算一遍算歪。
        """
        return LwdCommRequest(
            channel=LwdChannelType.UP,
            op="recv",
            num_elements=hash_payload_numel(
                num_tokens, self._lwd_up_hidden_size,
                hash_layer_count(self.model_config.hf_config),
                getattr(self.model_config.hf_config, "num_experts_per_tok", 0),
            ),
            seqno=int(seqno),
            # aux 帧(mrope positions [n,3] int64)与主帧同 seqno;
            # 0 = 纯文本批,零 aux 流量
            aux_num_elements=int(aux_rows) * 3,
        )

    def _start_lwd_early_recv_thread(self) -> None:
        """重建 hint MQ reader、建 done MQ writer,起提前收通信线程。

        起线程前后各打一行:"starting" 在起之前(证明这段代码走到了)、"up"
        在线程体里设完设备之后(证明线程真的活起来了)。只有前一行 = 线程在
        设置设备时挂了(异常栈在 stderr);两行都在但没 SUBMIT = 没收到提示
        (hint MQ 没挂上,日志里会写 attached/none)。
        """
        from vllm_ascend.distributed.lwd_comm.pre_recv import (
            attach_hint_mq,
            create_report_mq,
        )

        self._irecv_hint_mq = attach_hint_mq()
        self._irecv_done_mq = create_report_mq()
        self.irecv_done_mq = self._irecv_done_mq
        logger.info(
            "[Lwd][pre-recv] comm thread starting rank=%d hint MQ=%s",
            self.rank, "attached" if self._irecv_hint_mq is not None else "none",
        )
        self._lwd_early_recv_thread = threading.Thread(
            target=self._lwd_early_recv_loop,
            name="lwd-early-recv",
            daemon=True,
        )
        self._lwd_early_recv_thread.start()

    def _lwd_early_recv_loop(self) -> None:
        """通信线程主体:取提示 -> 提前挂 recv;轮询 future -> 上报收完。

        只做四件极轻的事:``dequeue`` / ``submit_recv`` / ``future.done()`` /
        ``enqueue`` —— 从不 ``wait()`` 任何 future、也不碰模型,所以不与
        busy_loop 的 HCCL 使用冲突(同通道上一端 irecv 一端跨线程 wait 是
        HCCL 不允许的)。
        """
        # 当前设备是线程级状态,通信线程要自己设一次(否则 irecv 落到 0 卡)
        torch.npu.set_device(self.device)
        logger.info(
            "[Lwd][pre-recv] comm thread up rank=%d device=%s hint MQ=%s",
            self.rank, self.device,
            "attached" if self._irecv_hint_mq is not None else "none",
        )
        hint_mq = self._irecv_hint_mq
        while True:
            if hint_mq is not None:
                try:
                    hint = hint_mq.dequeue(timeout=0.0005)
                except TimeoutError:
                    pass
                except Exception:
                    # 关停时 ring 被取消等:记一条、小睡,别热转(daemon 线程
                    # 随进程退出)
                    logger.exception("[Lwd][pre-recv] hint MQ dequeue failed")
                    time.sleep(0.01)
                else:
                    self.start_early_irecv(hint)
            self._report_irecv_completions()
            self._warn_if_early_recv_stuck()
            # ring 读端在"刚读过"的时间窗内是 sched_yield 自旋(busy_loop_s=1s),
            # 光给 0.5ms 超时并不封顶 CPU —— 这行 sleep 才是封顶,代价是
            # 提示/水位各约 2ms 的可见延迟。
            time.sleep(0.002)

    _LWD_EARLY_RECV_STUCK_LOG_INTERVAL_S = 5.0

    def _warn_if_early_recv_stuck(self) -> None:
        """队列挂不上时每 5s 一条告警:排队有提示、名额却全被占住。

        名额被占住 = 那几条已挂 recv 既没被消费也没释放(水位因此不前进、
        闸门不再放行 prefill)。这条日志把"谁占着名额"直接写出来,避免出现
        "prefill 不动了但不知道为什么"。
        """
        with self._early_recv_lock:
            if (
                not self._early_recv_pending
                or len(self._early_recv_futures) < self._early_recv_max_inflight
            ):
                return
            now = time.monotonic()
            if (
                now - self._early_recv_stuck_log_at
                < self._LWD_EARLY_RECV_STUCK_LOG_INTERVAL_S
            ):
                return
            self._early_recv_stuck_log_at = now
            oldest = next(iter(self._early_recv_pending))[1]
            holders = sorted(key[1] for key in self._early_recv_futures)
        logger.warning(
            "[Lwd][pre-recv] queue stuck: oldest queued seqno=%d, %d slot(s) "
            "held by seqno=%s (slots held = those batches posted recv but "
            "were never consumed; downstream dispatch is the pacer)",
            oldest, len(holders), holders,
        )

    def start_early_irecv(self, hint: dict) -> None:
        """收下一条引擎提示;名额没满就立刻挂 recv,满了先排队。

        幂等:同一 (channel, seqno) 的重复提示、或消费点已经处理过的,都不再
        接受 —— 同一通道上两条 irecv 会抢发送端那一条 isend。
        """
        from vllm_ascend.distributed.lwd_comm.pre_recv import hint_fields

        fields = hint_fields(hint)
        if fields is None:
            logger.warning("[Lwd][pre-recv] malformed hint %s, skipped", hint)
            return
        seqno, _num_tokens, _aux_rows = fields
        key = (hint["channel"], seqno)
        with self._early_recv_lock:
            if (
                key in self._early_recv_futures
                or key in self._early_recv_consumed
                or key in self._early_recv_pending
            ):
                return
            self._early_recv_pending[key] = hint
        self._early_recv_pump()

    def _early_recv_pump(self) -> None:
        """把排队的提示按到达序挂成 recv,直到用满在飞名额。

        名额 = "已挂 recv、还没随 SO 被消费"的条数(每条占一块整批接收 buffer),
        所以这里的上限就是显存上限;``get_or_post_early_recv`` 消费一条后调用
        本函数补挂下一条,提示队列因此不会因为上限而丢。
        """
        from vllm_ascend.distributed.lwd_comm.pre_recv import hint_fields

        while True:
            with self._early_recv_lock:
                if (
                    not self._early_recv_pending
                    or len(self._early_recv_futures)
                    >= self._early_recv_max_inflight
                ):
                    return
                key, hint = next(iter(self._early_recv_pending.items()))
                del self._early_recv_pending[key]
                seqno, num_tokens, aux_rows = hint_fields(hint)
                future = get_lwd_comm_service().submit_recv(
                    self._lwd_build_up_recv_request(seqno, num_tokens, aux_rows)
                )
                self._early_recv_futures[key] = future
                inflight = len(self._early_recv_futures)
                queued = len(self._early_recv_pending)
            logger.info(
                "[Lwd][pre-recv] SUBMIT channel=%s seqno=%d num_tokens=%d "
                "aux_rows=%d inflight=%d queued=%d ts=%.3f",
                key[0].value, seqno, num_tokens, aux_rows, inflight, queued,
                time.monotonic(),
            )

    def _report_irecv_completions(self) -> None:
        """把已收完的 recv 以 (channel, seqno) 上报到 done MQ。

        每通道完成序 = seqno 序,所以引擎侧只维护一个最大水位就能精确判就绪。
        已被消费点取走的条目继续轮询:消费意味着前向正在用这份数据,``done()``
        很快转真,水位照样前进;条目"既上报过又被消费过"才移除。
        """
        done_mq = self._irecv_done_mq
        if done_mq is None:
            return
        with self._early_recv_lock:
            pending = [
                (key, future)
                for key, future in self._early_recv_futures.items()
                if key not in self._early_recv_reported
            ]
        for key, future in pending:
            if not future.done():
                continue
            # ① 收完:这条 recv 的数据(主帧 + aux 帧)已完整落地
            logger.info(
                "[Lwd][pre-recv] DONE channel=%s seqno=%d ts=%.3f",
                key[0].value, key[1], time.monotonic(),
            )
            # ② 上报:把 (channel, seqno) 写进 done MQ,引擎每步排空它推进水位
            done_mq.enqueue(key)
            logger.info(
                "[Lwd][pre-recv] REPORT channel=%s seqno=%d ts=%.3f",
                key[0].value, key[1], time.monotonic(),
            )
            with self._early_recv_lock:
                self._early_recv_reported.add(key)
                if key in self._early_recv_consumed:
                    self._early_recv_futures.pop(key, None)
                    self._early_recv_reported.discard(key)
                    self._early_recv_consumed.discard(key)
                    released = True
                else:
                    released = False
            if released:
                self._early_recv_pump()

    def get_or_post_early_recv(
        self, seqno: int, num_tokens: int, aux_rows: int
    ) -> LwdCommFuture | None:
        """取通信线程提前挂好的 UP recv future;没有就现场补挂一条并登记。

        非端点 rank 无跨机通道(数据在注入时刻由 TP 广播取到),返回 None。
        条目不在本处移除:上报线程要继续轮询它推进水位,只有"既上报又消费"
        才移除,所以水位不会因为被消费而断档。

        本处是**名额归还点**(与旧版 `note_scheduled` 同款):SO 一到,这一条
        的 buffer 马上要交给前向,名额立刻转给还在排队的提示(提示还排着队时
        先就地挂上,闸门/前向不可能跑到 recv 前面)。
        """
        if not self._lwd_early_recv_endpoint:
            return None

        key = (LwdChannelType.UP, seqno)
        with self._early_recv_lock:
            self._early_recv_consumed.add(key)
            future = self._early_recv_futures.get(key)
            if future is None:
                # 名额满时该提示还在排队:本批已下发,现在必须挂上
                hint = self._early_recv_pending.pop(key, None)
                if hint is not None:
                    future = get_lwd_comm_service().submit_recv(
                        self._lwd_build_up_recv_request(
                            seqno, num_tokens, aux_rows
                        )
                    )
                    self._early_recv_futures[key] = future
            if future is not None:
                if key in self._early_recv_reported:
                    # 完成已上报:消费即条目终点(上报线程不会再碰它)
                    self._early_recv_futures.pop(key, None)
                    self._early_recv_reported.discard(key)
                    self._early_recv_consumed.discard(key)
            else:
                # 提示彻底没到(或到得比本 SO 还晚):用同一份信息现场补挂,
                # 保证上报链不缺号,水位能继续前进
                logger.info("[Lwd][pre-recv] submit at consume seqno=%d", seqno)
                future = get_lwd_comm_service().submit_recv(
                    self._lwd_build_up_recv_request(seqno, num_tokens, aux_rows)
                )
                self._early_recv_futures[key] = future
            inflight = len(self._early_recv_futures)
            queued = len(self._early_recv_pending)
        # 名额已归还:补挂排队中的提示
        self._early_recv_pump()
        logger.info(
            "[Lwd][pre-recv] stats seqno=%d inflight=%d queued=%d",
            seqno, inflight, queued,
        )
        return future

    # 注:take_lwd_up_embeds(host 阻塞收割)不随移植恢复——本线上 UP
    # 消费已收进 runner(_lwd_inject_remote_embeds:端点过门 + TP 组内
    # 现场广播),worker 层只保留 post 侧;aux(mrope)帧同路径消费。

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: "GrammarOutput") -> ModelRunnerOutput | AsyncModelRunnerOutput:
        _t0 = time.monotonic()
        output = self.model_runner.sample_tokens(grammar_output)
        _t_sample = time.monotonic()
        # rank-replay DOWN:发送 hidden 包(通道异步、流内有序);pinned 视图
        # 挂输出内层(就绪由 get_output 的 wait_stream(主流) 保证),
        # 引擎侧解码发布 meta。
        if self.enable_lwd:
            payload = self.model_runner.take_lwd_pending_down_payload()
            if payload is not None and output is not None:
                hidden, pinned, _event, req_ids, _accepted = payload
                seqno = self._lwd_next_down_seqno()
                logger.info(
                    "[Lwd][cloud-worker] DOWN send seqno=%d numel=%d",
                    seqno, hidden.numel(),
                )
                get_lwd_comm_service().submit_send(
                    LwdCommRequest(
                        channel=LwdChannelType.DOWN,
                        op="send",
                        num_elements=hidden.numel(),
                        tensor=hidden,
                        seqno=seqno,
                    )
                )
                # async 包装器下挂到内层,否则 get_output() 解包丢失
                target = getattr(output, "_model_runner_output", output)
                target.lwd_down_carrier = (
                    pinned, req_ids, hidden.numel(), seqno
                )
                # [Lwd][perf] 云侧每步计时:sample=采样+采集(含秩计算);
                # send=DOWN 提交(快照clone+isend+bridge wait);total=全步
                logger.info(
                    "[Lwd][perf] cloud-step seqno=%d sample=%.2f send=%.2f "
                    "total=%.2fms rows=%d",
                    seqno,
                    (_t_sample - _t0) * 1000,
                    (time.monotonic() - _t_sample) * 1000,
                    (time.monotonic() - _t0) * 1000,
                    hidden.shape[0] if hidden is not None else 0,
                )
        return output

    # ------------------------------------------------------------------ #
    # LWD housekeeping (control-plane glue)                               #
    # ------------------------------------------------------------------ #

    def _lwd_cloud_flush_finished(self, finished_req_ids) -> None:
        """Cloud side (streaming): a finished request's DOWN stream simply
        STOPS (its last step packet already went out with that step).
        No FIN packet -- request termination is signaled by the control
        plane (v2.6).  Here we drop the collector bookkeeping —
        finished_req_ids is the engine's own liveness signal, so this
        cleanup does not depend on the control plane.  Also releases the
        request's prompt-embeds assembly buffer (backstop for abort
        mid-prefill; normal prefill completion frees it in the runner's
        ``_lwd_release_consumed_prompt_embeds``)."""
        embeds_map = self.model_runner.input_batch.req_prompt_embeds
        req_id_to_index = self.model_runner.input_batch.req_id_to_index
        logger.info(
            "[Lwd][cloud-worker] flush finished reqs=%s", list(finished_req_ids)
        )
        for req_id in finished_req_ids:
            if self.model_runner.lwd_hash_state is not None:
                self.model_runner.lwd_hash_state.discard(req_id)
            idx = req_id_to_index.get(req_id)
            if idx is not None:
                embeds_map.pop(idx, None)

    def _register_lwd_prompt_embeds_provider(self) -> None:
        """Wire the draft proposer's first-pass prompt-embeds provider.

        The provider resolves the request's prompt embeds from the
        runner's ``input_batch.req_prompt_embeds`` (already injected by
        ``_lwd_inject_remote_embeds`` during prefill); a missing entry
        (not scheduled / not LWD) returns None and the proposer falls
        back to the token-id path.
        """
        from vllm_ascend.spec_decode.llm_base_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        runner = self.model_runner

        def _lwd_prompt_embeds_provider(req_id: str):
            idx = runner.input_batch.req_id_to_index.get(req_id)
            if idx is None:
                return None
            return runner.input_batch.req_prompt_embeds.get(idx)

        AscendSpecDecodeBaseProposer.set_lwd_prompt_embeds_provider(
            _lwd_prompt_embeds_provider
        )

    def _lwd_next_down_seqno(self) -> int:
        seqno = self._lwd_down_next_seqno
        self._lwd_down_next_seqno += 1
        return seqno
