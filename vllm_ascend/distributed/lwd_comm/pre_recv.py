# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD 提前收(early recv):引擎⇄worker 旁路链路 + 提示/水位协议。

数据路径只有一条(与既有提前收同款,不新增旁路)::

    云引擎 IO 线程                        云端点 worker 通信线程
    ───────────────                       ──────────────────────
    RangeNotify 到达(边侧预告)
      └─ post_irecv_hint ──hint MQ(下行)──►  start_early_irecv
                                              └─ submit_recv(UP, seqno, numel, aux)
                                                    │  irecv 已提前挂出
                                              future.done() 轮询
                                                    │
    record_irecv_completions ◄──done MQ(上行)── _report_irecv_completions
      └─ 水位 → is_irecv_complete(UP, seqno)
           └─ PD_mix 调度闸门:收完才下发这个 prefill 批

两条 MQ 都是旁路,都不能借 ``rpc_broadcast_mq``:那条队列由
``worker_busy_loop`` 单线程消费,它在 ``execute_model`` 里为一个 prefill 批
阻塞几百毫秒 —— 提示排在那条队列上,就等不到"提前"挂 recv 的机会。

* hint MQ:引擎建 writer(worker 创建之前把 handle 写进环境变量),云端点
  rank 按 handle 挂 reader;
* done MQ:云端点 rank 建 writer,handle 随 READY 握手回传,引擎挂 reader。

水位语义:``is_irecv_complete(channel, seqno)`` == ``seqno <= watermark``。
每通道完成序 = seqno 序(``channel.py`` 的 FIFO:队首没完成,后面不可能完成),
"已完成集合"恒为前缀,故一个最大水位即可精确表示。水位只是**就绪判据**,
不参与任务优先级。

本分支(chunked prefill + mm/hash)的提示字段:``num_tokens`` = 该批(本 chunk
批)flat token 数,``aux_rows`` = 批内**带 mrope 的请求**的 token 行数之和 ——
UP 主帧 numel 由 hash 载荷公式在 **worker 侧**按本机配置算(见
``lwd_cloud_worker._lwd_build_up_recv_request``),提示只带"事实"(token 数、
mrope 行数),不带尺寸公式,避免两侧各算一遍算歪;aux 帧 numel = aux_rows*3
(int64,[n,3],与主帧同 seqno)。

非端点 rank(云端其余 TP rank)没有跨机通道:它们的数据在注入时刻由 TP 组内
广播取到(见 ``lwd_cloud_model_runner._lwd_inject_remote_embeds``),既不挂
recv 也不上报 —— 本模块的提示/上报只走端点 rank 这一条线。
"""

from __future__ import annotations

import base64
import os
import pickle
import threading
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_ascend.distributed.lwd_comm.types import LwdChannelType

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

# 日志名必须落在 **vllm.** 命名空间内:vLLM 只给名为 "vllm" 的 logger 挂
# handler(DEFAULT_LOGGING_CONFIG,propagate=False),子 logger 靠向上传播拿到
# 那个 handler。vllm_ascend.* 的 logger 传不到 "vllm"(根 logger 无 handler)。
# 本模块自身的日志只留异常诊断(正常路径的 HINT/水位/闸门/worker 四段都在
# lwd_cloud_engine.py / lwd_cloud_mixed_scheduler.py / lwd_cloud_worker.py 里打)。
logger = init_logger("vllm.lwd.pre_recv")

#: 下行提示 MQ 的 handle 环境变量。执行器在创建 worker 之前导出,worker 在
#: ``init_device`` 里重建 reader —— 两侧只认这一处常量名。
LWD_RECV_HINT_MQ_ENV = "VLLM_ASCEND_LWD_RECV_HINT_HANDLE"

# 旁路 MQ 规格:每槽 1KB、共 64 槽。一批一条提示/上报,边侧 pending 闸门
# (``LwdEdgeScheduler._LWD_MAX_PENDING_CHUNKS = 3``)把未消费 chunk 数压在
# 个位数,而提示是正确性依赖(丢一条 = 该 recv 不提前挂、水位不前进、闸门后的
# 批永不下发),所以留足余量、不做流控。
_SIDEBAND_MAX_CHUNK_BYTES = 1024
_SIDEBAND_MAX_CHUNKS = 64


# ---------------------------------------------------------------------- #
# 提示(hint):调度侧生产,worker 通信线程消费                              #
# ---------------------------------------------------------------------- #
def make_recv_hint(seqno: int, num_tokens: int, aux_rows: int = 0) -> dict:
    """构造一条 UP 提前收提示。

    * ``num_tokens``:本批 flat token 数(主帧尺寸由 worker 按 hash 载荷公式推)
    * ``aux_rows``:本批带 mrope 的 token 行数(0 = 纯文本批,零 aux 流量)
    """
    return {
        "channel": LwdChannelType.UP,
        "seqno": int(seqno),
        "num_tokens": int(num_tokens),
        "aux_rows": int(aux_rows),
    }


def hint_fields(hint: dict) -> tuple[int, int, int] | None:
    """提示自检:返回 (seqno, num_tokens, aux_rows);坏提示返回 None。"""
    try:
        seqno = int(hint["seqno"])
        num_tokens = int(hint["num_tokens"])
        aux_rows = int(hint.get("aux_rows", 0))
    except (KeyError, TypeError, ValueError):
        return None
    if seqno < 0 or num_tokens <= 0 or aux_rows < 0 or aux_rows > num_tokens:
        return None
    return seqno, num_tokens, aux_rows


# ---------------------------------------------------------------------- #
# 旁路 MQ 生命周期                                                        #
# ---------------------------------------------------------------------- #
def _new_sideband_mq() -> "MessageQueue":
    """1 读 1 写的旁路 MQ(提示/上报同一规格,且都只有端点 rank 参与)。"""
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue(
        1, 1,
        max_chunk_bytes=_SIDEBAND_MAX_CHUNK_BYTES,
        max_chunks=_SIDEBAND_MAX_CHUNKS,
    )


def create_hint_mq() -> "MessageQueue":
    """引擎侧:建 hint MQ(writer),并把 handle 导出到环境变量。

    必须在 worker 被创建之前调用(worker 启动时按环境变量重建 reader);
    fork/spawn 两种启动方式都靠环境变量继承。
    """
    mq = _new_sideband_mq()
    os.environ[LWD_RECV_HINT_MQ_ENV] = base64.b64encode(
        pickle.dumps(mq.export_handle())
    ).decode("ascii")
    return mq


def attach_hint_mq() -> "MessageQueue | None":
    """worker 侧:按环境变量里的 handle 重建 hint MQ reader。

    没有 handle(该角色不接提示)返回 None:通信线程照旧跑上报,于是"提示
    没来"只影响提前挂 recv 的时机,不影响水位前进。
    """
    raw = os.environ.get(LWD_RECV_HINT_MQ_ENV)
    if raw is None:
        return None
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue.create_from_handle(pickle.loads(base64.b64decode(raw)), 0)


def create_report_mq() -> "MessageQueue":
    """worker 侧:建 done MQ(writer),handle 随后随 READY 握手回传引擎。"""
    return _new_sideband_mq()


# ---------------------------------------------------------------------- #
# 调度进程侧:提示下发 + 完成水位                                          #
# ---------------------------------------------------------------------- #
_hint_sender: Callable[[dict], None] | None = None
_watermarks: dict[LwdChannelType, int] = {}
_lock = threading.Lock()


def register_hint_sender(sender: Callable[[dict], None]) -> None:
    """装上下发提示用的传输(每个引擎进程装一次)。

    传进来的闭包必须**可靠投递**(把提示 enqueue 进 hint MQ):闸门开启时提示
    是正确性依赖 —— 丢一条,那条 recv 就不提前挂,水位不前进,被闸门拦下的批
    永不下发。
    """
    global _hint_sender
    _hint_sender = sender


def post_irecv_hint(hint: dict) -> None:
    """把一条 recv 提示发给本地 worker 的通信线程。

    成功路径不打日志(由调用方 ``lwd_cloud_engine`` 打 ``HINT ...``):本模块只留
    "本该不发生"的诊断,而且它必须能从日志里看到。
    """
    if _hint_sender is None:
        # 没有传输 = 本进程不参与提前收(如相位调度逃生通道):丢一条提示不影响
        # 正确性 —— 该分支没有闸门在等它,recv 由消费点自己挂。
        logger.warning(
            "[Lwd][pre-recv] HINT DROPPED: no sender registered seqno=%s",
            hint.get("seqno"),
        )
        return
    _hint_sender(hint)


def record_irecv_completions(
    items: Iterable[tuple[LwdChannelType, int]],
) -> list[tuple[LwdChannelType, int]]:
    """把 worker 上报的完成项并进每通道水位(引擎每步排空 done MQ 后调用)。

    返回**本次真正推进水位**的项,由调用方(``lwd_cloud_engine``)打日志:
    水位是否在动是定位"卡在哪一段"的第一手证据,日志放在已证明能打出来的
    模块里。
    """
    advanced: list[tuple[LwdChannelType, int]] = []
    with _lock:
        for channel, seqno in items:
            if seqno > _watermarks.get(channel, -1):
                _watermarks[channel] = seqno
                advanced.append((channel, seqno))
    return advanced


def watermark(channel: LwdChannelType) -> int:
    """当前水位:该通道上"已确认收完"的最大 seqno(-1 = 一条都没有)。"""
    with _lock:
        return _watermarks.get(channel, -1)


def is_irecv_complete(channel: LwdChannelType, seqno: int) -> bool:
    """该通道 seqno 这条 recv 收完了没 —— 只作就绪判据,不打日志。

    "在等谁"由闸门那一行给出(``lwd_cloud_mixed_scheduler`` 每步都打
    ``gate seqno=.. watermark=.. ready=..``),所以这里保持纯判据。
    """
    return seqno <= watermark(channel)
