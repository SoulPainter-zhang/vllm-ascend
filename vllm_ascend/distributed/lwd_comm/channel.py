# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdChannel: one strict FIFO per physical (direction, peer) wire.

Simplified port of the demo branch's ``lwd_comm/channel.py`` for
the prefill_only duplex data plane (two direction-only channels,
single-tensor payloads).

Mechanics preserved from the demo:
  * send payload snapshot at submit-entry (graph replay / staging buffer
    reuse safety), keepalive released by lazy reap — no background
    thread;
  * per-channel seqno reorder buffer: ops are posted to HCCL only once
    all lower seqnos have been submitted, so the send order and the
    peer's recv-post order always agree (HCCL P2P has no tags);
  * every wire op is bridged onto the channel stream via
    ``handle.wait()`` (non-blocking HCCL semantics) followed by an event
    record; the next op waits for the predecessor's event.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.distributed import lwd_wire
from vllm_ascend.distributed.lwd_comm.future import LwdCommFuture
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType
from vllm_ascend.distributed.lwd_comm.types import LwdCommRequest, LwdChannelType


class LwdChannel:
    """One FIFO of pending requests for a physical channel/peer wire."""

    def __init__(self, channel_type: LwdChannelType, op: str) -> None:
        self.channel_type = channel_type
        self.op = op  # one FIFO carries exactly one direction's seqno stream
        self._pending: deque[LwdCommFuture] = deque()
        self._lock = threading.Lock()
        # The pending-queue lock alone cannot prevent two host threads
        # from interleaving HCCL calls of separate requests.
        self._submission_lock = threading.Lock()
        self._next_seqno: int | None = None
        self._held: dict[int, tuple[LwdCommRequest, LwdCommFuture]] = {}
        # Abort holes: seqnos the peer will never submit (aborted
        # requests).  Skipped when advancing so one abort cannot wedge
        # every later request on this channel.
        self._skipped: set[int] = set()

    # ------------------------------------------------------------------ #
    # Abort holes                                                         #
    # ------------------------------------------------------------------ #

    def skip_seqno(self, seqno: int) -> None:
        """Mark a seqno as never-to-arrive (aborted request) and advance
        past it.  Both peers must skip the same seqno (driven by the
        control-plane abort on both sides) — skipping a seqno the peer
        later submits is a pairing error and raises at submission."""
        with self._submission_lock:
            if self._next_seqno is None:
                self._next_seqno = 0
            if seqno < self._next_seqno:
                return  # already passed
            # A held (out-of-order) entry for an aborted seqno must never
            # be executed: discard it and fail its deferred future so any
            # waiter fails promptly instead of waiting out the timeout.
            held = self._held.pop(seqno, None)
            if held is not None:
                held[1]._finalize(
                    error=RuntimeError(
                        f"seqno {seqno} aborted on channel "
                        f"{self.channel_type.value}"
                    )
                )
            self._skipped.add(seqno)
            self._advance_past_skipped()

    def _advance_past_skipped(self) -> None:
        """Caller holds the submission lock."""
        while self._next_seqno in self._skipped:
            self._skipped.discard(self._next_seqno)
            self._next_seqno += 1
            while self._next_seqno in self._held:
                held_request, held_future = self._held.pop(self._next_seqno)
                self._execute_next(held_request, into=held_future)
                self._next_seqno += 1

    # ------------------------------------------------------------------ #
    # Submission                                                          #
    # ------------------------------------------------------------------ #

    def submit(self, request: LwdCommRequest) -> LwdCommFuture:
        """Execute the wire op and enqueue the future."""
        if request.op != self.op:
            raise RuntimeError(
                f"op mismatch on {self.channel_type.value} FIFO: "
                f"channel carries {self.op!r}, got {request.op!r}"
            )
        finalized = self._reap()
        self._finalize_many(finalized)
        if request.op == "send":
            request = self._snapshot_send(request)
        with self._submission_lock:
            if request.seqno is not None:
                return self._submit_sequenced(request)
            return self._execute_next(request)

    @staticmethod
    def _snapshot_send(request: LwdCommRequest) -> LwdCommRequest:
        """Give the comm layer ownership of the send payload."""
        assert request.tensor is not None, "send requires tensor"
        owned = request.tensor.detach().clone()
        aux_owned = (
            request.aux_tensor.detach().clone()
            if request.aux_tensor is not None
            else None
        )
        return replace(request, tensor=owned, aux_tensor=aux_owned)

    # ------------------------------------------------------------------ #
    # Sequenced submission (reorder buffer)                               #
    # ------------------------------------------------------------------ #

    def _submit_sequenced(self, request: LwdCommRequest) -> LwdCommFuture:
        """Post in per-channel seqno order.  Caller holds the submission
        lock."""
        seqno = request.seqno
        assert seqno is not None
        if self._next_seqno is None:
            # Contract: per-channel seqno counters start at 0.
            self._next_seqno = 0
        self._advance_past_skipped()
        if seqno in self._skipped or seqno < self._next_seqno:
            # A skipped (aborted) seqno can still arrive when the peer's
            # abort raced its send — do NOT crash the process; complete
            # the request with an error result so the caller's readiness
            # gate fails this request promptly.
            logger.warning(
                "[lwd-comm] dropping submission of skipped/duplicate "
                "seqno %d on channel %s %s (next expected %d)",
                seqno, self.channel_type.value, self.op, self._next_seqno,
            )
            future = LwdCommFuture(request, [], None)
            future._finalize(
                error=RuntimeError(
                    f"seqno {seqno} was skipped (aborted) on channel "
                    f"{self.channel_type.value}"
                )
            )
            return future
        if seqno > self._next_seqno:
            if seqno in self._held:
                # Duplicate submission of an already-held seqno: return
                # the existing deferred future instead of overwriting it
                # (its waiter would otherwise spin until timeout).
                return self._held[seqno][1]
            future = LwdCommFuture.deferred(request)
            self._held[seqno] = (request, future)
            logger.debug(
                "[lwd-comm] held out-of-order %s %s seqno=%d (next=%d)",
                self.channel_type.value, request.op, seqno, self._next_seqno,
            )
            return future
        future = self._execute_next(request)
        self._next_seqno += 1
        while self._next_seqno in self._held:
            held_request, held_future = self._held.pop(self._next_seqno)
            self._execute_next(held_request, into=held_future)
            self._next_seqno += 1
        return future

    def _execute_next(
        self,
        request: LwdCommRequest,
        into: LwdCommFuture | None = None,
    ) -> LwdCommFuture:
        """Execute one wire op as the channel's new tail."""
        with self._lock:
            predecessor = self._pending[-1] if self._pending else None
        future = self._execute(request, predecessor, into=into)
        with self._lock:
            self._pending.append(future)
        return future

    # ------------------------------------------------------------------ #
    # Reaping (head-of-line query)                                        #
    # ------------------------------------------------------------------ #

    def reap(self) -> list[LwdCommFuture]:
        """Pop and finalize every completed request from the head.

        FIFO completion is in-order: if the head is not done, nothing
        behind it can be either, so one ``event.query()`` per call per
        channel suffices.
        """
        finalized = self._reap()
        self._finalize_many(finalized)
        return finalized

    def shutdown(self, timeout: float | None = None) -> list[LwdCommFuture]:
        """Wait for pending operations before releasing owned buffers."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._submission_lock:
            if self._held:
                logger.warning(
                    "[lwd-comm] shutdown on channel %s with %d held "
                    "(never-posted) request(s); lowest missing seqno=%s",
                    self.channel_type.value, len(self._held), self._next_seqno,
                )
                # Fail the deferred futures before dropping them: their
                # waiters must not hang until the readiness-gate timeout.
                for _, held_future in self._held.values():
                    held_future._finalize(
                        error=RuntimeError(
                            f"channel {self.channel_type.value} shut down "
                            "with the request still held (never posted)"
                        )
                    )
                self._held.clear()
            with self._lock:
                pending = list(self._pending)
            for future in pending:
                remaining = (
                    None
                    if deadline is None
                    else max(0.0, deadline - time.monotonic())
                )
                future.wait(timeout=remaining)
            finalized = self._reap()
            self._finalize_many(finalized)
            return finalized

    def _reap(self) -> list[LwdCommFuture]:
        done: list[LwdCommFuture] = []
        with self._lock:
            while self._pending and self._pending[0].done():
                done.append(self._pending.popleft())
        return done

    @staticmethod
    def _finalize_many(futures: list[LwdCommFuture]) -> None:
        for future in futures:
            future._finalize()

    # ------------------------------------------------------------------ #
    # Wire execution                                                      #
    # ------------------------------------------------------------------ #

    def _execute(
        self,
        req: LwdCommRequest,
        predecessor: LwdCommFuture | None,
        into: LwdCommFuture | None = None,
    ) -> LwdCommFuture:
        """Issue the wire op, bridge it onto the channel stream, record
        the completion event."""
        tensor: torch.Tensor | None = None
        aux_tensor: torch.Tensor | None = None
        keepalive: Any = None
        stream = self._stream()
        # Capture the producer stream BEFORE entering the channel-stream
        # context: inside the `with` block torch.npu.current_stream() IS
        # the channel stream, so waiting on it there would be a no-op
        # self-wait and the send could read a half-written snapshot.
        producer_stream = torch.npu.current_stream()
        # The wire op MUST be issued on the channel stream: seqno
        # ordering only holds if every op on this FIFO lands on the same
        # device stream (HCCL P2P has no tags; matching order = device
        # arrival order, not host submission order).
        with torch.npu.stream(stream):
            self._order_after(predecessor)
            if req.op == "send":
                assert req.tensor is not None, "send requires tensor"
                # The snapshot was produced on the producer stream; order
                # the channel stream after it.
                stream.wait_stream(producer_stream)
                handles = self._wire_send(req)
                keepalive = (req.tensor, req.aux_tensor)
            else:
                tensor, aux_tensor, handles = self._wire_recv(req)
            done_event = self._bridge_and_record(handles)
        future_request = replace(req, tensor=None) if req.op == "send" else req
        if into is None:
            return LwdCommFuture(
                request=future_request,
                handles=handles,
                done_event=done_event,
                tensor=tensor,
                keepalive=keepalive,
                aux_tensor=aux_tensor,
            )
        into._bind(
            handles=handles,
            done_event=done_event,
            tensor=tensor,
            keepalive=keepalive,
            aux_tensor=aux_tensor,
        )
        return into

    def _stream(self):
        return lwd_wire.get_lwd_channel_stream(self.channel_type)

    def _order_after(self, predecessor: LwdCommFuture | None) -> None:
        if predecessor is None:
            return
        event = predecessor._done_event
        if event is not None:
            self._stream().wait_event(event)

    def _wire_send(self, req: LwdCommRequest) -> list[Any]:
        group = lwd_wire.get_lwd_channel_device_group(self.channel_type)
        peer = req.src_dst
        if peer is None:
            peer = lwd_wire.get_lwd_channel_peer(self.channel_type)
        tensor = req.tensor
        assert tensor is not None
        logger.debug(
            "[lwd-comm] SEND post channel=%s my_rank=%d peer=%s group_ranks=%s "
            "shape=%s dtype=%s op=%s",
            self.channel_type, dist.get_rank(), peer,
            dist.get_process_group_ranks(group),
            list(tensor.shape), tensor.dtype,
            "world_broadcast(src=0)" if self.channel_type == LwdChannelType.UP
            else "isend",
        )
        # UP(边→云)两层通信:第一段边→云端点 P2P(只等端点 join,
        # 不再被 8 卡 join 拖住);第二段端点在云 TP 组内 broadcast
        # (见 _wire_recv)。DOWN(云 leader→边)保持点对点。
        # aux 帧(mrope positions)为紧跟主帧的第二条 P2P op——同
        # seqno 管两帧,skip/drain 语义不变;第二段(云 TP 组内广播)
        # 由云 runner 在消费时刻对 aux 同样补发(见 lwd_cloud_model_runner)。
        if self.channel_type == LwdChannelType.UP:
            handles = [dist.isend(tensor.contiguous(), dst=peer, group=group)]
            if req.aux_tensor is not None:
                handles.append(
                    dist.isend(
                        req.aux_tensor.contiguous(), dst=peer, group=group
                    )
                )
            return handles
        if req.aux_tensor is not None:
            raise RuntimeError(
                "aux payload is only supported on the UP channel"
            )
        return [dist.isend(tensor.contiguous(), dst=peer, group=group)]

    def _wire_recv(self, req: LwdCommRequest):
        group = lwd_wire.get_lwd_channel_device_group(self.channel_type)
        peer = req.src_dst
        if peer is None:
            peer = lwd_wire.get_lwd_channel_peer(self.channel_type)
        logger.debug(
            "[lwd-comm] RECV post channel=%s my_rank=%d src=%s group_ranks=%s "
            "num_elements=%d aux_elements=%d op=%s",
            self.channel_type, dist.get_rank(), peer,
            dist.get_process_group_ranks(group), req.num_elements,
            req.aux_num_elements,
            "world_broadcast(src=0)" if self.channel_type == LwdChannelType.UP
            else "irecv",
        )
        # Exact-size buffer: HCCL P2P requires matching numel on both
        # ends; the size is learned from the control-plane notification.
        buffer = torch.empty(
            req.num_elements, dtype=torch.bfloat16, device="npu"
        )
        aux_buffer = None
        if self.channel_type == LwdChannelType.UP and req.aux_num_elements > 0:
            # aux 帧尺寸 = num_tokens*3(mrope positions);dtype 取
            # 请求的 aux_dtype(缺省 int64),与发送端严格同 numel。
            aux_buffer = torch.empty(
                req.aux_num_elements,
                dtype=req.aux_dtype or torch.int64,
                device="npu",
            )
            return buffer, aux_buffer, [
                dist.irecv(buffer, src=peer, group=group),
                dist.irecv(aux_buffer, src=peer, group=group),
            ]
        if req.aux_num_elements > 0:
            raise RuntimeError(
                "aux payload is only supported on the UP channel"
            )
        return buffer, None, [dist.irecv(buffer, src=peer, group=group)]

    def _bridge_and_record(self, handles: list[Any]):
        """Bridge HCCL completion onto the channel stream and record an
        event right behind it (CPU returns immediately).  Caller already
        holds the channel stream as current, so bridging binds to the
        right stream."""
        if not handles:
            return None
        _t = time.monotonic()
        for handle in handles:
            handle.wait()
        # [Lwd][perf] 临时探针:测 handle.wait() 是否真如注释所述"CPU 立即
        # 返回"——dur 大 = 发送/接收 posting 阻塞主机(锁定步判定依据)
        logger.info(
            "[Lwd][perf] bridge-wait op=%s ch=%s dur=%.2fms",
            self.op, self.channel_type, (time.monotonic() - _t) * 1000,
        )
        logger.debug(
            "[lwd-comm] DONE channel=%s my_rank=%d handles=%d",
            self.channel_type, dist.get_rank(), len(handles),
        )
        event = torch.npu.Event()
        event.record()
        return event
