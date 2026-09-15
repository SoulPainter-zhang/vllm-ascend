# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdCommFuture: completion handle for one duplex comm request.

Ported (simplified) from the demo branch's ``lwd_comm/future.py``:
pure-CPU ``done()`` via ``event.query()``, blocking ``wait()`` for the
debug/timeout path, in-place ``_bind`` for deferred (out-of-order
sequenced) futures, and exactly-once ``_finalize`` releasing the send
keepalive.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import torch

from vllm_ascend.distributed.lwd_comm.types import (
    LwdCommRequest,
    LwdCommResult,
    LwdCommStatus,
)


class LwdCommFuture:
    """Tracks one posted (or held) wire op.

    Completion means the cross-node P2P op has finished on the device,
    witnessed by ``done_event`` recorded on the channel stream right
    behind the bridged HCCL op.
    """

    def __init__(
        self,
        request: LwdCommRequest,
        handles: list[Any],
        done_event: torch.npu.Event | None,
        tensor: torch.Tensor | None = None,
        keepalive: Any = None,
        aux_tensor: torch.Tensor | None = None,
    ) -> None:
        self._request = request
        self._handles = handles
        self._done_event = done_event
        self._tensor = tensor
        self._aux_tensor = aux_tensor
        self._keepalive = keepalive
        self._status = LwdCommStatus.PENDING
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._done_cond = threading.Condition(self._lock)
        self._finalized = False

    # ------------------------------------------------------------------ #
    # Deferred (out-of-order sequenced) support                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def deferred(cls, request: LwdCommRequest) -> "LwdCommFuture":
        """A future for a held request; bound in place when its seqno
        turn arrives."""
        return cls(request, [], None)

    def _bind(
        self,
        *,
        handles: list[Any],
        done_event: "torch.npu.Event | None",
        tensor: torch.Tensor | None = None,
        keepalive: Any = None,
        aux_tensor: torch.Tensor | None = None,
    ) -> None:
        with self._done_cond:
            self._handles = handles
            self._done_event = done_event
            self._tensor = tensor
            self._aux_tensor = aux_tensor
            self._keepalive = keepalive
            self._done_cond.notify_all()

    # ------------------------------------------------------------------ #
    # Status                                                              #
    # ------------------------------------------------------------------ #

    @property
    def request(self) -> LwdCommRequest:
        return self._request

    def done(self) -> bool:
        """Pure CPU query; never blocks, never touches the device."""
        with self._lock:
            if self._status is not LwdCommStatus.PENDING:
                return True
            event = self._done_event
        if event is None:
            return False
        try:
            if event.query():
                with self._lock:
                    if self._status is LwdCommStatus.PENDING:
                        self._status = LwdCommStatus.OK
                        self._done_cond.notify_all()
                return True
        except Exception as exc:  # device error surfaces as ERROR
            with self._lock:
                if self._status is LwdCommStatus.PENDING:
                    self._status = LwdCommStatus.ERROR
                    self._error = exc
                    self._done_cond.notify_all()
            return True
        return False

    def wait(self, timeout: float | None = None) -> LwdCommResult:
        """Block until completion (readiness-gate path)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._done_cond:
            while self._status is LwdCommStatus.PENDING:
                if self._done_event is None:
                    # Deferred future not yet bound; wait for _bind.
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError(
                            f"lwd-comm wait timed out (unbound) on "
                            f"{self._request.channel.value} seqno="
                            f"{self._request.seqno}"
                        )
                    self._done_cond.wait(timeout=remaining or 0.05)
                    continue
                if self._done_event.query():
                    self._status = LwdCommStatus.OK
                    break
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f"lwd-comm wait timed out on "
                        f"{self._request.channel.value} seqno="
                        f"{self._request.seqno}"
                    )
                self._done_cond.wait(timeout=min(0.05, remaining or 0.05))
            if self._status is LwdCommStatus.ERROR:
                raise RuntimeError(
                    f"lwd-comm op failed on {self._request.channel.value} "
                    f"seqno={self._request.seqno}"
                ) from self._error
        return LwdCommResult(
            status=self._status, tensor=self._tensor,
            aux_tensor=self._aux_tensor,
        )

    def result(self) -> LwdCommResult:
        with self._lock:
            status, tensor, aux_tensor, error = (
                self._status, self._tensor, self._aux_tensor, self._error,
            )
        if status is LwdCommStatus.ERROR:
            raise RuntimeError(
                f"lwd-comm op failed on {self._request.channel.value} "
                f"seqno={self._request.seqno}"
            ) from error
        return LwdCommResult(status=status, tensor=tensor, aux_tensor=aux_tensor)

    # ------------------------------------------------------------------ #
    # Device-side ordering for consumers                                  #
    # ------------------------------------------------------------------ #

    def wait_for_comm(self) -> None:
        """Order the *current* stream after the channel-stream completion
        event (pure device-side; CPU returns immediately)."""
        event = self._done_event
        if event is not None:
            torch.npu.current_stream().wait_event(event)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def _finalize(self, error: BaseException | None = None) -> None:
        """Exactly-once terminal bookkeeping: release the send keepalive.
        With ``error`` the future completes in ERROR state immediately
        (abort/skip path) so waiters fail promptly instead of timing out."""
        with self._done_cond:
            if self._finalized:
                return
            self._finalized = True
            self._keepalive = None
            self._handles = []
            if error is not None:
                self._status = LwdCommStatus.ERROR
                self._error = error
                self._done_cond.notify_all()
                return
            if self._status is LwdCommStatus.PENDING:
                self._status = (
                    LwdCommStatus.OK
                    if (self._done_event is not None and self._done_event.query())
                    else self._status
                )
