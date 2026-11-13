"""Three-state circuit breaker for LLM and external dependency calls.

改造前项目里有两份熔断器实现（`app/core/circuit_breaker.py` 与本文件），`LLMService` 用一份、
其它路径用另一份，失败计数互不可见 —— 等于熔断阈值被悄悄放大一倍。现在只保留本实现，
并通过 `get_llm_circuit_breaker()` 让所有走同一个 LLM 供应商的调用共享同一个计数器。
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class LLMUnavailable(RuntimeError):
    """LLM 供应商暂时不可用（熔断打开或调用失败）。编排层据此走降级路径而不是抛 500。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """CLOSED: normal. Failures reach threshold -> OPEN. After cooldown -> HALF_OPEN probe."""

    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    half_open_max_calls: int = 1
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _half_open_inflight: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            await self._transition_on_cooldown()
            if self._state is CircuitState.OPEN:
                raise RuntimeError("circuit breaker is OPEN")
            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_inflight >= self.half_open_max_calls:
                    raise RuntimeError("circuit breaker HALF_OPEN probe already running")
                self._half_open_inflight += 1

        try:
            result = await fn()
        except Exception:
            async with self._lock:
                await self._on_failure()
            raise

        async with self._lock:
            await self._on_success()
        return result

    async def _transition_on_cooldown(self) -> None:
        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return
        if time.monotonic() - self._opened_at >= self.cooldown_seconds:
            self._state = CircuitState.HALF_OPEN
            self._half_open_inflight = 0

    async def _on_failure(self) -> None:
        self._failures += 1
        if self._state is CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            self._half_open_inflight = 0
            self._failures = self.failure_threshold
            return
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    async def _on_success(self) -> None:
        self._failures = 0
        if self._state is CircuitState.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
        self._state = CircuitState.CLOSED

    async def record_external_failure(self) -> None:
        """记录一次在 `call()` 之外发生的失败。

        流式调用的失败通常出现在**消费流**的阶段，而 `call()` 只覆盖到"建立流"那一步。
        没有这个入口，流中断就永远不会计入熔断，阈值形同虚设。
        """
        async with self._lock:
            await self._on_failure()

    async def reset(self) -> None:
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = None
            self._half_open_inflight = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state is CircuitState.OPEN


_llm_breaker: CircuitBreaker | None = None


def get_llm_circuit_breaker() -> CircuitBreaker:
    """所有 LLM 调用共享的熔断器（单进程内单例）。"""
    global _llm_breaker
    if _llm_breaker is None:
        from app.config import settings

        _llm_breaker = CircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            cooldown_seconds=float(settings.circuit_breaker_recovery_timeout),
            half_open_max_calls=settings.circuit_breaker_half_open_max_calls,
        )
    return _llm_breaker
