"""LLM 网关：统一流式调用、工具调用 delta 归一化、熔断与指标。

改造要点：

- **真流式**。原来 `chat_completion_stream` 写好了却没人用，编排层先 `await` 完整结果再逐
  字符 yield，首 token 延迟等于全量延迟。现在编排层只走流式，内部把 `tool_calls` 的分片
  按 `index` 累积成完整调用。
- **单一熔断器**。与 `LLMClient` 共享 `get_llm_circuit_breaker()`，失败计数不再被拆成两份。
- **usage 容错**。`stream_options.include_usage` 并非所有兼容实现都支持，首次被拒后自动
  降级为不带该参数，并回退到本地 token 估算。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]

from app.config import settings
from app.core.logging import get_logger
from app.infrastructure.llm.circuit_breaker import (
    CircuitBreaker,
    LLMUnavailable,
    get_llm_circuit_breaker,
)
from app.infrastructure.observability.metrics import MetricsCollector

logger = get_logger(__name__)


@dataclass
class ToolCallAccumulator:
    """按 `index` 累积一次工具调用的分片。

    流式协议下 `id` / `name` / `arguments` 会分散在多个 chunk 里，`arguments` 更是逐字符
    到达。不累积就会得到半截 JSON。
    """

    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""

    def merge(self, delta: Any) -> None:
        if getattr(delta, "id", None):
            self.id = delta.id
        fn = getattr(delta, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                self.name += fn.name
            if getattr(fn, "arguments", None):
                self.arguments += fn.arguments

    @property
    def complete(self) -> bool:
        return bool(self.id and self.name)


@dataclass
class StreamedTurn:
    """一次 LLM 流式调用的归一化结果。"""

    content: str = ""
    tool_calls: list[ToolCallAccumulator] = field(default_factory=list)
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    @property
    def has_tool_calls(self) -> bool:
        return any(tc.complete for tc in self.tool_calls)


class LLMService:
    """DashScope 兼容接口的流式网关。"""

    def __init__(
        self,
        *,
        metrics: MetricsCollector | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is not installed")
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key or "dummy",
            base_url=settings.llm_api_base,
        )
        self._model = settings.llm_model
        self._breaker = breaker or get_llm_circuit_breaker()
        self._metrics = metrics
        self._supports_stream_usage = True

    @property
    def model(self) -> str:
        return self._model

    @property
    def circuit_open(self) -> bool:
        return self._breaker.is_open

    def _build_kwargs(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        temperature: float,
        stream: bool,
        include_usage: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if stream:
            kwargs["stream"] = True
            if include_usage:
                kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    async def stream_turn(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """产出 `("content", str)` / `("turn", StreamedTurn)` 两类事件。

        `content` 边到边发给前端；`turn` 在流结束时给出完整的工具调用与用量。
        """
        turn = StreamedTurn()
        accumulators: dict[int, ToolCallAccumulator] = {}

        async def _open_stream() -> Any:
            kwargs = self._build_kwargs(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                stream=True,
                include_usage=self._supports_stream_usage,
            )
            try:
                return await self._client.chat.completions.create(**kwargs, timeout=timeout)
            except TypeError:
                # 部分 SDK / 兼容实现不接受 stream_options，降级重试一次。
                self._supports_stream_usage = False
                kwargs.pop("stream_options", None)
                return await self._client.chat.completions.create(**kwargs, timeout=timeout)

        with (self._metrics.time_block("llm_stream") if self._metrics else _NullCtx()):
            try:
                stream = await self._breaker.call(_open_stream)
            except Exception as exc:  # noqa: BLE001
                if self._metrics:
                    self._metrics.record_success("llm", False)
                raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc

            try:
                async for chunk in stream:
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        turn.prompt_tokens = getattr(usage, "prompt_tokens", None)
                        turn.completion_tokens = getattr(usage, "completion_tokens", None)

                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if getattr(choice, "finish_reason", None):
                        turn.finish_reason = choice.finish_reason

                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue

                    piece = getattr(delta, "content", None)
                    if piece:
                        turn.content += piece
                        yield "content", piece

                    for tc_delta in getattr(delta, "tool_calls", None) or []:
                        idx = getattr(tc_delta, "index", 0) or 0
                        acc = accumulators.setdefault(idx, ToolCallAccumulator(index=idx))
                        acc.merge(tc_delta)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if self._metrics:
                    self._metrics.record_success("llm", False)
                # 流中断发生在 breaker.call() 之外，必须显式回报，否则永远不计入熔断。
                await self._breaker.record_external_failure()
                raise LLMUnavailable(f"流式读取中断 {type(exc).__name__}: {exc}") from exc

        turn.tool_calls = [accumulators[i] for i in sorted(accumulators)]
        if self._metrics:
            self._metrics.record_success("llm", True)
            self._metrics.record_tokens(turn.prompt_tokens or 0, turn.completion_tokens or 0)
        yield "turn", turn

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> str:
        """非流式的一次性文本补全，用于摘要、反思等内部阶段。"""

        async def _call() -> Any:
            return await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )

        try:
            resp = await self._breaker.call(_call)
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
        return resp.choices[0].message.content or ""


class _NullCtx:
    def __enter__(self) -> "_NullCtx":
        return self

    def __exit__(self, *exc: object) -> None:
        return None
