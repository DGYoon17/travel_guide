"""DashScope-compatible async chat client for chat completions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]

from app.infrastructure.llm.circuit_breaker import LLMUnavailable, get_llm_circuit_breaker

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatMessage:
    role: Role
    content: str
    name: str | None = None


@dataclass
class ChatCompletionResult:
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    raw: dict[str, Any]


class LLMClient:
    """DashScope-compatible async chat client using the OpenAI-compatible API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        default_model: str = "qwen-max",
        timeout_s: float = 120.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is not installed")
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            default_headers=default_headers,
        )
        self._default_model = default_model
        # 与 LLMService 共享同一个熔断器：同一个供应商的失败必须计入同一个计数器。
        self._breaker = get_llm_circuit_breaker()

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        mname = model or self._default_model
        payload: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": mname,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        async def _call() -> Any:
            return await self._client.chat.completions.create(**kwargs)

        try:
            resp = await self._breaker.call(_call)
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = resp.usage
        return ChatCompletionResult(
            content=content,
            model=resp.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            raw=resp.model_dump(),
        )
