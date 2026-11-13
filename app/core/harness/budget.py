"""单次请求的运行预算：轮次、工具调用数、token、墙钟截止。

改造前唯一的边界是 `max_react_iterations`，而且触顶后返回一句硬编码道歉并**丢弃已经拿到
的全部工具输出**——用户等了 5 轮工具调用，最后什么都没拿到。

这里把边界补齐，并且触顶不再等于"白跑"：`exhausted_reason()` 只负责停止继续调用模型，
由编排层拿已有的工具结果做 best-effort 收尾。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings


class BudgetLimit(str, Enum):
    ITERATIONS = "max_iterations"
    TOOL_CALLS = "max_tool_calls"
    TOKENS = "max_total_tokens"
    DEADLINE = "deadline"


_LIMIT_MESSAGES: dict[BudgetLimit, str] = {
    BudgetLimit.ITERATIONS: "已达到最大推理轮次",
    BudgetLimit.TOOL_CALLS: "已达到单次请求的工具调用上限",
    BudgetLimit.TOKENS: "已达到单次请求的 token 预算",
    BudgetLimit.DEADLINE: "已达到单次请求的时间上限",
}


@dataclass
class RunBudget:
    """一次 `/chat` 请求的资源账本。"""

    max_iterations: int = field(default_factory=lambda: settings.harness_max_iterations)
    max_tool_calls: int = field(default_factory=lambda: settings.harness_max_tool_calls)
    max_total_tokens: int = field(default_factory=lambda: settings.harness_max_total_tokens)
    deadline_seconds: float = field(default_factory=lambda: settings.harness_deadline_seconds)

    iterations: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _started_at: float = field(default_factory=time.monotonic, init=False)

    # ------------------------------------------------------------------ 计量
    def charge_iteration(self) -> None:
        self.iterations += 1

    def charge_tool_calls(self, count: int = 1) -> None:
        self.tool_calls += count

    def charge_tokens(self, prompt: int | None, completion: int | None) -> None:
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_seconds - self.elapsed_seconds)

    # ------------------------------------------------------------------ 判定
    def exhausted(self) -> BudgetLimit | None:
        """返回首个触顶的限制项；None 表示还可以继续。"""
        if self.iterations >= self.max_iterations:
            return BudgetLimit.ITERATIONS
        if self.tool_calls >= self.max_tool_calls:
            return BudgetLimit.TOOL_CALLS
        if self.total_tokens >= self.max_total_tokens:
            return BudgetLimit.TOKENS
        if self.remaining_seconds() <= 0:
            return BudgetLimit.DEADLINE
        return None

    def describe(self, limit: BudgetLimit) -> str:
        return _LIMIT_MESSAGES.get(limit, limit.value)

    def llm_timeout(self, *, floor: float = 5.0) -> float:
        """给单次 LLM 调用的超时：不超过剩余预算，避免整体 deadline 被单次调用吃掉。"""
        return max(floor, min(self.remaining_seconds(), self.deadline_seconds))

    def snapshot(self) -> dict[str, float | int]:
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_ms": round(self.elapsed_seconds * 1000, 1),
        }
