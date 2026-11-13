"""L1 短期会话记忆：滑动窗口 + token 预算。

改造前有三个会导致线上错误的问题，这里逐个从结构上消掉：

1. **摘要被裁掉**：编排层先做摘要再按窗口截断，把开头那条 `[历史摘要]` 直接裁走。
   现在摘要存在独立的 `_pinned` 区，`_pinned` 不参与裁剪，物理上不可能被截断。
2. **tool_calls 被裁断**：原来按单条消息 `popleft`，会把 assistant 的 `tool_calls` 和它
   对应的 `role=tool` 结果拆开，OpenAI 兼容接口收到孤立 `tool_call_id` 直接返回 400。
   现在裁剪单位是"消息组"（assistant+tool_calls 与其全部 tool 结果绑定），不可拆。
3. **token 计数 O(n²)**：`total_tokens()` 每轮重算全量。现在每条消息缓存 token 数，
   组级汇总，增删时增量维护。
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]

Role = Literal["system", "user", "assistant", "tool"]

#: 每条消息除正文外的固定协议开销（role / 分隔符等）的粗略估算。
_PER_MESSAGE_OVERHEAD_TOKENS = 4


class ToolCallRef(BaseModel):
    """assistant 消息里的一次工具调用。"""

    id: str
    name: str
    arguments: str = "{}"

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments or "{}"},
        }


class StoredMessage(BaseModel):
    """可持久化的会话消息。字段与 OpenAI 兼容接口一一对应。"""

    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCallRef] = Field(default_factory=list)
    turn: int = 0

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return payload

    def billable_text(self) -> str:
        parts = [self.content or ""]
        for tc in self.tool_calls:
            parts.append(tc.name)
            parts.append(tc.arguments or "")
        return "\n".join(parts)


# 兼容旧调用点：原来的 ChatTurn 只有 role/content。
class ChatTurn(StoredMessage):
    pass


class _TokenCounter:
    """带编码回退的 token 计数器。tiktoken 缺失时用字符数近似。"""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        self._enc = None
        if tiktoken is not None:
            try:
                self._enc = tiktoken.get_encoding(encoding)
            except Exception:  # noqa: BLE001
                self._enc = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is None:
            # 中文约 1.5 字/token，英文约 4 字符/token；取折中的保守估计。
            return max(1, len(text) // 2)
        return len(self._enc.encode(text))


class _Group:
    """裁剪的原子单位。

    要么是一条独立消息，要么是「assistant(tool_calls) + 全部对应 tool 结果」的整体。
    `pending_tool_ids` 非空表示这组还没收齐工具结果，此时不能发给模型。
    """

    __slots__ = ("messages", "tokens", "pending_tool_ids")

    def __init__(self, messages: list[StoredMessage], tokens: int, pending: set[str]) -> None:
        self.messages = messages
        self.tokens = tokens
        self.pending_tool_ids = pending

    @property
    def complete(self) -> bool:
        return not self.pending_tool_ids


class ShortTermMemory:
    """最近对话的滑动窗口，带独立的不可裁剪摘要区。"""

    def __init__(
        self,
        *,
        model_encoding: str = "cl100k_base",
        max_tokens: int = 8000,
        max_groups: int = 24,
    ) -> None:
        self._counter = _TokenCounter(model_encoding)
        self._max_tokens = max_tokens
        self._max_groups = max_groups
        self._groups: deque[_Group] = deque()
        self._group_tokens = 0
        self._pinned: list[StoredMessage] = []
        self._pinned_tokens = 0

    # ------------------------------------------------------------------ 计数
    def _tokens_of(self, msg: StoredMessage) -> int:
        return self._counter.count(msg.billable_text()) + _PER_MESSAGE_OVERHEAD_TOKENS

    def total_tokens(self) -> int:
        return self._pinned_tokens + self._group_tokens

    def group_count(self) -> int:
        return len(self._groups)

    # ------------------------------------------------------------------ 摘要区
    def set_summary(self, text: str) -> None:
        """写入滚动摘要。摘要区不参与裁剪，因此不会像改造前那样被窗口截断丢弃。"""
        if not text.strip():
            self.clear_summary()
            return
        msg = StoredMessage(role="system", content=f"[历史摘要]\n{text.strip()}")
        self._pinned = [msg]
        self._pinned_tokens = self._tokens_of(msg)

    def clear_summary(self) -> None:
        self._pinned = []
        self._pinned_tokens = 0

    def summary_text(self) -> str:
        if not self._pinned:
            return ""
        return self._pinned[0].content.removeprefix("[历史摘要]\n")

    # ------------------------------------------------------------------ 写入
    def append(self, msg: StoredMessage) -> None:
        """追加一条消息，自动归入正确的消息组。"""
        tokens = self._tokens_of(msg)

        if msg.role == "tool" and msg.tool_call_id:
            group = self._find_group_awaiting(msg.tool_call_id)
            if group is not None:
                group.messages.append(msg)
                group.tokens += tokens
                group.pending_tool_ids.discard(msg.tool_call_id)
                self._group_tokens += tokens
                self._trim()
                return
            # 找不到配对的 assistant 消息，这条 tool 结果单独发出去会被接口拒绝，直接丢弃。
            return

        pending = {tc.id for tc in msg.tool_calls} if msg.role == "assistant" else set()
        self._groups.append(_Group([msg], tokens, pending))
        self._group_tokens += tokens
        self._trim()

    def extend(self, messages: Iterable[StoredMessage]) -> None:
        for msg in messages:
            self.append(msg)

    def _find_group_awaiting(self, tool_call_id: str) -> _Group | None:
        for group in reversed(self._groups):
            if tool_call_id in group.pending_tool_ids:
                return group
        return None

    # ------------------------------------------------------------------ 裁剪
    def _trim(self) -> None:
        """按组裁剪。最后一组永不裁剪，否则会丢掉当前这轮的用户输入。"""
        while len(self._groups) > self._max_groups and len(self._groups) > 1:
            self._drop_oldest()
        while self.total_tokens() > self._max_tokens and len(self._groups) > 1:
            self._drop_oldest()

    def _drop_oldest(self) -> None:
        group = self._groups.popleft()
        self._group_tokens -= group.tokens

    def evict_oldest_groups(self, keep_recent: int) -> list[StoredMessage]:
        """摘要用：摘掉除最近 `keep_recent` 组以外的历史，返回被摘掉的消息。"""
        evicted: list[StoredMessage] = []
        while len(self._groups) > max(1, keep_recent):
            group = self._groups.popleft()
            self._group_tokens -= group.tokens
            evicted.extend(group.messages)
        return evicted

    # ------------------------------------------------------------------ 读取
    def as_messages(self, *, include_incomplete: bool = False) -> list[dict[str, Any]]:
        """转成 OpenAI 兼容消息列表。

        默认跳过未收齐工具结果的组：把孤立的 `tool_calls` 发出去会被接口拒绝，
        这种残缺组通常来自上一轮执行中途异常。
        """
        out: list[dict[str, Any]] = [m.to_openai() for m in self._pinned]
        for group in self._groups:
            if not group.complete and not include_incomplete:
                continue
            out.extend(m.to_openai() for m in group.messages)
        return out

    def snapshot(self) -> list[StoredMessage]:
        out: list[StoredMessage] = list(self._pinned)
        for group in self._groups:
            out.extend(group.messages)
        return out

    def transcript_messages(self) -> list[StoredMessage]:
        """不含摘要区的原始消息，供摘要环节使用。"""
        out: list[StoredMessage] = []
        for group in self._groups:
            out.extend(group.messages)
        return out

    def clear(self) -> None:
        self._groups.clear()
        self._group_tokens = 0
        self.clear_summary()

    # ------------------------------------------------------------------ 序列化
    def dump(self) -> dict[str, Any]:
        return {
            "summary": self.summary_text(),
            "messages": [m.model_dump(mode="json") for m in self.transcript_messages()],
        }

    def load(self, payload: dict[str, Any]) -> None:
        self.clear()
        summary = str(payload.get("summary") or "")
        if summary:
            self.set_summary(summary)
        for raw in payload.get("messages") or []:
            try:
                self.append(StoredMessage.model_validate(raw))
            except Exception:  # noqa: BLE001
                continue

    def to_json(self) -> str:
        return json.dumps(self.dump(), ensure_ascii=False)
