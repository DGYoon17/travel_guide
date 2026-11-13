"""差旅对话状态机 —— **只决定提示策略与可用工具集，不授予执行权限**。

预订/提交审批能否执行，只由确认门对具体 `action_id` 的显式确认决定。
`BOOKING` 状态只能由确认流程写入，正则永远推不进去。
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from app.core.intent.travel import TravelIntent


class TravelState(str, Enum):
    GREETING = "greeting"
    COLLECTING = "collecting"
    PLANNING = "planning"
    CONFIRMING = "confirming"
    BOOKING = "booking"
    COMPLETED = "completed"
    CHANGING = "changing"
    CANCELLING = "cancelling"


_CONFIRM_RE = re.compile(r"(就这个|确认|可以|没问题|按这个来|订吧)")
_DENY_RE = re.compile(r"(换一个|不对|改一下|取消|再看看)")
_GREETING_RE = re.compile(r"(你好|您好|在吗|帮忙|谢谢)")
_CHANGE_RE = re.compile(r"(改签|改期|换航班|换酒店)")
_CANCEL_RE = re.compile(r"(退订|取消订单|退票)")

#: 各状态下允许模型看到的工具。COLLECTING 阶段看不到预订与提交，避免模型"跳步"。
STATE_TOOL_ACCESS: dict[TravelState, frozenset[str]] = {
    TravelState.GREETING: frozenset({"read"}),
    TravelState.COLLECTING: frozenset({"read", "draft"}),
    TravelState.PLANNING: frozenset({"read", "draft"}),
    #: 确认阶段允许"发起写操作"——但只是登记 PendingAction，真正执行仍由确认门决定。
    TravelState.CONFIRMING: frozenset({"read", "draft", "write"}),
    TravelState.BOOKING: frozenset({"read", "draft", "write"}),
    TravelState.COMPLETED: frozenset({"read", "write"}),
    TravelState.CHANGING: frozenset({"read", "draft", "write"}),
    TravelState.CANCELLING: frozenset({"read", "write"}),
}


class TravelStateMachine:
    def __init__(self, initial: TravelState = TravelState.GREETING) -> None:
        self._state = initial

    @property
    def state(self) -> TravelState:
        return self._state

    def reset(self, state: TravelState = TravelState.GREETING) -> None:
        self._state = state

    def mark_booked(self) -> TravelState:
        """由确认门在预订成功后调用；这是进入 BOOKING 的唯一途径。"""
        self._state = TravelState.BOOKING
        return self._state

    def mark_completed(self) -> TravelState:
        self._state = TravelState.COMPLETED
        return self._state

    def transition(
        self,
        user_text: str,
        intent: Optional[TravelIntent] = None,
    ) -> TravelState:
        text = (user_text or "").strip()
        if not text:
            return self._state

        if _GREETING_RE.search(text) and self._state is TravelState.GREETING:
            self._state = TravelState.COLLECTING
            return self._state

        if _CANCEL_RE.search(text):
            self._state = TravelState.CANCELLING
            return self._state
        if _CHANGE_RE.search(text):
            self._state = TravelState.CHANGING
            return self._state

        if intent in (TravelIntent.TRIP_PLANNING, TravelIntent.FLIGHT_SEARCH, TravelIntent.HOTEL_SEARCH):
            if self._state in (TravelState.GREETING, TravelState.COLLECTING):
                self._state = TravelState.PLANNING
            return self._state

        if intent in (TravelIntent.BOOKING, TravelIntent.APPROVAL):
            if _DENY_RE.search(text):
                self._state = TravelState.PLANNING
            elif self._state in (TravelState.GREETING, TravelState.PLANNING, TravelState.COLLECTING):
                # 最多到 CONFIRMING；BOOKING 只能由确认门写入。
                self._state = TravelState.CONFIRMING
            return self._state

        if self._state is TravelState.PLANNING and _CONFIRM_RE.search(text):
            self._state = TravelState.CONFIRMING
            return self._state

        if self._state is TravelState.COMPLETED and len(text) > 6:
            self._state = TravelState.COLLECTING
        return self._state

    def allowed_access(self) -> frozenset[str]:
        return STATE_TOOL_ACCESS.get(self._state, frozenset({"read"}))


__all__ = ["STATE_TOOL_ACCESS", "TravelState", "TravelStateMachine"]
