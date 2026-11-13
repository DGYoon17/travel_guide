"""动态提示状态机 —— **仅用于选择提示词语气，不具备任何执行授权**。

改造前这里用正则 `(确认|同意|可以|就这样|没问题|提交|发起)` 把状态推进到 `EXECUTING`，
而"是否真的提交审批"又与这个状态若即若离。用正则判断用户是否同意提交一笔上万元的采购，
"这个方案可以吗？"就会命中。

现在执行授权归 `app/core/harness/confirmation.py` 的确认门：只有用户针对具体 `action_id`
的显式确认才会执行写操作。本模块降级为提示词策略，`EXECUTING` 只能由确认流程通过
`mark_executing()` 显式置入，正则不再能推进到该状态。
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from app.core.intent.recognizer import BusinessIntent


class ConversationState(str, Enum):
    """对话生命周期状态。"""

    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    PLANNING = "planning"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    COMPLETED = "completed"


_CONFIRM_RE = re.compile(r"(确认|同意|可以|就这样|没问题|提交|发起)")
_DENY_RE = re.compile(r"(修改|不对|取消|再改|换一个)")
_GREETING_RE = re.compile(r"(你好|您好|在吗|帮忙|谢谢)")


class PromptStateMachine:
    """
    根据当前状态、用户输入与意图，计算下一状态与是否进入执行。
    """

    def __init__(self, initial: ConversationState = ConversationState.GREETING) -> None:
        self._state = initial

    @property
    def state(self) -> ConversationState:
        return self._state

    def reset(self, state: ConversationState = ConversationState.GREETING) -> None:
        self._state = state

    def mark_executing(self) -> ConversationState:
        """由确认门在用户显式确认后调用。这是进入 `EXECUTING` 的唯一途径。"""
        self._state = ConversationState.EXECUTING
        return self._state

    def mark_completed(self) -> ConversationState:
        self._state = ConversationState.COMPLETED
        return self._state

    def transition(
        self,
        user_text: str,
        intent: Optional[BusinessIntent] = None,
    ) -> ConversationState:
        """
        根据用户文本与（可选）意图更新内部状态并返回新状态。
        """
        text = user_text.strip()
        next_state = self._infer_next(self._state, text, intent)
        self._state = next_state
        return next_state

    def _infer_next(
        self,
        current: ConversationState,
        text: str,
        intent: Optional[BusinessIntent],
    ) -> ConversationState:
        if not text:
            return current

        if _GREETING_RE.search(text) and current == ConversationState.GREETING:
            return ConversationState.COLLECTING_INFO

        if intent == BusinessIntent.REQUISITION_PLANNING and current in (
            ConversationState.GREETING,
            ConversationState.COLLECTING_INFO,
        ):
            return ConversationState.PLANNING

        if intent in (BusinessIntent.APPROVAL, BusinessIntent.APPLICATION) and current in (
            ConversationState.PLANNING,
            ConversationState.CONFIRMING,
        ):
            # 注意：这里最多推进到 CONFIRMING。正则不再能推进到 EXECUTING —— 执行授权只来自
            # 确认门，见模块 docstring。
            if _DENY_RE.search(text):
                return ConversationState.COLLECTING_INFO
            return ConversationState.CONFIRMING

        if current == ConversationState.PLANNING and _CONFIRM_RE.search(text):
            return ConversationState.CONFIRMING

        if current == ConversationState.EXECUTING and ("完成" in text or "好了" in text):
            return ConversationState.COMPLETED

        if current == ConversationState.COMPLETED and len(text) > 6:
            return ConversationState.COLLECTING_INFO

        if intent == BusinessIntent.INFO_QUERY and current == ConversationState.GREETING:
            return ConversationState.COLLECTING_INFO

        return current

    def should_use_structured_planning(self, state: ConversationState) -> bool:
        return state in (ConversationState.PLANNING, ConversationState.CONFIRMING)
