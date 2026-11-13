from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    """对话请求。

    历史消息由服务端按 `session_id` 维护，客户端只需发本轮输入。仍接受 `messages` 是为了
    兼容旧客户端，但其中的 assistant/tool 消息会被忽略：会话状态的真相在服务端，否则前端
    就能改写"用户已确认的行程与订单"。
    """

    message: Optional[str] = Field(None, description="本轮用户输入（推荐）")
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="兼容字段：旧客户端传的完整历史；服务端只取最后一条 user 消息",
    )
    stream: bool = False
    session_id: Optional[str] = Field(None, description="会话 ID；缺省由服务端生成并回传")
    locale: str = Field("zh-CN", description="语言区域")

    def user_text(self) -> str:
        """取本轮用户输入。"""
        if self.message and self.message.strip():
            return self.message.strip()
        for msg in reversed(self.messages):
            if msg.role is MessageRole.USER and msg.content.strip():
                return msg.content.strip()
        return ""


class ToolCallInfo(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error_code: Optional[str] = None


class PendingActionInfo(BaseModel):
    """需要用户显式确认才能执行的写操作。"""

    action_id: str
    tool_name: str
    preview: str
    expires_at: str
    requires_explicit_confirmation: bool = True


class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: Optional[dict[str, int]] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    #: 只暴露工具名与成败，不回传内部推理轨迹。
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    pending_action: Optional[PendingActionInfo] = None
    missing_fields: list[str] = Field(default_factory=list)
    degraded: bool = False
    finish_reason: Optional[str] = None


class StreamChunkType(str, Enum):
    STATUS = "status"
    CONTENT = "content"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONFIRMATION_REQUIRED = "confirmation_required"
    DONE = "done"
    ERROR = "error"


class StreamChunk(BaseModel):
    """SSE 事件。

    工具调用与执行结果作为独立事件推给前端，用户能看到「正在校验采购制度…」这类中间态，
    而不是在多轮工具调用期间盯着空白屏。
    """

    type: StreamChunkType
    index: int = 0
    delta: Optional[str] = None
    status: Optional[str] = None
    tool_name: Optional[str] = None
    tool_ok: Optional[bool] = None
    tool_error_code: Optional[str] = None
    pending_action: Optional[PendingActionInfo] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    degraded: Optional[bool] = None
    usage: Optional[dict[str, int]] = None
    finish_reason: Optional[str] = None
    #: 本轮意图与执行范式，便于前端与离线评测归因。
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    paradigm: Optional[str] = None
    route_reason: Optional[str] = None
    error: Optional[str] = None


class ConfirmDecision(str, Enum):
    CONFIRM = "confirm"
    REJECT = "reject"


class ConfirmActionRequest(BaseModel):
    """确认或拒绝一个待执行的写操作。

    刻意做成独立接口而不是对话内的自然语言判定：正则匹配「可以」来决定是否向 OA 提交
    上万元的采购审批，风险过高（「这个方案可以吗？」会命中）。
    """

    action_id: str = Field(..., min_length=1)
    decision: ConfirmDecision = ConfirmDecision.CONFIRM
    note: Optional[str] = Field(None, max_length=500)


class ConfirmActionResponse(BaseModel):
    action_id: str
    status: str
    replayed: bool = Field(False, description="True 表示命中幂等，返回的是首次执行结果")
    message: str
    result: Optional[dict[str, Any]] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None


class DocumentStatus(str, Enum):
    """知识库文档审核态。只有 ACTIVE 参与检索。"""

    DRAFT = "draft"
    REVIEWED = "reviewed"
    ACTIVE = "active"
    ARCHIVED = "archived"


class DocumentIngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1, description="纯文本正文，用于向量检索")
    doc_type: Literal["policy", "sop", "travel_policy", "visa", "other"] = "policy"
    policy_version: Optional[str] = Field(None, description="制度版本；缺省取当前配置版本")
    effective_from: Optional[str] = Field(None, description="生效日期 YYYY-MM-DD")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestResponse(BaseModel):
    doc_id: str
    collection: str
    inserted_at: datetime
    vector_dim: Optional[int] = None
    status: DocumentStatus = DocumentStatus.DRAFT
    policy_version: Optional[str] = None


class DocumentPublishRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    status: DocumentStatus = DocumentStatus.ACTIVE


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "unhealthy"]
    version: str = "0.1.0"
    checks: dict[str, bool] = Field(default_factory=dict)
    detail: Optional[str] = None
