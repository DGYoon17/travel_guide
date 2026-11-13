"""工具规格：一份声明同时驱动 function-calling schema、参数校验、权限与确认门。

`ToolAccess` 是这里的核心：查差标和"订一张不可退的机票"的风险不在一个量级，
harness 必须能按级别区分对待，而不是把所有 function 一视同仁地执行。

`ToolContext` 的 `draft` / `policy` 刻意声明为 `Any`：采购与差旅共用同一个注册表与
确认门，但各自持有不同的草稿与制度模型。内核不应该知道某个业务域的具体字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol

from pydantic import BaseModel

from app.core.memory.audit import AuditLog
from app.core.memory.profile import SubjectProfile
from app.core.security.principal import Principal, Scope


class ToolAccess(str, Enum):
    """工具风险级别。

    READ  —— 只读查询，模型可自由调用。
    DRAFT —— 产生服务端草稿状态，可自由调用，但会更新 L2 并留痕。
    WRITE —— 对外部系统产生不可逆影响，必须先经用户确认门。
    """

    READ = "read"
    DRAFT = "draft"
    WRITE = "write"


@dataclass
class ToolContext:
    """工具执行上下文。主体字段、制度版本、草稿状态都由编排层提供，工具自身不去猜。"""

    principal: Principal
    profile: SubjectProfile
    #: 当前业务域的 L2 草稿（采购 `RequisitionDraft` 或差旅 `TripDraft`）。
    draft: Any
    #: 当前业务域的制度模型（采购 `ProcurementPolicy` 或差旅 `TravelPolicy`）。
    policy: Any
    session_id: str
    trace_id: str
    today: date
    audit: AuditLog | None = None
    policy_version: str = ""
    turn: int = 0
    #: 知识库检索后端（`KnowledgeSearch` 协议）；为 None 时知识检索工具会返回明确的不可用提示。
    knowledge: Any | None = None
    #: 差旅供应方（航班/酒店/预订）。采购域不使用。
    supplier: Any | None = None
    #: 实时数据网关（MCP）。为 None 时实时数据工具不在注册表中，模型不会看到它们。
    mcp: Any | None = None
    #: 预订结果持久化（PostgreSQL）。为 None 时只保留供应方侧订单，不影响主链路。
    booking_repo: Any | None = None

    @property
    def tenant_id(self) -> str:
        return self.principal.tenant_id


class KnowledgeSearch(Protocol):
    """知识库检索后端。实现方负责只返回已发布（active）的文档。"""

    async def search_active(
        self, query: str, *, top_k: int, tenant_id: str
    ) -> list[dict[str, Any]]: ...


@dataclass
class ToolReturn:
    """工具返回值。

    `rendered` 是喂给模型的文本；`data` 是结构化结果，进审计与前端。

    回写 L2 槽位记忆的两个通道刻意分开，因为来源不同、可信度不同：
    - `draft_updates`：模型从用户表述里抽出来的值，记为 `INFERRED`，待用户确认后才锁定；
    - `draft_facts`：工具自己算出来的事实（预估总额、审批路径），记为 `TOOL`。
    """

    data: dict[str, Any] = field(default_factory=dict)
    rendered: str = ""
    draft_updates: dict[str, Any] = field(default_factory=dict)
    draft_facts: dict[str, Any] = field(default_factory=dict)
    #: 直接写入草稿的非槽位属性（如差旅的 `last_plan` / `selected_flight` / `booking_ids`）。
    #: 只允许写入草稿已声明的字段，避免工具往状态里塞任意键。
    draft_attributes: dict[str, Any] = field(default_factory=dict)


class ToolHandler(Protocol):
    async def __call__(self, args: Any, ctx: ToolContext) -> ToolReturn: ...


@dataclass(frozen=True)
class ToolSpec:
    """单个工具的完整声明。"""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any, ToolContext], Awaitable[ToolReturn]]
    access: ToolAccess = ToolAccess.READ
    #: 由服务端从 Principal / Profile 注入、并从模型可见 schema 中剔除的字段。
    injected_fields: frozenset[str] = frozenset()
    #: 调用该工具所需的授权范围。
    required_scopes: tuple[Scope, ...] = ()
    timeout_seconds: float | None = None
    #: True 时，参数中缺失的字段用 L2 草稿里已知的值补齐。
    #: 这是"用户第 1 轮说了单价、第 5 轮模型忘了"这类问题的正解——状态在服务端，不靠模型记。
    draft_backfill: bool = False
    #: WRITE 工具在确认门里向用户展示的摘要生成器。
    preview: Callable[[Any, ToolContext], str] | None = None

    @property
    def needs_confirmation(self) -> bool:
        return self.access is ToolAccess.WRITE
