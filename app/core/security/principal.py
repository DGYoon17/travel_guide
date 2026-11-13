"""调用者身份（Principal）：主体字段的唯一可信来源。

设计约束：`subject_id` / `department` / `requester_level` / `tenant_id` 决定审批链和
数据可见范围，因此它们只能来自已验证的凭据，**不能**出现在 LLM 可填写的工具参数里。
编排层在执行工具前把这些字段注入，模型既看不到也改不了。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.procurement.models import RequesterLevel


class Scope(str, Enum):
    """细粒度授权范围，按"读 / 起草 / 预订 / 提交审批 / 知识库"划分。"""

    CHAT = "chat"
    REQUISITION_DRAFT = "requisition:draft"
    REQUISITION_SUBMIT = "requisition:submit"
    TRIP_DRAFT = "trip:draft"
    TRIP_BOOK = "trip:book"
    TRIP_APPROVE = "trip:approve"
    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_WRITE = "knowledge:write"
    KNOWLEDGE_PUBLISH = "knowledge:publish"
    METRICS_READ = "metrics:read"


DEFAULT_EMPLOYEE_SCOPES: frozenset[Scope] = frozenset(
    {
        Scope.CHAT,
        Scope.REQUISITION_DRAFT,
        Scope.REQUISITION_SUBMIT,
        Scope.TRIP_DRAFT,
        Scope.TRIP_BOOK,
        Scope.TRIP_APPROVE,
        Scope.KNOWLEDGE_READ,
    }
)


class Principal(BaseModel):
    """已验证的调用主体。不可变，避免在请求链路中被下游改写。"""

    model_config = ConfigDict(frozen=True)

    subject_id: str = Field(..., min_length=1, description="员工工号，审批单的发起人身份")
    display_name: str = Field("", description="展示名")
    tenant_id: str = Field("default", min_length=1, description="法人/租户，决定数据隔离边界")
    department: str = Field(..., min_length=1, description="部门，来自主数据而非用户自述")
    requester_level: RequesterLevel = RequesterLevel.STAFF
    cost_center: str | None = None
    manager_id: str | None = None
    budget_codes: tuple[str, ...] = ()
    scopes: frozenset[Scope] = DEFAULT_EMPLOYEE_SCOPES

    def has_scope(self, scope: Scope) -> bool:
        return scope in self.scopes

    def missing_scopes(self, required: tuple[Scope, ...]) -> tuple[Scope, ...]:
        return tuple(s for s in required if s not in self.scopes)

    @property
    def audit_actor(self) -> str:
        """审计日志里的行为主体标识。"""
        return f"{self.tenant_id}:{self.subject_id}"
