"""审批提交工具（写操作）。

两处与改造前不同：

1. 单号由幂等键确定性派生，而不是 `uuid4()`。网络重试或用户重复点击确认时，同一份草稿
   得到同一个审批号，不会在 OA 里产生两张请购单。
2. 提交结果带上审批路径与合规提示的快照，作为审计留痕的一部分。
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ApprovalSubmitRequest(BaseModel):
    requisition_id: str
    requester_id: str
    tenant_id: str = "default"
    department: str = ""
    budget_code: str | None = None
    estimated_total_cny: float
    approval_roles: list[str] = Field(default_factory=list)
    compliance_warnings: list[str] = Field(default_factory=list)
    idempotency_key: str
    note: str | None = None


class ApprovalSubmitResult(BaseModel):
    approval_id: str
    status: str
    approval_no: str
    created_at: datetime
    current_stage: str


def derive_approval_no(idempotency_key: str) -> str:
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:8].upper()
    return f"APR-{digest}"


def derive_approval_id(idempotency_key: str) -> str:
    return "apv_" + hashlib.sha256(f"id:{idempotency_key}".encode("utf-8")).hexdigest()[:24]


async def submit_approval(req: ApprovalSubmitRequest) -> dict[str, Any]:
    """向审批系统提交请购单。

    当前为进程内模拟实现；接 OA/SRM 时把这里换成带幂等头的下游调用，
    `idempotency_key` 直接作为下游的 `Idempotency-Key`。
    """
    approval_no = derive_approval_no(req.idempotency_key)
    result = ApprovalSubmitResult(
        approval_id=derive_approval_id(req.idempotency_key),
        status="submitted",
        approval_no=approval_no,
        created_at=datetime.now(timezone.utc),
        current_stage=req.approval_roles[0] if req.approval_roles else "部门负责人",
    )
    return {
        "ok": True,
        "approval": result.model_dump(mode="json"),
        "requisition_id": req.requisition_id,
        "requester_id": req.requester_id,
        "tenant_id": req.tenant_id,
        "approval_roles": req.approval_roles,
        "compliance_warnings": req.compliance_warnings,
        "estimated_total_cny": req.estimated_total_cny,
        "idempotency_key": req.idempotency_key,
    }
