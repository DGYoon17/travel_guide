"""写操作的确认执行接口。

预订、取消、提交审批的**唯一**执行入口。刻意与 `/chat` 分离：确认必须是用户在界面上针对
一个具体 `action_id` 的显式动作，而不是从自然语言里正则匹配「可以」「没问题」推断出来的。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request, Response

from app.agent.travel_orchestrator import TravelOrchestrator
from app.api.routes.chat import get_orchestrator
from app.core.security.auth import require_scopes
from app.core.security.principal import Principal, Scope
from app.domain.schemas import ConfirmActionRequest, ConfirmActionResponse

router = APIRouter(tags=["approvals"])


@router.post("/approvals/confirm", response_model=ConfirmActionResponse)
async def confirm_action(
    body: ConfirmActionRequest,
    request: Request,
    response: Response,
    principal: Principal = Depends(require_scopes(Scope.TRIP_BOOK)),
    orchestrator: TravelOrchestrator = Depends(get_orchestrator),
) -> ConfirmActionResponse:
    trace_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    response.headers["X-Trace-Id"] = trace_id
    return await orchestrator.confirm_action(principal, body, trace_id=trace_id)


@router.post("/sessions/{session_id}/confirm-slots")
async def confirm_slots(
    session_id: str,
    request: Request,
    principal: Principal = Depends(require_scopes(Scope.TRIP_DRAFT)),
    orchestrator: TravelOrchestrator = Depends(get_orchestrator),
) -> dict[str, list[str]]:
    """把行程要素标记为用户已确认。

    确认后的字段在后续轮次里不允许被模型静默改写——用户确认过的日期与舱位，
    不能因为模型"顺手优化"而变化。
    """
    payload = await request.json() if await request.body() else {}
    raw_fields = payload.get("fields") if isinstance(payload, dict) else None
    fields = tuple(str(f) for f in raw_fields) if isinstance(raw_fields, list) else None
    changed = await orchestrator.mark_slots_confirmed(principal, session_id, fields)
    return {"confirmed": list(changed)}
