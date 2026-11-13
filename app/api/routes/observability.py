"""可观测性接口：指标快照与近期审计。

对应"AI 系统排障要能分辨是模型慢、检索慢还是工具慢"这个诉求 —— 改造前 `MetricsCollector`
写好了但没有任何采集点和出口，等于有可观测性代码、无可观测性。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from app.core.security.auth import require_scopes
from app.core.security.principal import Principal, Scope

router = APIRouter(tags=["observability"])


@router.get("/metrics")
async def metrics(
    request: Request,
    principal: Principal = Depends(require_scopes(Scope.METRICS_READ)),
) -> dict:
    collector = getattr(request.app.state, "metrics", None)
    breaker = getattr(request.app.state, "llm_breaker", None)
    return {
        "counters": collector.snapshot() if collector is not None else {},
        "llm_circuit_state": breaker.state.value if breaker is not None else "unknown",
        "tools": list(getattr(request.app.state, "tool_names", ())),
    }


@router.get("/audit/recent")
async def recent_audit(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    principal: Principal = Depends(require_scopes(Scope.METRICS_READ)),
) -> dict:
    audit = getattr(request.app.state, "audit", None)
    if audit is None:
        return {"events": [], "detail": "审计未启用"}
    events = await audit.recent(tenant_id=principal.tenant_id, limit=limit)
    return {"events": [e.model_dump(mode="json") for e in events]}
