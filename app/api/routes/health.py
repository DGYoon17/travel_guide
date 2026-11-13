"""健康检查。

改造前这里把"依赖没连上"报成健康：`app.state.redis is None` 时 `checks["redis"] = True`。
而 lifespan 恰恰是在 ping 失败后把它置成 None 的 —— 于是 Redis 挂掉时探针显示 ok，
而会话状态已经退化到进程内存（多副本不共享）。现在区分三种情况：连通、降级、未配置。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.config import settings
from app.domain.schemas import HealthStatus

router = APIRouter(tags=["health"])

#: 这些依赖不可用只造成降级（功能可用但有损），不算 unhealthy。
_DEGRADING = ("redis", "milvus", "database", "orchestrator")


@router.get("/health", response_model=HealthStatus)
async def health(request: Request) -> HealthStatus:
    checks: dict[str, bool] = {}
    details: list[str] = []
    state = request.app.state

    # ---------------------------------------------------------------- Redis
    redis_client = getattr(state, "redis", None)
    if redis_client is None:
        checks["redis"] = False
        details.append("redis:未连接（会话与确认状态已退化为进程内存，多副本不共享）")
    else:
        try:
            await redis_client.ping()
            checks["redis"] = True
        except Exception as exc:  # noqa: BLE001
            checks["redis"] = False
            details.append(f"redis:{exc!s}")

    # ---------------------------------------------------------------- Milvus
    milvus = getattr(state, "milvus", None)
    checks["milvus"] = bool(milvus is not None and milvus.connected)
    if not checks["milvus"]:
        details.append("milvus:未连接（知识库检索不可用）")

    # ---------------------------------------------------------------- Postgres
    engine = getattr(state, "db_engine", None)
    if engine is None:
        checks["database"] = False
        details.append("database:未创建引擎（审计已退化为内存，重启即丢）")
    else:
        try:
            from sqlalchemy import text

            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            checks["database"] = True
        except Exception as exc:  # noqa: BLE001
            checks["database"] = False
            details.append(f"database:{exc!s}")

    # ---------------------------------------------------------------- 应用自身
    checks["orchestrator"] = getattr(state, "orchestrator", None) is not None
    if not checks["orchestrator"]:
        details.append("orchestrator:未初始化（/chat 将返回 503，通常是缺少 API key）")

    resolver = getattr(state, "principal_resolver", None)
    checks["auth"] = resolver is not None
    if not checks["auth"]:
        details.append("auth:鉴权组件未初始化")

    # 鉴权不可用是硬故障：接口会全线 401/503，不应报成 degraded。
    if not checks["auth"]:
        status: Any = "unhealthy"
    elif all(checks.get(name, False) for name in _DEGRADING):
        status = "ok"
    else:
        status = "degraded"

    return HealthStatus(
        status=status,
        checks=checks,
        detail="; ".join(details) if details else None,
    )


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """存活探针：只表示进程还在，不探测下游。用于 k8s livenessProbe。"""
    return {"status": "alive", "service": settings.app_name}
