"""Bearer token 鉴权与 scope 校验。

`PrincipalResolver` 是可替换点：生产接 SSO / JWT 校验或 OA 的 introspect 接口，
本地演示用 `StaticTokenResolver`。默认**失败关闭**——没有配置任何 token 时接口返回
401 而不是放开，避免"忘了配鉴权"退化成匿名可写。
"""

from __future__ import annotations

import hmac
import json
from typing import Any, Protocol

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.core.logging import get_logger
from app.core.security.principal import DEFAULT_EMPLOYEE_SCOPES, Principal, Scope
from app.domain.procurement.models import RequesterLevel

logger = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False, description="企业 SSO 下发的访问令牌")

DEV_PRINCIPAL_CLAIMS: dict[str, Any] = {
    "subject_id": "E10086",
    "display_name": "演示用户",
    "tenant_id": "demo",
    "department": "市场部",
    "requester_level": RequesterLevel.MANAGER.value,
    "cost_center": "CC-MKT-01",
    "manager_id": "E10001",
    "budget_codes": ["BUD-MKT-2026", "BUD-MKT-SW"],
    "scopes": sorted(s.value for s in DEFAULT_EMPLOYEE_SCOPES),
}


class PrincipalResolver(Protocol):
    async def resolve(self, token: str) -> Principal | None: ...


def _principal_from_claims(claims: dict[str, Any]) -> Principal:
    raw_scopes = claims.get("scopes") or []
    scopes: set[Scope] = set()
    for item in raw_scopes:
        try:
            scopes.add(Scope(item))
        except ValueError:
            logger.warning("auth.unknown_scope", scope=item)
    payload = dict(claims)
    payload["scopes"] = frozenset(scopes) if scopes else DEFAULT_EMPLOYEE_SCOPES
    payload["budget_codes"] = tuple(claims.get("budget_codes") or ())
    return Principal.model_validate(payload)


class StaticTokenResolver:
    """从配置加载 token → 主体声明的静态映射。

    token 比较用 `hmac.compare_digest`，避免按前缀短路的时序差异。
    """

    def __init__(self, tokens: dict[str, dict[str, Any]]) -> None:
        self._tokens = tokens

    @classmethod
    def from_settings(cls) -> "StaticTokenResolver":
        tokens: dict[str, dict[str, Any]] = {}
        if settings.auth_tokens_json.strip():
            try:
                parsed = json.loads(settings.auth_tokens_json)
                if isinstance(parsed, dict):
                    tokens.update({str(k): dict(v) for k, v in parsed.items()})
                else:
                    logger.error("auth.tokens_json_not_object")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.error("auth.tokens_json_invalid", error=str(exc))

        dev_token = settings.auth_dev_token.strip()
        if dev_token:
            if settings.is_production:
                logger.error("auth.dev_token_rejected_in_production")
            else:
                tokens.setdefault(dev_token, dict(DEV_PRINCIPAL_CLAIMS))
                logger.warning("auth.dev_token_enabled", subject=DEV_PRINCIPAL_CLAIMS["subject_id"])

        if not tokens:
            logger.error("auth.no_tokens_configured", hint="设置 AUTH_TOKENS_JSON 或 AUTH_DEV_TOKEN")
        return cls(tokens)

    async def resolve(self, token: str) -> Principal | None:
        for candidate, claims in self._tokens.items():
            if hmac.compare_digest(candidate, token):
                try:
                    return _principal_from_claims(claims)
                except Exception as exc:  # noqa: BLE001
                    logger.error("auth.claims_invalid", error=str(exc))
                    return None
        return None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    """FastAPI 依赖：解析并校验 Bearer token，返回可信主体。"""
    if not settings.auth_required:
        if settings.is_production:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="生产环境不允许关闭鉴权",
            )
        logger.warning("auth.disabled_returning_dev_principal")
        return _principal_from_claims(DEV_PRINCIPAL_CLAIMS)

    if credentials is None or not credentials.credentials:
        raise _unauthorized("缺少 Bearer 访问令牌")

    resolver: PrincipalResolver | None = getattr(request.app.state, "principal_resolver", None)
    if resolver is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="鉴权组件未初始化",
        )

    principal = await resolver.resolve(credentials.credentials)
    if principal is None:
        raise _unauthorized("访问令牌无效或已过期")
    return principal


def require_scopes(*required: Scope):
    """依赖工厂：在 `require_principal` 之上追加 scope 检查。"""

    async def _dep(principal: Principal = Depends(require_principal)) -> Principal:
        missing = principal.missing_scopes(required)
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"缺少所需权限：{', '.join(s.value for s in missing)}",
            )
        return principal

    return _dep
