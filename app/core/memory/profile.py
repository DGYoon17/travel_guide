"""L3 主体画像：部门、成本中心、职级、审批上级、可用预算编码。

这一层的值来自 HR / OA 主数据，属于"记忆"里**不允许由模型生成也不允许被摘要压缩**的部分。
工具执行前由编排层注入，因此用户不需要每轮重复交代"我是市场部主管"。

`ProfileProvider` 是替换点：生产接 HR 主数据或 OA 组织接口；默认实现直接采用凭据里的
声明（`ClaimsProfileProvider`），另提供 `StaticMasterDataProvider` 用配置模拟主数据覆盖。
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger
from app.core.security.principal import Principal
from app.domain.procurement.models import RequesterLevel

logger = get_logger(__name__)


class SubjectProfile(BaseModel):
    """请购人画像。字段与 `ProcurementRequest` 的主体字段一一对应。"""

    model_config = ConfigDict(frozen=True)

    subject_id: str
    display_name: str = ""
    tenant_id: str = "default"
    department: str
    requester_level: RequesterLevel = RequesterLevel.STAFF
    cost_center: str | None = None
    manager_id: str | None = None
    budget_codes: tuple[str, ...] = ()
    source: str = Field("claims", description="画像来源，用于审计区分主数据与凭据声明")

    @property
    def default_budget_code(self) -> str | None:
        return self.budget_codes[0] if self.budget_codes else None

    def as_injected_arguments(self) -> dict[str, Any]:
        """转成工具参数注入值。键名与工具入参模型的字段名一致。

        采购域用 `requester_*`，差旅域用 `traveler_*`；两套键同时给出，由工具 schema 决定取哪些。
        """
        return {
            "requester_id": self.subject_id,
            "requester_level": self.requester_level.value,
            "traveler_id": self.subject_id,
            "traveler_level": self.requester_level.value,
            "department": self.department,
            "tenant_id": self.tenant_id,
        }


class ProfileProvider(Protocol):
    async def get(self, principal: Principal) -> SubjectProfile | None: ...


class ClaimsProfileProvider:
    """默认实现：画像即凭据声明。适合 SSO 已下发完整组织信息的场景。"""

    async def get(self, principal: Principal) -> SubjectProfile | None:
        return SubjectProfile(
            subject_id=principal.subject_id,
            display_name=principal.display_name,
            tenant_id=principal.tenant_id,
            department=principal.department,
            requester_level=principal.requester_level,
            cost_center=principal.cost_center,
            manager_id=principal.manager_id,
            budget_codes=principal.budget_codes,
            source="claims",
        )


class StaticMasterDataProvider:
    """用配置模拟 HR/OA 主数据；命中则覆盖凭据声明，未命中回落到声明。

    覆盖方向是刻意的：主数据比 token 里的快照更新更及时（调岗、升级、预算编码变更）。
    """

    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self._records = records
        self._fallback = ClaimsProfileProvider()

    @classmethod
    def from_json(cls, raw: str) -> "StaticMasterDataProvider":
        records: dict[str, dict[str, Any]] = {}
        if raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    records = {str(k): dict(v) for k, v in parsed.items()}
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.error("profile.master_data_invalid", error=str(exc))
        return cls(records)

    async def get(self, principal: Principal) -> SubjectProfile | None:
        key = f"{principal.tenant_id}:{principal.subject_id}"
        record = self._records.get(key) or self._records.get(principal.subject_id)
        if record is None:
            return await self._fallback.get(principal)
        payload: dict[str, Any] = {
            "subject_id": principal.subject_id,
            "display_name": principal.display_name,
            "tenant_id": principal.tenant_id,
            "department": principal.department,
            "requester_level": principal.requester_level.value,
            "cost_center": principal.cost_center,
            "manager_id": principal.manager_id,
            "budget_codes": list(principal.budget_codes),
            **record,
            "source": "master_data",
        }
        payload["budget_codes"] = tuple(payload.get("budget_codes") or ())
        try:
            return SubjectProfile.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("profile.master_data_record_invalid", subject=principal.subject_id, error=str(exc))
            return await self._fallback.get(principal)


async def resolve_profile(principal: Principal, provider: ProfileProvider | None) -> SubjectProfile:
    """取画像；provider 缺失或异常时回落到凭据声明，保证主体字段永远有确定来源。"""
    if provider is not None:
        try:
            profile = await provider.get(principal)
            if profile is not None:
                return profile
        except Exception as exc:  # noqa: BLE001
            logger.warning("profile.provider_failed", error=str(exc))
    fallback = await ClaimsProfileProvider().get(principal)
    assert fallback is not None
    return fallback
