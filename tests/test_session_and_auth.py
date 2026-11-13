"""会话存储与鉴权测试。"""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from app.core.memory.draft import RequisitionDraft, SlotSource
from app.core.memory.session_store import (
    ConcurrentSessionUpdate,
    InMemorySessionStore,
    SessionState,
)
from app.core.security.auth import StaticTokenResolver, require_principal
from app.core.security.principal import Scope
from app.domain.procurement.models import RequesterLevel


# --------------------------------------------------------------------- 会话存储


async def test_session_roundtrip_preserves_draft_slots() -> None:
    store = InMemorySessionStore()
    draft = RequisitionDraft(session_id="s1", tenant_id="acme", subject_id="E1")
    draft.merge({"unit_price_cny": "15000"}, source=SlotSource.USER, turn=1)
    draft.confirm(("unit_price_cny",))

    await store.save(
        SessionState(session_id="s1", tenant_id="acme", subject_id="E1", draft=draft)
    )
    loaded = await store.load("s1", tenant_id="acme")

    assert loaded is not None
    assert loaded.draft.value("unit_price_cny") == "15000"
    assert "unit_price_cny" in loaded.draft.confirmed_fields()


async def test_tenant_isolation_in_session_keys() -> None:
    """不同法人的同名 session_id 不能互相读到。"""
    store = InMemorySessionStore()
    await store.save(SessionState(session_id="shared", tenant_id="acme", subject_id="E1"))
    assert await store.load("shared", tenant_id="other-corp") is None
    assert await store.load("shared", tenant_id="acme") is not None


async def test_optimistic_lock_rejects_stale_write() -> None:
    """并发写不能互相覆盖对方的槽位确认。"""
    store = InMemorySessionStore()
    saved = await store.save(SessionState(session_id="s1", tenant_id="acme", subject_id="E1"))

    stale = await store.load("s1", tenant_id="acme")
    assert stale is not None
    # 另一个请求先写入，把版本推进了。
    await store.save(saved)

    with pytest.raises(ConcurrentSessionUpdate):
        await store.save(stale)


async def test_version_increments_on_save() -> None:
    store = InMemorySessionStore()
    state = SessionState(session_id="s1", tenant_id="acme", subject_id="E1")
    assert state.version == 0
    await store.save(state)
    assert state.version == 1


# --------------------------------------------------------------------- 鉴权


async def test_static_resolver_maps_token_to_principal() -> None:
    resolver = StaticTokenResolver(
        {
            "tok-a": {
                "subject_id": "E10086",
                "display_name": "张三",
                "tenant_id": "acme",
                "department": "市场部",
                "requester_level": "manager",
                "budget_codes": ["BUD-1"],
                "scopes": ["chat", "requisition:draft"],
            }
        }
    )
    principal = await resolver.resolve("tok-a")
    assert principal is not None
    assert principal.subject_id == "E10086"
    assert principal.department == "市场部"
    assert principal.requester_level is RequesterLevel.MANAGER
    assert principal.budget_codes == ("BUD-1",)
    assert principal.has_scope(Scope.REQUISITION_DRAFT)
    assert not principal.has_scope(Scope.REQUISITION_SUBMIT)


async def test_unknown_token_is_rejected() -> None:
    resolver = StaticTokenResolver({"tok-a": {"subject_id": "E1", "department": "D"}})
    assert await resolver.resolve("tok-b") is None


async def test_unknown_scope_in_claims_is_ignored_not_granted() -> None:
    resolver = StaticTokenResolver(
        {
            "t": {
                "subject_id": "E1",
                "department": "D",
                "scopes": ["chat", "requisition:god-mode"],
            }
        }
    )
    principal = await resolver.resolve("t")
    assert principal is not None
    assert principal.scopes == frozenset({Scope.CHAT})


async def test_resolver_from_settings_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    payload = json.dumps(
        {"tok-x": {"subject_id": "E7", "department": "IT", "scopes": ["chat"]}}
    )
    monkeypatch.setattr(settings, "auth_tokens_json", payload)
    monkeypatch.setattr(settings, "auth_dev_token", "")

    resolver = StaticTokenResolver.from_settings()
    principal = await resolver.resolve("tok-x")
    assert principal is not None and principal.subject_id == "E7"


async def test_dev_token_is_refused_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "auth_tokens_json", "")
    monkeypatch.setattr(settings, "auth_dev_token", "dev-token")
    monkeypatch.setattr(settings, "app_env", "production")

    resolver = StaticTokenResolver.from_settings()
    assert await resolver.resolve("dev-token") is None, "生产环境不得接受演示 token"


async def test_missing_credentials_yields_401() -> None:
    class _Req:
        class app:  # noqa: N801
            class state:  # noqa: N801
                principal_resolver = StaticTokenResolver({})

    with pytest.raises(HTTPException) as excinfo:
        await require_principal(_Req(), None)  # type: ignore[arg-type]
    assert excinfo.value.status_code == 401
    assert excinfo.value.headers["WWW-Authenticate"] == "Bearer"


async def test_auth_disabled_is_blocked_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """AUTH_REQUIRED=false 在生产必须直接报错，而不是静默放开。"""
    from app.config import settings

    monkeypatch.setattr(settings, "auth_required", False)
    monkeypatch.setattr(settings, "app_env", "production")

    class _Req:
        class app:  # noqa: N801
            class state:  # noqa: N801
                principal_resolver = StaticTokenResolver({})

    with pytest.raises(HTTPException) as excinfo:
        await require_principal(_Req(), None)  # type: ignore[arg-type]
    assert excinfo.value.status_code == 500
