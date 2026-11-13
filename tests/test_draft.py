"""L2 槽位记忆测试：已确认字段不可被模型静默改写。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.core.memory.draft import RequisitionDraft, SlotSource


def _draft() -> RequisitionDraft:
    return RequisitionDraft(session_id="s1", tenant_id="acme", subject_id="E1")


def test_merge_normalizes_values_for_persistence() -> None:
    draft = _draft()
    draft.merge(
        {
            "unit_price_cny": Decimal("15000.00"),
            "needed_by": date(2026, 9, 15),
            "quantity": 3,
        },
        source=SlotSource.INFERRED,
        turn=1,
    )
    # 槽位值必须是 JSON 原生类型，否则进不了 Redis 也无法稳定哈希。
    assert draft.value("unit_price_cny") == "15000.00"
    assert draft.value("needed_by") == "2026-09-15"
    assert draft.value("quantity") == 3


def test_confirmed_slot_rejects_silent_overwrite() -> None:
    draft = _draft()
    draft.merge({"unit_price_cny": "15000"}, source=SlotSource.USER, turn=1)
    draft.confirm(("unit_price_cny",))

    applied, rejected = draft.merge(
        {"unit_price_cny": "18000"}, source=SlotSource.INFERRED, turn=3
    )

    assert applied == ()
    assert len(rejected) == 1
    assert rejected[0].field == "unit_price_cny"
    assert draft.value("unit_price_cny") == "15000", "已确认金额不能被后续轮次改掉"
    assert draft.rejected_changes, "被拒绝的变更要留痕，供提示模型和审计"


def test_explicit_overwrite_is_allowed_and_logged() -> None:
    draft = _draft()
    draft.merge({"unit_price_cny": "15000"}, source=SlotSource.USER, turn=1)
    draft.confirm(("unit_price_cny",))

    applied, rejected = draft.merge(
        {"unit_price_cny": "18000"},
        source=SlotSource.USER,
        turn=4,
        allow_overwrite_confirmed=True,
    )
    assert applied == ("unit_price_cny",)
    assert rejected == ()
    assert draft.value("unit_price_cny") == "18000"
    assert draft.overwrite_log[-1].old_value == "15000"


def test_unconfirm_releases_the_lock() -> None:
    draft = _draft()
    draft.merge({"item_name": "设计协作软件"}, source=SlotSource.USER, turn=1)
    draft.confirm(("item_name",))
    draft.unconfirm(("item_name",))
    applied, _ = draft.merge({"item_name": "协同办公软件"}, source=SlotSource.USER, turn=2)
    assert applied == ("item_name",)


def test_missing_required_and_prompt() -> None:
    draft = _draft()
    assert set(draft.missing_required()) == {
        "category", "item_name", "quantity", "unit_price_cny", "needed_by", "reason",
    }
    draft.merge(
        {
            "category": "software",
            "item_name": "设计协作软件",
            "quantity": 1,
            "unit_price_cny": "15000",
        },
        source=SlotSource.INFERRED,
        turn=1,
    )
    assert set(draft.missing_required()) == {"needed_by", "reason"}

    prompt = draft.missing_prompt()
    assert "需求到货/上线日期" in prompt
    assert "采购事由" in prompt
    assert "YYYY-MM-DD" in prompt, "追问应带取值格式，避免再问一轮"


def test_unknown_slots_are_ignored() -> None:
    draft = _draft()
    applied, _ = draft.merge({"totally_unknown": "x"}, source=SlotSource.INFERRED, turn=1)
    assert applied == ()
    assert draft.version == 0


def test_fingerprint_changes_with_content_and_is_stable_otherwise() -> None:
    draft = _draft()
    draft.merge({"item_name": "A", "quantity": 1}, source=SlotSource.USER, turn=1)
    first = draft.fingerprint()
    assert draft.fingerprint() == first, "同一份内容指纹必须稳定（幂等键依赖它）"

    draft.merge({"quantity": 2}, source=SlotSource.USER, turn=2)
    assert draft.fingerprint() != first, "内容变了就是一笔新的采购"


def test_version_increments_only_on_change() -> None:
    draft = _draft()
    draft.merge({"item_name": "A"}, source=SlotSource.USER, turn=1)
    version = draft.version
    draft.merge({"item_name": "A"}, source=SlotSource.USER, turn=2)
    assert draft.version == version, "重复相同值不应制造新版本"


def test_known_amounts_feeds_summary_verification() -> None:
    draft = _draft()
    draft.merge({"unit_price_cny": "15000.00"}, source=SlotSource.USER, turn=1)
    draft.last_requisition = {"estimated_total_cny": "45000.00"}
    amounts = draft.known_amounts()
    assert "15000.00" in amounts
    assert "15000" in amounts
    assert "45000.00" in amounts
