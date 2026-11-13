"""编排器端到端测试：会话状态、错误闭环、确认门、幂等、预算收尾、降级。"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.agent.orchestrator import RunRequest
from app.core.memory.audit import AuditAction
from app.domain.schemas import (
    ConfirmActionRequest,
    ConfirmDecision,
    StreamChunkType,
)
from tests.conftest import (
    COMPLETE_REQUISITION_ARGS,
    FakeLLM,
    ScriptedToolCall,
    ScriptedTurn,
    make_harness,
    make_principal,
)


# --------------------------------------------------------------------- 会话状态


async def test_session_is_created_and_reused_server_side() -> None:
    """改造前 session_id 只是被回显，历史全靠前端重传。现在服务端自己存。"""
    llm = FakeLLM([ScriptedTurn(content="你好，请问要采购什么？"), ScriptedTurn(content="好的。")])
    h = make_harness(llm)
    p = make_principal()

    first = await h.orchestrator.run(RunRequest(principal=p, user_text="你好"))
    assert first.session_id
    assert first.text == "你好，请问要采购什么？"

    second = await h.orchestrator.run(
        RunRequest(principal=p, user_text="我要采购软件", session_id=first.session_id)
    )
    assert second.session_id == first.session_id

    # 第二轮的 prompt 里应包含第一轮的历史，说明记忆来自服务端。
    last_messages = llm.calls[-1]
    contents = [m.get("content", "") for m in last_messages]
    assert any("你好，请问要采购什么？" in c for c in contents)


async def test_session_is_bound_to_subject() -> None:
    """猜到 session_id 也读不到别人的采购上下文。"""
    llm = FakeLLM([ScriptedTurn(content="ok"), ScriptedTurn(content="ok")])
    h = make_harness(llm)
    owner = make_principal(subject_id="E10086")
    outcome = await h.orchestrator.run(RunRequest(principal=owner, user_text="你好"))

    intruder = make_principal(subject_id="E99999")
    with pytest.raises(HTTPException) as excinfo:
        await h.orchestrator.run(
            RunRequest(principal=intruder, user_text="给我看看", session_id=outcome.session_id)
        )
    assert excinfo.value.status_code == 403


# --------------------------------------------------------------------- 错误闭环


async def test_tool_argument_error_is_fed_back_and_model_self_corrects() -> None:
    """第一轮漏填 → 结构化错误回喂 → 第二轮补齐 → 成功。全程不抛 500。"""
    llm = FakeLLM(
        [
            ScriptedTurn(
                tool_calls=[
                    ScriptedToolCall("plan_purchase_requisition", {"category": "software"})
                ]
            ),
            ScriptedTurn(
                tool_calls=[
                    ScriptedToolCall(
                        "plan_purchase_requisition", dict(COMPLETE_REQUISITION_ARGS)
                    )
                ]
            ),
            ScriptedTurn(content="草稿已生成，总额 15000 元。"),
        ]
    )
    h = make_harness(llm)
    outcome = await h.orchestrator.run(
        RunRequest(principal=make_principal(), user_text="帮我采购一套设计协作软件")
    )

    assert outcome.finish_reason == "stop"
    assert "15000" in outcome.text
    assert [tc.ok for tc in outcome.tool_calls] == [False, True]

    # 第二轮的 prompt 里必须能看到字段级错误说明，模型才有依据修正。
    repair_prompt = str(llm.calls[1])
    assert "missing_arguments" in repair_prompt
    assert "field_issues" in repair_prompt
    assert "needed_by" in repair_prompt


async def test_repeated_unrepairable_args_stop_and_ask_the_user() -> None:
    """反复填不对时不该无限重试，而应转为向用户提问。"""
    bad = ScriptedTurn(
        tool_calls=[ScriptedToolCall("plan_purchase_requisition", {"category": "software"})]
    )
    llm = FakeLLM([bad, bad, bad, bad, bad, bad])
    h = make_harness(llm)
    outcome = await h.orchestrator.run(
        RunRequest(principal=make_principal(), user_text="帮我采购软件")
    )
    assert outcome.finish_reason == "tool_args_unrecoverable"
    assert "需要你补充" in outcome.text
    assert "采购事由" in outcome.text or "需求到货" in outcome.text


async def test_permission_denied_is_not_retried() -> None:
    from app.core.security.principal import Scope

    llm = FakeLLM(
        [
            ScriptedTurn(
                tool_calls=[
                    ScriptedToolCall(
                        "plan_purchase_requisition", dict(COMPLETE_REQUISITION_ARGS)
                    )
                ]
            )
        ]
    )
    h = make_harness(llm)
    limited = make_principal(scopes=frozenset({Scope.CHAT}))
    outcome = await h.orchestrator.run(
        RunRequest(principal=limited, user_text="帮我采购软件")
    )
    assert outcome.finish_reason == "tool_failed"
    assert "无法完成" in outcome.text
    assert len(llm.calls) == 1, "不可修复的失败不应触发重试"


# --------------------------------------------------------------------- 确认门


async def _reach_pending_submit(monkeypatch: pytest.MonkeyPatch | None = None):
    llm = FakeLLM(
        [
            ScriptedTurn(
                tool_calls=[
                    ScriptedToolCall(
                        "plan_purchase_requisition", dict(COMPLETE_REQUISITION_ARGS)
                    )
                ]
            ),
            ScriptedTurn(content="草稿已生成。"),
            ScriptedTurn(
                tool_calls=[ScriptedToolCall("submit_purchase_approval", {})]
            ),
        ],
        complete_reply="无",
    )
    h = make_harness(llm)
    p = make_principal()

    first = await h.orchestrator.run(
        RunRequest(principal=p, user_text="帮我生成请购单：设计协作软件，15000 元，9月15日要")
    )
    submit = await h.orchestrator.run(
        RunRequest(principal=p, user_text="提交审批", session_id=first.session_id)
    )
    return h, p, submit


async def test_write_tool_is_gated_not_executed() -> None:
    """核心安全属性：模型调用提交工具时不执行，只登记待确认动作。"""
    h, p, outcome = await _reach_pending_submit()

    assert outcome.finish_reason == "confirmation_required"
    assert outcome.pending_action is not None
    assert outcome.pending_action.tool_name == "submit_purchase_approval"
    assert outcome.pending_action.requires_explicit_confirmation is True

    # 确认摘要必须包含金额与审批链，且明确说明尚未提交。
    preview = outcome.pending_action.preview
    assert "15000.00" in preview
    assert "审批路径" in preview
    assert "确认后" in preview

    # 还没有审批单号产生。
    state = await h.sessions.load(outcome.session_id, tenant_id=p.tenant_id)
    assert state is not None
    assert state.draft.submitted_approval_no is None

    events = await h.audit.recent(tenant_id=p.tenant_id, limit=50)
    assert any(e.action is AuditAction.ACTION_PENDING for e in events)
    assert not any(e.action is AuditAction.APPROVAL_SUBMITTED for e in events)


async def test_model_is_told_not_to_claim_submission() -> None:
    h, p, outcome = await _reach_pending_submit()
    # 回复文案是确定性拼装的预览，不是模型自由生成的"已提交"。
    assert "已提交" not in outcome.text.replace("提交后", "")
    assert outcome.text.startswith("即将提交")


async def test_confirm_executes_once_and_replays_on_retry() -> None:
    h, p, outcome = await _reach_pending_submit()
    action_id = outcome.pending_action.action_id  # type: ignore[union-attr]

    first = await h.orchestrator.confirm_action(
        p, ConfirmActionRequest(action_id=action_id), trace_id="t-confirm"
    )
    assert first.status == "confirmed"
    assert first.replayed is False
    approval_no = first.result["approval"]["approval_no"]  # type: ignore[index]
    assert approval_no.startswith("APR-")

    state = await h.sessions.load(outcome.session_id, tenant_id=p.tenant_id)
    assert state is not None
    assert state.draft.submitted_approval_no == approval_no
    # 确认后槽位被锁定，后续轮次不能静默改金额。
    assert "unit_price_cny" in state.draft.confirmed_fields()

    events = await h.audit.recent(tenant_id=p.tenant_id, limit=50)
    assert any(e.action is AuditAction.APPROVAL_SUBMITTED for e in events)


async def test_double_confirm_is_idempotent() -> None:
    """网络重试或用户重复点击不应产生第二张请购单。"""
    h, p, outcome = await _reach_pending_submit()
    action_id = outcome.pending_action.action_id  # type: ignore[union-attr]

    first = await h.orchestrator.confirm_action(
        p, ConfirmActionRequest(action_id=action_id), trace_id="t1"
    )
    # 第一次确认后 pending 已删除，重放要靠幂等键；这里重新登记同一个动作模拟重试。
    await h.confirmations.put(
        (await _rebuild_pending(h, p, outcome, action_id))
    )
    second = await h.orchestrator.confirm_action(
        p, ConfirmActionRequest(action_id=action_id), trace_id="t2"
    )

    assert second.replayed is True
    assert (
        second.result["approval"]["approval_no"]  # type: ignore[index]
        == first.result["approval"]["approval_no"]  # type: ignore[index]
    )


async def _rebuild_pending(h, p, outcome, action_id):
    """复制一份同 action_id、同幂等键的待确认动作，用于模拟重复确认。"""
    from app.core.harness.confirmation import PendingAction
    from app.core.tools.builtin import build_submit_idempotency_key
    from app.core.tools.spec import ToolContext
    from app.core.memory.profile import ClaimsProfileProvider
    from app.core.clock import business_today
    from app.domain.procurement.policy import default_procurement_policy

    state = await h.sessions.load(outcome.session_id, tenant_id=p.tenant_id)
    profile = await ClaimsProfileProvider().get(p)
    ctx = ToolContext(
        principal=p,
        profile=profile,
        draft=state.draft,
        policy=default_procurement_policy(),
        session_id=state.session_id,
        trace_id="t2",
        today=business_today(),
    )
    return PendingAction(
        action_id=action_id,
        tool_name="submit_purchase_approval",
        arguments={},
        preview="重复确认",
        session_id=state.session_id,
        tenant_id=p.tenant_id,
        subject_id=p.subject_id,
        idempotency_key=build_submit_idempotency_key(ctx),
        draft_fingerprint=state.draft.fingerprint(),
    )


async def test_reject_cancels_without_submitting() -> None:
    h, p, outcome = await _reach_pending_submit()
    action_id = outcome.pending_action.action_id  # type: ignore[union-attr]

    result = await h.orchestrator.confirm_action(
        p,
        ConfirmActionRequest(action_id=action_id, decision=ConfirmDecision.REJECT),
        trace_id="t-reject",
    )
    assert result.status == "rejected"

    state = await h.sessions.load(outcome.session_id, tenant_id=p.tenant_id)
    assert state is not None and state.draft.submitted_approval_no is None

    events = await h.audit.recent(tenant_id=p.tenant_id, limit=50)
    assert any(e.action is AuditAction.ACTION_REJECTED for e in events)


async def test_other_user_cannot_confirm_someone_elses_action() -> None:
    h, p, outcome = await _reach_pending_submit()
    action_id = outcome.pending_action.action_id  # type: ignore[union-attr]
    intruder = make_principal(subject_id="E99999")

    with pytest.raises(HTTPException) as excinfo:
        await h.orchestrator.confirm_action(
            intruder, ConfirmActionRequest(action_id=action_id), trace_id="t"
        )
    assert excinfo.value.status_code == 403


async def test_draft_changed_after_preview_blocks_confirmation() -> None:
    """用户确认的是当时那份内容；确认前草稿被改过就必须重新生成。"""
    h, p, outcome = await _reach_pending_submit()
    action_id = outcome.pending_action.action_id  # type: ignore[union-attr]

    state = await h.sessions.load(outcome.session_id, tenant_id=p.tenant_id)
    assert state is not None
    from app.core.memory.draft import SlotSource

    state.draft.merge({"unit_price_cny": "99999"}, source=SlotSource.USER, turn=99)
    await h.sessions.save(state)

    with pytest.raises(HTTPException) as excinfo:
        await h.orchestrator.confirm_action(
            p, ConfirmActionRequest(action_id=action_id), trace_id="t"
        )
    assert excinfo.value.status_code == 409


async def test_submit_without_draft_is_rejected_with_guidance() -> None:
    llm = FakeLLM(
        [
            ScriptedTurn(tool_calls=[ScriptedToolCall("submit_purchase_approval", {})]),
            ScriptedTurn(content="请先补充采购信息。"),
        ]
    )
    h = make_harness(llm)
    outcome = await h.orchestrator.run(
        RunRequest(principal=make_principal(), user_text="直接提交审批")
    )
    assert outcome.pending_action is None
    assert [tc.ok for tc in outcome.tool_calls] == [False]
    assert "plan_purchase_requisition" in str(llm.calls[-1])


# --------------------------------------------------------------------- 预算与降级


async def test_budget_exhaustion_preserves_work_instead_of_discarding() -> None:
    """改造前触顶返回一句硬编码道歉并丢弃全部工具产出。"""
    plan_turn = ScriptedTurn(
        tool_calls=[
            ScriptedToolCall("plan_purchase_requisition", dict(COMPLETE_REQUISITION_ARGS))
        ]
    )
    llm = FakeLLM([plan_turn] * 12)
    h = make_harness(llm)
    h.orchestrator._policy = h.orchestrator._policy  # noqa: SLF001 - 保持默认制度

    import app.core.harness.budget as budget_mod

    original = budget_mod.RunBudget

    class TinyBudget(original):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(max_iterations=2, max_tool_calls=99, max_total_tokens=10**9)

    import app.agent.orchestrator as orch_mod

    orch_mod.RunBudget = TinyBudget  # type: ignore[assignment]
    try:
        outcome = await h.orchestrator.run(
            RunRequest(principal=make_principal(), user_text="帮我生成请购单")
        )
    finally:
        orch_mod.RunBudget = original  # type: ignore[assignment]

    assert outcome.finish_reason == "budget_exhausted"
    assert "已达到最大推理轮次" in outcome.text
    # 关键：已生成的请购单内容必须保留下来。
    assert "15000.00" in outcome.text
    assert "尚未提交任何审批" in outcome.text


async def test_llm_unavailable_falls_back_to_rule_based_answer() -> None:
    """模型挂了也要给出确定性的制度答复，而不是 500。"""
    llm = FakeLLM([], fail_with="ConnectionError: upstream down")
    h = make_harness(llm)
    outcome = await h.orchestrator.run(
        RunRequest(principal=make_principal(), user_text="软件采购的审批制度和额度是多少")
    )

    assert outcome.degraded is True
    assert outcome.finish_reason == "degraded"
    assert "模型服务不可用" in outcome.text
    assert "审批路径" in outcome.text
    assert "多方报价阈值" in outcome.text

    events = await h.audit.recent(tenant_id="acme", limit=20)
    assert any(e.action is AuditAction.DEGRADED_RESPONSE for e in events)


# --------------------------------------------------------------------- 流式


async def test_stream_emits_intermediate_tool_events_before_content() -> None:
    """用户在多轮工具调用期间应能看到中间态，而不是盯着空白屏。"""
    llm = FakeLLM(
        [
            ScriptedTurn(
                tool_calls=[
                    ScriptedToolCall(
                        "plan_purchase_requisition", dict(COMPLETE_REQUISITION_ARGS)
                    )
                ]
            ),
            ScriptedTurn(content="草稿已生成，总额 15000 元。"),
        ]
    )
    h = make_harness(llm)

    types: list[StreamChunkType] = []
    contents: list[str] = []
    async for chunk in h.orchestrator.stream(
        RunRequest(principal=make_principal(), user_text="帮我生成请购单")
    ):
        types.append(chunk.type)
        if chunk.type is StreamChunkType.CONTENT and chunk.delta:
            contents.append(chunk.delta)

    assert types[0] is StreamChunkType.STATUS
    assert StreamChunkType.TOOL_CALL in types
    assert StreamChunkType.TOOL_RESULT in types
    assert types[-1] is StreamChunkType.DONE
    # 内容是分片推送的，不是一次性整段。
    assert len(contents) > 1
    assert "".join(contents) == "草稿已生成，总额 15000 元。"

    tool_call_pos = types.index(StreamChunkType.TOOL_CALL)
    first_content_pos = types.index(StreamChunkType.CONTENT)
    assert tool_call_pos < first_content_pos, "工具进度事件应先于最终内容到达"


async def test_stream_reports_usage_and_session_on_done() -> None:
    llm = FakeLLM([ScriptedTurn(content="好的。", prompt_tokens=120, completion_tokens=8)])
    h = make_harness(llm)
    done = None
    async for chunk in h.orchestrator.stream(
        RunRequest(principal=make_principal(), user_text="你好")
    ):
        if chunk.type is StreamChunkType.DONE:
            done = chunk
    assert done is not None
    assert done.session_id
    assert done.usage == {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128}
