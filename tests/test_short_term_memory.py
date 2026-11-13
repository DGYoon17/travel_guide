"""短期记忆的三个回归测试，对应改造前会导致线上错误的三个问题。"""

from __future__ import annotations

from app.core.memory.short_term import ShortTermMemory, StoredMessage, ToolCallRef


def _tool_turn(call_id: str, name: str = "check_procurement_policy") -> list[StoredMessage]:
    return [
        StoredMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCallRef(id=call_id, name=name, arguments='{"a":1}')],
        ),
        StoredMessage(role="tool", content='{"ok":true}', name=name, tool_call_id=call_id),
    ]


def test_summary_survives_window_trimming() -> None:
    """改造前：先摘要再按窗口截断，开头的 [历史摘要] 被直接裁走，等于白调一次 LLM。"""
    memory = ShortTermMemory(max_tokens=100000, max_groups=3)
    memory.set_summary("用户要采购设计协作软件，预算 15000 元。")
    for i in range(10):
        memory.append(StoredMessage(role="user", content=f"第 {i} 轮输入"))

    messages = memory.as_messages()
    assert messages[0]["role"] == "system"
    assert "[历史摘要]" in messages[0]["content"]
    # 窗口只保留 3 组普通消息，摘要不计入裁剪。
    assert len(messages) == 1 + 3


def test_tool_call_group_is_never_split() -> None:
    """改造前按单条 popleft，会把 assistant(tool_calls) 与 tool 结果拆开，接口直接 400。"""
    memory = ShortTermMemory(max_tokens=100000, max_groups=2)
    for i in range(4):
        for msg in _tool_turn(f"call_{i}"):
            memory.append(msg)

    messages = memory.as_messages()
    ids_declared: set[str] = set()
    ids_answered: set[str] = set()
    for msg in messages:
        for tc in msg.get("tool_calls", []) or []:
            ids_declared.add(tc["id"])
        if msg.get("tool_call_id"):
            ids_answered.add(msg["tool_call_id"])

    assert ids_declared == ids_answered, "每个 tool_call 都必须有配对的 tool 结果"
    assert ids_declared, "至少应保留一组完整的工具调用"


def test_multiple_tool_results_attach_to_same_group() -> None:
    memory = ShortTermMemory()
    memory.append(
        StoredMessage(
            role="assistant",
            tool_calls=[
                ToolCallRef(id="a", name="t1", arguments="{}"),
                ToolCallRef(id="b", name="t2", arguments="{}"),
            ],
        )
    )
    memory.append(StoredMessage(role="tool", content="r1", tool_call_id="a"))
    assert memory.as_messages() == [], "工具结果未收齐时不应把残缺组发给模型"

    memory.append(StoredMessage(role="tool", content="r2", tool_call_id="b"))
    messages = memory.as_messages()
    assert len(messages) == 3
    assert memory.group_count() == 1


def test_orphan_tool_result_is_dropped() -> None:
    """没有配对 assistant 消息的 tool 结果会被接口拒绝，直接丢弃。"""
    memory = ShortTermMemory()
    memory.append(StoredMessage(role="tool", content="orphan", tool_call_id="nope"))
    assert memory.as_messages() == []


def test_token_accounting_is_incremental_and_bounded() -> None:
    memory = ShortTermMemory(max_tokens=200, max_groups=100)
    for i in range(200):
        memory.append(StoredMessage(role="user", content="采购需求描述" * 20 + str(i)))
    assert memory.total_tokens() <= 200 or memory.group_count() == 1
    # 每条消息的 token 数被缓存，总量与逐条求和一致。
    recomputed = sum(
        memory._tokens_of(m) for m in memory.transcript_messages()  # noqa: SLF001
    )
    assert memory.total_tokens() == recomputed


def test_dump_load_roundtrip_preserves_summary_and_tool_groups() -> None:
    memory = ShortTermMemory()
    memory.set_summary("已确认：品类=software，单价=15000")
    for msg in _tool_turn("call_x"):
        memory.append(msg)
    memory.append(StoredMessage(role="assistant", content="草稿已生成"))

    restored = ShortTermMemory()
    restored.load(memory.dump())

    assert restored.summary_text() == memory.summary_text()
    assert restored.as_messages() == memory.as_messages()


def test_last_group_is_never_trimmed() -> None:
    """裁剪不能把当前这轮的用户输入丢掉。"""
    memory = ShortTermMemory(max_tokens=1, max_groups=1)
    memory.append(StoredMessage(role="user", content="我要采购一批笔记本电脑" * 50))
    assert memory.group_count() == 1
    assert memory.as_messages()
