"""
各 Agent 与对话状态对应的提示词模板库。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.core.prompt.state_machine import ConversationState


@dataclass(frozen=True)
class PromptBundle:
    """单场景下的系统提示与用户消息前缀。"""

    system: str
    user_prefix: str = ""


class PromptTemplateLibrary:
    """按子域与对话状态组装提示片段。"""

    def __init__(self) -> None:
        self._base_system = (
            "你是企业采购与审批智能助手，回答需准确、合规、可执行。"
            "涉及预算、合同、供应商准入和审批要求时必须说明依据。"
        )

    def system_for_state(self, state: ConversationState) -> str:
        extras: dict[ConversationState, str] = {
            ConversationState.GREETING: "当前为初次问候，简洁自我介绍并询问采购需求。",
            ConversationState.COLLECTING_INFO: "当前需收集采购要素：品类、数量、预算、需求日期、供应商偏好。一次只追问缺失项。",
            ConversationState.PLANNING: "当前正在规划采购方案，给出请购建议、审批路径与风险提示。",
            ConversationState.CONFIRMING: "当前等待用户确认，列出摘要并请用户明确确认或修改。",
            ConversationState.EXECUTING: "当前执行请购提交或系统操作，逐步反馈进度与结果。",
            ConversationState.COMPLETED: "本轮任务已完成，可询问是否需要其它帮助。",
        }
        return f"{self._base_system}\n{extras[state]}"

    def react_tools_section(self, tool_names: tuple[str, ...]) -> str:
        names = ", ".join(tool_names) if tool_names else "（暂无工具）"
        return f"可用工具名称：{names}。每次只选择一个工具或给出最终答复。"

    def planner_decomposition(self, goal: str) -> PromptBundle:
        system = (
            f"{self._base_system}\n"
            "你是任务分解专家。将用户目标拆成有序、可验证的步骤，每步一行，使用动词开头。"
        )
        user = f"用户目标：{goal}\n请输出步骤列表（纯文本，每行一步）。"
        return PromptBundle(system=system, user_prefix=user)

    def reflection_critique(self, draft: str, criteria: Mapping[str, Any]) -> PromptBundle:
        crit = ", ".join(f"{k}={v}" for k, v in criteria.items())
        system = (
            f"{self._base_system}\n"
            "你是审查员。根据给定标准评估草稿，指出问题并给出修订建议。"
        )
        user = f"草稿：\n{draft}\n\n标准：{crit}\n输出：问题列表 + 修订版摘要。"
        return PromptBundle(system=system, user_prefix=user)

    def subagent_system_prompt(self, domain: str) -> str:
        domain_hints = {
            "procurement_planning": "专注采购方案、金额拆解、审批顺序与供应商选择。",
            "info_query": "专注事实查询与引用来源，不确定则说明。",
            "policy": "专注采购制度与合规解读，引用制度条款。",
            "approval": "专注审批流程、状态说明与提交条件，不虚构系统结果。",
            "rag": "基于检索片段回答，缺失则说明并建议下一步。",
            "general": "通用采购协助，保持专业与礼貌。",
        }
        hint = domain_hints.get(domain, domain_hints["general"])
        return f"{self._base_system}\n{hint}"

    def confirmation_summary(self, bullets: tuple[str, ...]) -> str:
        body = "\n".join(f"- {b}" for b in bullets) if bullets else "- （无摘要项）"
        return f"请用户确认下列摘要；若需修改请说明具体项：\n{body}"

    def policy_explain_prefix(self, topic: str) -> str:
        return f"采购制度主题：{topic}。回答须区分「硬性规定」与「建议性说明」。"

    def streaming_ack(self, partial_topic: str) -> str:
        return f"正在处理：{partial_topic}…"
