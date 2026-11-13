"""工具层结构化错误：让工具失败变成模型能读懂并自我修正的观察值，而不是 HTTP 500。

关键设计：`ToolFailure.to_observation()` 输出的不只是"错了"，还包含**哪个字段错了、期望
什么格式、示例是什么**。没有这个，模型只会原样重试一次然后放弃。
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolErrorCode(str, Enum):
    MISSING_ARGUMENTS = "missing_arguments"
    INVALID_ARGUMENTS = "invalid_arguments"
    MALFORMED_ARGUMENTS_JSON = "malformed_arguments_json"
    UNKNOWN_TOOL = "unknown_tool"
    PERMISSION_DENIED = "permission_denied"
    CONFIRMATION_REQUIRED = "confirmation_required"
    #: 前置步骤没做（例如还没生成请购单草稿就想提交）。模型可以自己补上前置调用。
    PRECONDITION_NOT_MET = "precondition_not_met"
    NOT_FOUND = "not_found"
    UPSTREAM_ERROR = "upstream_error"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERNAL = "internal"


#: 模型自修正有意义的错误码；其余直接终止该工具的重试。
REPAIRABLE_CODES: frozenset[ToolErrorCode] = frozenset(
    {
        ToolErrorCode.MISSING_ARGUMENTS,
        ToolErrorCode.INVALID_ARGUMENTS,
        ToolErrorCode.MALFORMED_ARGUMENTS_JSON,
        ToolErrorCode.UNKNOWN_TOOL,
        ToolErrorCode.PRECONDITION_NOT_MET,
    }
)


class FieldIssue(BaseModel):
    """单个字段的问题描述，直接面向模型的下一轮修正。"""

    field: str
    problem: str
    expected: str | None = None
    example: str | None = None


class ToolError(Exception):
    """工具内部抛出的受控错误。编排层捕获后转成观察值回喂模型。"""

    def __init__(
        self,
        code: ToolErrorCode,
        message: str,
        *,
        field_issues: list[FieldIssue] | None = None,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field_issues = field_issues or []
        self.hint = hint
        self.details = details or {}

    @property
    def repairable(self) -> bool:
        return self.code in REPAIRABLE_CODES


class ToolFailure(BaseModel):
    """序列化后的工具失败信息。"""

    tool: str
    code: ToolErrorCode
    message: str
    field_issues: list[FieldIssue] = Field(default_factory=list)
    hint: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def repairable(self) -> bool:
        return self.code in REPAIRABLE_CODES

    @classmethod
    def from_error(cls, tool: str, err: ToolError) -> "ToolFailure":
        return cls(
            tool=tool,
            code=err.code,
            message=err.message,
            field_issues=err.field_issues,
            hint=err.hint,
            details=err.details,
        )

    def to_observation(self) -> str:
        """给模型看的观察值。刻意用 JSON，便于模型定位字段而不是猜自然语言。"""
        payload: dict[str, Any] = {
            "ok": False,
            "error_code": self.code.value,
            "message": self.message,
        }
        if self.field_issues:
            payload["field_issues"] = [i.model_dump(exclude_none=True) for i in self.field_issues]
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        payload["retry_allowed"] = self.repairable
        if self.repairable:
            payload["next_step"] = (
                "请修正上述字段后重新调用同一工具。若信息来自用户且确实缺失，"
                "改为向用户提问，不要臆造数值。"
            )
        else:
            payload["next_step"] = "不要重复调用该工具；向用户说明当前无法完成的原因。"
        return json.dumps(payload, ensure_ascii=False)


class ToolSuccess(BaseModel):
    """工具成功结果。`rendered` 是喂给模型的文本，`data` 保留结构化值供审计与前端使用。"""

    tool: str
    data: dict[str, Any] = Field(default_factory=dict)
    rendered: str = ""

    def to_observation(self) -> str:
        if self.rendered:
            return json.dumps({"ok": True, "result": self.rendered}, ensure_ascii=False)
        return json.dumps({"ok": True, "data": self.data}, ensure_ascii=False)


ToolOutcome = ToolSuccess | ToolFailure


def missing_arguments_error(tool: str, fields: dict[str, FieldIssue]) -> ToolError:
    """构造"必填字段缺失"错误，附带每个字段的期望格式与示例。"""
    names = "、".join(fields.keys())
    return ToolError(
        ToolErrorCode.MISSING_ARGUMENTS,
        f"调用 {tool} 缺少必填字段：{names}",
        field_issues=list(fields.values()),
        hint="这些字段无法推断时应向用户提问，不要填默认值。",
    )
