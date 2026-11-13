"""工具注册表：function-calling 定义、参数校验、权限与超时的唯一事实源。

原来的问题是"三份真相"：手写的 function schema、pydantic 领域模型、编排器里的 if/else
分发。这里合成一份 —— 注册一个 `ToolSpec`，schema、校验、注入、权限、确认门、审计全部
由它派生。
"""

from __future__ import annotations

import asyncio
import importlib
import json
import pkgutil
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ValidationError

from app.config import settings
from app.core.logging import get_logger
from app.core.tools.errors import (
    FieldIssue,
    ToolError,
    ToolErrorCode,
    ToolFailure,
    ToolSuccess,
)
from app.core.tools.schema import build_function_definition, build_parameters_schema
from app.core.tools.spec import ToolAccess, ToolContext, ToolReturn, ToolSpec

logger = get_logger(__name__)


class ToolRegistry:
    """注册并执行工具，导出 function-calling 定义与 MCP catalog。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------ 注册
    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            logger.warning("tool.re_registered", tool=spec.name)
        self._tools[spec.name] = spec
        return spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolError(
                ToolErrorCode.UNKNOWN_TOOL,
                f"不存在名为 {name} 的工具",
                hint=f"可用工具：{', '.join(sorted(self._tools)) or '（无）'}",
            )
        return self._tools[name]

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    # ------------------------------------------------------------------ 导出
    def function_definitions(self, *, access_levels: Iterable[ToolAccess] | None = None) -> list[dict[str, Any]]:
        """导出 OpenAI 兼容的 `tools[]`。注入字段已从 schema 中剔除。"""
        allowed = set(access_levels) if access_levels is not None else set(ToolAccess)
        return [
            build_function_definition(
                spec.name,
                spec.description,
                spec.args_model,
                exclude=spec.injected_fields,
            )
            for spec in self._tools.values()
            if spec.access in allowed
        ]

    def mcp_tool_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "inputSchema": build_parameters_schema(spec.args_model, exclude=spec.injected_fields),
                "annotations": {
                    "access": spec.access.value,
                    "requiresConfirmation": spec.needs_confirmation,
                },
            }
            for spec in self._tools.values()
        ]

    def discover_package(self, package: str) -> int:
        """导入包内定义了 `register_tools(registry)` 的子模块。"""
        pkg = importlib.import_module(package)
        path = getattr(pkg, "__path__", None)
        if path is None:
            return 0
        count = 0
        for _finder, modname, _ispkg in pkgutil.walk_packages(path, pkg.__name__ + "."):
            try:
                mod = importlib.import_module(modname)
            except Exception as exc:  # noqa: BLE001
                logger.warning("tool.discovery_import_failed", module=modname, error=str(exc))
                continue
            fn = getattr(mod, "register_tools", None)
            if callable(fn):
                fn(self)
                count += 1
        return count

    # ------------------------------------------------------------------ 执行
    def parse_arguments(self, spec: ToolSpec, raw_arguments: str | Mapping[str, Any] | None, ctx: ToolContext) -> BaseModel:
        """解析 + 注入 + 校验。失败一律抛 `ToolError`，由调用方转成观察值回喂模型。"""
        if isinstance(raw_arguments, Mapping):
            payload: dict[str, Any] = dict(raw_arguments)
        else:
            text = (raw_arguments or "").strip() or "{}"
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ToolError(
                    ToolErrorCode.MALFORMED_ARGUMENTS_JSON,
                    f"工具参数不是合法 JSON：{exc.msg}（位置 {exc.pos}）",
                    hint="请输出严格的 JSON 对象，不要包含注释、尾随逗号或 Markdown 代码块。",
                ) from exc
            if not isinstance(parsed, dict):
                raise ToolError(
                    ToolErrorCode.MALFORMED_ARGUMENTS_JSON,
                    "工具参数必须是 JSON 对象",
                    hint='形如 {"item_name": "...", "quantity": 1}',
                )
            payload = parsed

        # L2 回填：模型漏填的字段用草稿里已确认/已知的值补上，减少无谓的重复追问。
        if spec.draft_backfill:
            known = ctx.draft.values()
            model_fields = spec.args_model.model_fields
            for field_name, value in known.items():
                if field_name in model_fields and payload.get(field_name) in (None, ""):
                    payload[field_name] = value

        # 剔除模型试图自行填写的注入字段，再用可信来源覆盖。
        injected = ctx.profile.as_injected_arguments()
        for name in spec.injected_fields:
            if name in payload:
                logger.warning("tool.injected_field_from_model_dropped", tool=spec.name, field=name)
                payload.pop(name, None)
            if name in injected:
                payload[name] = injected[name]

        try:
            return spec.args_model.model_validate(payload)
        except ValidationError as exc:
            raise _validation_to_tool_error(spec, exc) from exc

    async def invoke(
        self,
        name: str,
        raw_arguments: str | Mapping[str, Any] | None,
        ctx: ToolContext,
    ) -> ToolSuccess | ToolFailure:
        """执行工具。所有异常都被转成 `ToolFailure`，不向上冒泡成 HTTP 错误。"""
        try:
            spec = self.get(name)
        except ToolError as err:
            return ToolFailure.from_error(name, err)

        missing_scopes = ctx.principal.missing_scopes(spec.required_scopes)
        if missing_scopes:
            return ToolFailure.from_error(
                name,
                ToolError(
                    ToolErrorCode.PERMISSION_DENIED,
                    f"当前用户缺少调用 {name} 所需权限：{', '.join(s.value for s in missing_scopes)}",
                    hint="向用户说明权限不足，并建议联系管理员开通，不要尝试其它工具绕过。",
                ),
            )

        try:
            args = self.parse_arguments(spec, raw_arguments, ctx)
        except ToolError as err:
            return ToolFailure.from_error(name, err)

        timeout = spec.timeout_seconds or settings.tool_timeout_seconds
        try:
            result = await asyncio.wait_for(spec.handler(args, ctx), timeout=timeout)
        except asyncio.TimeoutError:
            return ToolFailure.from_error(
                name,
                ToolError(
                    ToolErrorCode.TIMEOUT,
                    f"工具 {name} 执行超过 {timeout:.0f} 秒被中止",
                    hint="告知用户下游系统响应超时，可稍后重试；不要假设操作已成功。",
                ),
            )
        except ToolError as err:
            return ToolFailure.from_error(name, err)
        except Exception as exc:  # noqa: BLE001
            logger.exception("tool.unhandled_exception", tool=name)
            return ToolFailure.from_error(
                name,
                ToolError(
                    ToolErrorCode.INTERNAL,
                    f"工具 {name} 执行异常：{type(exc).__name__}",
                    hint="向用户说明系统异常，不要重复调用同一工具。",
                ),
            )

        from app.core.memory.slots import SlotSource

        if result.draft_updates:
            ctx.draft.merge(result.draft_updates, source=SlotSource.INFERRED, turn=ctx.turn)
        if result.draft_facts:
            ctx.draft.merge(result.draft_facts, source=SlotSource.TOOL, turn=ctx.turn)
        if result.draft_attributes:
            changed = False
            for attr, value in result.draft_attributes.items():
                if hasattr(ctx.draft, attr):
                    setattr(ctx.draft, attr, value)
                    changed = True
                else:
                    logger.warning("tool.unknown_draft_attribute", tool=name, attribute=attr)
            if changed:
                ctx.draft.version += 1

        return ToolSuccess(tool=name, data=result.data, rendered=result.rendered)


def _validation_to_tool_error(spec: ToolSpec, exc: ValidationError) -> ToolError:
    """把 pydantic 校验错误翻译成模型能照着改的字段级说明。"""
    schema = build_parameters_schema(spec.args_model, exclude=spec.injected_fields)
    properties: dict[str, Any] = schema.get("properties", {})

    issues: list[FieldIssue] = []
    missing = False
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "__root__") or "(root)"
        prop = properties.get(loc, {})
        expected_parts: list[str] = []
        if "enum" in prop:
            expected_parts.append("取值之一：" + "、".join(str(v) for v in prop["enum"]))
        elif prop.get("type"):
            expected_parts.append(f"类型 {prop['type']}")
        if prop.get("description"):
            expected_parts.append(prop["description"])
        if err.get("type") == "missing":
            missing = True
        issues.append(
            FieldIssue(
                field=loc,
                problem=err.get("msg", "校验失败"),
                expected="；".join(expected_parts) or None,
                example=str(prop["enum"][0]) if prop.get("enum") else None,
            )
        )

    code = ToolErrorCode.MISSING_ARGUMENTS if missing else ToolErrorCode.INVALID_ARGUMENTS
    names = "、".join(dict.fromkeys(i.field for i in issues))
    return ToolError(
        code,
        f"调用 {spec.name} 的参数有误：{names}",
        field_issues=issues,
        hint=(
            "缺失字段若用户没提供，请直接向用户提问，不要填占位值或猜测的金额与日期。"
            if missing
            else "按 field_issues 的期望格式修正后重试。"
        ),
    )


__all__ = [
    "ToolRegistry",
    "ToolAccess",
    "ToolContext",
    "ToolReturn",
    "ToolSpec",
]
