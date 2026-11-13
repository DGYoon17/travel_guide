"""从 pydantic 入参模型生成 function-calling 的 JSON Schema。

替代原来手写 95 行 schema 的做法：手写 schema 与 pydantic 模型是两份真相，改了一处
另一处静默漂移，模型按旧 schema 填参、校验按新模型执行，失败原因极难定位。

同时做三件适配 DashScope 兼容模式的处理：
1. 内联 `$ref` / `$defs`（部分兼容实现不解引用）；
2. 把 `anyOf: [T, null]` 折叠成 `T`（可选性由 `required` 表达）；
3. 剔除服务端注入字段（主体身份等），模型看不到就填不了。
"""

from __future__ import annotations

import copy
from typing import Any, Iterable

from pydantic import BaseModel

#: 对模型无用、只增加 prompt token 的键。
_DROPPED_KEYS: frozenset[str] = frozenset({"title", "additionalProperties"})
_MAX_DEPTH = 12


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    key = ref.rsplit("/", 1)[-1]
    return copy.deepcopy(defs.get(key, {}))


def _clean(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    if depth > _MAX_DEPTH or not isinstance(node, dict):
        return node

    if "$ref" in node:
        merged = _resolve_ref(node["$ref"], defs)
        for key, value in node.items():
            if key != "$ref":
                merged.setdefault(key, value)
        return _clean(merged, defs, depth + 1)

    if "allOf" in node and len(node["allOf"]) == 1:
        merged = _clean(node["allOf"][0], defs, depth + 1)
        if isinstance(merged, dict):
            for key, value in node.items():
                if key != "allOf":
                    merged.setdefault(key, value)
            return _clean_keys(merged, defs, depth)
        return merged

    if "anyOf" in node:
        branches = [b for b in node["anyOf"] if not (isinstance(b, dict) and b.get("type") == "null")]
        if len(branches) == 1:
            merged = _clean(branches[0], defs, depth + 1)
            if isinstance(merged, dict):
                for key, value in node.items():
                    if key != "anyOf":
                        merged.setdefault(key, value)
                return _clean_keys(merged, defs, depth)
            return merged
        node = {**node, "anyOf": [_clean(b, defs, depth + 1) for b in branches]}

    return _clean_keys(node, defs, depth)


def _clean_keys(node: dict[str, Any], defs: dict[str, Any], depth: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED_KEYS or key in {"$defs", "$ref"}:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {k: _clean(v, defs, depth + 1) for k, v in value.items()}
        elif key == "items":
            out[key] = _clean(value, defs, depth + 1)
        elif key in {"anyOf", "oneOf"} and isinstance(value, list):
            out[key] = [_clean(v, defs, depth + 1) for v in value]
        else:
            out[key] = value
    return out


def build_parameters_schema(
    model: type[BaseModel],
    *,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """生成 function 的 `parameters` 对象，排除服务端注入字段。"""
    excluded = frozenset(exclude)
    raw = model.model_json_schema()
    defs: dict[str, Any] = raw.get("$defs", {})

    properties: dict[str, Any] = {}
    for name, sub in (raw.get("properties") or {}).items():
        if name in excluded:
            continue
        cleaned = _clean(sub, defs)
        properties[name] = cleaned if isinstance(cleaned, dict) else {}

    required = [name for name in raw.get("required", []) if name not in excluded]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def build_function_definition(
    name: str,
    description: str,
    model: type[BaseModel],
    *,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """生成 OpenAI 兼容的 `tools[]` 单项。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": build_parameters_schema(model, exclude=exclude),
        },
    }
