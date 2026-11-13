"""OpenAI 兼容的 mock 模型服务，用于 HTTP/SSE 压测。

它不生成智能回答，只复现协议：
- 流式 `chat.completion.chunk`（content 或 tool_calls）；
- 非流式 `chat.completion`（意图分类、摘要等内部调用）；
- 固定时延，便于比较"路由 on/off 时模型被调用几次"。

用法：
    uvicorn benchmarks.mock_openai_server:app --port 18081
    MOCK_LLM_DELAY_MS=50 uvicorn benchmarks.mock_openai_server:app --port 18081

然后把应用的 DASHSCOPE_API_BASE 指到 http://127.0.0.1:18081/v1。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Mock OpenAI API")

DELAY_MS = float(os.getenv("MOCK_LLM_DELAY_MS", "50"))
CHUNKS = int(os.getenv("MOCK_LLM_CHUNKS", "5"))
MODEL = os.getenv("MOCK_LLM_MODEL", "mock-qwen")

ANSWER = "按公司差旅标准，经理职级国内可订经济舱，北京住宿上限 800 元/晚。"


def _last_user(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names = set()
    for item in tools or []:
        function = item.get("function") if isinstance(item, dict) else None
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
    return names


def _has_tool_result(messages: list[dict[str, Any]]) -> bool:
    return any(message.get("role") == "tool" for message in messages)


def _guess_intent(text: str) -> str:
    if "标准" in text or "差标" in text or "制度" in text:
        return "policy"
    if "安排" in text or "行程" in text:
        return "trip_planning"
    if "订" in text or "预订" in text:
        return "booking"
    if "审批" in text:
        return "approval"
    return "general"


def _chunks(text: str) -> list[str]:
    size = max(1, len(text) // max(1, CHUNKS))
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _chunk(
    *,
    content: str | None = None,
    tool_name: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "id": f"chatcmpl-mock-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [],
    }
    if tool_name:
        payload["choices"] = [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_mock_1",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": None,
            }
        ]
    elif content is not None:
        payload["choices"] = [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ]
    if finish_reason:
        payload["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
    if usage:
        payload["usage"] = usage
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None):
    await asyncio.sleep(DELAY_MS / 1000.0)
    text = _last_user(messages)
    names = _tool_names(tools)

    if names and not _has_tool_result(messages):
        tool_name = None
        if "get_travel_policy" in names and ("标准" in text or "差标" in text):
            tool_name = "get_travel_policy"
        elif "plan_trip" in names and ("安排" in text or "行程" in text):
            tool_name = "plan_trip"
        elif "book_trip" in names and ("订" in text or "预订" in text):
            tool_name = "book_trip"
        if tool_name:
            yield _chunk(tool_name=tool_name)
            yield _chunk(finish_reason="tool_calls", usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
            yield "data: [DONE]\n\n"
            return

    pieces = _chunks(ANSWER)
    for piece in pieces:
        yield _chunk(content=piece)
        await asyncio.sleep(DELAY_MS / 1000.0 / max(1, len(pieces)))
    yield _chunk(
        finish_reason="stop",
        usage={"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
    )
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    tools = body.get("tools")
    stream = bool(body.get("stream"))
    system = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system")

    if "意图分类器" in system:
        content = json.dumps(
            {"intent_slug": _guess_intent(_last_user(messages)), "confidence": 0.9, "rationale": "mock"},
            ensure_ascii=False,
        )
    elif "摘要器" in system:
        content = json.dumps(
            {
                "goals": ["mock summary"],
                "confirmed_facts": {},
                "amounts_cny": {},
                "dates": {},
                "open_questions": [],
                "preferences": "",
            },
            ensure_ascii=False,
        )
    else:
        content = ANSWER

    if not stream:
        return JSONResponse(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            }
        )

    return StreamingResponse(
        _stream_chat(messages, tools),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/models")
async def models():
    return {"data": [{"id": MODEL, "object": "model"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "delay_ms": DELAY_MS}
