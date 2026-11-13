from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from app.agent.travel_orchestrator import RunRequest, TravelOrchestrator
from app.core.logging import get_logger
from app.core.security.auth import require_scopes
from app.core.security.principal import Principal, Scope
from app.domain.schemas import (
    ChatRequest,
    ChatResponse,
    StreamChunk,
    StreamChunkType,
)

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])


def get_orchestrator(request: Request) -> TravelOrchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="编排器未初始化（通常是缺少 DASHSCOPE_API_KEY 或 OPENAI_API_KEY）",
        )
    return orchestrator


def _trace_id(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex


@router.post("/chat", response_model=None)
async def chat(
    body: ChatRequest,
    request: Request,
    response: Response,
    principal: Principal = Depends(require_scopes(Scope.CHAT)),
    orchestrator: TravelOrchestrator = Depends(get_orchestrator),
) -> Union[ChatResponse, StreamingResponse]:
    user_text = body.user_text()
    if not user_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="缺少本轮用户输入（message 或 messages 中的最后一条 user 消息）",
        )

    trace_id = _trace_id(request)
    run_request = RunRequest(
        principal=principal,
        user_text=user_text,
        session_id=body.session_id,
        trace_id=trace_id,
    )

    if body.stream:

        async def event_gen() -> AsyncIterator[str]:
            index = 0
            try:
                async for chunk in orchestrator.stream(run_request):
                    index = chunk.index
                    payload = chunk.model_dump(mode="json", exclude_none=True)
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except HTTPException as exc:
                err = StreamChunk(
                    type=StreamChunkType.ERROR,
                    index=index + 1,
                    error=str(exc.detail),
                    trace_id=trace_id,
                )
                yield f"data: {json.dumps(err.model_dump(mode='json', exclude_none=True), ensure_ascii=False)}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception("chat.stream_failed", trace_id=trace_id)
                err = StreamChunk(
                    type=StreamChunkType.ERROR,
                    index=index + 1,
                    error=f"服务内部错误（trace {trace_id}）",
                    trace_id=trace_id,
                )
                _ = exc
                yield f"data: {json.dumps(err.model_dump(mode='json', exclude_none=True), ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "X-Trace-Id": trace_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    outcome = await orchestrator.run(run_request)
    response.headers["X-Trace-Id"] = trace_id
    return ChatResponse(
        id=f"chatcmpl_{trace_id[:16]}",
        created=int(time.time()),
        model=orchestrator.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": outcome.text},
                "finish_reason": outcome.finish_reason,
            }
        ],
        usage=outcome.usage or None,
        session_id=outcome.session_id,
        trace_id=outcome.trace_id,
        tool_calls=outcome.tool_calls,
        pending_action=outcome.pending_action,
        missing_fields=outcome.missing_fields,
        degraded=outcome.degraded,
        finish_reason=outcome.finish_reason,
    )
