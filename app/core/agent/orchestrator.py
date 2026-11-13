"""RAG 检索编排：意图 → 多路检索 → 生成。

改造前这个模块里还有一个 `AgentOrchestrator`：自带一套 `ShortTermMemory` 的进程内会话字典、
自研 ReAct 文本解析、以及把 `react_trace` 直接回传客户端的元数据。它从未被 API 层接入
（`main.py` 只装配 `ProcurementOrchestrator`），是纯死代码，且与主链路的记忆、工具注册表
互不相通 —— 两套骨架并存是排障时最费时间的一类问题。

现在主对话统一走 `app/agent/orchestrator.py` 的原生 function-calling 编排；本模块只保留
纯 RAG 管道，供知识问答与离线评测使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.core.intent.recognizer import IntentRecognizer, ProcurementIntent
from app.core.rag.generator import RAGAnswerGenerator
from app.core.rag.retriever import MultiChannelRetriever, RetrievedChunk


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


@dataclass
class OrchestratorResult:
    answer: str
    intent: ProcurementIntent
    retrieved: list[RetrievedChunk]
    metadata: dict[str, Any]


class ProcurementAgentOrchestrator:
    """意图识别 → 多路检索 → 受约束生成。"""

    def __init__(
        self,
        *,
        retriever: MultiChannelRetriever,
        generator: RAGAnswerGenerator,
        intent_recognizer: IntentRecognizer | None = None,
        embedder: Embedder | None = None,
        top_k: int = 8,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._intent = intent_recognizer or IntentRecognizer()
        self._embedder = embedder
        self._top_k = top_k

    async def _embedding(self, text: str) -> list[float]:
        if self._embedder is None:
            raise RuntimeError("embedder is required for retrieval")
        return await self._embedder.embed(text)

    async def run(self, user_message: str) -> OrchestratorResult:
        ir = await self._intent.recognize(user_message)
        vector = await self._embedding(user_message)
        chunks = await self._retriever.retrieve(
            user_message,
            vector,
            intent=ir.intent.value,
            top_k=self._top_k,
        )
        generated = await self._generator.generate(user_message, chunks)
        return OrchestratorResult(
            answer=generated.text,
            intent=ir.intent,
            retrieved=chunks,
            metadata={
                "intent_confidence": ir.confidence,
                "retrieved_count": len(chunks),
                "prompt_tokens": generated.prompt_tokens,
                "completion_tokens": generated.completion_tokens,
            },
        )
