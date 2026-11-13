"""Unit tests for procurement agent orchestrator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock


from app.core.agent.orchestrator import ProcurementAgentOrchestrator
from app.core.intent.recognizer import ProcurementIntent
from app.core.rag.generator import RAGAnswerGenerator
from app.core.rag.retriever import MultiChannelRetriever, RetrievedChunk
from app.infrastructure.llm.client import ChatCompletionResult, LLMClient


class _Embed:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_orchestrator_runs_rag_and_generation() -> None:
    milvus = MagicMock()
    milvus.search_vectors = AsyncMock(
        return_value=[
            {"id": "1", "text": "软件采购超过 8000 元需补充三家报价。", "score": 0.9},
        ]
    )
    from app.infrastructure.vector.milvus_client import MilvusVectorClient

    mc = MagicMock(spec=MilvusVectorClient)
    mc.search_vectors = milvus.search_vectors

    class KwBackend:
        async def search(self, q: str, *, top_k: int) -> list[dict]:
            return [{"chunk_id": "k1", "text": "关键词命中", "score": 0.5}]

    class IntentBackend:
        async def search_by_intent(
            self, q: str, intent: str | None, *, top_k: int
        ) -> list[dict]:
            return []

    retriever = MultiChannelRetriever(
        milvus=mc,
        keyword_backend=KwBackend(),
        intent_backend=IntentBackend(),
    )

    llm = MagicMock(spec=LLMClient)
    llm.chat = AsyncMock(
        return_value=ChatCompletionResult(
            content="根据资料，应选择经济舱。",
            model="gpt-test",
            prompt_tokens=10,
            completion_tokens=20,
            raw={},
        )
    )
    gen = RAGAnswerGenerator(llm=llm)  # type: ignore[arg-type]

    orch = ProcurementAgentOrchestrator(
        retriever=retriever,
        generator=gen,
        embedder=_Embed(),
    )
    result = asyncio.run(orch.run("采购一套协同办公软件需要走什么流程"))
    assert result.answer
    assert result.intent in (
        ProcurementIntent.CATALOG_SEARCH,
        ProcurementIntent.GENERAL,
        ProcurementIntent.POLICY,
        ProcurementIntent.REQUISITION_PLANNING,
    )
    assert isinstance(result.retrieved[0], RetrievedChunk)
