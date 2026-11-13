"""按租户/主体装配 L4 长期记忆。

`LongTermMemory` 的 `tenant_id` 在构造时固定，而编排器是单例、请求才带租户。
这里做一层轻量工厂：按请求取实例，底层 Milvus 客户端与 embedding 复用。
"""

from __future__ import annotations

from typing import Any, Sequence

from app.config import settings
from app.core.memory.long_term import LongTermMemory, MemoryKind, MemoryRecord


class TenantLongTermMemory:
    def __init__(
        self,
        client: Any,
        embed: Any,
        *,
        collection: str | None = None,
        ttl_days: int | None = None,
        policy_version: str | None = None,
    ) -> None:
        self._client = client
        self._embed = embed
        self._collection = collection or settings.ltm_collection
        self._ttl_days = ttl_days if ttl_days is not None else settings.ltm_default_ttl_days
        self._policy_version = policy_version or settings.travel_policy_version
        self._cache: dict[tuple[str, str], LongTermMemory] = {}

    def _for(self, tenant_id: str, subject_id: str) -> LongTermMemory:
        key = (tenant_id, subject_id)
        memory = self._cache.get(key)
        if memory is None:
            memory = LongTermMemory(
                self._client,
                self._embed,
                tenant_id=tenant_id,
                subject_id=subject_id,
                collection=self._collection,
                ttl_days=self._ttl_days,
                policy_version=self._policy_version,
            )
            self._cache[key] = memory
        return memory

    async def recall(
        self,
        query: str,
        *,
        tenant_id: str,
        subject_id: str,
        top_k: int = 5,
        kind: MemoryKind | None = None,
        min_score: float = 0.0,
    ) -> list[MemoryRecord]:
        return await self._for(tenant_id, subject_id).recall(
            query, top_k=top_k, kind=kind, min_score=min_score
        )

    async def remember(
        self,
        text: str,
        *,
        tenant_id: str,
        subject_id: str,
        kind: MemoryKind = MemoryKind.PREFERENCE,
        intent: str = "",
    ) -> str | None:
        return await self._for(tenant_id, subject_id).remember(text, kind=kind, intent=intent)


__all__ = ["TenantLongTermMemory"]
