"""采购知识库的 Milvus 存储。

改造要点：

- 加 `status` 字段并且**检索只命中 active**。知识库文档会被拼进 prompt，开放写入 + 直接
  参与检索等于一条间接 prompt injection 通道；现在新入库文档默认 `draft`，需显式发布。
- 加 `tenant_id` 并在检索时强制过滤，避免多法人场景下互相读到对方的制度。
- 加 `policy_version` / `effective_from`，让制度改版后旧版本可以被过滤掉。
- 向量维度改为从配置读取，与 `EmbeddingService` 共用同一个值。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.config import settings
from app.core.logging import get_logger
from app.domain.schemas import DocumentStatus

logger = get_logger(__name__)

COLLECTION_NAME = "travel_knowledge"


def _load_pymilvus() -> tuple[Any, ...]:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    return (Collection, CollectionSchema, DataType, FieldSchema, connections, utility)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class MilvusDocumentStore:
    host: str
    port: int
    collection_name: str = COLLECTION_NAME
    embedding_dim: int = field(default_factory=lambda: settings.embedding_dim)
    _collection: Any = None
    _connected: bool = field(default=False, init=False)

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        if self._connected and self._collection is not None:
            return True
        try:
            (
                MilvusCollection,
                MilvusCollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            ) = _load_pymilvus()
            alias = "default"
            connections.connect(alias=alias, host=self.host, port=str(self.port))
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                    FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
                    FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=16),
                    FieldSchema(name="policy_version", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="effective_from", dtype=DataType.VARCHAR, max_length=16),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(
                        name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim
                    ),
                ]
                schema = MilvusCollectionSchema(
                    fields, description="Travel policy knowledge base"
                )
                col = MilvusCollection(name=self.collection_name, schema=schema)
                col.create_index(
                    field_name="embedding",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": "COSINE",
                        "params": {"nlist": 128},
                    },
                )
            self._collection = MilvusCollection(self.collection_name)
            self._collection.load()
            self._connected = True
            return True
        except Exception as exc:
            logger.warning("milvus.connect_failed", error=str(exc))
            self._connected = False
            self._collection = None
            return False

    # ------------------------------------------------------------------ 写入
    def insert_vector(
        self,
        doc_id: str,
        title: str,
        doc_type: str,
        content: str,
        vector: List[float],
        *,
        tenant_id: str = "default",
        status: str = DocumentStatus.DRAFT.value,
        policy_version: str = "",
        effective_from: str = "",
    ) -> None:
        if not self._collection:
            raise RuntimeError("Milvus not connected")
        if len(vector) != self.embedding_dim:
            raise RuntimeError(
                f"向量维度 {len(vector)} 与 collection 定义 {self.embedding_dim} 不一致"
            )
        self._collection.insert(
            [
                [doc_id],
                [tenant_id[:64]],
                [title[:512]],
                [doc_type[:32]],
                [status[:16]],
                [policy_version[:64]],
                [effective_from[:16]],
                [content[:65530]],
                [vector],
            ]
        )
        self._collection.flush()

    def set_status(self, doc_id: str, status: str, *, tenant_id: str = "default") -> bool:
        """发布/下架文档。Milvus 不支持字段级 update，这里删旧插新。"""
        if not self._collection:
            raise RuntimeError("Milvus not connected")
        rows = self._collection.query(
            expr=f'id == "{_escape(doc_id)}" and tenant_id == "{_escape(tenant_id)}"',
            output_fields=[
                "id", "tenant_id", "title", "doc_type", "status",
                "policy_version", "effective_from", "content", "embedding",
            ],
        )
        if not rows:
            return False
        row = rows[0]
        self._collection.delete(expr=f'id == "{_escape(doc_id)}"')
        self._collection.flush()
        self.insert_vector(
            doc_id=doc_id,
            title=str(row.get("title") or ""),
            doc_type=str(row.get("doc_type") or "other"),
            content=str(row.get("content") or ""),
            vector=list(row.get("embedding") or []),
            tenant_id=tenant_id,
            status=status,
            policy_version=str(row.get("policy_version") or ""),
            effective_from=str(row.get("effective_from") or ""),
        )
        return True

    # ------------------------------------------------------------------ 检索
    def search(
        self,
        vector: List[float],
        top_k: int = 5,
        *,
        tenant_id: str | None = None,
        statuses: tuple[str, ...] = (DocumentStatus.ACTIVE.value,),
    ) -> List[Dict[str, Any]]:
        if not self._collection:
            raise RuntimeError("Milvus not connected")
        self._collection.load()

        clauses: list[str] = []
        if statuses:
            allowed = ", ".join(f'"{_escape(s)}"' for s in statuses)
            clauses.append(f"status in [{allowed}]")
        if tenant_id is not None:
            clauses.append(f'tenant_id == "{_escape(tenant_id)}"')
        expr = " and ".join(clauses) or None

        res = self._collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=expr,
            output_fields=["title", "doc_type", "content", "status", "policy_version", "tenant_id"],
        )
        hits: List[Dict[str, Any]] = []
        for hit in res[0]:
            ent = _entity_to_dict(getattr(hit, "entity", None))
            dist = getattr(hit, "distance", None)
            hits.append(
                {
                    "id": getattr(hit, "id", None),
                    "score": float(dist) if dist is not None else 0.0,
                    "title": ent.get("title"),
                    "doc_type": ent.get("doc_type"),
                    "status": ent.get("status"),
                    "policy_version": ent.get("policy_version"),
                    "content": str(ent.get("content") or "")[:2000],
                }
            )
        return hits


def _entity_to_dict(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if hasattr(raw, "to_dict"):
        try:
            return dict(raw.to_dict())
        except Exception:  # noqa: BLE001
            return {}
    if isinstance(raw, dict):
        return raw
    try:
        return dict(raw)
    except Exception:  # noqa: BLE001
        return {}


class MilvusKnowledgeSearch:
    """`KnowledgeSearch` 协议的实现：只返回已发布文档，并强制租户过滤。"""

    def __init__(self, store: MilvusDocumentStore, embedder: Any) -> None:
        self._store = store
        self._embedder = embedder

    async def search_active(
        self, query: str, *, top_k: int, tenant_id: str
    ) -> list[dict[str, Any]]:
        if not self._store.connected:
            return []
        vector = await self._embedder.embed_text(query)
        return await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._store.search(
                vector,
                top_k=top_k,
                tenant_id=tenant_id,
                statuses=(DocumentStatus.ACTIVE.value,),
            ),
        )


def get_milvus_store() -> MilvusDocumentStore:
    return MilvusDocumentStore(host=settings.milvus_host, port=settings.milvus_port)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_doc_id() -> str:
    return str(uuid.uuid4())
