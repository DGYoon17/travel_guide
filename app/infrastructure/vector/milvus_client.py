"""Milvus client for dense vector search and metadata filters."""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from app.infrastructure.observability.metrics import MetricsCollector

try:
    from pymilvus import Collection, connections, utility
except ImportError:  # pragma: no cover
    Collection = Any  # type: ignore[assignment]
    connections = None  # type: ignore[assignment]
    utility = None  # type: ignore[assignment]


class MilvusVectorClient:
    """Async-friendly wrapper around pymilvus for embedding search and insert."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 19530,
        alias: str = "default",
        collection_name: str = "procurement_docs",
        vector_field: str = "embedding",
        text_field: str = "text",
        id_field: str = "chunk_id",
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._alias = alias
        self._collection_name = collection_name
        self._vector_field = vector_field
        self._text_field = text_field
        self._id_field = id_field
        self._metrics = metrics
        self._collection: Collection | None = None
        self._connected = False

    def _ensure_connection(self) -> None:
        if self._connected:
            return
        if connections is None:
            raise RuntimeError("pymilvus is not installed")
        connections.connect(alias=self._alias, host=self._host, port=self._port)
        self._connected = True

    def _collection_or_raise(self) -> Collection:
        self._ensure_connection()
        if self._collection is None:
            if utility is None:
                raise RuntimeError("pymilvus is not installed")
            if not utility.has_collection(self._collection_name, using=self._alias):
                raise RuntimeError(
                    f"Milvus collection '{self._collection_name}' missing; run ETL first."
                )
            col = Collection(self._collection_name, using=self._alias)
            col.load()
            self._collection = col
        return self._collection

    async def search_vectors(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        intent_filter: str | None = None,
        output_fields: list[str] | None = None,
        expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索。

        `expr` 是通用过滤表达式；长期记忆需要按租户与过期时间过滤，只有 `intent_filter`
        一个维度不够（改造前 `LongTermMemory` 因此只能传 `None`，等于不过滤）。
        """

        def _search() -> list[dict[str, Any]]:
            col = self._collection_or_raise()
            fields = output_fields or [self._id_field, self._text_field, "intent"]
            params = {"metric_type": "IP", "params": {"nprobe": 16}}
            clauses: list[str] = []
            if intent_filter:
                safe = intent_filter.replace('"', '\\"')
                clauses.append(f'intent == "{safe}"')
            if expr:
                clauses.append(f"({expr})")
            filter_expr = " and ".join(clauses) or None
            res = col.search(
                data=[query_vector],
                anns_field=self._vector_field,
                param=params,
                limit=top_k,
                expr=filter_expr,
                output_fields=fields,
            )
            rows: list[dict[str, Any]] = []
            for hit in res[0]:
                ent = hit.entity
                row: dict[str, Any] = {
                    "id": ent.get(self._id_field),
                    "text": ent.get(self._text_field, ""),
                    "score": float(hit.score),
                }
                for extra in fields:
                    if extra in {self._id_field, self._text_field}:
                        continue
                    row[extra] = ent.get(extra)
                rows.append(row)
            return rows

        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(None, _search)
        if self._metrics:
            self._metrics.increment_counter("milvus_search_total")
        return rows

    async def insert_vectors(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        chunk_ids: list[str],
        intents: list[str] | None = None,
    ) -> None:
        if not (len(embeddings) == len(texts) == len(chunk_ids)):
            raise ValueError("embeddings, texts, chunk_ids length mismatch")
        if intents is None:
            intents = [""] * len(texts)

        def _insert() -> None:
            col = self._collection_or_raise()
            col.insert([chunk_ids, texts, intents, embeddings])
            col.flush()

        await asyncio.get_running_loop().run_in_executor(None, _insert)

    async def insert_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        field_order: Sequence[str],
    ) -> None:
        """按显式字段顺序写入。

        长期记忆需要 `tenant_id` / `expires_at_ts` / `policy_version` 等字段，固定四列的
        `insert_vectors` 表达不了。
        """
        if not rows:
            return

        def _insert() -> None:
            col = self._collection_or_raise()
            col.insert([[row.get(name) for row in rows] for name in field_order])
            col.flush()

        await asyncio.get_running_loop().run_in_executor(None, _insert)


#: 长期记忆 collection 的字段顺序（与 `ensure_memory_collection` 的定义一致）。
MEMORY_FIELD_ORDER: tuple[str, ...] = (
    "chunk_id",
    "tenant_id",
    "subject_id",
    "text",
    "intent",
    "policy_version",
    "expires_at_ts",
    "embedding",
)


def ensure_memory_collection(
    *,
    host: str,
    port: int,
    name: str,
    dim: int,
    alias: str = "default",
) -> bool:
    """创建长期记忆 collection（与制度文档分开，避免互相召回）。"""
    try:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            utility,
        )
    except ImportError:
        return False

    try:
        connections.connect(alias=alias, host=host, port=str(port))
        if utility.has_collection(name, using=alias):
            return True
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="subject_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="intent", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="policy_version", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="expires_at_ts", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        col = Collection(
            name=name,
            schema=CollectionSchema(fields, description="Long-term agent memory"),
            using=alias,
        )
        col.create_index(
            field_name="embedding",
            index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}},
        )
        return True
    except Exception:  # noqa: BLE001
        return False
