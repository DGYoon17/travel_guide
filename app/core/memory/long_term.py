"""L4 长期记忆：跨会话的稳定偏好与已确认事实。

改造前有三个问题，都会造成"记错"或"看到别人的数据"：

1. **与制度文档共用同一个 collection**，只靠 `intents` 字段区分，而 `recall` 传的是
   `intent_filter=None` —— 查"用户记忆"会召回制度文档，反之亦然。现在独立 collection。
2. **`_namespaced_id` 只影响写入的 id，检索时完全不按 namespace 过滤** —— 多部门/多法人
   场景下是跨租户泄漏。现在检索强制带 `tenant_id` 过滤表达式。
3. **`remember` 无门槛、无 TTL、无遗忘**，任何文本都写。现在有写入判别、过期时间与
   `policy_version` 标记：制度改版后依赖旧版本的记忆会被自动排除。
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Sequence

from app.config import settings
from app.core.logging import get_logger
from app.infrastructure.vector.milvus_client import MEMORY_FIELD_ORDER, MilvusVectorClient

logger = get_logger(__name__)

EmbedFn = Callable[[str], Awaitable[Sequence[float]] | Sequence[float]]


class MemoryKind(str, Enum):
    """记忆类型。只有稳定、可复用的信息值得跨会话保留。"""

    PREFERENCE = "preference"
    CONFIRMED_FACT = "confirmed_fact"
    ORG_CONTEXT = "org_context"


#: 一次性问句、寒暄、明显的临时内容不写长期记忆。
_TRANSIENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(你好|您好|谢谢|在吗|嗯+|好的|收到)[。！!？?~]*$"),
    re.compile(r"[？?]$"),
    re.compile(r"^(今天|明天|刚才|现在)"),
)

#: 出现这些内容一律不入库（PII / 凭据）。
_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{15,19}\b"),                      # 银行卡 / 长号
    re.compile(r"\b1[3-9]\d{9}\b"),                    # 手机号
    re.compile(r"\b\d{17}[\dXx]\b"),                   # 身份证
    re.compile(r"(?i)\b(sk|api[_-]?key|token|password)\b"),
)


@dataclass
class MemoryRecord:
    memory_id: str
    text: str
    score: float
    kind: MemoryKind = MemoryKind.PREFERENCE
    policy_version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WriteDecision:
    should_write: bool
    reason: str = ""


def evaluate_write(text: str, *, min_length: int | None = None) -> WriteDecision:
    """写入门槛。宁可少记，也不要把一次性问句和 PII 沉淀成"长期事实"。"""
    stripped = text.strip()
    threshold = min_length or settings.ltm_min_text_length
    if len(stripped) < threshold:
        return WriteDecision(False, f"文本过短（<{threshold} 字）")
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(stripped):
            return WriteDecision(False, "含敏感信息（PII 或凭据）")
    for pattern in _TRANSIENT_PATTERNS:
        if pattern.search(stripped):
            return WriteDecision(False, "疑似一次性表述")
    return WriteDecision(True)


class LongTermMemory:
    """按租户隔离、带过期与制度版本标记的长期记忆。"""

    def __init__(
        self,
        milvus: MilvusVectorClient,
        embed: EmbedFn,
        *,
        tenant_id: str,
        subject_id: str = "",
        collection: str | None = None,
        ttl_days: int | None = None,
        policy_version: str | None = None,
    ) -> None:
        self._milvus = milvus
        self._embed = embed
        self._tenant_id = tenant_id
        self._subject_id = subject_id
        self._collection = collection or settings.ltm_collection
        self._ttl_days = ttl_days if ttl_days is not None else settings.ltm_default_ttl_days
        self._policy_version = policy_version or settings.policy_version

    @property
    def namespace(self) -> str:
        return f"{self._tenant_id}:{self._subject_id}" if self._subject_id else self._tenant_id

    def _memory_id(self, text: str, kind: MemoryKind) -> str:
        digest = hashlib.sha256(f"{self.namespace}:{kind.value}:{text}".encode()).hexdigest()[:24]
        return f"ltm_{digest}"

    async def _vector(self, text: str) -> list[float]:
        result = self._embed(text)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[misc]
        return list(result)  # type: ignore[arg-type]

    def _filter_expr(self, *, include_expired: bool = False) -> str:
        # 租户隔离是硬边界：即使调用方忘了传，也不会退化成全量检索。
        clauses = [f'tenant_id == "{_escape(self._tenant_id)}"']
        if self._subject_id:
            clauses.append(f'(subject_id == "{_escape(self._subject_id)}" or subject_id == "")')
        if not include_expired:
            clauses.append(f"expires_at_ts > {int(time.time())}")
        clauses.append(f'policy_version == "{_escape(self._policy_version)}"')
        return " and ".join(clauses)

    async def remember(
        self,
        text: str,
        *,
        kind: MemoryKind = MemoryKind.PREFERENCE,
        intent: str = "",
    ) -> str | None:
        """写入一条记忆。未通过写入判别时返回 None 并记录原因。"""
        decision = evaluate_write(text)
        if not decision.should_write:
            logger.info("ltm.write_skipped", reason=decision.reason, namespace=self.namespace)
            return None

        vector = await self._vector(text)
        memory_id = self._memory_id(text, kind)
        expires_at_ts = int(time.time()) + self._ttl_days * 86400
        try:
            await self._milvus.insert_rows(
                [
                    {
                        "chunk_id": memory_id,
                        "tenant_id": self._tenant_id,
                        "subject_id": self._subject_id,
                        "text": text.strip()[:8000],
                        "intent": (intent or kind.value)[:32],
                        "policy_version": self._policy_version,
                        "expires_at_ts": expires_at_ts,
                        "embedding": vector,
                    }
                ],
                field_order=MEMORY_FIELD_ORDER,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ltm.write_failed", error=str(exc), namespace=self.namespace)
            return None
        logger.info(
            "ltm.written",
            memory_id=memory_id,
            kind=kind.value,
            namespace=self.namespace,
            expires_at_ts=expires_at_ts,
        )
        return memory_id

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        kind: MemoryKind | None = None,
        min_score: float = 0.0,
    ) -> list[MemoryRecord]:
        """按租户+主体+未过期+当前制度版本召回。"""
        vector = await self._vector(query)
        try:
            rows = await self._milvus.search_vectors(
                vector,
                top_k=top_k,
                intent_filter=kind.value if kind else None,
                output_fields=["chunk_id", "text", "intent", "tenant_id", "policy_version"],
                expr=self._filter_expr(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ltm.recall_failed", error=str(exc), namespace=self.namespace)
            return []

        out: list[MemoryRecord] = []
        for row in rows:
            score = float(row.get("score", 0.0))
            if score < min_score:
                continue
            raw_kind = str(row.get("intent") or MemoryKind.PREFERENCE.value)
            try:
                parsed_kind = MemoryKind(raw_kind)
            except ValueError:
                parsed_kind = MemoryKind.PREFERENCE
            out.append(
                MemoryRecord(
                    memory_id=str(row.get("id") or row.get("chunk_id") or ""),
                    text=str(row.get("text") or ""),
                    score=score,
                    kind=parsed_kind,
                    policy_version=str(row.get("policy_version") or ""),
                    metadata={
                        k: v
                        for k, v in row.items()
                        if k not in {"id", "chunk_id", "text", "score", "intent"}
                    },
                )
            )
        return out


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
