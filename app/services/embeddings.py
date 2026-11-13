from __future__ import annotations

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """向量化服务。

    显式传 `dimensions`，让输出维度与 Milvus collection 的维度定义强绑定。改造前两处各自
    取默认值（模型默认 1024 / collection 写死 1536），入库时才报维度不匹配。
    """

    def __init__(self, model: str | None = None, *, dim: int | None = None) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is not installed")
        self._client = AsyncOpenAI(
            api_key=settings.embedding_api_key or "dummy",
            base_url=settings.embedding_api_base,
        )
        self._model = model or settings.embedding_model
        self._dim = dim or settings.embedding_dim
        self._supports_dimensions = True

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_text(self, text: str) -> list[float]:
        payload = text[:8000]
        kwargs: dict[str, object] = {"model": self._model, "input": payload}
        if self._supports_dimensions:
            kwargs["dimensions"] = self._dim
        try:
            resp = await self._client.embeddings.create(**kwargs)  # type: ignore[arg-type]
        except TypeError:
            self._supports_dimensions = False
            resp = await self._client.embeddings.create(model=self._model, input=payload)

        vector = list(resp.data[0].embedding)
        if len(vector) != self._dim:
            # 维度不符会在写库时才炸，且报错信息在 pymilvus 层，很难定位；这里提前拦住。
            raise RuntimeError(
                f"embedding 维度不匹配：模型返回 {len(vector)}，配置要求 {self._dim}。"
                f"请调整 EMBEDDING_DIM 或重建 Milvus collection。"
            )
        return vector
