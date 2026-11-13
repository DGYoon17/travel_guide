from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.agent.travel_orchestrator import TravelOrchestrator
from app.api.routes import approvals as approval_routes
from app.api.routes import chat as chat_routes
from app.api.routes import documents as documents_routes
from app.api.routes import health as health_routes
from app.api.routes import observability as observability_routes
from app.config import settings
from app.core.harness.confirmation import build_confirmation_store, build_idempotency_store
from app.core.intent.travel import TravelIntentRecognizer
from app.core.agent.router import AgentRouter, BaselineRouter
from app.core.logging import configure_logging, get_logger
from app.core.memory.audit import InMemoryAuditLog, NullAuditLog
from app.core.memory.profile import ClaimsProfileProvider
from app.core.memory.travel_memory import TenantLongTermMemory
from app.core.memory.travel_session import build_travel_session_store
from app.core.memory.summary import MemorySummarizer
from app.core.security.auth import StaticTokenResolver
from app.core.tools.travel_builtin import build_travel_registry
from app.domain.travel.policy import default_travel_policy
from app.infrastructure.llm.circuit_breaker import get_llm_circuit_breaker
from app.infrastructure.llm.client import LLMClient
from app.infrastructure.observability.metrics import MetricsCollector
from app.infrastructure.suppliers.mock_travel import MockTravelSupplier
from app.infrastructure.vector.milvus_client import MilvusVectorClient, ensure_memory_collection
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.milvus_store import MilvusKnowledgeSearch, get_milvus_store

logger = get_logger(__name__)


def _setup_tracing() -> None:
    """配置 OTel。

    改造前只建了 `TracerProvider` 却没有任何 exporter，也没做 FastAPI/httpx 自动埋点，
    span 生成后直接丢弃。这里补上 exporter 与埋点，且在依赖缺失时优雅跳过。
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        logger.warning("otel.sdk_missing")
        return

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": settings.app_name,
                "service.version": "0.1.0",
                "deployment.environment": settings.app_env,
            }
        )
    )

    endpoint = settings.otel_exporter_otlp_endpoint.strip()
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            logger.info("otel.otlp_exporter_enabled", endpoint=endpoint)
        except ImportError:
            logger.warning("otel.otlp_exporter_missing", hint="pip install opentelemetry-exporter-otlp-proto-http")
    elif settings.otel_console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("otel.console_exporter_enabled")
    else:
        logger.info("otel.no_exporter_configured", hint="设置 OTEL_EXPORTER_OTLP_ENDPOINT 以导出 trace")

    trace.set_tracer_provider(provider)


def _instrument_app(app: FastAPI) -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="/api/v1/health")
        logger.info("otel.fastapi_instrumented")
    except ImportError:
        logger.info("otel.fastapi_instrumentation_missing")
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
        logger.info("otel.httpx_instrumented")
    except ImportError:
        logger.info("otel.httpx_instrumentation_missing")


async def _build_audit(engine: AsyncEngine | None) -> Any:
    if not settings.audit_enabled:
        if settings.is_production:
            logger.error("audit.disabled_in_production")
        return NullAuditLog()
    if engine is not None:
        try:
            from app.infrastructure.database.audit_repo import SqlAuditLog

            audit = SqlAuditLog(engine)
            await audit.ensure_schema()
            return audit
        except Exception as exc:  # noqa: BLE001
            logger.error("audit.sql_unavailable_degraded_to_memory", error=str(exc))
    else:
        logger.warning("audit.no_database_degraded_to_memory")
    return InMemoryAuditLog()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    _setup_tracing()

    app.state.metrics = MetricsCollector()
    app.state.principal_resolver = StaticTokenResolver.from_settings()
    app.state.llm_breaker = get_llm_circuit_breaker()

    # ---------------------------------------------------------------- Redis
    app.state.redis = None
    try:
        client = redis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        app.state.redis = client
        logger.info("redis.connected")
    except Exception as exc:
        logger.warning("redis.unavailable", error=str(exc))
        app.state.redis = None

    # ---------------------------------------------------------------- Postgres
    app.state.db_engine: Optional[AsyncEngine] = None
    try:
        app.state.db_engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        logger.info("database.engine_created")
    except Exception as exc:
        logger.warning("database.engine_failed", error=str(exc))

    app.state.audit = await _build_audit(app.state.db_engine)

    # ---------------------------------------------------------------- 订单持久化
    app.state.booking_repo = None
    if app.state.db_engine is not None:
        try:
            from app.infrastructure.database.booking_repo import SqlBookingRepository

            repo = SqlBookingRepository(app.state.db_engine)
            await repo.ensure_schema()
            app.state.booking_repo = repo
            logger.info("booking.repo_ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning("booking.repo_unavailable", error=str(exc))

    # ---------------------------------------------------------------- Milvus
    store = get_milvus_store()
    store.connect()
    app.state.milvus = store
    logger.info("milvus.status", connected=store.connected)

    # ---------------------------------------------------------------- 差旅供应方与工具
    registry = build_travel_registry()
    app.state.tool_registry = registry
    app.state.tool_names = registry.names()
    app.state.supplier = MockTravelSupplier()

    app.state.embedder = None
    app.state.orchestrator = None
    knowledge = None
    long_term = None
    try:
        app.state.embedder = EmbeddingService()
        if store.connected:
            knowledge = MilvusKnowledgeSearch(store, app.state.embedder)
            if ensure_memory_collection(
                host=settings.milvus_host,
                port=settings.milvus_port,
                name=settings.ltm_collection,
                dim=settings.embedding_dim,
            ):
                long_term = TenantLongTermMemory(
                    MilvusVectorClient(
                        host=settings.milvus_host,
                        port=settings.milvus_port,
                        collection_name=settings.ltm_collection,
                    ),
                    app.state.embedder.embed_text,
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("embeddings.unavailable", error=str(exc))

    try:
        llm_service = LLMService(metrics=app.state.metrics)
        summarizer_client = LLMClient(
            api_key=settings.llm_api_key or "dummy",
            base_url=settings.llm_api_base,
            default_model=settings.llm_model,
        )
        app.state.orchestrator = TravelOrchestrator(
            llm=llm_service,
            registry=registry,
            session_store=build_travel_session_store(app.state.redis),
            confirmation_store=build_confirmation_store(app.state.redis),
            idempotency_store=build_idempotency_store(app.state.redis),
            audit=app.state.audit,
            metrics=app.state.metrics,
            intent_recognizer=TravelIntentRecognizer.with_llm(summarizer_client),
            summarizer=MemorySummarizer(summarizer_client),
            profile_provider=ClaimsProfileProvider(),
            policy=default_travel_policy(version=settings.travel_policy_version),
            knowledge=knowledge,
            supplier=app.state.supplier,
            booking_repo=app.state.booking_repo,
            long_term=long_term,
            router=(
                BaselineRouter()
                if settings.router_mode.lower() == "off"
                else AgentRouter(confidence_threshold=settings.router_confidence_threshold)
            ),
        )
        logger.info(
            "orchestrator.ready",
            tools=registry.names(),
            provider=settings.llm_provider,
            model=settings.llm_model,
            long_term=long_term is not None,
        )
    except Exception as exc:  # noqa: BLE001
        # 缺 API key 时不阻塞启动：/health 仍可用，/chat 返回 503 并说明原因。
        logger.error("orchestrator.init_failed", error=str(exc))

    if settings.auth_required and not settings.auth_tokens_json.strip() and not settings.auth_dev_token.strip():
        logger.error("startup.no_auth_tokens", hint="所有需鉴权接口将返回 401")

    yield

    if app.state.redis is not None:
        await app.state.redis.aclose()
    engine = getattr(app.state, "db_engine", None)
    if engine is not None:
        await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    # allow_origins=["*"] 与 allow_credentials=True 组合会被浏览器直接拒绝，
    # 且一旦接入鉴权就是实质漏洞；改为显式白名单。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
        expose_headers=["X-Trace-Id"],
    )

    @app.middleware("http")
    async def trace_header(request: Request, call_next):
        trace_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers.setdefault("X-Trace-Id", trace_id)
        return response

    _instrument_app(app)

    app.include_router(health_routes.router, prefix="/api/v1")
    app.include_router(chat_routes.router, prefix="/api/v1")
    app.include_router(approval_routes.router, prefix="/api/v1")
    app.include_router(documents_routes.router, prefix="/api/v1")
    app.include_router(observability_routes.router, prefix="/api/v1")

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": settings.app_name, "docs": "/docs"}

    return app


app = create_app()
