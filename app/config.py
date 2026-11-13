from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用与 Agent 运行时配置（自环境变量与 `.env` 加载）。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "TravelMind AI"
    app_env: str = "dev"
    #: 主模型供应方：dashscope（qwen-max）或 openai。两者都走 OpenAI 兼容协议。
    llm_provider: str = "dashscope"
    dashscope_api_key: str = ""
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"
    #: 向量化可以独立于对话模型选择供应方；留空则跟随 llm_provider。
    embedding_provider: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    # 必须与 Milvus collection 的向量维度一致。text-embedding-v4 默认输出 1024 维；
    # OpenAI text-embedding-3-small 是 1536 维，切换 embedding 时必须重建 collection。
    embedding_dim: int = 1024
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/procurement_agent"
    redis_url: str = "redis://localhost:6379/0"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    log_level: str = "INFO"

    # ---------------------------------------------------------------- 鉴权与来源
    # 认证一律走 Bearer token；主体字段（工号/部门/职级）只从这里来，不接受 LLM 抽取。
    auth_required: bool = True
    # JSON：{"<token>": {"subject_id": "...", "department": "...", ...}}
    auth_tokens_json: str = ""
    # 本地演示用单一 token；生产必须留空（留空则仅 auth_tokens_json 生效）。
    auth_dev_token: str = ""
    # 逗号分隔白名单；不使用 "*"，因为与 allow_credentials=True 组合会被浏览器拒绝。
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # ---------------------------------------------------------------- 记忆（L1/L2）
    session_ttl_seconds: int = 7 * 24 * 3600
    memory_max_groups: int = 24
    memory_max_tokens: int = 8000
    # 触发摘要的 token 水位，必须小于 memory_max_tokens，否则窗口先裁剪、摘要永不触发。
    memory_summary_trigger_tokens: int = 5200
    # 摘要时保留的最近消息组数量。
    memory_keep_recent_groups: int = 8
    memory_summary_max_tokens: int = 640

    # ---------------------------------------------------------------- 记忆（L4/L5）
    ltm_collection: str = "travel_memory"
    ltm_default_ttl_days: int = 180
    ltm_min_text_length: int = 8
    policy_version: str = "2026.08-v1"
    travel_policy_version: str = "2026.11-v1"
    audit_enabled: bool = True

    # ---------------------------------------------------------------- 差旅编排
    #: 意图置信度低于该值时走 CLARIFY，不猜。
    router_confidence_threshold: float = 0.85
    #: on = 多范式路由；off = 单 ReAct 基线（仅用于压测/离线 A/B）。
    router_mode: str = "on"
    #: 预订确认前允许的报价波动；超过即中止并要求重新确认。
    booking_price_tolerance: float = 0.02

    # ---------------------------------------------------------------- Harness 预算
    harness_max_iterations: int = 8
    harness_max_tool_calls: int = 12
    harness_deadline_seconds: float = 90.0
    harness_max_total_tokens: int = 32000
    tool_timeout_seconds: float = 15.0
    # 参数校验失败后允许模型自修正的额外轮次。
    tool_max_repair_attempts: int = 2
    confirmation_ttl_seconds: int = 1800
    idempotency_ttl_seconds: int = 7 * 24 * 3600

    # ---------------------------------------------------------------- 可选推理阶段
    enable_planner: bool = False
    enable_write_reflection: bool = True

    # ---------------------------------------------------------------- 熔断
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 30
    circuit_breaker_half_open_max_calls: int = 3

    # ---------------------------------------------------------------- 可观测性
    otel_exporter_otlp_endpoint: str = ""
    otel_console_export: bool = False
    prompt_version: str = "2026.08.20-1"

    @property
    def llm_api_key(self) -> str:
        return self.openai_api_key if self.llm_provider.lower() == "openai" else self.dashscope_api_key

    @property
    def llm_api_base(self) -> str:
        return self.openai_api_base if self.llm_provider.lower() == "openai" else self.dashscope_api_base

    @property
    def llm_model(self) -> str:
        return self.openai_model if self.llm_provider.lower() == "openai" else self.dashscope_model

    @property
    def _embedding_provider(self) -> str:
        return (self.embedding_provider or self.llm_provider).lower()

    @property
    def embedding_api_key(self) -> str:
        return self.openai_api_key if self._embedding_provider == "openai" else self.dashscope_api_key

    @property
    def embedding_api_base(self) -> str:
        return self.openai_api_base if self._embedding_provider == "openai" else self.dashscope_api_base

    @property
    def embedding_model(self) -> str:
        return (
            self.openai_embedding_model
            if self._embedding_provider == "openai"
            else self.dashscope_embedding_model
        )

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"prod", "production", "staging"}

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


settings = Settings()
