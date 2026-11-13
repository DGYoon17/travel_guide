"""MCP 服务注册与调用网关。

这一层只做四件事，把"外部服务不可靠"关在业务代码之外：

1. **服务发现**：从环境变量 / JSON 配置 / JSON 文件三种来源解析服务器列表；
2. **传输选择**：URL 以 `/sse` 结尾走 SSE，其余走 HTTP JSON-RPC；
3. **调用韧性**：超时、有限重试（只重试传输层错误）、失败后重建连接；
4. **可读降级**：不可用时抛 `MCPError`，由工具层翻译成"实时数据服务暂不可用"，
   而不是把栈回溯丢给模型或让整轮对话 500。

协议级错误（JSON-RPC error）不重试：同样的参数再发一次只会再错一次。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from app.core.logging import get_logger
from app.core.tools.mcp_client import (
    MCPHttpClient,
    MCPRpcError,
    MCPSseClient,
    MCPTransportError,
    extract_tool_result,
)

logger = get_logger(__name__)

#: 服务器逻辑名。工具层按这些名字路由，与具体 URL 解耦。
TRAIN_SERVER = "12306 Server"
GAODE_SERVER = "Gaode Server"
BAZI_SERVER = "bazi Server"
FLIGHT_SERVER = "flight Server"

#: 环境变量未显式给出服务器名单时的默认四件套：(名称, 配置字段, 说明, 是否必需)
DEFAULT_SERVERS: tuple[tuple[str, str, str, bool], ...] = (
    (TRAIN_SERVER, "mcp_12306_url", "12306 火车票查询（车次、时刻、票价）", True),
    (GAODE_SERVER, "mcp_gaode_url", "高德地图（地理编码、自驾路线、POI/酒店、天气）", True),
    (BAZI_SERVER, "mcp_bazi_url", "八字黄历（农历、宜忌、出行吉日）", False),
    (FLIGHT_SERVER, "mcp_flight_url", "航班查询（长途 >800km 自动触发）", False),
)


class MCPError(RuntimeError):
    """MCP 调用失败的受控错误。`retryable` 只描述"换一时刻重试是否有意义"。"""

    def __init__(
        self,
        message: str,
        *,
        server: str = "",
        tool: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.server = server
        self.tool = tool
        self.retryable = retryable


@dataclass(frozen=True)
class MCPServerConfig:
    """一个 MCP 服务器的连接配置。"""

    name: str
    url: str
    transport: str = "auto"  # auto | http | sse
    description: str = ""
    required: bool = False
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def resolved_transport(self) -> str:
        if self.transport in ("http", "sse"):
            return self.transport
        path = urlparse(self.url).path.rstrip("/").lower()
        return "sse" if path.endswith(("/sse", "/events", "/stream")) else "http"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "transport": self.resolved_transport,
            "description": self.description,
            "required": self.required,
        }


#: 参考项目示例里的占位符 URL。识别出来后直接跳过：注册一个必定失败的工具
#: 只会让模型在"服务不可用"上浪费一轮，不如当作未配置。
_PLACEHOLDER_URL_HINTS = ("your-", "your_", "<your", "placeholder", "请替换", "example-placeholder")


def _coerce_headers(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def parse_server_configs(payload: Any) -> list[MCPServerConfig]:
    """解析 `MCP_SERVERS_JSON` / `servers_config.json` 两种结构。

    兼容参考项目的 `{"mcp_servers": [{"name","url"}]}` 与纯数组两种写法；
    缺少 name 的条目按顺序补 `mcp-server-N`，避免整份配置因为一个占位符而失效。
    """
    if isinstance(payload, Mapping):
        entries = payload.get("mcp_servers") or payload.get("servers") or []
    elif isinstance(payload, list):
        entries = payload
    else:
        return []

    configs: list[MCPServerConfig] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            continue
        url = str(entry.get("url") or "").strip()
        lowered = url.lower()
        if not url or any(hint in lowered for hint in _PLACEHOLDER_URL_HINTS):
            # 占位符/未配置的服务器直接跳过：让"少配一个"退化为功能缺失，而不是启动失败。
            continue
        name = str(entry.get("name") or f"mcp-server-{index}").strip()
        configs.append(
            MCPServerConfig(
                name=name,
                url=url,
                transport=str(entry.get("transport") or "auto").strip().lower(),
                description=str(entry.get("description") or ""),
                required=bool(entry.get("required", False)),
                headers=_coerce_headers(entry.get("headers")),
            )
        )
    return configs


def resolve_server_configs(settings_obj: Any) -> list[MCPServerConfig]:
    """按优先级解析服务器配置：JSON 字符串 > 配置文件 > 单服务器 URL 字段。"""
    raw_json = str(getattr(settings_obj, "mcp_servers_json", "") or "").strip()
    if raw_json:
        try:
            return parse_server_configs(json.loads(raw_json))
        except json.JSONDecodeError as exc:
            logger.warning("mcp.servers_json_invalid", error=str(exc))
            return []

    config_path = str(getattr(settings_obj, "mcp_servers_config_path", "") or "").strip()
    if config_path:
        path = Path(config_path)
        if path.is_file():
            try:
                return parse_server_configs(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("mcp.config_file_invalid", path=config_path, error=str(exc))
        else:
            logger.warning("mcp.config_file_missing", path=config_path)

    default_transport = str(getattr(settings_obj, "mcp_transport", "auto") or "auto").lower()
    configs: list[MCPServerConfig] = []
    for name, attr, description, required in DEFAULT_SERVERS:
        url = str(getattr(settings_obj, attr, "") or "").strip()
        if not url:
            continue
        configs.append(
            MCPServerConfig(
                name=name,
                url=url,
                transport=default_transport,
                description=description,
                required=required,
            )
        )
    return configs


class MCPToolHub:
    """多 MCP 服务器的连接池 + 调用入口。同一服务器按名称复用连接。"""

    def __init__(
        self,
        servers: Iterable[MCPServerConfig],
        *,
        timeout_s: float = 25.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.4,
        client_name: str = "TravelMind AI",
    ) -> None:
        self._servers: dict[str, MCPServerConfig] = {s.name: s for s in servers}
        self._timeout_s = max(1.0, float(timeout_s))
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._client_name = client_name
        self._clients: dict[str, Any] = {}
        self._init_lock: dict[str, asyncio.Lock] = {name: asyncio.Lock() for name in self._servers}
        self._initialized: set[str] = set()
        self._tool_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ 元信息
    @property
    def servers(self) -> tuple[MCPServerConfig, ...]:
        return tuple(self._servers.values())

    @property
    def server_names(self) -> tuple[str, ...]:
        return tuple(self._servers)

    def has(self, server: str) -> bool:
        return server in self._servers

    def describe(self) -> list[dict[str, Any]]:
        return [config.to_dict() for config in self._servers.values()]

    # ------------------------------------------------------------------ 连接
    def _build_client(self, config: MCPServerConfig) -> Any:
        if config.resolved_transport == "sse":
            return MCPSseClient(
                config.url, timeout_s=self._timeout_s, headers=dict(config.headers)
            )
        return MCPHttpClient(config.url, timeout_s=self._timeout_s, headers=dict(config.headers))

    async def _invalidate(self, server: str) -> None:
        client = self._clients.pop(server, None)
        self._initialized.discard(server)
        self._tool_cache.pop(server, None)
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响重连
                pass

    async def _client(self, server: str) -> Any:
        config = self._servers.get(server)
        if config is None:
            raise MCPError(
                f"未配置 MCP 服务器 {server}",
                server=server,
                retryable=False,
            )
        async with self._init_lock[server]:
            client = self._clients.get(server)
            if client is None:
                client = self._build_client(config)
                self._clients[server] = client
            if server not in self._initialized:
                try:
                    await asyncio.wait_for(
                        client.initialize(self._client_name), timeout=self._timeout_s
                    )
                except Exception as exc:  # noqa: BLE001 - 初始化失败要能重建
                    await self._invalidate(server)
                    raise MCPError(
                        f"MCP 服务器 {server} 初始化失败：{type(exc).__name__}: {exc}",
                        server=server,
                        retryable=True,
                    ) from exc
                self._initialized.add(server)
                # 协议要求初始化后告知服务端"我准备好了"；部分实现会拒绝未通知的调用。
                notify = getattr(client, "notify", None)
                if callable(notify):
                    try:
                        await asyncio.wait_for(
                            notify("notifications/initialized", {}), timeout=self._timeout_s
                        )
                    except Exception as exc:  # noqa: BLE001 - 通知失败不阻塞主流程
                        logger.warning("mcp.initialized_notify_failed", server=server, error=str(exc))
            return client

    # ------------------------------------------------------------------ 调用
    async def call(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        max_retries: int | None = None,
        timeout_s: float | None = None,
    ) -> str:
        """调用远程工具，返回纯文本。失败抛 `MCPError`（不回退成空字符串）。"""
        payload = dict(arguments or {})
        timeout = self._timeout_s if timeout_s is None else max(1.0, float(timeout_s))
        retries = self._max_retries if max_retries is None else max(0, int(max_retries))
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            if attempt:
                await asyncio.sleep(self._retry_backoff_s * attempt)
            try:
                client = await self._client(server)
                raw = await asyncio.wait_for(
                    client.call_tool(tool, payload), timeout=timeout
                )
                text, is_error = extract_tool_result(raw)
                if is_error:
                    raise MCPError(
                        f"{server}.{tool} 返回错误：{text[:400]}",
                        server=server,
                        tool=tool,
                        retryable=False,
                    )
                return text
            except asyncio.TimeoutError as exc:
                last_error = exc
                logger.warning("mcp.timeout", server=server, tool=tool, attempt=attempt + 1)
                await self._invalidate(server)
            except MCPTransportError as exc:
                last_error = exc
                logger.warning(
                    "mcp.transport_error",
                    server=server,
                    tool=tool,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                await self._invalidate(server)
            except MCPRpcError as exc:
                raise MCPError(
                    f"{server}.{tool} 协议错误：{exc.message}",
                    server=server,
                    tool=tool,
                    retryable=False,
                ) from exc
            except MCPError as exc:
                # 例如"初始化失败"被标记为 retryable：重建连接后重试一次是有意义的。
                if exc.retryable and attempt < retries:
                    last_error = exc
                    await self._invalidate(server)
                    continue
                raise
            except Exception as exc:  # noqa: BLE001 - 未知异常不重试，直接给可读错误
                raise MCPError(
                    f"{server}.{tool} 调用异常：{type(exc).__name__}: {exc}",
                    server=server,
                    tool=tool,
                    retryable=False,
                ) from exc

        detail = f"{type(last_error).__name__}: {last_error}" if last_error else "未知原因"
        raise MCPError(
            f"{server}.{tool} 重试 {retries} 次后仍失败（{detail}）",
            server=server,
            tool=tool,
            retryable=True,
        )

    async def list_tools(self, server: str, *, refresh: bool = False) -> list[str]:
        """列出服务器工具名。失败时返回空列表而不是抛错：健康检查要能继续跑。"""
        if not refresh and server in self._tool_cache:
            return list(self._tool_cache[server])
        try:
            client = await self._client(server)
            tools = await asyncio.wait_for(client.list_tools(), timeout=self._timeout_s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mcp.list_tools_failed", server=server, error=str(exc))
            await self._invalidate(server)
            return []

        names: list[str] = []
        for tool in tools:
            if isinstance(tool, Mapping):
                name = tool.get("name")
            else:
                name = getattr(tool, "name", None)
            if name:
                names.append(str(name))
        self._tool_cache[server] = names
        return names

    async def health(self, *, list_tools: bool = True) -> list[dict[str, Any]]:
        """逐服务器探测，返回可直接序列化的状态。不抛错。"""
        results: list[dict[str, Any]] = []
        for config in self._servers.values():
            entry: dict[str, Any] = {
                "name": config.name,
                "transport": config.resolved_transport,
                "required": config.required,
                "configured": True,
                "connected": False,
                "tools": [],
                "error": None,
            }
            try:
                client = await self._client(config.name)
                entry["connected"] = True
                if list_tools:
                    entry["tools"] = await self.list_tools(config.name, refresh=True)
            except MCPError as exc:
                entry["error"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(entry)
        return results

    async def aclose(self) -> None:
        for server in list(self._clients):
            await self._invalidate(server)


def build_mcp_hub(settings_obj: Any) -> MCPToolHub | None:
    """按配置构建 hub；未配置任何服务器或显式关闭时返回 None。

    返回 None 是"没有实时数据能力"，不是错误：工具注册表据此不注册 MCP 工具，
    模型也就不会看到一批永远失败的工具。
    """
    if not bool(getattr(settings_obj, "mcp_enabled", False)):
        return None
    configs = resolve_server_configs(settings_obj)
    if not configs:
        # 未配置是正常状态（实时数据是可选增强），用 info 避免每次启动都告警。
        logger.info("mcp.no_servers_configured", hint="设置 MCP_GAODE_URL / MCP_12306_URL 可接入实时数据")
        return None
    return MCPToolHub(
        configs,
        timeout_s=float(getattr(settings_obj, "mcp_timeout_seconds", 25.0)),
        max_retries=int(getattr(settings_obj, "mcp_max_retries", 2)),
        client_name=str(getattr(settings_obj, "app_name", "TravelMind AI")),
    )


__all__ = [
    "BAZI_SERVER",
    "DEFAULT_SERVERS",
    "FLIGHT_SERVER",
    "GAODE_SERVER",
    "MCPError",
    "MCPServerConfig",
    "MCPToolHub",
    "TRAIN_SERVER",
    "build_mcp_hub",
    "parse_server_configs",
    "resolve_server_configs",
]
