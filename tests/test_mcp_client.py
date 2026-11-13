"""MCP 传输层测试：SSE 解析、两种客户端的 RPC 语义、hub 重试与配置解析。

传输层是唯一需要"假装成服务器"的部分：用 `httpx.MockTransport` 模拟 SSE 事件流，
验证"握手 → POST → 响应回到事件流 → 按 id 配对"这条最容易被写错的链路。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

from app.core.tools.mcp_client import (
    MCPHttpClient,
    MCPRpcError,
    MCPSseClient,
    MCPTransportError,
    extract_tool_result,
    parse_sse_events,
)
from app.infrastructure.mcp.hub import (
    GAODE_SERVER,
    TRAIN_SERVER,
    MCPError,
    MCPServerConfig,
    MCPToolHub,
    build_mcp_hub,
    parse_server_configs,
    resolve_server_configs,
)


# --------------------------------------------------------------------------- 纯解析


def test_parse_sse_events_handles_comments_and_multiline_data() -> None:
    text = (
        ": keep-alive\n"
        "event: endpoint\n"
        "data: /messages?session=abc\n\n"
        "event: message\n"
        'data: {"jsonrpc": "2.0",\n'
        'data:  "id": 1, "result": {}}\n\n'
    )
    events = parse_sse_events(text)
    assert events[0] == ("endpoint", "/messages?session=abc")
    assert events[1][0] == "message"
    assert json.loads(events[1][1])["id"] == 1


def test_extract_tool_result_text_blocks() -> None:
    text, is_error = extract_tool_result(
        {"content": [{"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}]}
    )
    assert text == "第一段\n第二段"
    assert is_error is False


def test_extract_tool_result_structured_content() -> None:
    text, _ = extract_tool_result({"structuredContent": {"distance": 123}})
    assert json.loads(text) == {"distance": 123}


def test_extract_tool_result_error_flag() -> None:
    text, is_error = extract_tool_result(
        {"isError": True, "content": [{"type": "text", "text": "server exploded"}]}
    )
    assert text == "server exploded"
    assert is_error is True


# --------------------------------------------------------------------------- HTTP 客户端


async def test_http_client_parses_sse_response_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        body = {
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {"tools": [{"name": "maps_geo"}]},
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"event: message\ndata: {json.dumps(body)}\n\n".encode(),
        )

    client = MCPHttpClient("http://test/rpc")
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        tools = await client.list_tools()
    finally:
        await client.close()
    assert tools == [{"name": "maps_geo"}]


async def test_http_client_raises_rpc_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "error": {"code": -32601, "message": "method not found"},
            },
        )

    client = MCPHttpClient("http://test/rpc")
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(MCPRpcError) as excinfo:
            await client.call_tool("missing", {})
    finally:
        await client.close()
    assert excinfo.value.code == -32601


async def test_http_client_reports_empty_202_as_sse_hint() -> None:
    client = MCPHttpClient("http://test/rpc")
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(202, content=b""))
    )
    try:
        with pytest.raises(MCPTransportError) as excinfo:
            await client.list_tools()
    finally:
        await client.close()
    assert "SSE" in str(excinfo.value)


# --------------------------------------------------------------------------- SSE 客户端


class _QueueStream(httpx.AsyncByteStream):
    """把异步生成器包装成 httpx 的流式响应体。"""

    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._source:
            yield chunk

    async def aclose(self) -> None:
        await self._source.aclose()


class FakeSSEServer:
    """最小 SSE MCP 服务：GET 建流 + endpoint，POST 投递 JSON-RPC 响应到流里。"""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.posted: list[dict[str, Any]] = []
        self.stream_closed = asyncio.Event()

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_QueueStream(self._events()),
            )
        body = json.loads(request.content)
        self.posted.append(body)
        method = body.get("method")
        if method == "initialize":
            await self.queue.put(
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {"name": "fake"},
                    },
                }
            )
        elif method == "tools/list":
            await self.queue.put(
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"tools": [{"name": "maps_geo"}]},
                }
            )
        elif method == "tools/call" and body.get("params", {}).get("name") == "maps_geo":
            await self.queue.put(
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"content": [{"type": "text", "text": "geo ok"}]},
                }
            )
        return httpx.Response(202, headers={"content-type": "application/json"}, content=b"")

    async def _events(self) -> AsyncIterator[bytes]:
        yield b"event: endpoint\ndata: /messages?session=abc\n\n"
        try:
            while True:
                message = await self.queue.get()
                yield f"event: message\ndata: {json.dumps(message)}\n\n".encode()
        finally:
            self.stream_closed.set()


async def _wired_sse_client(server: FakeSSEServer) -> MCPSseClient:
    client = MCPSseClient("http://test/sse", timeout_s=5.0)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(server.handle),
        timeout=httpx.Timeout(5.0),
    )
    return client


async def test_sse_client_handshake_and_calls() -> None:
    server = FakeSSEServer()
    client = await _wired_sse_client(server)
    try:
        await asyncio.wait_for(client.initialize(), timeout=5.0)
        tools = await asyncio.wait_for(client.list_tools(), timeout=5.0)
        result = await asyncio.wait_for(client.call_tool("maps_geo", {"address": "上海"}), timeout=5.0)
        await asyncio.wait_for(client.notify("notifications/initialized"), timeout=5.0)
    finally:
        await client.close()

    assert tools == [{"name": "maps_geo"}]
    assert extract_tool_result(result) == ("geo ok", False)
    assert [p["method"] for p in server.posted] == [
        "initialize",
        "tools/list",
        "tools/call",
        "notifications/initialized",
    ]


async def test_sse_client_close_fails_pending_requests() -> None:
    server = FakeSSEServer()
    client = await _wired_sse_client(server)
    await asyncio.wait_for(client.connect(), timeout=5.0)

    pending = asyncio.ensure_future(client.call_tool("tools/never", {}))
    await asyncio.sleep(0.05)
    await client.close()

    with pytest.raises((MCPTransportError, asyncio.CancelledError)):
        await asyncio.wait_for(pending, timeout=5.0)


async def test_sse_client_handshake_without_endpoint_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b": keep-alive\n\n",
        )

    client = MCPSseClient("http://test/sse", timeout_s=0.5)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=httpx.Timeout(0.5)
    )
    try:
        with pytest.raises(MCPTransportError) as excinfo:
            await client.initialize()
    finally:
        await client.close()
    assert "endpoint" in str(excinfo.value)


# --------------------------------------------------------------------------- Hub


class _FlakyClient:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False
        self.notified: list[str] = []

    async def initialize(self, client_name: str = "x") -> dict[str, Any]:
        return {}

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self.notified.append(method)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls += 1
        if self.calls == 1:
            raise MCPTransportError("peer closed connection")
        return {"content": [{"type": "text", "text": "ok"}]}

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "maps_geo"}]

    async def close(self) -> None:
        self.closed = True


async def test_hub_retries_transport_failures() -> None:
    hub = MCPToolHub(
        [MCPServerConfig(name=GAODE_SERVER, url="http://test/rpc")],
        max_retries=1,
        retry_backoff_s=0.0,
    )
    flaky = _FlakyClient()
    hub._build_client = lambda config: flaky  # type: ignore[assignment]

    text = await hub.call(GAODE_SERVER, "maps_geo", {"address": "上海"})

    assert text == "ok"
    assert flaky.calls == 2
    assert flaky.closed is True  # 传输失败后旧连接被丢弃重建
    assert flaky.notified == ["notifications/initialized", "notifications/initialized"]
    await hub.aclose()


async def test_hub_does_not_retry_protocol_errors() -> None:
    class ProtocolErrorClient(_FlakyClient):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            self.calls += 1
            raise MCPRpcError(-32602, "invalid params")

    hub = MCPToolHub(
        [MCPServerConfig(name=GAODE_SERVER, url="http://test/rpc")],
        max_retries=3,
        retry_backoff_s=0.0,
    )
    client = ProtocolErrorClient()
    hub._build_client = lambda config: client  # type: ignore[assignment]

    with pytest.raises(MCPError) as excinfo:
        await hub.call(GAODE_SERVER, "maps_geo", {})
    assert "协议错误" in str(excinfo.value)
    assert client.calls == 1
    await hub.aclose()


async def test_hub_reports_tool_level_error_result() -> None:
    class ErrorResultClient(_FlakyClient):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            self.calls += 1
            return {"isError": True, "content": [{"type": "text", "text": "日期格式错误"}]}

    hub = MCPToolHub([MCPServerConfig(name=TRAIN_SERVER, url="http://test/rpc")])
    hub._build_client = lambda config: ErrorResultClient()  # type: ignore[assignment]

    with pytest.raises(MCPError) as excinfo:
        await hub.call(TRAIN_SERVER, "get-tickets", {})
    assert "日期格式错误" in str(excinfo.value)
    await hub.aclose()


async def test_hub_unconfigured_server_is_controlled_error() -> None:
    hub = MCPToolHub([MCPServerConfig(name=GAODE_SERVER, url="http://test/rpc")])
    with pytest.raises(MCPError) as excinfo:
        await hub.call(TRAIN_SERVER, "get-tickets", {})
    assert "未配置" in str(excinfo.value)


# --------------------------------------------------------------------------- 配置


class _Settings:
    def __init__(self, **overrides: Any) -> None:
        self.mcp_enabled = True
        self.mcp_transport = "auto"
        self.mcp_servers_json = ""
        self.mcp_servers_config_path = ""
        self.mcp_12306_url = ""
        self.mcp_gaode_url = ""
        self.mcp_bazi_url = ""
        self.mcp_flight_url = ""
        for key, value in overrides.items():
            setattr(self, key, value)


def test_resolve_servers_from_individual_urls() -> None:
    configs = resolve_server_configs(
        _Settings(
            mcp_12306_url="https://train.example/sse",
            mcp_gaode_url="https://gaode.example/rpc",
            mcp_flight_url="",
        )
    )
    assert [c.name for c in configs] == [TRAIN_SERVER, GAODE_SERVER]
    assert configs[0].resolved_transport == "sse"
    assert configs[1].resolved_transport == "http"


def test_resolve_servers_from_json_overrides_urls() -> None:
    configs = resolve_server_configs(
        _Settings(
            mcp_12306_url="https://ignored/sse",
            mcp_servers_json=json.dumps(
                [
                    {"name": "Gaode Server", "url": "https://gaode.example/sse", "transport": "sse"},
                    {"name": "flight Server", "url": ""},
                ]
            ),
        )
    )
    assert [c.name for c in configs] == [GAODE_SERVER]
    assert configs[0].resolved_transport == "sse"


def test_resolve_servers_from_reference_style_config_file() -> None:
    # 不用 tmp_path：部分受限环境无法写系统临时目录，改用工作区内的临时文件。
    path = Path(".mcp_test_servers.json")
    path.write_text(
        json.dumps(
            {
                "mcp_servers": [
                    {"name": "12306 Server", "url": "https://train.example/sse", "required": True},
                    {"name": "Gaode Server", "url": "https://gaode.example/sse"},
                ],
                "agent": {"name": "TravelPlannerAssistant"},
            }
        ),
        encoding="utf-8",
    )
    try:
        configs = resolve_server_configs(_Settings(mcp_servers_config_path=str(path.resolve())))
    finally:
        path.unlink(missing_ok=True)
    assert [c.name for c in configs] == [TRAIN_SERVER, GAODE_SERVER]
    assert configs[0].required is True


def test_parse_server_configs_skips_placeholders() -> None:
    configs = parse_server_configs(
        {
            "mcp_servers": [
                {"name": "Gaode Server", "url": "https://your-gaode-mcp-server-url/sse"},
                {"name": "Bazi Server", "url": ""},
                {"name": "12306 Server", "url": "https://train.example/sse"},
            ]
        }
    )
    # 占位符与空 URL 都不注册：否则模型会看到一批必定失败的工具。
    assert [c.name for c in configs] == [TRAIN_SERVER]


def test_build_hub_returns_none_when_disabled_or_unconfigured() -> None:
    assert build_mcp_hub(_Settings(mcp_enabled=False, mcp_gaode_url="https://x/sse")) is None
    assert build_mcp_hub(_Settings()) is None


async def test_build_hub_creates_configured_hub() -> None:
    hub = build_mcp_hub(_Settings(mcp_gaode_url="https://gaode.example/sse"))
    assert hub is not None
    assert hub.server_names == (GAODE_SERVER,)
    assert hub.describe()[0]["transport"] == "sse"
    await hub.aclose()
