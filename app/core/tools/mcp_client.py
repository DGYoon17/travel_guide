"""MCP (Model Context Protocol) 客户端：HTTP JSON-RPC 与 SSE 两种传输。

项目里真实可用的 12306 / 高德 / 八字 / 航班 MCP 服务大多以 SSE 暴露（`.../sse`），
而自建/内网服务更常见的是普通 JSON-RPC POST。这里把两种传输统一成同一个极小的
接口（`initialize` / `list_tools` / `call_tool` / `close`），上层 hub 不关心传输差异。

刻意不引入官方 `mcp` SDK：依赖体积大、版本演进快，而协议需要的只是 JSON-RPC 2.0
与 SSE 事件解析这两件小事；自己实现也让"重试/超时/降级"这些生产约束可控可测。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from contextlib import suppress
from typing import Any
from urllib.parse import urljoin

import httpx

PROTOCOL_VERSION = "2024-11-05"


class MCPTransportError(RuntimeError):
    """传输层失败（连接中断、超时、非 2xx）。上层可据此决定是否重试。"""


class MCPRpcError(RuntimeError):
    """MCP 服务端返回的 JSON-RPC 错误。协议级错误通常重试无意义。"""

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(f"MCP RPC error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


# --------------------------------------------------------------------------- 解析


def parse_sse_events(text: str) -> list[tuple[str, str]]:
    """把一段 SSE 报文解析成 `(event, data)` 列表。"""
    events: list[tuple[str, str]] = []
    event = "message"
    data_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.rstrip()
        if not line:
            if data_lines:
                events.append((event, "\n".join(data_lines)))
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value or "message"
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        events.append((event, "\n".join(data_lines)))
    return events


async def iter_sse_events(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, str]]:
    """流式解析 SSE。断流时把已缓冲的事件交给调用方，不静默丢失。"""
    event = "message"
    data_lines: list[str] = []
    async for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value or "message"
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield event, "\n".join(data_lines)


def _body_result(body: Any) -> Any:
    if not isinstance(body, dict):
        return body
    if "error" in body:
        err = body["error"]
        if isinstance(err, dict):
            raise MCPRpcError(
                code=int(err.get("code", -1)),
                message=str(err.get("message", "unknown")),
                data=err.get("data"),
            )
        raise MCPRpcError(-1, str(err))
    return body.get("result")


def extract_tool_result(result: Any) -> tuple[str, bool]:
    """把 `tools/call` 的 result 归一化成 `(文本, 是否错误)`。

    MCP 规范允许三种形态：`content` 文本块列表、`structuredContent` 结构化结果、
    以及直接返回字符串。调用方（hub）只看文本，结构化结果由业务层另行解析。
    """
    if result is None:
        return "", False
    if isinstance(result, str):
        return result, False
    if isinstance(result, (int, float, bool)):
        return str(result), False
    if isinstance(result, list):
        return json.dumps(result, ensure_ascii=False), False
    if not isinstance(result, dict):
        return str(result), False

    is_error = bool(result.get("isError"))
    blocks = result.get("content")
    texts: list[str] = []
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, str):
                texts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind in (None, "text"):
                text = block.get("text")
                if text is not None:
                    texts.append(str(text))
            elif kind == "resource":
                resource = block.get("resource") or {}
                text = resource.get("text") if isinstance(resource, dict) else None
                if text is not None:
                    texts.append(str(text))
            else:
                texts.append(json.dumps(block, ensure_ascii=False))
    if texts:
        return "\n".join(texts), is_error

    structured = result.get("structuredContent")
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False), is_error
    return json.dumps(result, ensure_ascii=False), is_error


# --------------------------------------------------------------------------- HTTP


class MCPHttpClient:
    """JSON-RPC over HTTP POST。适配 `.../rpc`、`/mcp` 这类无状态 HTTP MCP 服务。"""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._rpc_url = base_url.rstrip("/")
        merged = {"Accept": "application/json, text/event-stream"}
        merged.update(headers or {})
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=merged)
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    @staticmethod
    def _payload(method: str, params: dict[str, Any] | None, req_id: int) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}

    async def _rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        req_id = self._next_id()
        payload = self._payload(method, params, req_id)
        try:
            response = await self._client.post(self._rpc_url, json=payload)
        except httpx.HTTPError as exc:
            raise MCPTransportError(f"{type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise MCPTransportError(f"HTTP {response.status_code} from {self._rpc_url}")
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            # 少数服务端把响应也包成 SSE：取与本次 id 匹配的那条。
            for event, data in parse_sse_events(response.text):
                if event not in ("message", ""):
                    continue
                try:
                    body = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(body, dict) and body.get("id") in (req_id, None):
                    return _body_result(body)
            raise MCPTransportError("SSE 响应中没有匹配的 JSON-RPC 结果")
        try:
            body = response.json()
        except ValueError as exc:
            hint = ""
            if not response.text.strip():
                hint = "（空响应：该服务可能使用 SSE 传输，请把 URL 配成 /sse 或设置 MCP_TRANSPORT=sse）"
            raise MCPTransportError(f"响应不是合法 JSON{hint}：{response.text[:200]}") from exc
        return _body_result(body)

    async def initialize(self, client_name: str = "TravelMind AI") -> dict[str, Any]:
        return await self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": "0.2.0"},
            },
        )

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """JSON-RPC 通知（无 id、无响应）。`notifications/initialized` 用得到。"""
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        try:
            await self._client.post(self._rpc_url, json=payload)
        except httpx.HTTPError as exc:
            raise MCPTransportError(f"{type(exc).__name__}: {exc}") from exc

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._rpc("tools/list")
        if isinstance(result, dict) and "tools" in result:
            return list(result["tools"])
        if isinstance(result, list):
            return result
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._rpc("tools/call", {"name": name, "arguments": arguments})

    async def list_resources(self) -> list[dict[str, Any]]:
        result = await self._rpc("resources/list")
        if isinstance(result, dict) and "resources" in result:
            return list(result["resources"])
        if isinstance(result, list):
            return result
        return []

    async def read_resource(self, uri: str) -> dict[str, Any]:
        raw = await self._rpc("resources/read", {"uri": uri})
        return raw if isinstance(raw, dict) else {"contents": raw}

    async def close(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- SSE


class MCPSseClient:
    """MCP 经典 SSE 传输。

    流程遵循协议：GET `<url>` 建立事件流 → 服务端先下发 `endpoint` 事件告诉客户端
    消息投递地址 → 客户端把 JSON-RPC 请求 POST 到该地址 → 响应异步回到事件流，
    按 JSON-RPC `id` 配对。因此必须有一个后台任务持续读流，否则永远收不到结果。
    """

    def __init__(
        self,
        sse_url: str,
        *,
        timeout_s: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._sse_url = sse_url
        self._timeout_s = timeout_s
        self._headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            **(headers or {}),
        }
        self._post_headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **(headers or {}),
        }
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 15.0), read=None),
            headers=headers or {},
        )
        self._post_url: str | None = None
        self._endpoint_ready = asyncio.Event()
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._req_id = 0
        self._task: asyncio.Task[None] | None = None
        self._error: Exception | None = None
        self._connect_lock = asyncio.Lock()

    # ------------------------------------------------------------------ 连接
    @property
    def connected(self) -> bool:
        return self._post_url is not None and self._task is not None and not self._task.done()

    async def connect(self) -> None:
        async with self._connect_lock:
            if self.connected:
                return
            self._endpoint_ready.clear()
            self._error = None
            self._post_url = None
            self._task = asyncio.create_task(self._stream_loop(), name="mcp-sse-stream")
            try:
                await asyncio.wait_for(self._endpoint_ready.wait(), timeout=self._timeout_s)
            except asyncio.TimeoutError as exc:
                await self._reset_task()
                raise MCPTransportError(f"SSE 握手超时：{self._sse_url}") from exc
            if self._post_url is None:
                detail = f"：{self._error}" if self._error is not None else ""
                await self._reset_task()
                raise MCPTransportError(f"SSE 握手失败，未收到 endpoint 事件{detail}")

    async def _reset_task(self) -> None:
        """停止读流任务。握手失败时也要清理，否则重连会泄漏一个后台任务。"""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task

    async def _stream_loop(self) -> None:
        try:
            async with self._client.stream(
                "GET", self._sse_url, headers=self._headers
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise MCPTransportError(f"HTTP {response.status_code} from {self._sse_url}")
                async for event, data in iter_sse_events(response.aiter_lines()):
                    if event == "endpoint":
                        self._post_url = urljoin(self._sse_url, data.strip())
                        self._endpoint_ready.set()
                    elif event in ("message", ""):
                        self._dispatch(data)
                    elif event == "error":
                        self._fail_pending(MCPTransportError(f"SSE error: {data[:200]}"))
        except asyncio.CancelledError:  # pragma: no cover - 正常关闭路径
            raise
        except Exception as exc:  # noqa: BLE001 - 任何读流异常都要让等待方失败，而不是挂死
            self._error = exc
            self._fail_pending(exc)
        finally:
            self._endpoint_ready.set()

    def _dispatch(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            return
        if not isinstance(body, dict):
            return
        req_id = body.get("id")
        if not isinstance(req_id, int) or req_id not in self._pending:
            return
        future = self._pending.pop(req_id)
        if future.done():
            return
        try:
            future.set_result(_body_result(body))
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)

    def _fail_pending(self, exc: Exception) -> None:
        for req_id in list(self._pending):
            future = self._pending.pop(req_id)
            if not future.done():
                future.set_exception(exc)

    # ------------------------------------------------------------------ 请求
    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        await self.connect()
        req_id = self._next_id()
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[req_id] = future
        try:
            response = await self._client.post(
                self._post_url or self._sse_url,
                json=payload,
                headers=self._post_headers,
                timeout=self._timeout_s,
            )
            if response.status_code >= 400:
                raise MCPTransportError(f"HTTP {response.status_code}（{method}）")
            text = response.text.strip()
            if text:
                # 少数实现直接在 POST 响应里返回 JSON 或 SSE，而不是走事件流。
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    for event, data in parse_sse_events(text):
                        if event in ("message", ""):
                            self._dispatch(data)
                else:
                    try:
                        self._dispatch(text)
                    except json.JSONDecodeError:
                        pass
            return await asyncio.wait_for(future, timeout=self._timeout_s)
        except httpx.HTTPError as exc:
            raise MCPTransportError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self._pending.pop(req_id, None)

    async def initialize(self, client_name: str = "TravelMind AI") -> dict[str, Any]:
        return await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": "0.2.0"},
            },
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._request("tools/list")
        if isinstance(result, dict) and "tools" in result:
            return list(result["tools"])
        if isinstance(result, list):
            return result
        return []

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """SSE 传输下的通知：POST 后不等结果（服务端通常回 202）。"""
        await self.connect()
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        try:
            await self._client.post(
                self._post_url or self._sse_url,
                json=payload,
                headers=self._post_headers,
                timeout=self._timeout_s,
            )
        except httpx.HTTPError as exc:
            raise MCPTransportError(f"{type(exc).__name__}: {exc}") from exc

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._request("tools/call", {"name": name, "arguments": arguments})

    async def close(self) -> None:
        self._fail_pending(MCPTransportError("客户端已关闭"))
        await self._reset_task()
        with suppress(Exception):
            await self._client.aclose()

    async def __aenter__(self) -> "MCPSseClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


def tool_names_from_listing(tools: Iterable[Any]) -> list[str]:
    """从 `tools/list` 结果里取出工具名，兼容 dict / 对象两种形态。"""
    names: list[str] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name")
        else:
            name = getattr(tool, "name", None)
        if name:
            names.append(str(name))
    return names


MCPClient = MCPHttpClient


__all__ = [
    "MCPClient",
    "MCPHttpClient",
    "MCPRpcError",
    "MCPTransportError",
    "MCPSseClient",
    "PROTOCOL_VERSION",
    "extract_tool_result",
    "iter_sse_events",
    "parse_sse_events",
    "tool_names_from_listing",
]
