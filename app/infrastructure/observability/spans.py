"""OpenTelemetry Span 封装。

只在有 OTel 依赖时生效；缺失时退化为 no-op，保证单测与本地运行不因观测组件失败。

用法：
    with span("tool.book_trip", tool="book_trip", paradigm="react") as sp:
        ...
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

try:  # pragma: no cover - 依赖缺失时走 no-op
    from opentelemetry import trace as _otel_trace
except ImportError:  # pragma: no cover
    _otel_trace = None  # type: ignore[assignment]

_TRACER_NAME = "travelmind.agent"


def _coerce(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any | None]:
    """创建一个当前上下文的 span；异常会被记录后重新抛出。"""
    if _otel_trace is None:
        yield None
        return
    tracer = _otel_trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, _coerce(value))
        try:
            yield current
        except Exception as exc:  # noqa: BLE001 - 记录后必须原样抛出
            current.record_exception(exc)
            raise


def add_event(name: str, **attributes: Any) -> None:
    """给当前 span 挂一个事件；没有活动 span 时静默忽略。"""
    if _otel_trace is None:
        return
    current = _otel_trace.get_current_span()
    if current is None:
        return
    current.add_event(name, {k: _coerce(v) for k, v in attributes.items() if v is not None})


__all__ = ["add_event", "span"]
