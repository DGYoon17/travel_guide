import logging
import sys
from typing import Any

try:  # pragma: no cover - 依赖缺失时退化为标准库日志
    import structlog
except ImportError:  # pragma: no cover
    structlog = None  # type: ignore[assignment]


class _StdlibLogger:
    """`structlog` 缺失时的最小适配器：保留 event + 结构化字段的可读输出。"""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _log(self, level: int, event: str, **fields: Any) -> None:
        if fields:
            detail = " ".join(f"{key}={value}" for key, value in fields.items())
            message = f"{event} {detail}"
        else:
            message = event
        self._logger.log(level, message)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level, logging.INFO))
    if structlog is None:
        return
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level, logging.INFO),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    if structlog is None:
        return _StdlibLogger(name)
    return structlog.get_logger(name)
