"""HTTP/SSE 压测客户端。

只依赖 httpx，不依赖 locust。用法：

    python benchmarks/load_http.py \
      --url http://127.0.0.1:18080 \
      --token bench-token \
      --concurrency 5 \
      --requests 20 \
      --message "公司差旅住宿标准是多少"

指标：
- TTFT：从发出请求到第一个 content 事件；
- total：到 done 事件；
- RPS、P50、P95、错误率。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmarks" / "reports"


@dataclass
class Sample:
    ttft_ms: float
    total_ms: float
    ok: bool
    error: str = ""


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


async def one_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    token: str,
    message: str,
    session_id: str,
) -> Sample:
    started = time.perf_counter()
    ttft = None
    payload = {"message": message, "stream": True, "session_id": session_id}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        async with client.stream("POST", f"{url}/api/v1/chat", json=payload, headers=headers) as response:
            if response.status_code != 200:
                body = await response.aread()
                return Sample(0.0, (time.perf_counter() - started) * 1000, False, f"HTTP {response.status_code}: {body[:200]!r}")
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = event.get("type")
                if kind == "content" and event.get("delta") and ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                if kind == "done":
                    break
                if kind == "error":
                    return Sample(ttft or 0.0, (time.perf_counter() - started) * 1000, False, str(event.get("error")))
        total = (time.perf_counter() - started) * 1000
        return Sample(ttft or total, total, True)
    except Exception as exc:  # noqa: BLE001
        return Sample(0.0, (time.perf_counter() - started) * 1000, False, f"{type(exc).__name__}: {exc}")


async def worker(
    *,
    url: str,
    token: str,
    message: str,
    requests: int,
    worker_id: int,
    reuse_session: bool,
) -> list[Sample]:
    samples: list[Sample] = []
    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index in range(requests):
            session_id = f"load-{worker_id}" if reuse_session else f"load-{worker_id}-{index}"
            samples.append(
                await one_request(
                    client,
                    url=url,
                    token=token,
                    message=message,
                    session_id=session_id,
                )
            )
    return samples


async def run(
    *,
    url: str,
    token: str,
    message: str,
    concurrency: int,
    requests: int,
    reuse_session: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    batches = await asyncio.gather(
        *(
            worker(
                url=url,
                token=token,
                message=message,
                requests=requests,
                worker_id=i,
                reuse_session=reuse_session,
            )
            for i in range(concurrency)
        )
    )
    wall = time.perf_counter() - started
    samples = [sample for batch in batches for sample in batch]
    ok = [sample for sample in samples if sample.ok]
    ttft = [sample.ttft_ms for sample in ok]
    total = [sample.total_ms for sample in ok]
    return {
        "url": url,
        "message": message,
        "concurrency": concurrency,
        "requests": len(samples),
        "rps": round(len(samples) / wall, 2) if wall else 0.0,
        "ok": len(ok),
        "errors": len(samples) - len(ok),
        "ttft_p50_ms": round(statistics.median(ttft), 1) if ttft else 0.0,
        "ttft_p95_ms": round(p95(ttft), 1),
        "total_p50_ms": round(statistics.median(total), 1) if total else 0.0,
        "total_p95_ms": round(p95(total), 1),
        "wall_s": round(wall, 2),
        "error_samples": [sample.error for sample in samples if not sample.ok][:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="HTTP/SSE 压测")
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--token", default="bench-token")
    parser.add_argument("--message", default="公司差旅住宿标准是多少")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--requests", type=int, default=10, help="每个并发 worker 的请求数")
    parser.add_argument("--reuse-session", action="store_true", help="同一 worker 复用会话，压记忆增长")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    for concurrency in args.concurrency:
        row = asyncio.run(
            run(
                url=args.url,
                token=args.token,
                message=args.message,
                concurrency=concurrency,
                requests=args.requests,
                reuse_session=args.reuse_session,
            )
        )
        rows.append(row)
        print(
            f"c={concurrency:>3d} rps={row['rps']:>6.2f} "
            f"ttft p50={row['ttft_p50_ms']:>7.1f} p95={row['ttft_p95_ms']:>7.1f} "
            f"total p50={row['total_p50_ms']:>7.1f} p95={row['total_p95_ms']:>7.1f} "
            f"ok={row['ok']}/{row['requests']}"
        )
        if row["error_samples"]:
            print("  errors:", row["error_samples"])

    payload = {
        "mode": "http_sse",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else REPORTS / f"http_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report:", path)
    return 0 if all(row["errors"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
