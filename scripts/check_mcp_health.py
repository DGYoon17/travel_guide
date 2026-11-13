"""MCP 实时数据服务健康检查。

用法：
    python scripts/check_mcp_health.py            # 人类可读
    python scripts/check_mcp_health.py --json     # 机器可读（CI/监控）
    python scripts/check_mcp_health.py --no-tools # 只探连通性，不拉工具列表

退出码：0 全部可用；1 有服务器不可用；2 未配置任何 MCP 服务。
必需服务器（12306/高德）不可用才算失败；可选服务器（黄历/航班）失败只告警。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.infrastructure.mcp.hub import build_mcp_hub  # noqa: E402

# Windows 默认 GBK 控制台无法输出部分符号；统一改用 UTF-8，失败则退化为纯 ASCII 文案。
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 - 老环境没有 reconfigure 也能继续
    pass


async def _run(*, list_tools: bool, as_json: bool) -> int:
    hub = build_mcp_hub(settings)
    if hub is None:
        message = (
            "未配置 MCP 实时数据服务。请在 .env 中设置 MCP_GAODE_URL / MCP_12306_URL "
            "（或 MCP_SERVERS_JSON / MCP_SERVERS_CONFIG_PATH）后重试。"
        )
        if as_json:
            print(json.dumps({"configured": False, "servers": [], "message": message}, ensure_ascii=False))
        else:
            print(f"[!] {message}")
        return 2

    try:
        results = await hub.health(list_tools=list_tools)
    finally:
        await hub.aclose()

    if as_json:
        print(json.dumps({"configured": True, "servers": results}, ensure_ascii=False, indent=2))
    else:
        print("MCP 实时数据服务健康检查\n")
        for entry in results:
            icon = "[OK]  " if entry["connected"] else "[FAIL]"
            required = "必需" if entry["required"] else "可选"
            print(f"{icon} {entry['name']}（{entry['transport']}，{required}）")
            if entry["connected"]:
                tools = entry["tools"]
                print(f"   工具 {len(tools)} 个：" + ("、".join(tools) if tools else "（未返回）"))
            else:
                print(f"   错误：{entry['error']}")
            print()

    failed_required = [
        entry for entry in results if entry["required"] and not entry["connected"]
    ]
    failed_optional = [
        entry for entry in results if not entry["required"] and not entry["connected"]
    ]
    if not as_json:
        if failed_required:
            print("[FAIL] 必需服务不可用：" + "、".join(entry["name"] for entry in failed_required))
        if failed_optional:
            print("[!] 可选服务不可用（不影响火车/自驾/天气）：" + "、".join(e["name"] for e in failed_optional))
        if not failed_required:
            print("[OK] 必需服务连接正常")
    return 1 if failed_required else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 MCP 实时数据服务连接与工具列表")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--no-tools", action="store_true", help="不拉取工具列表")
    args = parser.parse_args(argv)
    return asyncio.run(_run(list_tools=not args.no_tools, as_json=args.json))


if __name__ == "__main__":
    raise SystemExit(main())
