"""最小测试运行器：在没有 pytest 的环境里跑仓库测试。

支持本项目用到的少量 pytest 能力：`raises`、`monkeypatch.setattr`、
`mark.parametrize`。不是 pytest 的替代品，只用于离线环境自检。

用法：
    python scripts/run_tests_lite.py tests/test_travel_policy.py tests/test_travel_router.py
    python scripts/run_tests_lite.py            # 跑全部可发现的测试
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
import traceback
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- pytest 桩


class _Raises:
    def __init__(self, expected: Any) -> None:
        self.expected = expected
        self.value: BaseException | None = None

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError(f"期望抛出 {self.expected}，但没有异常")
        if not issubclass(exc_type, self.expected):
            return False
        self.value = exc
        return True


def _raises(expected: Any) -> _Raises:
    return _Raises(expected)


class MonkeyPatch:
    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, Any]] = []

    def setattr(self, target: Any, name: Any, value: Any = None) -> None:
        if isinstance(target, str) and value is None:
            module_name, _, attr = target.rpartition(".")
            obj = importlib.import_module(module_name)
            new_value = name
            attr_name = attr
        else:
            obj = target
            attr_name = name
            new_value = value
        old = getattr(obj, attr_name)
        setattr(obj, attr_name, new_value)
        self._undo.append((obj, attr_name, old))

    def undo(self) -> None:
        for obj, attr, old in reversed(self._undo):
            setattr(obj, attr, old)
        self._undo.clear()


class _Mark:
    def parametrize(self, names: Any, values: Any) -> Callable:
        def decorator(fn: Callable) -> Callable:
            fn._params = (names, values)  # type: ignore[attr-defined]
            return fn

        return decorator


pytest_stub = types.ModuleType("pytest")
pytest_stub.raises = _raises  # type: ignore[attr-defined]
pytest_stub.MonkeyPatch = MonkeyPatch  # type: ignore[attr-defined]
pytest_stub.mark = _Mark()  # type: ignore[attr-defined]
sys.modules.setdefault("pytest", pytest_stub)


def _param_cases(fn: Callable) -> list[dict[str, Any]]:
    params = getattr(fn, "_params", None)
    if params is None:
        return [{}]
    names, values = params
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    cases = []
    for value in values:
        row = value if isinstance(value, (tuple, list)) else (value,)
        cases.append(dict(zip(names, row)))
    return cases


def _invoke(fn: Callable, case: dict[str, Any]) -> None:
    signature = inspect.signature(fn)
    kwargs = dict(case)
    monkeypatch: MonkeyPatch | None = None
    if "monkeypatch" in signature.parameters:
        monkeypatch = MonkeyPatch()
        kwargs["monkeypatch"] = monkeypatch
    try:
        result = fn(**kwargs)
        if inspect.isawaitable(result):
            asyncio.run(result)
    finally:
        if monkeypatch is not None:
            monkeypatch.undo()


def run_module(path: Path) -> tuple[int, int, list[str]]:
    module_name = ".".join(path.with_suffix("").parts[-2:]) if path.parent.name == "tests" else path.stem
    if path.parent.name == "tests":
        module_name = f"tests.{path.stem}"
    module = importlib.import_module(module_name)
    passed = failed = 0
    failures: list[str] = []
    for name, fn in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        for index, case in enumerate(_param_cases(fn)):
            label = f"{path.name}::{name}" + (f"[{index}]" if len(_param_cases(fn)) > 1 else "")
            try:
                _invoke(fn, case)
                passed += 1
                print(f"  [ok] {label}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
                print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
    return passed, failed, failures


def discover() -> list[Path]:
    return sorted((ROOT / "tests").glob("test_*.py"))


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv] if argv else discover()
    total_passed = total_failed = 0
    all_failures: list[str] = []
    for path in paths:
        if not path.is_absolute():
            path = ROOT / path
        print(f"\n== {path.name}")
        passed, failed, failures = run_module(path)
        total_passed += passed
        total_failed += failed
        all_failures.extend(failures)
    print(f"\n{total_passed} passed, {total_failed} failed")
    if all_failures:
        print("\n失败明细：")
        for item in all_failures:
            print(f"- {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
