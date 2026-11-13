"""Unit tests for procurement intent recognizer."""

from __future__ import annotations

import asyncio


from app.core.intent.recognizer import IntentRecognizer, ProcurementIntent


def test_recognize_catalog_intent() -> None:
    r = IntentRecognizer()
    out = asyncio.run(r.recognize("帮我查一下笔记本采购目录和报价"))
    assert out.intent is ProcurementIntent.CATALOG_SEARCH
    assert out.confidence >= 0.4


def test_recognize_policy_intent() -> None:
    r = IntentRecognizer()
    out = asyncio.run(r.recognize("公司的采购审批政策和预算上限是多少"))
    assert out.intent is ProcurementIntent.POLICY


def test_recognize_general_fallback() -> None:
    r = IntentRecognizer()
    out = asyncio.run(r.recognize("你好"))
    assert out.intent is ProcurementIntent.GENERAL
