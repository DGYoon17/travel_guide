"""差旅金标门禁：意图快车道、槽位解析、差标召回、审批链。"""

from __future__ import annotations

import pytest

from app.core.intent.travel import TravelRuleEngine
from app.core.memory.trip_draft import TripDraft
from app.domain.travel.policy import default_travel_policy, evaluate_trip
from tests.golden.travel_dataset import (
    APPROVAL_CASES,
    INTENT_CASES,
    MIN_APPROVAL_PATH_ACCURACY,
    MIN_INTENT_ACCURACY,
    MIN_POLICY_RECALL,
    MIN_SLOT_ACCURACY,
    POLICY_CASES,
    REFERENCE_TODAY,
    SLOT_CASES,
    SLOW_LANE_TEXTS,
    MetricReport,
    build_trip,
)


def test_intent_fast_lane_accuracy() -> None:
    engine = TravelRuleEngine()
    report = MetricReport("intent_fast_lane")
    for case in INTENT_CASES:
        match = engine.classify(case.text)
        ok = bool(match and match[0] == case.expected and match[1] >= 0.85)
        report.record(ok, f"{case.text!r} -> {match} expected {case.expected}")
    assert report.score >= MIN_INTENT_ACCURACY, report.summary()


def test_slow_lane_inputs_are_not_fast_laned() -> None:
    engine = TravelRuleEngine()
    for text in SLOW_LANE_TEXTS:
        match = engine.classify(text)
        assert match is None or match[1] < 0.85, f"{text!r} 不应被快车道直接放行：{match}"


def test_slot_extraction_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.core.tools.coercion.business_today", lambda: REFERENCE_TODAY
    )
    report = MetricReport("slot_extraction")
    for case in SLOT_CASES:
        draft = TripDraft()
        from app.core.memory.slots import SlotSource

        draft.merge(case.raw, source=SlotSource.USER, turn=1)
        actual = {k: draft.value(k) for k in case.expected}
        ok = actual == case.expected
        report.record(ok, f"{case.name}: {actual} != {case.expected}")
    assert report.score >= MIN_SLOT_ACCURACY, report.summary()


def test_policy_recall() -> None:
    policy = default_travel_policy(version="golden-v1")
    report = MetricReport("policy_recall")
    for case in POLICY_CASES:
        trip = build_trip(case)
        result = evaluate_trip(
            policy,
            trip,
            hotel_nightly_cny=case.hotel_nightly_cny,
            total_cny=case.total_cny,
            today=REFERENCE_TODAY,
        )
        codes = {f.code for f in result.findings}
        missing_codes = [c for c in case.expected_codes if c not in codes]
        missing_roles = [r for r in case.expected_roles if r not in result.required_roles]
        ok = not missing_codes and not missing_roles and result.compliant is case.expect_compliant
        report.record(
            ok,
            f"{case.name}: missing_codes={missing_codes} missing_roles={missing_roles} "
            f"compliant={result.compliant}",
        )
    assert report.score >= MIN_POLICY_RECALL, report.summary()


def test_approval_path_accuracy() -> None:
    policy = default_travel_policy(version="golden-v1")
    report = MetricReport("approval_path")
    for case in APPROVAL_CASES:
        trip = build_trip(case)
        result = evaluate_trip(policy, trip, total_cny=case.total_cny, today=REFERENCE_TODAY)
        ok = tuple(result.required_roles) == case.expected_roles
        report.record(ok, f"{case.name}: {result.required_roles} != {case.expected_roles}")
    assert report.score >= MIN_APPROVAL_PATH_ACCURACY, report.summary()
