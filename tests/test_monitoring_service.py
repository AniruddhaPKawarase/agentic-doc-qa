"""Tests for monitoring service — metrics tracking, alerts, summaries."""
import pytest
from datetime import datetime, timezone
from services.monitoring_service import MonitoringService, QueryMetrics, AlertCheck


def _make_metrics(**overrides) -> QueryMetrics:
    defaults = {
        "session_id": "test-sess",
        "query_type": "specific",
        "context_strategy": "full_context",
        "model_used": "gpt-4o",
        "total_tokens": 5000,
        "estimated_cost_usd": 0.01,
        "latency_ms": 3000,
        "groundedness_score": 0.85,
        "guard_passed": True,
        "cached": False,
        "file_count": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    defaults.update(overrides)
    return QueryMetrics(**defaults)


class TestRecord:
    def test_record_stores_metrics(self):
        svc = MonitoringService()
        svc.record(_make_metrics())
        assert len(svc.get_recent(100)) == 1

    def test_max_history_limit(self):
        svc = MonitoringService(max_history=3)
        for i in range(5):
            svc.record(_make_metrics(session_id=f"s{i}"))
        assert len(svc.get_recent(100)) == 3


class TestAlerts:
    def test_latency_warning(self):
        svc = MonitoringService(latency_warning_ms=15000)
        alerts = svc.check_alerts(_make_metrics(latency_ms=16000))
        assert any(
            a.metric == "latency_ms" and a.level == "warning" for a in alerts
        )

    def test_latency_critical(self):
        svc = MonitoringService(latency_critical_ms=30000)
        alerts = svc.check_alerts(_make_metrics(latency_ms=35000))
        assert any(
            a.metric == "latency_ms" and a.level == "critical" for a in alerts
        )

    def test_cost_warning(self):
        svc = MonitoringService(cost_warning_usd=0.50)
        alerts = svc.check_alerts(_make_metrics(estimated_cost_usd=0.60))
        assert any(
            a.metric == "cost_usd" and a.level == "warning" for a in alerts
        )

    def test_groundedness_warning(self):
        svc = MonitoringService(groundedness_warning=0.40)
        alerts = svc.check_alerts(_make_metrics(groundedness_score=0.35))
        assert any(
            a.metric == "groundedness" and a.level == "warning" for a in alerts
        )

    def test_no_alerts_normal(self):
        svc = MonitoringService()
        alerts = svc.check_alerts(_make_metrics())
        assert alerts == []


class TestSummary:
    def test_empty_summary(self):
        svc = MonitoringService()
        s = svc.get_summary(hours=1.0)
        assert s["total_queries"] == 0
        assert s["avg_latency_ms"] == 0.0

    def test_summary_with_data(self):
        svc = MonitoringService()
        svc.record(_make_metrics(latency_ms=1000, estimated_cost_usd=0.01))
        svc.record(_make_metrics(latency_ms=3000, estimated_cost_usd=0.03))
        svc.record(_make_metrics(latency_ms=2000, estimated_cost_usd=0.02))
        s = svc.get_summary(hours=1.0)
        assert s["total_queries"] == 3
        assert s["avg_latency_ms"] == 2000.0

    def test_strategy_distribution(self):
        svc = MonitoringService()
        svc.record(_make_metrics(context_strategy="full_context"))
        svc.record(_make_metrics(context_strategy="full_context"))
        svc.record(_make_metrics(context_strategy="retrieval_only"))
        s = svc.get_summary(hours=1.0)
        assert s["strategy_distribution"]["full_context"] == 2
        assert s["strategy_distribution"]["retrieval_only"] == 1

    def test_get_recent_limit(self):
        svc = MonitoringService()
        for i in range(10):
            svc.record(_make_metrics(session_id=f"s{i}"))
        recent = svc.get_recent(5)
        assert len(recent) == 5
