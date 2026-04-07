"""
Monitoring Service — Per-query metrics tracking, aggregation, and alerting.
"""

import collections
import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("docqa.monitoring")


@dataclass(frozen=True)
class QueryMetrics:
    session_id: str
    query_type: str
    context_strategy: str
    model_used: str
    total_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    groundedness_score: float
    guard_passed: bool
    cached: bool
    file_count: int
    timestamp: str


@dataclass(frozen=True)
class AlertCheck:
    metric: str
    value: float
    threshold: float
    level: str
    triggered: bool


class MonitoringService:
    def __init__(
        self,
        latency_warning_ms: float = 15000,
        latency_critical_ms: float = 30000,
        cost_warning_usd: float = 0.50,
        cost_critical_usd: float = 1.00,
        groundedness_warning: float = 0.40,
        groundedness_critical: float = 0.20,
        max_history: int = 1000,
    ):
        self._latency_warning = latency_warning_ms
        self._latency_critical = latency_critical_ms
        self._cost_warning = cost_warning_usd
        self._cost_critical = cost_critical_usd
        self._groundedness_warning = groundedness_warning
        self._groundedness_critical = groundedness_critical
        self._history: collections.deque = collections.deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._alert_count = 0

    def record(self, metrics: QueryMetrics) -> List[AlertCheck]:
        with self._lock:
            self._history.append(metrics)
        alerts = self.check_alerts(metrics)
        if alerts:
            self._alert_count += len(alerts)
            for a in alerts:
                logger.warning(
                    f"ALERT [{a.level}] {a.metric}: {a.value} "
                    f"(threshold: {a.threshold})"
                )
        logger.info(
            f"Query recorded: strategy={metrics.context_strategy}, "
            f"tokens={metrics.total_tokens}, "
            f"latency={metrics.latency_ms:.0f}ms, "
            f"cost=${metrics.estimated_cost_usd:.4f}"
        )
        return alerts

    def check_alerts(self, metrics: QueryMetrics) -> List[AlertCheck]:
        alerts: List[AlertCheck] = []
        # Latency
        if metrics.latency_ms >= self._latency_critical:
            alerts.append(
                AlertCheck(
                    "latency_ms",
                    metrics.latency_ms,
                    self._latency_critical,
                    "critical",
                    True,
                )
            )
        elif metrics.latency_ms >= self._latency_warning:
            alerts.append(
                AlertCheck(
                    "latency_ms",
                    metrics.latency_ms,
                    self._latency_warning,
                    "warning",
                    True,
                )
            )
        # Cost
        if metrics.estimated_cost_usd >= self._cost_critical:
            alerts.append(
                AlertCheck(
                    "cost_usd",
                    metrics.estimated_cost_usd,
                    self._cost_critical,
                    "critical",
                    True,
                )
            )
        elif metrics.estimated_cost_usd >= self._cost_warning:
            alerts.append(
                AlertCheck(
                    "cost_usd",
                    metrics.estimated_cost_usd,
                    self._cost_warning,
                    "warning",
                    True,
                )
            )
        # Groundedness (low is bad)
        if metrics.groundedness_score <= self._groundedness_critical:
            alerts.append(
                AlertCheck(
                    "groundedness",
                    metrics.groundedness_score,
                    self._groundedness_critical,
                    "critical",
                    True,
                )
            )
        elif metrics.groundedness_score <= self._groundedness_warning:
            alerts.append(
                AlertCheck(
                    "groundedness",
                    metrics.groundedness_score,
                    self._groundedness_warning,
                    "warning",
                    True,
                )
            )
        return alerts

    def get_summary(self, hours: float = 1.0) -> dict:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self._lock:
            recent = [
                m
                for m in self._history
                if datetime.fromisoformat(m.timestamp) >= cutoff
            ]

        if not recent:
            return {
                "period_hours": hours,
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "avg_cost_usd": 0.0,
                "avg_groundedness": 0.0,
                "cache_hit_rate": 0.0,
                "strategy_distribution": {},
                "model_distribution": {},
                "error_count": 0,
                "alerts_triggered": self._alert_count,
            }

        n = len(recent)
        avg_latency = sum(m.latency_ms for m in recent) / n
        avg_cost = sum(m.estimated_cost_usd for m in recent) / n
        avg_ground = sum(m.groundedness_score for m in recent) / n
        cache_hits = sum(1 for m in recent if m.cached)
        errors = sum(1 for m in recent if not m.guard_passed)

        strategy_dist: Dict[str, int] = {}
        model_dist: Dict[str, int] = {}
        for m in recent:
            strategy_dist[m.context_strategy] = (
                strategy_dist.get(m.context_strategy, 0) + 1
            )
            model_dist[m.model_used] = model_dist.get(m.model_used, 0) + 1

        return {
            "period_hours": hours,
            "total_queries": n,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_cost_usd": round(avg_cost, 6),
            "avg_groundedness": round(avg_ground, 3),
            "cache_hit_rate": round(cache_hits / n, 3),
            "strategy_distribution": strategy_dist,
            "model_distribution": model_dist,
            "error_count": errors,
            "alerts_triggered": self._alert_count,
        }

    def get_recent(self, count: int = 20) -> List[dict]:
        with self._lock:
            items = list(self._history)
        return [asdict(m) for m in items[-count:]]
