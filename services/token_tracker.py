"""
Token Tracker — Per-step granular token + cost tracking.
──────────────────────────────────────────────────────────────────────────────
Tracks embedding, prompt, and completion tokens per pipeline step.
Estimates cost based on OpenAI pricing.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger("docqa.tokens")

# Pricing per 1M tokens (as of 2025)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02},
}


@dataclass
class PipelineTokenLog:
    """Granular per-step token tracking."""
    steps: Dict[str, Dict] = field(default_factory=dict)
    total_embedding_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def record_step(
        self,
        step_name: str,
        embedding_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        elapsed_ms: float = 0.0,
    ) -> None:
        self.steps[step_name] = {
            "embedding_tokens": embedding_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_ms": round(elapsed_ms, 1),
        }
        self.total_embedding_tokens += embedding_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

    @property
    def total_tokens(self) -> int:
        return (
            self.total_embedding_tokens
            + self.total_prompt_tokens
            + self.total_completion_tokens
        )

    def to_dict(self) -> Dict:
        return {
            "steps": self.steps,
            "totals": {
                "embedding_tokens": self.total_embedding_tokens,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            },
        }


class TokenTracker:
    """Token budget enforcement and cost estimation."""

    def __init__(self, chat_model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def estimate_cost(self, log: PipelineTokenLog) -> float:
        """Estimate cost in USD for a pipeline run."""
        chat_pricing = PRICING.get(self.chat_model, {"input": 0.15, "output": 0.60})
        embed_pricing = PRICING.get(self.embedding_model, {"input": 0.02})

        cost = 0.0
        cost += (log.total_embedding_tokens / 1_000_000) * embed_pricing["input"]
        cost += (log.total_prompt_tokens / 1_000_000) * chat_pricing["input"]
        cost += (log.total_completion_tokens / 1_000_000) * chat_pricing["output"]

        return round(cost, 6)

    def check_budget(
        self,
        current_tokens: int,
        max_tokens: int,
        warning_threshold: float = 0.85,
    ) -> Dict:
        """Check if we're within token budget."""
        usage_pct = current_tokens / max_tokens if max_tokens > 0 else 0.0
        return {
            "current_tokens": current_tokens,
            "max_tokens": max_tokens,
            "usage_pct": round(usage_pct, 3),
            "within_budget": current_tokens <= max_tokens,
            "warning": usage_pct >= warning_threshold,
        }
