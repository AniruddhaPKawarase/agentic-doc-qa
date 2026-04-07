"""
Hallucination Guard — Token-overlap groundedness scoring.
──────────────────────────────────────────────────────────────────────────────
Compares LLM output tokens against source context tokens.
If groundedness < threshold, replaces answer with clarification questions.
"""

import re
import logging
from typing import Dict, List, Set

logger = logging.getLogger("docqa.guard")

# Common English stopwords to exclude from overlap scoring
STOPWORDS: Set[str] = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "some", "them",
    "than", "its", "over", "such", "that", "this", "with", "will", "each",
    "make", "from", "what", "when", "which", "their", "there", "about",
    "would", "these", "other", "into", "more", "also", "been", "could",
    "does", "just", "like", "they", "very", "your", "most", "only", "where",
    "here", "should", "being", "those", "after", "before", "between",
}

# Minimum word length for scoring
MIN_WORD_LENGTH = 4


class HallucinationGuard:
    """Check LLM output groundedness against source context."""

    def __init__(
        self,
        threshold: float = 0.35,
        enabled: bool = True,
        general_threshold: float = 0.20,
        specific_threshold: float = 0.30,
        comparison_threshold: float = 0.25,
        marginal_low: float = 0.25,
        marginal_high: float = 0.50,
    ):
        self.threshold = threshold
        self.enabled = enabled
        self.general_threshold = general_threshold
        self.specific_threshold = specific_threshold
        self.comparison_threshold = comparison_threshold
        self.marginal_low = marginal_low
        self.marginal_high = marginal_high

    def _get_threshold(self, query_type: str) -> float:
        """Resolve groundedness threshold based on query type."""
        thresholds = {
            "general": self.general_threshold,
            "comparison": self.comparison_threshold,
            "specific": self.specific_threshold,
        }
        return thresholds.get(query_type, self.specific_threshold)

    def _extract_tokens(self, text: str) -> Set[str]:
        """Extract meaningful tokens from text."""
        words = re.findall(r'\b\w{%d,}\b' % MIN_WORD_LENGTH, text.lower())
        return {w for w in words if w not in STOPWORDS}

    def check(self, answer: str, context: str, query_type: str = "specific") -> Dict:
        """
        Score groundedness of answer against context.
        Returns {groundedness, passed, answer_tokens, matched_tokens, tier}.
        """
        if not self.enabled:
            return {
                "groundedness": 1.0,
                "passed": True,
                "answer_tokens": 0,
                "matched_tokens": 0,
                "tier": "disabled",
            }

        answer_tokens = self._extract_tokens(answer)
        context_tokens = self._extract_tokens(context)

        if not answer_tokens:
            return {
                "groundedness": 1.0,
                "passed": True,
                "answer_tokens": 0,
                "matched_tokens": 0,
                "tier": "tier1",
            }

        matched = answer_tokens & context_tokens
        groundedness = len(matched) / len(answer_tokens)

        if len(answer_tokens) < 10:
            groundedness = max(groundedness, 0.40)

        threshold = self._get_threshold(query_type)

        result = {
            "groundedness": round(groundedness, 3),
            "answer_tokens": len(answer_tokens),
            "matched_tokens": len(matched),
        }

        if groundedness >= self.marginal_high:
            result.update({"passed": True, "tier": "tier1"})
        elif groundedness < threshold:
            logger.warning(
                f"Guard triggered: groundedness={groundedness:.2f}, "
                f"threshold={threshold}, query_type={query_type}"
            )
            result.update({"passed": False, "tier": "tier1"})
        else:
            logger.info(
                f"Guard marginal: groundedness={groundedness:.2f}, "
                f"threshold={threshold}, query_type={query_type}"
            )
            result.update({"passed": True, "tier": "tier1_marginal"})

        return result

    def generate_clarification_questions(
        self,
        context: str,
        count: int = 3,
    ) -> List[str]:
        """
        Generate clarification questions from context when guard triggers.
        These are rule-based fallbacks (no LLM call).
        """
        # Extract key topics from context
        tokens = self._extract_tokens(context)
        # Get most common meaningful words (rough frequency proxy: longer = more specific)
        topic_words = sorted(tokens, key=len, reverse=True)[:10]

        questions = [
            "Could you be more specific about which part of the document you're asking about?",
            "I want to make sure I give you accurate information. Can you rephrase your question with more context?",
            "Which specific section or topic in the uploaded documents are you referring to?",
        ]

        if topic_words:
            questions.append(
                f"Are you asking about topics like {', '.join(topic_words[:3])}? "
                "Please clarify so I can give you a precise answer."
            )

        return questions[:count]
