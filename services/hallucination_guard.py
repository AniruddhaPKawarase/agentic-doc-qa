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

    def __init__(self, threshold: float = 0.65, enabled: bool = True):
        self.threshold = threshold
        self.enabled = enabled

    def _extract_tokens(self, text: str) -> Set[str]:
        """Extract meaningful tokens from text."""
        words = re.findall(r'\b\w{%d,}\b' % MIN_WORD_LENGTH, text.lower())
        return {w for w in words if w not in STOPWORDS}

    def check(self, answer: str, context: str) -> Dict:
        """
        Score groundedness of answer against context.
        Returns {groundedness, passed, answer_tokens, matched_tokens}.
        """
        if not self.enabled:
            return {
                "groundedness": 1.0,
                "passed": True,
                "answer_tokens": 0,
                "matched_tokens": 0,
            }

        answer_tokens = self._extract_tokens(answer)
        context_tokens = self._extract_tokens(context)

        if not answer_tokens:
            return {
                "groundedness": 1.0,
                "passed": True,
                "answer_tokens": 0,
                "matched_tokens": 0,
            }

        matched = answer_tokens & context_tokens
        groundedness = len(matched) / len(answer_tokens)

        # Floor at 0.40 — very short answers shouldn't score too low
        groundedness = max(groundedness, 0.40) if len(answer_tokens) < 10 else groundedness

        passed = groundedness >= self.threshold

        if not passed:
            logger.warning(
                f"Hallucination guard triggered: groundedness={groundedness:.2f} "
                f"(threshold={self.threshold}), "
                f"{len(matched)}/{len(answer_tokens)} tokens matched"
            )

        return {
            "groundedness": round(groundedness, 3),
            "passed": passed,
            "answer_tokens": len(answer_tokens),
            "matched_tokens": len(matched),
        }

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
