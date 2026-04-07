"""
Generation Service — LLM call with streaming + follow-up question generation.
──────────────────────────────────────────────────────────────────────────────
Builds prompts from context + history, calls the configured LLM, streams tokens.
Generates follow-up questions from source context.

v2 (2026-03-26): File-aware prompt builder. When the user uploads files,
the system prompt explicitly names them and instructs the LLM to scope
its answer to those files.

v2.1 (2026-04-07): Citation-enforced prompt, model routing (gpt-4o primary,
gpt-4o-mini for follow-ups), query-type-aware instructions.
"""

import json
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI

from config import Settings

logger = logging.getLogger("docqa.generation")

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """You are a Document Q&A Assistant. You answer questions based on the provided document context.

CITATION RULES (MANDATORY):
1. Every factual claim MUST include a citation: [Source: {filename}, page {N}] or [Source: {filename}, section "{heading}"]
2. If you cannot cite a claim from the provided context, do not include it.
3. For summaries, cite the overall document and specific pages for key facts.
4. When information comes from multiple documents, cite each source separately.

RESPONSE RULES:
1. Answer questions thoroughly using ONLY the provided document context.
2. If the context does not contain enough information, say so clearly with what IS available.
3. Match the user's question style - concise for simple questions, detailed for complex ones.
4. Use bullet points, tables, or paragraphs as appropriate for the content.
5. For summaries and overviews, cover ALL major topics in the document.
6. For specific questions, quote relevant text when helpful.

OUT-OF-CONTEXT:
If the question is completely unrelated to the provided documents, respond:
"This question doesn't appear to be covered in the uploaded documents. The documents contain information about: [list key topics]. Could you ask about one of these topics?\""""

FILE_SCOPE_INSTRUCTION = """
FILE AWARENESS (CRITICAL — FOLLOW STRICTLY):
The user just uploaded: {file_names}
Your answer must be based PRIMARILY on the content from {scope_desc}.
- When the user says "this document", "this file", "about this" — they mean: {file_names}
- Do NOT answer from other files unless the user explicitly asks to compare or reference them.
- Always state which file your answer is based on."""

GLOBAL_SCOPE_INSTRUCTION = """
FILE AWARENESS:
The user has multiple documents in this session. Answer using ALL available documents.
Always clearly indicate which file each piece of information comes from."""

QUERY_TYPE_INSTRUCTIONS = {
    "general": "\nQUERY TYPE: This is a general/overview question. Provide a comprehensive answer covering all relevant aspects of the document(s). Cite the document name and major sections.",
    "specific": "\nQUERY TYPE: This is a specific factual question. Provide a precise answer with exact citations (page numbers, section headings).",
    "comparison": "\nQUERY TYPE: This is a comparison question. Structure your answer to clearly compare/contrast the items, citing each source separately.",
}

FOLLOWUP_PROMPT = """Based on the document context provided, generate exactly {count} follow-up questions that:
1. Are directly answerable from the document content
2. Would help the user explore the documents further
3. Are specific and actionable (not generic)

Return ONLY a JSON array of strings, no other text.
Example: ["What are the specific requirements for...", "How does the document address...", "Can you compare..."]"""


class GenerationService:
    """LLM generation with streaming support and file-aware prompting."""

    def __init__(self, settings: Settings):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_chat_model
        self.max_output_tokens = settings.openai_max_output_tokens
        self.min_followups = settings.min_followup_questions
        # v2.1 model routing
        self.primary_model = settings.primary_model
        self.secondary_model = settings.secondary_model
        self.primary_max_output_tokens = settings.primary_max_output_tokens

    def _build_system_prompt(
        self,
        current_files: Optional[List[str]] = None,
        scope_mode: str = "global",
        query_type: str = "specific",
    ) -> str:
        """Build system prompt with file awareness and query-type instruction."""
        prompt = SYSTEM_PROMPT_BASE

        if current_files and scope_mode in ("current_upload", "referenced_file"):
            file_list = ", ".join(f'"{f}"' for f in current_files)
            if scope_mode == "current_upload":
                scope_desc = "the file(s) the user just uploaded"
            else:
                scope_desc = "the file the user is referencing"
            prompt += FILE_SCOPE_INSTRUCTION.format(
                file_names=file_list,
                scope_desc=scope_desc,
            )
        else:
            prompt += GLOBAL_SCOPE_INSTRUCTION

        # Append query-type-aware instruction
        prompt += QUERY_TYPE_INSTRUCTIONS.get(query_type, "")

        return prompt

    def _build_messages(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict]] = None,
        current_files: Optional[List[str]] = None,
        scope_mode: str = "global",
        query_type: str = "specific",
    ) -> List[Dict]:
        """Build message list for the LLM call."""
        system_prompt = self._build_system_prompt(current_files, scope_mode, query_type)
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (already filtered by session service)
        if history:
            messages.extend(history)

        # Add current context + query
        user_content = f"""DOCUMENT CONTEXT:
─────────────────────────────────────────
{context}
─────────────────────────────────────────

USER QUESTION: {query}"""

        messages.append({"role": "user", "content": user_content})
        return messages

    async def generate(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict]] = None,
        current_files: Optional[List[str]] = None,
        scope_mode: str = "global",
        query_type: str = "specific",
        model_override: Optional[str] = None,
    ) -> Dict:
        """
        Blocking generation. Returns full response + token usage.

        Uses primary_model by default (gpt-4o). Pass model_override to
        route to a different model (e.g. secondary for cheaper calls).
        """
        messages = self._build_messages(
            query, context, history, current_files, scope_mode, query_type,
        )
        model = model_override or self.primary_model
        start = time.perf_counter()

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=self.primary_max_output_tokens,
            temperature=0.1,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        usage = response.usage

        return {
            "answer": choice.message.content or "",
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "llm_ms": elapsed_ms,
        }

    async def generate_stream(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict]] = None,
        current_files: Optional[List[str]] = None,
        scope_mode: str = "global",
        query_type: str = "specific",
        model_override: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming generation. Yields text chunks as they arrive.

        Uses primary_model by default. Pass model_override to route elsewhere.
        """
        messages = self._build_messages(
            query, context, history, current_files, scope_mode, query_type,
        )
        model = model_override or self.primary_model

        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=self.primary_max_output_tokens,
            temperature=0.1,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def generate_followups(
        self,
        context: str,
        answer: str,
        count: int = 3,
    ) -> List[str]:
        """Generate follow-up questions based on document context."""
        count = max(count, self.min_followups)

        messages = [
            {"role": "system", "content": FOLLOWUP_PROMPT.format(count=count)},
            {"role": "user", "content": f"Document context:\n{context[:3000]}\n\nPrevious answer:\n{answer[:1000]}"},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.secondary_model,
                messages=messages,
                max_tokens=300,
                temperature=0.3,
            )

            text = response.choices[0].message.content or "[]"
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            questions = json.loads(text)
            if isinstance(questions, list):
                return [str(q) for q in questions[:count]]
        except Exception as e:
            logger.warning(f"Failed to generate follow-ups: {e}")

        return [
            "Could you provide more details about a specific section?",
            "What other aspects of this document would you like to explore?",
            "Are there any specific terms or data points you'd like explained?",
        ]
