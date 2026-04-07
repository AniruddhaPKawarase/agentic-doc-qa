"""
Tests for v2 generation service enhancements — citations, model routing, query types.
──────────────────────────────────────────────────────────────────────────────
Validates the citation-enforced system prompt, query-type instructions,
and model routing (primary vs secondary) introduced in v2.1.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.generation_service import (
    GenerationService,
    QUERY_TYPE_INSTRUCTIONS,
    SYSTEM_PROMPT_BASE,
)
from config import Settings


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _settings(**overrides) -> Settings:
    """Create a Settings instance without reading .env files."""
    defaults = {
        "openai_api_key": "test-key",
        "primary_model": "gpt-4o",
        "secondary_model": "gpt-4o-mini",
        "primary_max_output_tokens": 4096,
        "_env_file": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def mock_settings() -> Settings:
    return _settings()


@pytest.fixture
def service(mock_settings: Settings) -> GenerationService:
    """Build a GenerationService with a mocked OpenAI client."""
    svc = GenerationService(mock_settings)
    svc.client = MagicMock()
    return svc


def _mock_chat_response(content: str = "test answer") -> MagicMock:
    """Create a mock that mimics openai chat.completions.create response."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ── Prompt Content Tests ─────────────────────────────────────────────────────


class TestSystemPromptCitations:
    """The v2.1 system prompt must enforce citation rules."""

    def test_system_prompt_has_citation_rules(self) -> None:
        assert "CITATION RULES" in SYSTEM_PROMPT_BASE
        assert "MANDATORY" in SYSTEM_PROMPT_BASE
        assert "[Source:" in SYSTEM_PROMPT_BASE

    def test_system_prompt_has_response_rules(self) -> None:
        assert "RESPONSE RULES" in SYSTEM_PROMPT_BASE

    def test_system_prompt_has_out_of_context(self) -> None:
        assert "OUT-OF-CONTEXT" in SYSTEM_PROMPT_BASE


class TestQueryTypeInstructions:
    """All three query types must have corresponding instructions."""

    def test_query_type_instructions_exist(self) -> None:
        assert "general" in QUERY_TYPE_INSTRUCTIONS
        assert "specific" in QUERY_TYPE_INSTRUCTIONS
        assert "comparison" in QUERY_TYPE_INSTRUCTIONS

    def test_general_instruction_content(self) -> None:
        assert "general/overview" in QUERY_TYPE_INSTRUCTIONS["general"]

    def test_specific_instruction_content(self) -> None:
        assert "specific factual" in QUERY_TYPE_INSTRUCTIONS["specific"]

    def test_comparison_instruction_content(self) -> None:
        assert "comparison question" in QUERY_TYPE_INSTRUCTIONS["comparison"]


# ── System Prompt Builder Tests ──────────────────────────────────────────────


class TestBuildSystemPrompt:
    """_build_system_prompt should include query type instructions."""

    def test_build_system_prompt_includes_query_type_general(
        self, service: GenerationService,
    ) -> None:
        result = service._build_system_prompt(query_type="general")
        assert "general/overview" in result

    def test_build_system_prompt_includes_query_type_comparison(
        self, service: GenerationService,
    ) -> None:
        result = service._build_system_prompt(query_type="comparison")
        assert "comparison question" in result

    def test_build_system_prompt_includes_query_type_specific(
        self, service: GenerationService,
    ) -> None:
        result = service._build_system_prompt(query_type="specific")
        assert "specific factual" in result

    def test_build_system_prompt_unknown_type_no_crash(
        self, service: GenerationService,
    ) -> None:
        result = service._build_system_prompt(query_type="unknown")
        # Should still contain the base prompt, just no query type suffix
        assert "CITATION RULES" in result
        assert "QUERY TYPE" not in result


# ── Model Routing Tests ──────────────────────────────────────────────────────


class TestModelRouting:
    """generate() should use primary_model, follow-ups should use secondary."""

    @pytest.mark.asyncio
    async def test_generate_uses_primary_model(
        self, service: GenerationService,
    ) -> None:
        mock_resp = _mock_chat_response("answer with citations")
        service.client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await service.generate(query="What is this?", context="Some context")

        call_kwargs = service.client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_with_model_override(
        self, service: GenerationService,
    ) -> None:
        mock_resp = _mock_chat_response("overridden model answer")
        service.client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await service.generate(
            query="Quick q", context="ctx", model_override="gpt-4o-mini",
        )

        call_kwargs = service.client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_generate_uses_primary_max_output_tokens(
        self, service: GenerationService,
    ) -> None:
        mock_resp = _mock_chat_response("tokens test")
        service.client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await service.generate(query="test", context="ctx")

        call_kwargs = service.client.chat.completions.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_generate_followups_uses_secondary_model(
        self, service: GenerationService,
    ) -> None:
        followup_resp = _mock_chat_response('["Q1?", "Q2?", "Q3?"]')
        service.client.chat.completions.create = AsyncMock(return_value=followup_resp)

        result = await service.generate_followups(
            context="Some doc context", answer="Previous answer",
        )

        call_kwargs = service.client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
        assert len(result) == 3


# ── Query Type Passthrough Tests ─────────────────────────────────────────────


class TestQueryTypePassthrough:
    """generate() should pass query_type through to the system prompt."""

    @pytest.mark.asyncio
    async def test_generate_passes_query_type_general(
        self, service: GenerationService,
    ) -> None:
        mock_resp = _mock_chat_response("general answer")
        service.client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await service.generate(
            query="Summarize everything", context="ctx", query_type="general",
        )

        # Extract the system message from the call
        call_kwargs = service.client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "general/overview" in system_msg

    @pytest.mark.asyncio
    async def test_generate_passes_query_type_comparison(
        self, service: GenerationService,
    ) -> None:
        mock_resp = _mock_chat_response("comparison answer")
        service.client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await service.generate(
            query="Compare X and Y", context="ctx", query_type="comparison",
        )

        call_kwargs = service.client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "comparison question" in system_msg

    @pytest.mark.asyncio
    async def test_generate_default_query_type_is_specific(
        self, service: GenerationService,
    ) -> None:
        mock_resp = _mock_chat_response("specific answer")
        service.client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await service.generate(query="What is X?", context="ctx")

        call_kwargs = service.client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "specific factual" in system_msg
