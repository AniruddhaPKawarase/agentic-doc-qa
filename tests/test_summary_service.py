"""
Tests for SummaryService and SummaryStore.
──────────────────────────────────────────────────────────────────────────────
Validates summary generation (with mocked OpenAI), storage, retrieval,
filtering, isolation, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.summary_service import SummaryService, SummaryStore


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> SummaryStore:
    return SummaryStore()


@pytest.fixture
def populated_store(store: SummaryStore) -> SummaryStore:
    store.store("session-1", "spec.pdf", "Summary of spec document.")
    store.store("session-1", "drawing.pdf", "Summary of drawing document.")
    store.store("session-2", "other.docx", "Summary of other document.")
    return store


# ── SummaryStore Tests ───────────────────────────────────────────────────


class TestSummaryStoreBasic:

    def test_store_and_retrieve(self, store: SummaryStore) -> None:
        """Store a summary and retrieve it back."""
        store.store("s1", "file.pdf", "This is a test summary.")
        result = store.get("s1", "file.pdf")
        assert result == "This is a test summary."

    def test_get_missing_returns_none(self, store: SummaryStore) -> None:
        """Missing file returns None, not an error."""
        result = store.get("s1", "nonexistent.pdf")
        assert result is None

    def test_get_missing_session_returns_none(self, store: SummaryStore) -> None:
        """Missing session also returns None."""
        result = store.get("nonexistent-session", "file.pdf")
        assert result is None


class TestSummaryStoreSessionSummaries:

    def test_get_session_summaries_all(
        self, populated_store: SummaryStore
    ) -> None:
        """Get combined summaries for all files in a session."""
        combined = populated_store.get_session_summaries("session-1")
        assert "=== SUMMARY: spec.pdf ===" in combined
        assert "=== SUMMARY: drawing.pdf ===" in combined
        assert "Summary of spec document." in combined
        assert "Summary of drawing document." in combined

    def test_get_session_summaries_filtered(
        self, populated_store: SummaryStore
    ) -> None:
        """Filter combined summaries by file names."""
        combined = populated_store.get_session_summaries(
            "session-1", file_names=["spec.pdf"]
        )
        assert "=== SUMMARY: spec.pdf ===" in combined
        assert "Summary of spec document." in combined
        assert "drawing.pdf" not in combined

    def test_get_session_summaries_empty_session(
        self, store: SummaryStore
    ) -> None:
        """Empty/unknown session returns empty string."""
        result = store.get_session_summaries("nonexistent")
        assert result == ""


class TestSummaryStoreSessionManagement:

    def test_delete_session(self, populated_store: SummaryStore) -> None:
        """Delete clears all summaries for that session."""
        populated_store.delete_session("session-1")
        assert populated_store.get("session-1", "spec.pdf") is None
        assert populated_store.get("session-1", "drawing.pdf") is None
        # Session 2 should be unaffected
        assert populated_store.get("session-2", "other.docx") is not None

    def test_session_isolation(self, populated_store: SummaryStore) -> None:
        """Different sessions do not see each other's summaries."""
        s1_combined = populated_store.get_session_summaries("session-1")
        s2_combined = populated_store.get_session_summaries("session-2")
        assert "other.docx" not in s1_combined
        assert "spec.pdf" not in s2_combined


# ── SummaryService Tests (mocked OpenAI) ─────────────────────────────────


class TestSummaryServiceGenerate:

    @pytest.mark.asyncio
    async def test_generate_summary_calls_openai(self) -> None:
        """SummaryService should call OpenAI with the summary prompt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "This is a construction spec document..."
        )

        with patch("services.summary_service.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            service = SummaryService(api_key="test-key", model="gpt-4o-mini")
            result = await service.generate_summary(
                "test.docx", "Some document content here"
            )

            assert result == "This is a construction spec document..."
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args
            assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
            assert call_kwargs.kwargs["temperature"] == 0.1
            assert call_kwargs.kwargs["max_tokens"] == 2048
            # Verify system prompt is present
            messages = call_kwargs.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert "document analysis expert" in messages[0]["content"]
            # Verify user message contains file name and content
            assert "test.docx" in messages[1]["content"]
            assert "Some document content here" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_generate_summary_handles_error(self) -> None:
        """On API error, return a fallback string instead of raising."""
        with patch("services.summary_service.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )

            service = SummaryService(api_key="test-key", model="gpt-4o-mini")
            result = await service.generate_summary(
                "report.pdf", "Some content"
            )

            assert "Summary generation failed for report.pdf" in result
            assert "tokens" in result

    @pytest.mark.asyncio
    async def test_generate_summary_truncates_long_text(self) -> None:
        """Text over max_input_tokens gets truncated with a note."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Truncated summary result"

        # Use a very small max to trigger truncation easily
        with patch("services.summary_service.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            service = SummaryService(
                api_key="test-key",
                model="gpt-4o-mini",
                max_input_tokens=10,  # very small limit to force truncation
            )
            long_text = "word " * 500  # well over 10 tokens
            result = await service.generate_summary("big.pdf", long_text)

            assert result == "Truncated summary result"
            # Verify the user message includes the truncation note
            call_kwargs = mock_client.chat.completions.create.call_args
            user_msg = call_kwargs.kwargs["messages"][1]["content"]
            assert "[NOTE: Document was truncated" in user_msg
            assert "10 tokens" in user_msg
