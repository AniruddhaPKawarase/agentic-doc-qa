"""
Tests for FullTextStore — in-memory full document text storage.
──────────────────────────────────────────────────────────────────────────────
Validates storage, retrieval, filtering, isolation, and cleanup operations.
"""

import pytest

from services.fulltext_store import FullTextStore, StoredDocument


@pytest.fixture
def store() -> FullTextStore:
    return FullTextStore()


@pytest.fixture
def populated_store(store: FullTextStore) -> FullTextStore:
    store.store("session-1", "doc_a.pdf", "Hello world from doc A.", ["Hello world from doc A."])
    store.store("session-1", "doc_b.pdf", "Content of doc B is here.", ["Content of doc B is here."])
    store.store("session-2", "doc_c.pdf", "Isolated session two text.", ["Isolated session two text."])
    return store


class TestStoreAndRetrieve:

    def test_store_and_retrieve(self, store: FullTextStore) -> None:
        store.store("s1", "file.pdf", "Some document text.", ["Some document text."])
        result = store.get_session_text("s1")
        assert "Some document text." in result

    def test_get_document(self, populated_store: FullTextStore) -> None:
        doc = populated_store.get_document("session-1", "doc_a.pdf")
        assert doc is not None
        assert isinstance(doc, StoredDocument)
        assert doc.file_name == "doc_a.pdf"
        assert doc.full_text == "Hello world from doc A."
        assert doc.token_count > 0

    def test_get_document_missing(self, populated_store: FullTextStore) -> None:
        doc = populated_store.get_document("session-1", "nonexistent.pdf")
        assert doc is None


class TestSessionText:

    def test_get_session_text_all(self, populated_store: FullTextStore) -> None:
        text = populated_store.get_session_text("session-1")
        assert "=== FILE: doc_a.pdf ===" in text
        assert "=== FILE: doc_b.pdf ===" in text
        assert "Hello world from doc A." in text
        assert "Content of doc B is here." in text

    def test_get_session_text_filtered(self, populated_store: FullTextStore) -> None:
        text = populated_store.get_session_text("session-1", file_names=["doc_a.pdf"])
        assert "=== FILE: doc_a.pdf ===" in text
        assert "doc_b.pdf" not in text


class TestTokenCounting:

    def test_get_session_token_count(self, populated_store: FullTextStore) -> None:
        count = populated_store.get_session_token_count("session-1")
        assert count > 0
        # Both docs should contribute tokens
        count_a = populated_store.get_session_token_count("session-1", file_names=["doc_a.pdf"])
        count_b = populated_store.get_session_token_count("session-1", file_names=["doc_b.pdf"])
        assert count == count_a + count_b

    def test_get_session_token_count_filtered(self, populated_store: FullTextStore) -> None:
        count_all = populated_store.get_session_token_count("session-1")
        count_a = populated_store.get_session_token_count("session-1", file_names=["doc_a.pdf"])
        assert 0 < count_a < count_all


class TestSessionManagement:

    def test_delete_session(self, populated_store: FullTextStore) -> None:
        populated_store.delete_session("session-1")
        assert populated_store.get_session_text("session-1") == ""
        # Session 2 should be unaffected
        assert "Isolated session two text." in populated_store.get_session_text("session-2")

    def test_session_isolation(self, populated_store: FullTextStore) -> None:
        text_s1 = populated_store.get_session_text("session-1")
        text_s2 = populated_store.get_session_text("session-2")
        assert "Isolated session two text." not in text_s1
        assert "Hello world from doc A." not in text_s2

    def test_get_file_names(self, populated_store: FullTextStore) -> None:
        names = populated_store.get_file_names("session-1")
        assert set(names) == {"doc_a.pdf", "doc_b.pdf"}

    def test_empty_session(self, store: FullTextStore) -> None:
        assert store.get_session_text("nonexistent") == ""
        assert store.get_session_token_count("nonexistent") == 0
        assert store.get_file_names("nonexistent") == []
        # delete_session on nonexistent should not raise
        store.delete_session("nonexistent")
