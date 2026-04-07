"""Tests for BM25 keyword search service."""

from __future__ import annotations

import pytest

from services.bm25_service import BM25Service, tokenize
from services.file_processor import Chunk


# -- Helpers ------------------------------------------------------------------

def _make_chunk(text: str, file_name: str = "test.docx", idx: int = 0) -> Chunk:
    return Chunk(
        text=text,
        file_name=file_name,
        file_type=".docx",
        chunk_index=idx,
        token_count=len(text.split()),
    )


# -- Tokenizer tests ---------------------------------------------------------

def test_tokenize_basic():
    result = tokenize("Fire Alarm Systems!")
    assert result == ["fire", "alarm", "systems"]


def test_tokenize_removes_stopwords():
    result = tokenize("the fire and the alarm")
    assert result == ["fire", "alarm"]


def test_tokenize_removes_short_words():
    result = tokenize("I am a PM")
    assert result == []


# -- BM25Service tests -------------------------------------------------------

@pytest.fixture()
def bm25() -> BM25Service:
    return BM25Service()


def test_index_and_search(bm25: BM25Service):
    chunks = [
        _make_chunk("fire alarm system installation and testing", idx=0),
        _make_chunk("electrical panel wiring and conduit routing", idx=1),
        _make_chunk("plumbing water supply pipe sizing", idx=2),
    ]
    bm25.index_chunks("s1", chunks)
    results = bm25.search("s1", "fire alarm")
    assert len(results) > 0
    assert results[0][0] == 0  # fire chunk should be first


def test_search_returns_normalized_scores(bm25: BM25Service):
    chunks = [
        _make_chunk("fire alarm system installation and testing", idx=0),
        _make_chunk("electrical panel wiring and conduit routing", idx=1),
        _make_chunk("plumbing water supply pipe sizing", idx=2),
    ]
    bm25.index_chunks("s1", chunks)
    results = bm25.search("s1", "fire alarm")
    assert len(results) > 0
    # Top result must be normalized to 1.0
    assert results[0][1] == pytest.approx(1.0)


def test_search_empty_session(bm25: BM25Service):
    results = bm25.search("nonexistent", "fire alarm")
    assert results == []


def test_search_top_k_limit(bm25: BM25Service):
    chunks = [
        _make_chunk("fire alarm system installation and testing procedures", idx=0),
        _make_chunk("fire alarm detector placement guidelines ceiling mount", idx=1),
        _make_chunk("electrical panel wiring fire rated conduit", idx=2),
        _make_chunk("plumbing water supply pipe sizing residential", idx=3),
        _make_chunk("hvac duct insulation fire damper requirements", idx=4),
    ]
    bm25.index_chunks("s1", chunks)
    results = bm25.search("s1", "fire alarm", top_k=2)
    assert len(results) == 2


def test_append_chunks(bm25: BM25Service):
    batch_1 = [
        _make_chunk("fire alarm system installation", idx=0),
        _make_chunk("electrical panel wiring", idx=1),
    ]
    batch_2 = [
        _make_chunk("plumbing water supply", idx=2),
        _make_chunk("hvac duct insulation requirements", idx=3),
    ]
    bm25.index_chunks("s1", batch_1)
    bm25.index_chunks("s1", batch_2)

    # Should find results from both batches
    fire_results = bm25.search("s1", "fire alarm")
    assert any(idx == 0 for idx, _ in fire_results)

    hvac_results = bm25.search("s1", "hvac duct insulation")
    assert any(idx == 3 for idx, _ in hvac_results)


def test_delete_session(bm25: BM25Service):
    chunks = [_make_chunk("fire alarm system", idx=0)]
    bm25.index_chunks("s1", chunks)
    bm25.delete_session("s1")
    results = bm25.search("s1", "fire alarm")
    assert results == []


def test_session_isolation(bm25: BM25Service):
    bm25.index_chunks("s1", [_make_chunk("fire alarm system", idx=0)])
    bm25.index_chunks("s2", [_make_chunk("plumbing water supply", idx=0)])

    s1_results = bm25.search("s1", "plumbing water")
    s2_results = bm25.search("s2", "fire alarm")

    # Neither session should return results for the other's content
    assert s1_results == []
    assert s2_results == []
