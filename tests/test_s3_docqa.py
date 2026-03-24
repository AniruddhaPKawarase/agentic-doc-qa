"""Phase 7.5: Document QA Agent S3 migration tests."""
import json
import os
import sys
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

AGENT_ROOT = Path(__file__).resolve().parent.parent
PROD_ROOT = AGENT_ROOT.parent
sys.path.insert(0, str(PROD_ROOT))
sys.path.insert(0, str(AGENT_ROOT))

TEST_BUCKET = "test-vcs-agents"


@pytest.fixture(autouse=True)
def s3_env(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.setenv("S3_BUCKET_NAME", TEST_BUCKET)
    monkeypatch.setenv("S3_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("S3_AGENT_PREFIX", "document-qa-agent")
    monkeypatch.setenv("S3_ENDPOINT_URL", "")
    from s3_utils.config import get_s3_config
    from s3_utils.client import get_s3_client
    get_s3_config.cache_clear()
    get_s3_client.cache_clear()


@pytest.fixture
def s3_bucket():
    with mock_aws():
        from s3_utils.config import get_s3_config
        from s3_utils.client import get_s3_client
        get_s3_config.cache_clear()
        get_s3_client.cache_clear()
        conn = boto3.client("s3", region_name="us-east-1")
        conn.create_bucket(Bucket=TEST_BUCKET)
        yield conn
        get_s3_config.cache_clear()
        get_s3_client.cache_clear()


class TestSessionPersistence:
    def test_upload_session_meta(self, s3_bucket):
        from s3_utils.operations import upload_bytes, download_bytes
        from s3_utils.helpers import docqa_session_meta_key
        session = {
            "session_id": "abc123",
            "created_at": "2026-03-23T00:00:00Z",
            "file_count": 2,
            "total_chunks": 50,
            "history": [{"role": "user", "content": "What is this?"}],
        }
        s3_key = docqa_session_meta_key("document-qa-agent", "abc123")
        assert upload_bytes(json.dumps(session).encode(), s3_key) is True
        restored = json.loads(download_bytes(s3_key))
        assert restored["session_id"] == "abc123"
        assert restored["total_chunks"] == 50

    def test_list_sessions(self, s3_bucket):
        from s3_utils.operations import upload_bytes, list_objects
        from s3_utils.helpers import docqa_session_meta_key
        for i in range(3):
            s3_key = docqa_session_meta_key("document-qa-agent", f"sess_{i}")
            upload_bytes(json.dumps({"id": i}).encode(), s3_key)
        objects = list_objects("document-qa-agent/session_data/")
        assert len(objects) == 3


class TestFAISSIndexPersistence:
    def test_upload_faiss_index(self, s3_bucket, tmp_path):
        from s3_utils.operations import upload_file, object_exists
        from s3_utils.helpers import docqa_session_index_key
        idx = tmp_path / "faiss_index.bin"
        idx.write_bytes(b"\x00" * 256)
        s3_key = docqa_session_index_key("document-qa-agent", "sess123")
        assert upload_file(str(idx), s3_key) is True
        assert object_exists(s3_key) is True

    def test_upload_chunks_jsonl(self, s3_bucket):
        from s3_utils.operations import upload_bytes, download_bytes
        from s3_utils.helpers import docqa_session_chunks_key
        chunks = [json.dumps({"text": f"chunk {i}", "page": i}) for i in range(5)]
        s3_key = docqa_session_chunks_key("document-qa-agent", "sess123")
        assert upload_bytes("\n".join(chunks).encode(), s3_key) is True
        content = download_bytes(s3_key).decode()
        assert len(content.strip().split("\n")) == 5


class TestSessionDelete:
    def test_delete_session_prefix(self, s3_bucket):
        from s3_utils.operations import upload_bytes, delete_prefix, list_objects
        from s3_utils.helpers import docqa_session_meta_key, docqa_session_index_key, docqa_session_chunks_key
        sid = "delete_me"
        upload_bytes(b'{}', docqa_session_meta_key("document-qa-agent", sid))
        upload_bytes(b'\x00', docqa_session_index_key("document-qa-agent", sid))
        upload_bytes(b'{}', docqa_session_chunks_key("document-qa-agent", sid))
        deleted = delete_prefix(f"document-qa-agent/session_data/{sid}/")
        assert deleted == 3
        assert list_objects(f"document-qa-agent/session_data/{sid}/") == []


class TestDocQARollback:
    def test_local_mode(self, monkeypatch):
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        from s3_utils.config import get_s3_config
        get_s3_config.cache_clear()
        assert get_s3_config().is_s3_enabled is False

    def test_no_s3_calls_in_local_mode(self, monkeypatch):
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        from s3_utils.config import get_s3_config
        from s3_utils.client import get_s3_client
        get_s3_config.cache_clear()
        get_s3_client.cache_clear()
        from s3_utils.operations import upload_bytes
        assert upload_bytes(b"test", "key") is False
