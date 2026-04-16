"""
FastAPI integration tests using TestClient.
All external dependencies (RAG pipeline, MLflow) are mocked.
"""
import base64
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_pipeline():
    """A fully-mocked HybridRAGPipeline singleton."""
    pipeline = MagicMock()
    pipeline.semantic_search_with_scores.return_value = []
    pipeline.process_query.return_value = {
        "query": "test query",
        "summary": "Test summary.",
        "results": [
            {"content": "chunk text", "metadata": {"filename": "test.pdf"}}
        ],
        "metadata": {
            "num_results": 1,
            "page": 1,
            "page_size": 5,
            "source_files": ["test.pdf"],
            "search_type": "hybrid",
            "timestamp": "2026-01-01T00:00:00",
        },
    }
    return pipeline


@pytest.fixture()
def client(mock_pipeline):
    """TestClient with rag_pipeline injected and MLflow disabled."""
    # Build a mock MLflow run context manager that returns a string run_id.
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id-123"
    mock_mlflow_cm = MagicMock()
    mock_mlflow_cm.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow_cm.__exit__ = MagicMock(return_value=False)

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value = mock_mlflow_cm

    import main as main_module
    original_pipeline = main_module.rag_pipeline
    main_module.rag_pipeline = mock_pipeline
    try:
        with patch("main.mlflow", mock_mlflow):
            from main import app
            with TestClient(app) as c:
                yield c, mock_pipeline
    finally:
        main_module.rag_pipeline = original_pipeline


# ---------------------------------------------------------------------------
# /retrieve
# ---------------------------------------------------------------------------

class TestRetrieveEndpoint:
    def test_retrieve_returns_200(self, client):
        c, pipeline = client
        # is_context_relevant → True so local search is used
        with patch("main.is_context_relevant", return_value=True):
            resp = c.post("/retrieve", json={"user_query": "What is GST?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "results" in data

    def test_retrieve_short_query_rejected(self, client):
        c, _ = client
        resp = c.post("/retrieve", json={"user_query": "Hi"})
        assert resp.status_code == 422  # min_length=3

    def test_retrieve_invalid_search_type_rejected(self, client):
        c, _ = client
        resp = c.post(
            "/retrieve",
            json={"user_query": "What is GST?", "search_type": "fuzzy"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------

class TestIngestEndpoint:
    def _payload(self, filename="test.pdf"):
        tiny_pdf = b"%PDF-1.4 fake"
        return {
            "base64_text": base64.b64encode(tiny_pdf).decode(),
            "file_type": ".pdf",
            "filename": filename,
            "metadata": {},
        }

    def test_ingest_success(self, client):
        c, pipeline = client
        pipeline.vectorstore = MagicMock()
        with patch(
            "main.ingest_document_from_base64", return_value=5
        ):
            resp = c.post("/ingest", json=self._payload())
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        assert resp.json()["chunks_added"] == 5

    def test_ingest_duplicate(self, client):
        c, pipeline = client
        pipeline.vectorstore = MagicMock()
        with patch("main.ingest_document_from_base64", return_value=0):
            resp = c.post("/ingest", json=self._payload())
        assert resp.status_code == 200
        assert resp.json()["status"] == "duplicate"

    def test_ingest_oversized_file_returns_413(self, client):
        c, pipeline = client
        pipeline.vectorstore = MagicMock()
        with patch(
            "main.ingest_document_from_base64",
            side_effect=ValueError("exceeds the 50 MB limit"),
        ):
            resp = c.post("/ingest", json=self._payload())
        assert resp.status_code == 413

    def test_ingest_wrong_filetype_rejected(self, client):
        c, _ = client
        payload = self._payload()
        payload["file_type"] = ".docx"
        resp = c.post("/ingest", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /refresh-index
# ---------------------------------------------------------------------------

class TestRefreshIndexEndpoint:
    def test_refresh_returns_200(self, client):
        c, pipeline = client
        pipeline.documents = ["doc1", "doc2"]
        resp = c.post("/refresh-index")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        assert resp.json()["documents_indexed"] == 2
