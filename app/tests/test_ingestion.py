"""
Tests for ingestion_src: duplicate detection, file size, category.
"""
import base64
import hashlib
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.ingestion_src import (
    get_category,
    ingest_document_from_base64,
)


# ---------------------------------------------------------------------------
# get_category
# ---------------------------------------------------------------------------

class TestGetCategory:
    def test_gst_keyword(self):
        assert get_category("GST_rules_2024.pdf") == "GST"

    def test_income_tax_keyword(self):
        assert get_category("income_tax_act.pdf") == "Income Tax"

    def test_penal_code_keyword(self):
        assert get_category("IPC_sections.pdf") == "Penal Code"

    def test_company_act_keyword(self):
        assert get_category("companies_act_2013.pdf") == "Company Act"

    def test_fallback_other(self):
        assert get_category("some_random_document.pdf") == "Other"

    def test_case_insensitive(self):
        assert get_category("CGST_circular.pdf") == "GST"


# ---------------------------------------------------------------------------
# ingest_document_from_base64 — file size validation
# ---------------------------------------------------------------------------

class TestFileSizeValidation:
    def test_oversized_file_raises_value_error(self):
        # Build a fake PDF bytes payload that exceeds the limit.
        # We patch settings.max_file_size_mb to 1 to keep the test fast.
        large_bytes = b"x" * (2 * 1024 * 1024)  # 2 MB
        encoded = base64.b64encode(large_bytes).decode()
        mock_vs = MagicMock()

        with patch(
            "src.ingestion.ingestion_src.settings"
        ) as mock_settings:
            mock_settings.max_file_size_mb = 1  # 1 MB limit
            with pytest.raises(ValueError, match="exceeds the"):
                ingest_document_from_base64(
                    vectorstore=mock_vs,
                    base64_text=encoded,
                    filename="big_file.pdf",
                    file_type=".pdf",
                    metadata={},
                )


# ---------------------------------------------------------------------------
# ingest_document_from_base64 — duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def _make_tiny_pdf_b64(self) -> tuple[str, str]:
        """Return (base64_str, sha256_hex) for a minimal fake PDF."""
        content = b"%PDF-1.4 fake content"
        encoded = base64.b64encode(content).decode()
        digest = hashlib.sha256(content).hexdigest()
        return encoded, digest

    def test_duplicate_returns_zero(self):
        b64, file_hash = self._make_tiny_pdf_b64()
        mock_vs = MagicMock()
        # Simulate ChromaDB finding the hash already stored.
        mock_vs.get_by_metadata.return_value = {"ids": ["existing-id-1"]}

        with patch("src.ingestion.ingestion_src.settings") as mock_settings:
            mock_settings.max_file_size_mb = 50
            result = ingest_document_from_base64(
                vectorstore=mock_vs,
                base64_text=b64,
                filename="duplicate.pdf",
                file_type=".pdf",
                metadata={},
            )

        assert result == 0
        mock_vs.add_documents.assert_not_called()

    def test_new_document_proceeds_to_loader(self):
        b64, _ = self._make_tiny_pdf_b64()
        mock_vs = MagicMock()
        # No existing document found.
        mock_vs.get_by_metadata.return_value = {"ids": []}

        with patch("src.ingestion.ingestion_src.settings") as mock_settings:
            mock_settings.max_file_size_mb = 50
            # PyPDFLoader would fail on fake bytes, so patch it out.
            with patch(
                "src.ingestion.ingestion_src.PyPDFLoader"
            ) as mock_loader_cls:
                from langchain_core.documents import Document
                mock_loader = MagicMock()
                mock_loader.load.return_value = [
                    Document(page_content="Test content", metadata={})
                ]
                mock_loader_cls.return_value = mock_loader

                chunks_added = ingest_document_from_base64(
                    vectorstore=mock_vs,
                    base64_text=b64,
                    filename="new_doc.pdf",
                    file_type=".pdf",
                    metadata={},
                )

        assert chunks_added > 0
        mock_vs.add_documents.assert_called()
