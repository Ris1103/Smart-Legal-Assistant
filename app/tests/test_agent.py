"""
Tests for agent.py: relevance check, Perplexity fallback behavior.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.agent import is_context_relevant, search_perplexity


# ---------------------------------------------------------------------------
# is_context_relevant
# ---------------------------------------------------------------------------

class TestIsContextRelevant:
    def _mock_model(self, answer: str):
        model = MagicMock()
        model.generate_content.return_value = MagicMock(text=answer)
        return model

    def test_returns_true_when_llm_says_yes(self):
        model = self._mock_model("yes")
        docs = [Document(page_content="GST rate is 18% for electronics.")]
        assert is_context_relevant("What is the GST rate?", docs, model) is True

    def test_returns_false_when_llm_says_no(self):
        model = self._mock_model("no")
        docs = [Document(page_content="Unrelated content.")]
        assert is_context_relevant("What is the GST rate?", docs, model) is False

    def test_returns_false_when_no_documents(self):
        model = self._mock_model("yes")  # should not be called
        assert is_context_relevant("query", [], model) is False
        model.generate_content.assert_not_called()

    def test_returns_false_on_model_exception(self):
        model = MagicMock()
        model.generate_content.side_effect = RuntimeError("API error")
        docs = [Document(page_content="Some text.")]
        assert is_context_relevant("query", docs, model) is False

    def test_case_insensitive_yes(self):
        model = self._mock_model("  YES  ")
        docs = [Document(page_content="relevant content")]
        assert is_context_relevant("query", docs, model) is True


# ---------------------------------------------------------------------------
# search_perplexity
# ---------------------------------------------------------------------------

class TestSearchPerplexity:
    @pytest.mark.asyncio
    async def test_returns_fallback_when_no_api_key(self):
        with patch("src.agent.settings") as mock_settings:
            mock_settings.perplexity_api_key = ""
            result = await search_perplexity("What is Section 80C?")
        assert result["query"] == "What is Section 80C?"
        assert "not configured" in result["summary"]
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_successful_web_search(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Section 80C allows deductions."}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.agent.settings") as mock_settings:
            mock_settings.perplexity_api_key = "test-key"
            mock_settings.perplexity_model_name = "llama-3-sonar-large-32k-online"
            with patch("src.agent.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client_cls.return_value.__aenter__ = AsyncMock(
                    return_value=mock_client
                )
                mock_client_cls.return_value.__aexit__ = AsyncMock(
                    return_value=False
                )
                result = await search_perplexity("What is Section 80C?")

        assert "Section 80C" in result["summary"]
        assert len(result["results"]) == 1
