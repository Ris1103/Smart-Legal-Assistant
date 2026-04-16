"""
Tests for evaluation.py: sentinel value behavior, score parsing.
"""
from unittest.mock import MagicMock

from src.evaluation.evaluation import (
    FAITHFULNESS_ERROR_SENTINEL,
    calculate_faithfulness,
)


class TestFaithfulness:
    def _mock_model(self, response_text: str):
        model = MagicMock()
        model.generate_content.return_value = MagicMock(text=response_text)
        return model

    def test_valid_score_returned(self):
        model = self._mock_model("0.85")
        docs = [{"content": "The sky is blue."}]
        score = calculate_faithfulness("What color is sky?", docs, "The sky is blue.", model)
        assert score == pytest.approx(0.85)

    def test_unparseable_response_returns_sentinel(self):
        model = self._mock_model("I cannot evaluate this.")
        docs = [{"content": "Some legal text."}]
        score = calculate_faithfulness("query", docs, "summary", model)
        assert score == FAITHFULNESS_ERROR_SENTINEL

    def test_empty_docs_returns_zero(self):
        model = self._mock_model("1.0")
        score = calculate_faithfulness("query", [], "summary", model)
        assert score == 0.0

    def test_model_exception_returns_sentinel(self):
        model = MagicMock()
        model.generate_content.side_effect = RuntimeError("API error")
        docs = [{"content": "Some text."}]
        score = calculate_faithfulness("query", docs, "summary", model)
        assert score == FAITHFULNESS_ERROR_SENTINEL

    def test_sentinel_distinct_from_zero(self):
        assert FAITHFULNESS_ERROR_SENTINEL != 0.0
        assert FAITHFULNESS_ERROR_SENTINEL < 0


import pytest  # noqa: E402 — placed after class to satisfy import order checker
