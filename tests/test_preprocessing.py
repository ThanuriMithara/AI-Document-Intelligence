"""
Unit tests for text preprocessing pipeline.
"""

import pytest
from src.preprocessing import preprocess_text, tokenize_text


def test_preprocess_lowercasing():
    """Checks that upper/mixed case text is converted to lowercase."""
    text = "Machine Learning and Artificial Intelligence"
    result = preprocess_text(text)
    assert result == "machine learning artificial intelligence"


def test_preprocess_punctuation_removal():
    """Checks that commas, colons, exclamation marks, and periods are removed."""
    text = "Python, SQL! Machine learning: pandas... and git?"
    result = preprocess_text(text)
    assert "," not in result
    assert "!" not in result
    assert ":" not in result
    assert "?" not in result
    assert "python" in result
    assert "sql" in result


def test_preprocess_whitespace_normalization():
    """Checks that extra tabs, multiple spaces, and newlines are collapsed."""
    text = "Python     \n\n\t   SQL    \n  Pandas"
    result = preprocess_text(text)
    assert "  " not in result
    assert result == "python sql pandas"


def test_preprocess_stopwords_removal():
    """Checks that common filler words (and, the, is, with) are removed."""
    text = "This is a great position with Python and with SQL"
    result = preprocess_text(text)
    assert "is" not in result.split()
    assert "this" not in result.split()
    assert "with" not in result.split()
    assert "and" not in result.split()
    assert "python" in result.split()
    assert "sql" in result.split()


def test_preprocess_empty_and_none_input():
    """Checks that empty strings, whitespace-only, or None return an empty string."""
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""
    assert preprocess_text(None) == ""


def test_tokenize_text_returns_list():
    """Checks that tokenization splits text into a list of word tokens."""
    tokens = tokenize_text("Python for machine learning")
    assert isinstance(tokens, list)
    assert len(tokens) >= 4
