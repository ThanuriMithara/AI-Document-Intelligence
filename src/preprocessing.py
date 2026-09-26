"""
Text Preprocessing Module
--------------------------
This module prepares raw, messy text for NLP (Natural Language Processing).

Steps performed:
1. Lowercasing: Ensures "Python" and "python" are treated as the same word.
2. Punctuation Removal: Removes commas, periods, colons, brackets, etc.
3. Whitespace Normalization: Replaces multiple spaces/newlines with a single space.
4. Tokenization: Splits continuous text into individual words (tokens).
5. Stopword Removal: Filters out common English filler words ("and", "the", "in", "is")
   that add little semantic value for topic matching.
"""

import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure stopwords are available; fallback to a basic set if NLTK data is missing
try:
    ENGLISH_STOPWORDS = set(stopwords.words("english"))
except LookupError:
    ENGLISH_STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
        "does", "doing", "down", "during", "each", "few", "for", "from", "further",
        "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
        "let's", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off",
        "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
        "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "theirs", "them", "themselves", "then", "there", "these",
        "they", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself"
    }


def tokenize_text(text: str) -> List[str]:
    """
    Splits text into individual tokens using NLTK word_tokenize with a regex fallback.
    """
    try:
        return word_tokenize(text)
    except Exception:
        # Fallback to simple whitespace/word-boundary splitting if NLTK data is unavailable
        return re.findall(r"\b\w+\b", text)


def preprocess_text(text: str) -> str:
    """
    Cleans raw text for NLP processing.

    Args:
        text (str): Input text (such as a job description or student CV text).

    Returns:
        str: Cleaned, lowercased, and filtered text.

    Example:
        >>> preprocess_text("Seeking an AI Intern with Python, SQL, and Pandas experience!")
        'seeking ai intern python sql pandas experience'
    """
    # Step 0: Handle None, non-strings, or empty input
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Convert to lowercase
    # This prevents duplicate vocabulary entries like "Python" vs "python"
    cleaned = text.lower()

    # Step 2: Normalize whitespace (tabs, newlines, multiple spaces -> single space)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Step 3: Remove unwanted punctuation
    # We keep letters, numbers, spaces, and certain programming symbols like '+' or '#'
    # Replace standard punctuation marks with spaces so words don't get accidentally glued
    for char in string.punctuation:
        if char not in {"+", "#"}:  # Preserve '+' for C++ and '#' for C#
            cleaned = cleaned.replace(char, " ")

    # Re-normalize spaces after punctuation replacement
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Step 4: Tokenize into words
    tokens = tokenize_text(cleaned)

    # Step 5: Remove stopwords
    filtered_tokens = [
        token for token in tokens
        if token not in ENGLISH_STOPWORDS and len(token) > 0
    ]

    # Join filtered tokens back into a clean string for TF-IDF / vectorizers
    return " ".join(filtered_tokens)
