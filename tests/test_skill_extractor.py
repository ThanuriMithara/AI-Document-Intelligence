"""
Unit tests for skill extraction logic.
"""

import pytest
from src.skill_extractor import extract_skills


def test_extract_skills_basic():
    """Checks basic extraction of multiple skills from descriptive text."""
    text = "I have experience with Python, SQL, pandas and machine learning."
    skills = extract_skills(text)
    expected = ["Python", "SQL", "Pandas", "Machine Learning"]
    for s in expected:
        assert s in skills


def test_extract_skills_case_insensitivity():
    """Checks that skills in uppercase, lowercase, or mixed case are extracted."""
    text = "Proficient in PYTHON, sql, and ScIkiT-lEaRn."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "SQL" in skills
    assert "Scikit-learn" in skills


def test_extract_skills_deduplication():
    """Checks that repeated mentions of the same skill produce only 1 entry."""
    text = "Python python PYTHON is my favorite language. Python developers love Python."
    skills = extract_skills(text)
    assert skills.count("Python") == 1


def test_extract_skills_special_characters():
    """Checks skills with special symbols: C++, Node.js, Power BI, Scikit-learn."""
    text = "Core skills include C++, Node.js, Power BI, and Scikit-learn."
    skills = extract_skills(text)
    assert "C++" in skills
    assert "Node.js" in skills
    assert "Power BI" in skills
    assert "Scikit-learn" in skills


def test_extract_skills_avoid_false_substring_matches():
    """Checks that 'Java' does not accidentally trigger just because 'JavaScript' is present."""
    text = "Building frontend apps with JavaScript and React."
    skills = extract_skills(text)
    assert "JavaScript" in skills
    assert "React" in skills
    assert "Java" not in skills  # Java should NOT be extracted here


def test_extract_skills_empty_input():
    """Checks that empty strings or None return an empty list."""
    assert extract_skills("") == []
    assert extract_skills("   ") == []
    assert extract_skills(None) == []


def test_extract_skills_unrelated_text():
    """Checks that text without technical skills returns an empty list."""
    text = "The quick brown fox jumps over the lazy dog."
    assert extract_skills(text) == []
