"""
Unit tests for similarity matching, skill gap analysis, and learning recommendations.
"""

import pytest
from src.matcher import calculate_similarity, analyze_skill_gap
from src.recommendations import recommend_learning


def test_similarity_identical_texts():
    """Identical texts should yield a high similarity score near 1.0."""
    text = "Python developer working with FastAPI, Docker, and PostgreSQL databases."
    score = calculate_similarity(text, text)
    assert 0.95 <= score <= 1.0


def test_similarity_disjoint_texts():
    """Completely unrelated texts should have a similarity score near 0.0."""
    text_a = "Cooking delicious homemade pasta with tomato basil sauce."
    text_b = "Quantum computing algorithms using superconducting qubits and cryogenics."
    score = calculate_similarity(text_a, text_b)
    assert score == pytest.approx(0.0, abs=0.05)


def test_similarity_empty_text():
    """Empty inputs should safely return 0.0 without errors."""
    assert calculate_similarity("", "Python developer") == 0.0
    assert calculate_similarity("Data analyst", "") == 0.0
    assert calculate_similarity("", "") == 0.0


def test_analyze_skill_gap_standard():
    """Checks standard scenario: some matched, some missing."""
    required = ["Python", "SQL", "Machine Learning", "Docker"]
    student = ["Python", "SQL", "Machine Learning"]

    result = analyze_skill_gap(required, student)

    assert result["matched_skills"] == ["Python", "SQL", "Machine Learning"]
    assert result["missing_skills"] == ["Docker"]
    assert result["match_percentage"] == 75.0


def test_analyze_skill_gap_case_insensitive():
    """Checks that case differences (e.g. 'python' vs 'Python') match correctly."""
    required = ["Python", "SQL"]
    student = ["python", "sql"]

    result = analyze_skill_gap(required, student)
    assert result["matched_skills"] == ["Python", "SQL"]
    assert result["missing_skills"] == []
    assert result["match_percentage"] == 100.0


def test_analyze_skill_gap_zero_required_skills():
    """Checks that zero required skills returns 0.0% without division-by-zero crashes."""
    result = analyze_skill_gap([], ["Python", "SQL"])
    assert result["matched_skills"] == []
    assert result["missing_skills"] == []
    assert result["match_percentage"] == 0.0


def test_analyze_skill_gap_zero_student_skills():
    """Checks that zero student skills results in all skills missing and 0.0% match."""
    required = ["Python", "Docker", "Git"]
    result = analyze_skill_gap(required, [])
    assert result["matched_skills"] == []
    assert result["missing_skills"] == ["Python", "Docker", "Git"]
    assert result["match_percentage"] == 0.0


def test_recommend_learning_basic():
    """Checks that known missing skills yield appropriate recommendations."""
    missing = ["Docker", "PyTorch"]
    recs = recommend_learning(missing)

    assert len(recs) == 2
    assert "Docker fundamentals" in recs[0]
    assert "PyTorch fundamentals" in recs[1]


def test_recommend_learning_empty():
    """Checks that having 0 missing skills produces a positive congratulatory message."""
    recs = recommend_learning([])
    assert len(recs) == 1
    assert "Great job" in recs[0]


def test_recommend_learning_unmapped_skill():
    """Checks that an unmapped custom skill gets a sensible fallback recommendation."""
    missing = ["CustomToolX"]
    recs = recommend_learning(missing)
    assert len(recs) == 1
    assert "CustomToolX fundamentals" in recs[0]
