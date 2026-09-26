"""
Integration tests validating the entire pipeline end-to-end across multiple job roles.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.skill_extractor import extract_skills
from src.matcher import calculate_similarity, analyze_skill_gap
from src.recommendations import recommend_learning


@pytest.fixture
def sample_jobs_df() -> pd.DataFrame:
    csv_path = Path(__file__).parent.parent / "data" / "sample_jobs.csv"
    assert csv_path.exists()
    return pd.read_csv(csv_path)


def test_scenario_aiml_intern(sample_jobs_df: pd.DataFrame):
    """Scenario 1: AI/ML Intern with partial match."""
    job = sample_jobs_df[sample_jobs_df["job_title"] == "AI/ML Intern"].iloc[0]
    required = extract_skills(job["job_description"])

    student_skills = ["Python", "Pandas", "NumPy", "Git"]
    gap = analyze_skill_gap(required, student_skills)
    sim = calculate_similarity(job["job_description"], "Python, Pandas, NumPy, Git")
    recs = recommend_learning(gap["missing_skills"])

    assert "Python" in gap["matched_skills"]
    assert "Pandas" in gap["matched_skills"]
    assert "Git" in gap["matched_skills"]
    assert len(gap["missing_skills"]) > 0
    assert 0.0 < gap["match_percentage"] < 100.0
    assert 0.0 < sim < 1.0
    assert len(recs) == len(gap["missing_skills"])


def test_scenario_data_analyst_intern(sample_jobs_df: pd.DataFrame):
    """Scenario 2: Data Analyst Intern with solid match."""
    job = sample_jobs_df[sample_jobs_df["job_title"] == "Data Analyst Intern"].iloc[0]
    required = extract_skills(job["job_description"])

    student_skills = ["SQL", "Excel", "Power BI", "Data Analysis", "Python", "Pandas"]
    gap = analyze_skill_gap(required, student_skills)

    assert gap["match_percentage"] >= 80.0
    assert "SQL" in gap["matched_skills"]
    assert "Excel" in gap["matched_skills"]


def test_scenario_web_dev_intern(sample_jobs_df: pd.DataFrame):
    """Scenario 3: Web Developer Intern with missing React."""
    job = sample_jobs_df[sample_jobs_df["job_title"] == "Web Developer Intern"].iloc[0]
    required = extract_skills(job["job_description"])

    student_skills = ["HTML", "CSS", "JavaScript"]
    gap = analyze_skill_gap(required, student_skills)

    assert "HTML" in gap["matched_skills"]
    assert "CSS" in gap["matched_skills"]
    assert "JavaScript" in gap["matched_skills"]
    assert "React" in gap["missing_skills"]

    recs = recommend_learning(gap["missing_skills"])
    assert any("React" in r for r in recs)


def test_scenario_zero_student_skills(sample_jobs_df: pd.DataFrame):
    """Scenario 4: Complete beginner with zero skills."""
    job = sample_jobs_df.iloc[0]
    required = extract_skills(job["job_description"])

    gap = analyze_skill_gap(required, [])
    sim = calculate_similarity(job["job_description"], "")
    recs = recommend_learning(gap["missing_skills"])

    assert gap["matched_skills"] == []
    assert len(gap["missing_skills"]) == len(required)
    assert gap["match_percentage"] == 0.0
    assert sim == 0.0
    assert len(recs) == len(required)


def test_scenario_all_matched_skills():
    """Scenario 5: Perfect match with 100% score."""
    required = ["Python", "SQL"]
    student = ["Python", "SQL"]

    gap = analyze_skill_gap(required, student)
    recs = recommend_learning(gap["missing_skills"])

    assert gap["match_percentage"] == 100.0
    assert gap["missing_skills"] == []
    assert "Great job" in recs[0]
