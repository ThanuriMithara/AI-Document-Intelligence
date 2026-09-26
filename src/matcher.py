"""
Matcher Module
---------------
Performs two types of matching:
1. TF-IDF & Cosine Similarity:
   Measures holistic semantic/textual overlap between raw job text and student text.
2. Direct Skill Gap Analysis:
   Compares required technical skills against candidate skills to find matches,
   missing requirements, and calculate the skill match percentage.

DISCLAIMER:
Text similarity and skill match percentages indicate textual overlap only.
They are NOT predictions of candidate quality, competency, or hiring probability.
"""

from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text


def calculate_similarity(job_description: str, student_profile: str) -> float:
    """
    Computes text similarity using TF-IDF vectorization and Cosine Similarity.

    HOW IT WORKS:
    1. TF-IDF (Term Frequency - Inverse Document Frequency):
       - Converts raw text into numerical feature vectors.
       - 'Term Frequency' (TF): How frequently a word appears in a document.
       - 'Inverse Document Frequency' (IDF): Reduces the weight of very common words
         and increases the weight of rare, informative words (e.g. 'PyTorch').
    2. Cosine Similarity:
       - Measures the cosine of the angle between two multi-dimensional vectors in space.
       - If two texts share similar key vocabulary in similar proportions, the angle is 0
         and cosine similarity is 1.0 (100%).
       - If they share no words, the vectors are orthogonal and similarity is 0.0 (0%).

    WHY THIS IS GREAT FOR BEGINNERS:
    - Fast, lightweight, and runs on any CPU without neural networks or GPU memory.
    - Completely transparent and mathematically interpretable.
    - A standard industry baseline for information retrieval and document matching.

    Args:
        job_description (str): Raw or preprocessed job description text.
        student_profile (str): Raw or preprocessed student CV or profile text.

    Returns:
        float: Similarity score between 0.0 and 1.0.
    """
    # Clean both texts first
    clean_job = preprocess_text(job_description)
    clean_student = preprocess_text(student_profile)

    # Edge case: If either text is blank after cleaning, similarity is 0
    if not clean_job or not clean_student:
        return 0.0

    try:
        # Fit TF-IDF vectorizer across both documents
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([clean_job, clean_student])

        # Compute cosine similarity between document 0 (job) and document 1 (student)
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Clip between 0.0 and 1.0 to guard against tiny floating-point rounding issues
        return float(max(0.0, min(1.0, score)))

    except Exception:
        return 0.0


def analyze_skill_gap(
    required_skills: List[str],
    student_skills: List[str]
) -> Dict[str, Any]:
    """
    Identifies matched and missing skills between job requirements and student skills.

    Formula:
        Skill Match % = (Matched Skills Count / Total Required Skills Count) * 100

    Args:
        required_skills (List[str]): List of skills required by the job.
        student_skills (List[str]): List of skills possessed by the student.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'matched_skills' (List[str]): Skills present in both lists.
            - 'missing_skills' (List[str]): Required skills missing from the student.
            - 'match_percentage' (float): Percentage of required skills met (0.0 to 100.0).

    Example:
        >>> analyze_skill_gap(["Python", "SQL", "Docker"], ["Python", "SQL"])
        {
            'matched_skills': ['Python', 'SQL'],
            'missing_skills': ['Docker'],
            'match_percentage': 66.7
        }
    """
    # Guard against None inputs
    required = required_skills or []
    student = student_skills or []

    # Build a lookup set of student skills in lowercase for case-insensitive matching
    student_lookup = {s.strip().lower() for s in student if s and isinstance(s, str)}

    matched: List[str] = []
    missing: List[str] = []

    # Categorize each required skill
    for req in required:
        if not req or not isinstance(req, str):
            continue
        cleaned_req = req.strip()
        if cleaned_req.lower() in student_lookup:
            matched.append(cleaned_req)
        else:
            missing.append(cleaned_req)

    # Calculate match percentage safely (avoid division by zero)
    total_required = len(matched) + len(missing)
    if total_required == 0:
        match_percentage = 0.0
    else:
        match_percentage = round((len(matched) / total_required) * 100.0, 1)

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": match_percentage,
    }
