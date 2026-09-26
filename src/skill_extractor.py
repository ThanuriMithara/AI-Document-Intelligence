"""
Skill Extractor Module
-----------------------
Extracts known technical and domain skills from raw text (such as job
descriptions, student resumes, or comma-separated skill lists).

Features:
- Controlled, curated dictionary of common student and entry-level skills.
- Case-insensitive matching (finds 'python', 'PYTHON', or 'Python').
- Preserves canonical display names (returns 'Python', 'Machine Learning').
- Avoids duplicates and handles punctuation/boundary edge cases (e.g., 'C++', 'Node.js').
"""

import re
from typing import List, Dict, Pattern

# Controlled dictionary of known skills mapped to their canonical display format.
# Patterns handle variations like 'scikit-learn' vs 'scikit learn' vs 'sklearn'.
KNOWN_SKILLS: List[str] = [
    "Python",
    "Java",
    "C++",
    "SQL",
    "HTML",
    "CSS",
    "JavaScript",
    "React",
    "Angular",
    "Node.js",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Machine Learning",
    "Deep Learning",
    "Data Science",
    "Data Analysis",
    "NLP",
    "Computer Vision",
    "Git",
    "GitHub",
    "Docker",
    "AWS",
    "Azure",
    "Power BI",
    "Excel",
    "FastAPI",
    "Flask",
]

# Regex patterns tailored for each skill to ensure precise matching
# without accidental substring collisions (e.g., 'Java' should not match 'JavaScript')
SKILL_PATTERNS: Dict[str, Pattern] = {
    "Python": re.compile(r"\bpython\b", re.IGNORECASE),
    "Java": re.compile(r"\bjava\b", re.IGNORECASE),
    "C++": re.compile(r"(?:^|[\s,;(\[])c\+\+(?:[\s,;.)\]]|$)", re.IGNORECASE),
    "SQL": re.compile(r"\bsql\b", re.IGNORECASE),
    "HTML": re.compile(r"\bhtml5?\b", re.IGNORECASE),
    "CSS": re.compile(r"\bcss3?\b", re.IGNORECASE),
    "JavaScript": re.compile(r"\bjavascript\b|\bjs\b", re.IGNORECASE),
    "React": re.compile(r"\breact(?:\.js)?\b", re.IGNORECASE),
    "Angular": re.compile(r"\bangular(?:\.js)?\b", re.IGNORECASE),
    "Node.js": re.compile(r"\bnode(?:\.js)?\b|\bnodejs\b", re.IGNORECASE),
    "Pandas": re.compile(r"\bpandas\b", re.IGNORECASE),
    "NumPy": re.compile(r"\bnumpy\b", re.IGNORECASE),
    "Scikit-learn": re.compile(r"\bscikit[- ]learn\b|\bsklearn\b", re.IGNORECASE),
    "TensorFlow": re.compile(r"\btensorflow\b|\btf\b", re.IGNORECASE),
    "PyTorch": re.compile(r"\bpytorch\b", re.IGNORECASE),
    "Machine Learning": re.compile(r"\bmachine learning\b|\bml\b", re.IGNORECASE),
    "Deep Learning": re.compile(r"\bdeep learning\b|\bdl\b", re.IGNORECASE),
    "Data Science": re.compile(r"\bdata science\b", re.IGNORECASE),
    "Data Analysis": re.compile(r"\bdata analysis\b|\bdata analytics\b", re.IGNORECASE),
    "NLP": re.compile(r"\bnlp\b|\bnatural language processing\b", re.IGNORECASE),
    "Computer Vision": re.compile(r"\bcomputer vision\b|\bcv\b", re.IGNORECASE),
    "Git": re.compile(r"\bgit\b", re.IGNORECASE),
    "GitHub": re.compile(r"\bgithub\b", re.IGNORECASE),
    "Docker": re.compile(r"\bdocker\b", re.IGNORECASE),
    "AWS": re.compile(r"\baws\b|\bamazon web services\b", re.IGNORECASE),
    "Azure": re.compile(r"\bazure\b", re.IGNORECASE),
    "Power BI": re.compile(r"\bpower[- ]?bi\b", re.IGNORECASE),
    "Excel": re.compile(r"\bexcel\b", re.IGNORECASE),
    "FastAPI": re.compile(r"\bfastapi\b", re.IGNORECASE),
    "Flask": re.compile(r"\bflask\b", re.IGNORECASE),
}


def extract_skills(text: str) -> List[str]:
    """
    Scans input text and returns a deduplicated list of recognized canonical skills.

    Args:
        text (str): Input text from job description, student profile, or CV text.

    Returns:
        List[str]: List of canonical skill names found in the text.

    Example:
        >>> extract_skills("I have experience with Python, SQL, pandas and machine learning.")
        ['Python', 'SQL', 'Pandas', 'Machine Learning']
    """
    if not text or not isinstance(text, str):
        return []

    found_skills: List[str] = []

    # Check each skill in our canonical dictionary using its pre-compiled pattern
    for skill_name, pattern in SKILL_PATTERNS.items():
        if pattern.search(text):
            found_skills.append(skill_name)

    return found_skills
