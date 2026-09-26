# AI Job Skill Gap Analyzer

[![CI Pipeline](https://github.com/ThanuriMithara/AI-Document-Intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/ThanuriMithara/AI-Document-Intelligence/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, beginner-friendly Natural Language Processing (NLP) system that compares job descriptions against student skills, identifies missing requirements, computes mathematical text similarity, and provides actionable learning roadmaps.

---

## 1. Project Title
**AI Job Skill Gap Analyzer** — Beginner AI/ML Portfolio Project

---

## 2. Project Overview
The **AI Job Skill Gap Analyzer** is designed to bridge the gap between academic learning and industry requirements. By analyzing unstructured text from technical internship job postings and candidate skill profiles, it identifies exact skill overlaps, highlights critical missing technologies, and offers a step-by-step roadmap to acquire them.

---

## 3. Problem Statement
Technical job postings frequently contain dense, unstructured text filled with diverse requirements, frameworks, and tools. Undergraduate students often struggle to:
- Quickly determine which specific skills are required.
- Identify where their profile falls short.
- Figure out what concrete topics they should learn next to become competitive.

---

## 4. Objectives
- Build a complete, modular, and beginner-friendly NLP application using pure Python.
- Apply foundational text preprocessing techniques (lowercasing, punctuation removal, whitespace normalization, tokenization, stopword filtering).
- Implement a controlled, deterministic skill extraction engine.
- Calculate vocabulary similarity using **TF-IDF Vectorization** and **Cosine Similarity**.
- Provide a responsive, interactive web application using **Streamlit**.
- Maintain 100% test coverage using **pytest** with defensive handling for edge cases.

---

## 5. Features
- 📋 **Pre-Loaded Sample Jobs:** 10 realistic internship descriptions (AI/ML, Data Science, Web Dev, QA, Data Engineering, etc.).
- ✍️ **Custom Job & Skill Input:** Paste any custom job description and list your own skills.
- 📄 **Optional CV Upload:** Upload a plain `.txt` resume or project profile.
- 🎯 **Skill Gap Analysis:** Instant categorization of **Matched Skills** (✓) vs. **Missing Skills** (⚠️).
- 📈 **Dual Metrics:** Displays both deterministic **Skill Match %** and semantic **TF-IDF Text Similarity %**.
- 📚 **Personalized Learning Roadmap:** Automatically maps missing skills to concrete fundamentals to study.
- 🛡️ **Defensive Engineering:** Gracefully handles empty inputs, zero-skill cases, and unusual punctuation without crashing.

---

## 6. Technologies
- **Python 3.10+**: Core programming language.
- **Pandas**: Structured dataset management for sample job roles.
- **NumPy**: Numerical operations.
- **Scikit-learn**: `TfidfVectorizer` and `cosine_similarity`.
- **NLTK**: Word tokenization and stopword removal corpus.
- **Streamlit**: Interactive web dashboard.
- **Pytest**: Automated unit testing.

---

## 7. Project Architecture

```
User
 │
 ▼
Streamlit Web UI (app.py)
 │
 ├───► Text Preprocessing (src/preprocessing.py)
 │         │
 │         ▼
 ├───► Skill Extraction (src/skill_extractor.py)
 │         │
 │         ├──────────────────────────┐
 │         ▼                          ▼
 │    Skill Gap Analysis         TF-IDF Vectorizer
 │    (src/matcher.py)           (src/matcher.py)
 │         │                          │
 │         ▼                          ▼
 │    Matched / Missing Skills   Cosine Similarity
 │         │                          │
 │         └────────────┬─────────────┘
 │                      ▼
 └───► Learning Recommendations (src/recommendations.py)
       │
       ▼
 Interactive Results Dashboard (Cards, Badges, Metrics)
```

---

## 8. How the System Works
1. **Input Stage:** The user enters a job description (or selects one of the 10 built-in templates) and enters their own skills or uploads a `.txt` CV.
2. **Preprocessing:** Text is lowercased, punctuation is stripped (preserving symbols like `+` for `C++`), whitespace is normalized, and English stopwords are removed.
3. **Extraction:** Regex patterns scan the text against a canonical dictionary to isolate known technical skills without duplicate entries or false substring matches (e.g. avoiding "Java" when "JavaScript" is present).
4. **Matching:** 
   - Compares required skills to candidate skills to find exact matches and missing skills.
   - Converts the cleaned texts into TF-IDF vectors and computes the cosine of the angle between them.
5. **Recommendations:** For every missing skill, the system looks up a curated learning objective.
6. **Presentation:** Streamlit displays visual metrics, progress bars, skill badges, and a numbered roadmap.

---

## 9. TF-IDF Explanation
**TF-IDF** (Term Frequency - Inverse Document Frequency) translates unstructured text into a numerical vector:
- **Term Frequency (TF):** Measures how frequently a word appears in a specific document.
- **Inverse Document Frequency (IDF):** Discounts words that appear everywhere (e.g., *"work"*, *"candidate"*) and boosts words that appear rarely and carry high technical meaning (e.g., *"PyTorch"*, *"Docker"*).

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$

This creates a high-dimensional vector where technical terms receive the highest mathematical weights.

---

## 10. Cosine Similarity Explanation
Cosine Similarity measures the cosine of the angle between two high-dimensional vectors:

$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

- If two documents discuss identical concepts with similar vocabulary, their vectors point in the same direction, yielding a score near **1.0 (100%)**.
- If two documents share zero vocabulary, their vectors are orthogonal ($90^\circ$), yielding a score of **0.0 (0%)**.
- Unlike Euclidean distance, Cosine Similarity evaluates **direction rather than magnitude**, making it immune to differences in document length.

---

## 11. Skill Matching Explanation
Skill matching is a deterministic set comparison:
$$\text{Skill Match } \% = \left(\frac{\text{Matched Skills Count}}{\text{Total Required Skills Count}}\right) \times 100$$
If a job requires 4 skills (`Python`, `SQL`, `Machine Learning`, `Docker`) and the student possesses 3 (`Python`, `SQL`, `Machine Learning`), the match score is:
$$\frac{3}{4} \times 100 = 75.0\%$$
Missing skills (`Docker`) are forwarded to the recommendation engine.

---

## 12. Installation Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.

### Setup Steps
```bash
# 1. Clone repository
git clone https://github.com/your-username/AI-Job-Skill-Gap-Analyzer.git
cd AI-Job-Skill-Gap-Analyzer

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 13. How to Run the Web Application
With your virtual environment activated, run:
```bash
streamlit run app.py
```
Your default web browser will automatically open `http://localhost:8501`.

---

## 14. Example Input

**Job Description:**
> "We are seeking an enthusiastic AI/ML Intern to join our applied intelligence team. In this role, you will assist in training and evaluating machine learning models, cleaning datasets using Pandas and NumPy, and building text processing pipelines with Python and Scikit-learn. Familiarity with PyTorch, Git, and Docker is a plus."

**Student Skills:**
> `Python, SQL, Pandas, NumPy, Git, Machine Learning`

---

## 15. Example Output

```
========================================
SKILL GAP ANALYSIS
========================================
Skill Match: 75.0%
TF-IDF Text Similarity: 68.4%

Matched Skills:
✓ Python
✓ Machine Learning
✓ Pandas
✓ NumPy
✓ Git

Missing Skills:
⚠️ Docker
⚠️ PyTorch
⚠️ Scikit-learn

========================================
LEARNING RECOMMENDATIONS
========================================
1. Learn Docker fundamentals (containers, images, and Dockerfiles)
2. Learn PyTorch fundamentals (tensors, autograd, and building basic neural networks)
3. Learn Practical Machine Learning with Scikit-learn (pipelines, classifiers, metrics)
```

---

## 16. Testing Instructions
Run the automated test suite using `pytest`:
```bash
pytest -v
```
All 23 unit tests should pass with 100% success rate:
- `tests/test_preprocessing.py`: Tests lowercasing, punctuation, whitespace, and stopwords.
- `tests/test_skill_extractor.py`: Tests case-insensitivity, deduplication, symbol handling, and substring collisions.
- `tests/test_matcher.py`: Tests TF-IDF similarity, skill gap formulas, edge cases (zero skills), and recommendations.

---

## 17. Limitations

> **Important Notice:**
> The similarity score and skill match percentage are indicators of text/skill overlap. They are not predictions of hiring success, candidate quality, or employment probability.

Additional technical limitations:
- **Exact Vocabulary Dependency:** TF-IDF cannot detect semantic synonyms (e.g., *"Amazon Web Services"* vs *"AWS"*) unless explicitly mapped.
- **Negation Blindness:** Phrases such as *"I have no experience with Docker"* will still extract *"Docker"*.
- **Dictionary Boundary:** Skills outside the pre-defined dictionary will not be extracted automatically.

---

## 18. Future Improvements
1. **Dense Vector Embeddings:** Integrate Sentence-Transformers (`all-MiniLM-L6-v2`) for deep semantic understanding.
2. **Grammar & Negation Handling:** Use `spaCy` dependency parsing to identify context and experience level.
3. **Native PDF Resume Parser:** Support direct PDF resume uploads.
4. **Live Job Search:** Connect with public job board APIs to fetch live internship postings.
