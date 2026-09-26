# Project Explanation: AI Job Skill Gap Analyzer

Welcome! This document provides a clear, conceptual, student-friendly explanation of how the **AI Job Skill Gap Analyzer** works under the hood. It is written to help you understand every step so you can explain it with confidence during internship interviews.

---

## 1. What Problem We Solve

When students and recent graduates apply for technical internships, job descriptions often contain a long list of requirements (programming languages, libraries, tools). It can be difficult to:
- Quickly understand which skills are most critical.
- Objectively compare your current profile or resume against those requirements.
- Identify the exact "skill gap" (what you are missing).
- Know what to study next to bridge that gap.

This project automates that comparison using Natural Language Processing (NLP), giving the student an instant breakdown of **Matched Skills**, **Missing Skills**, **Skill Match %**, **Vocabulary Similarity %**, and a **Personalized Learning Roadmap**.

---

## 2. How Text Preprocessing Works (`src/preprocessing.py`)

Computers do not understand English grammar or vocabulary out of the box; they see text as raw ASCII or Unicode characters. Before doing any NLP, we must clean and standardize the text.

Our pipeline performs five steps:
1. **Lowercasing:** Converts `"Python"`, `"PYTHON"`, and `"python"` to `"python"`. This ensures words are matched regardless of capitalization.
2. **Whitespace Normalization:** Replaces tabs, line breaks, and repeated spaces with a single space.
3. **Punctuation Removal:** Cleans away commas, semicolons, exclamation marks, and periods, while carefully protecting symbols that belong to programming terms (like `+` in `C++` or `#` in `C#`).
4. **Tokenization:** Breaks the continuous string of characters into individual words (called "tokens") using NLTK (`nltk.tokenize.word_tokenize`).
5. **Stopword Removal:** Removes high-frequency English helper words (such as `"the"`, `"and"`, `"in"`, `"with"`, `"is"`) that occur in almost every sentence but carry very little domain meaning.

**Result:** A clean string containing only meaningful keywords ready for vectorization.

---

## 3. How Skill Extraction Works (`src/skill_extractor.py`)

Instead of guessing or hallucinating words, we use a **controlled canonical skill dictionary** covering the most common entry-level and internship skills (e.g., Python, SQL, React, Docker, PyTorch, Pandas).

Key design features:
- **Pre-compiled Regular Expressions (Regex):** For each skill, we compile a regex pattern using word boundaries (`\b`). For example, `\bpython\b` ensures we match the standalone word `"python"` without accidentally matching `"pythonic"`.
- **Special Characters:** Skills like `C++`, `Node.js`, and `Power BI` contain symbols or spaces. We use customized regex patterns so they are recognized accurately.
- **Substring Collision Prevention:** `\bjava\b` prevents `"JavaScript"` from accidentally being counted as `"Java"`.
- **Canonical Standardization:** Whether the user writes `"pandas"`, `"PANDAS"`, or `"Pandas"`, the extractor always returns the clean canonical name `"Pandas"`.
- **Deduplication:** If `"Python"` is mentioned five times in a job description, it only appears once in the extracted list.

---

## 4. What TF-IDF Is (`src/matcher.py`)

**TF-IDF** stands for **Term Frequency - Inverse Document Frequency**. It is one of the most widely used baseline algorithms in search engines and information retrieval.

- **Term Frequency (TF):** Measures how often a word appears in a specific text. If `"data"` appears 6 times in a job description, its TF is high for that text.
- **Inverse Document Frequency (IDF):** Measures how rare or common a word is across all documents.
  - If a word appears everywhere (like `"experience"` or `"team"`), its IDF score is low.
  - If a word appears only in a specific document (like `"pytorch"` or `"fastapi"`), its IDF score is high.

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

**Why it matters:** TF-IDF assigns heavy mathematical weights to unique technical keywords and diminishes the weight of generic conversational words.

---

## 5. What Cosine Similarity Is (`src/matcher.py`)

Once TF-IDF transforms two text documents (the Job Description and the Student Profile) into numerical vectors, we can plot them as arrows in high-dimensional space.

**Cosine Similarity** calculates the cosine of the angle ($\theta$) between these two vectors:

$$\text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|}$$

- **Score = 1.0 (100%):** The angle is $0^\circ$. The two texts share the exact same vocabulary in similar proportions.
- **Score = 0.0 (0%):** The angle is $90^\circ$ (orthogonal). The two texts share zero words in common.

**Why Cosine Similarity instead of Euclidean Distance?**
Euclidean distance measures the physical length between two points, which is distorted if one text is 1,000 words long and the other is 50 words long. Cosine similarity evaluates the **direction** of the vectors, making it immune to document length differences!

---

## 6. How Skill Gap Analysis Works (`src/matcher.py`)

While TF-IDF measures general vocabulary similarity, **Skill Gap Analysis** is a deterministic, rule-based check:

1. Takes the list of **Required Skills** ($R$) and **Student Skills** ($S$).
2. Compares each skill in $R$ against $S$ (case-insensitively).
3. Divides them into:
   - **Matched Skills:** Present in both $R$ and $S$.
   - **Missing Skills:** Present in $R$, but absent from $S$.
4. Computes the **Skill Match Percentage**:

$$\text{Skill Match } \% = \left(\frac{\text{Count of Matched Skills}}{\text{Total Count of Required Skills}}\right) \times 100$$

Edge cases (like 0 required skills or 0 student skills) are handled safely to avoid division-by-zero errors.

---

## 7. How Recommendations Work (`src/recommendations.py`)

For every skill flagged as **Missing**, our recommendation engine looks up an actionable, beginner-level learning topic from a curated knowledge map.

Example:
- Missing: `"Docker"` $\rightarrow$ Suggests: `"Learn Docker fundamentals (containers, images, and Dockerfiles)"`
- Missing: `"PyTorch"` $\rightarrow$ Suggests: `"Learn PyTorch fundamentals (tensors, autograd, and building basic neural networks)"`

If a skill is outside our dictionary, a safe fallback is provided: `"Learn [Skill] fundamentals and build a hands-on project"`.
No cloud LLM API or external network call is needed, keeping the system free, private, and deterministic.

---

## 8. How Streamlit Works (`app.py`)

Streamlit is an open-source Python framework that allows data scientists and ML engineers to build interactive web apps directly in pure Python without writing HTML, CSS, or JavaScript.

- Every time an input widget changes or a button is clicked, Streamlit re-executes `app.py` from top to bottom.
- `@st.cache_data` caches the dataset loading so we do not re-read the CSV on every button click.
- Streamlit's `st.columns`, `st.metric`, and `st.progress` display real-time interactive dashboards effortlessly.

---

## 9. Project Limitations

It is critical to be honest about limitations in technical interviews:
1. **Vocabulary vs. Semantics:** TF-IDF only matches exact word forms or n-grams. It does not understand synonyms (e.g., `"AWS"` vs `"Amazon Web Services"` unless explicitly handled by rules, or `"k8s"` vs `"Kubernetes"`).
2. **Controlled Skill Dictionary:** The system only recognizes skills defined in our dictionary. Novel or niche tools will be missed unless added.
3. **Absence of Context:** A student who writes *"I have never used Docker"* would still have `"Docker"` extracted as a skill by a simple keyword matching rule. Negation handling requires deeper dependency parsing.
4. **Not a Hiring Predictor:** The match score reflects textual keyword overlap. It cannot assess problem-solving skills, project depth, soft skills, or culture fit.

---

## 10. Possible Future Improvements

1. **Semantic Embeddings:** Use Sentence-Transformers (`all-MiniLM-L6-v2`) to capture conceptual similarities even when words differ.
2. **Negation Detection:** Integrate dependency parsing (e.g., using spaCy) to distinguish between *"proficient in SQL"* and *"no prior experience with SQL"*.
3. **Automated Resume PDF Parsing:** Integrate PDF extraction (as built in Stage 1) to read PDF resumes directly.
4. **Dynamic Web Scraping:** Add an option to fetch live job postings via URL using BeautifulSoup.
