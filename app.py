"""
AI Job Skill Gap Analyzer — Streamlit Web Application
------------------------------------------------------
An interactive, beginner-friendly NLP tool that analyzes the skill overlap
between a job description and a student's profile.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.preprocessing import preprocess_text
from src.skill_extractor import extract_skills
from src.matcher import calculate_similarity, analyze_skill_gap
from src.recommendations import recommend_learning

# Page configuration
st.set_page_config(
    page_title="AI Job Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Responsive CSS for PC, Tablet, and Mobile Devices
st.markdown(
    """
    <style>
    /* Responsive Content Container */
    .main .block-container {
        max-width: 1240px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* Modern Card Layout */
    .stCard {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    /* Badges for Skills */
    .badge-matched {
        display: inline-block;
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 6px 14px;
        border-radius: 9999px;
        margin: 4px;
        font-weight: 600;
        font-size: 0.92rem;
        transition: transform 0.15s ease;
    }
    .badge-matched:hover {
        transform: translateY(-1px);
    }

    .badge-missing {
        display: inline-block;
        background-color: #fff1f2;
        color: #9f1239;
        border: 1px solid #fecdd3;
        padding: 6px 14px;
        border-radius: 9999px;
        margin: 4px;
        font-weight: 600;
        font-size: 0.92rem;
        transition: transform 0.15s ease;
    }
    .badge-missing:hover {
        transform: translateY(-1px);
    }

    /* Device-specific responsiveness */
    /* Tablets (iPad, 768px - 1024px) */
    @media (max-width: 1024px) {
        .main .block-container {
            max-width: 95%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .badge-matched, .badge-missing {
            font-size: 0.88rem;
            padding: 5px 12px;
        }
    }

    /* Mobile Phones (< 768px) */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            padding-top: 1rem;
        }
        h1 {
            font-size: 1.75rem !important;
        }
        h2, h3 {
            font-size: 1.25rem !important;
        }
        .badge-matched, .badge-missing {
            font-size: 0.82rem;
            padding: 4px 10px;
            margin: 2px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom header styling
st.title("🎯 AI Job Skill Gap Analyzer")
st.caption("📱 Optimized for PC, Tablet, and Mobile screens.")
st.markdown("---")

# Load sample jobs dataset
DATA_PATH = Path(__file__).parent / "data" / "sample_jobs.csv"


@st.cache_data
def load_sample_jobs():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


sample_jobs_df = load_sample_jobs()

# Two-column layout for input
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📋 Section 1: Job Description")

    # Convenience dropdown to populate sample jobs
    job_options = ["(Custom / Paste your own)"]
    if not sample_jobs_df.empty:
        job_options += sample_jobs_df["job_title"].tolist()

    selected_sample = st.selectbox(
        "Choose a sample job role (or paste your own below):",
        options=job_options,
        index=1 if len(job_options) > 1 else 0,
        help="Select a realistic pre-loaded role to test instantly, or choose 'Custom' to paste your own.",
    )

    default_job_desc = ""
    default_required_skills = ""

    if selected_sample != "(Custom / Paste your own)" and not sample_jobs_df.empty:
        match_row = sample_jobs_df[sample_jobs_df["job_title"] == selected_sample].iloc[0]
        default_job_desc = match_row["job_description"]
        default_required_skills = match_row["required_skills"].replace(";", ", ")

    job_text = st.text_area(
        "Job Description Text:",
        value=default_job_desc,
        height=220,
        placeholder="Paste full job description text here...",
    )

    # Optional: explicitly specified required skills if user prefers to list them
    custom_required = st.text_input(
        "Explicit Job Required Skills (comma-separated, optional):",
        value=default_required_skills,
        placeholder="e.g. Python, SQL, Machine Learning, Docker",
        help="If left empty, skills will be automatically extracted from the Job Description text above.",
    )

with col_right:
    st.subheader("🎓 Section 2: Your Skills & Profile")

    student_skills_input = st.text_input(
        "Your Skills (comma-separated):",
        value="Python, SQL, Pandas, Git, Machine Learning",
        placeholder="e.g. Python, SQL, Pandas, Git",
    )

    st.write("**Or upload your CV (PDF or TXT):**")
    uploaded_file = st.file_uploader(
        "Upload CV (.pdf or .txt)",
        type=["pdf", "txt"],
        help="Upload your CV/resume in PDF or TXT format. Extracted skills will be analyzed automatically.",
    )

    cv_text = ""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                import pymupdf
                doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
                cv_text = "\n".join([page.get_text() for page in doc])
                doc.close()
            else:
                cv_text = uploaded_file.read().decode("utf-8", errors="ignore")

            st.success(f"Loaded '{uploaded_file.name}' successfully!")
            with st.expander("Preview extracted CV text"):
                st.text(cv_text[:600] + ("..." if len(cv_text) > 600 else ""))
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Section 3: Analyze action
st.markdown("---")
analyze_button = st.button("🚀 Analyze Skill Gap", type="primary", use_container_width=True)

if analyze_button:
    if not job_text.strip() and not custom_required.strip():
        st.warning("Please provide a Job Description or explicit Required Skills to analyze.")
    else:
        # 1. Determine Required Skills
        # If user explicitly entered skills, parse those; also extract any from description
        job_skills = []
        if custom_required.strip():
            # Parse user comma-separated input
            for item in custom_required.split(","):
                clean_item = item.strip()
                if clean_item and clean_item not in job_skills:
                    job_skills.append(clean_item)

        # Also extract skills recognized in the description text
        extracted_from_desc = extract_skills(job_text)
        for s in extracted_from_desc:
            if s not in job_skills:
                job_skills.append(s)

        # 2. Determine Student Skills
        student_skills = []
        if student_skills_input.strip():
            for item in student_skills_input.split(","):
                clean_item = item.strip()
                if clean_item and clean_item not in student_skills:
                    student_skills.append(clean_item)

        # If CV text was uploaded, extract skills from it as well
        if cv_text.strip():
            cv_extracted = extract_skills(cv_text)
            for s in cv_extracted:
                if s not in student_skills:
                    student_skills.append(s)

        # Combine student profile text for TF-IDF
        student_combined_text = f"{student_skills_input} {cv_text}".strip()

        # 3. Perform Analysis
        gap_results = analyze_skill_gap(job_skills, student_skills)
        matched_skills = gap_results["matched_skills"]
        missing_skills = gap_results["missing_skills"]
        match_percentage = gap_results["match_percentage"]

        # 4. Perform TF-IDF text similarity
        similarity_score = calculate_similarity(job_text, student_combined_text)
        similarity_percentage = round(similarity_score * 100.0, 1)

        # 5. Generate learning recommendations
        recommendations = recommend_learning(missing_skills)

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        st.header("📊 Analysis Results")

        # Top metric cards
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(
                label="Skill Match",
                value=f"{match_percentage}%",
                help="Percentage of required job skills present in your skill set.",
            )
            st.progress(min(1.0, match_percentage / 100.0))

        with metric_col2:
            st.metric(
                label="TF-IDF Text Similarity",
                value=f"{similarity_percentage}%",
                help="Mathematical vocabulary overlap between job text and your profile.",
            )
            st.progress(min(1.0, similarity_percentage / 100.0))

        with metric_col3:
            st.metric(
                label="Matched Skills Count",
                value=f"{len(matched_skills)} / {len(job_skills)}",
            )

        # Important Ethical / Technical Disclaimer
        st.info(
            "ℹ️ **Disclaimer:** Text similarity and skill match percentages measure **textual overlap only**. "
            "They are **NOT** a prediction of candidate quality, competency, or probability of getting hired."
        )

        # Skill Breakdown Columns (Responsive badges)
        res_col1, res_col2 = st.columns(2, gap="medium")

        with res_col1:
            st.subheader(f"✅ Matched Skills ({len(matched_skills)})")
            if matched_skills:
                badges_html = "<div>" + "".join(
                    f'<span class="badge-matched">✓ {skill}</span>' for skill in matched_skills
                ) + "</div>"
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.info("No matching skills found.")

        with res_col2:
            st.subheader(f"⚠️ Missing Skills ({len(missing_skills)})")
            if missing_skills:
                badges_html = "<div>" + "".join(
                    f'<span class="badge-missing">⚠️ {skill}</span>' for skill in missing_skills
                ) + "</div>"
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.success("🎉 No missing skills! You meet all listed requirements.")

        # Learning Recommendations
        st.markdown("---")
        st.subheader("📚 Recommended Learning Roadmap")
        for idx, rec in enumerate(recommendations, start=1):
            st.markdown(f"{idx}. {rec}")

        # Inspection Expander for Technical Transparency
        with st.expander("🔍 Inspect Extracted Skills & NLP Details"):
            st.write("**Detected Job Skills:**", job_skills if job_skills else "None")
            st.write("**Detected Student Skills:**", student_skills if student_skills else "None")
            st.write(
                "**Preprocessed Job Text Preview:**",
                preprocess_text(job_text)[:300] + "..." if len(job_text) > 300 else preprocess_text(job_text),
            )
