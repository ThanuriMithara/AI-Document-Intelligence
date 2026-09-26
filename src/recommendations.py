"""
Learning Recommendations Module
--------------------------------
Generates rule-based, beginner-friendly learning recommendations for missing skills.

No complex LLMs or external APIs are used — this ensures the application remains
fast, reliable, free to run, and transparent for beginners.
"""

from typing import List, Dict

# Explicit curated mapping of skills to actionable beginner learning topics
SKILL_LEARNING_MAP: Dict[str, str] = {
    "Docker": "Docker fundamentals (containers, images, and Dockerfiles)",
    "PyTorch": "PyTorch fundamentals (tensors, autograd, and building basic neural networks)",
    "TensorFlow": "TensorFlow & Keras fundamentals (model training and evaluation)",
    "SQL": "SQL fundamentals (SELECT, JOINs, aggregations, and subqueries)",
    "Machine Learning": "Machine Learning fundamentals (Supervised/Unsupervised learning with Scikit-learn)",
    "Deep Learning": "Neural Networks and Deep Learning (CNNs, RNNs, and backpropagation)",
    "NLP": "Natural Language Processing (tokenization, TF-IDF, and word embeddings)",
    "Computer Vision": "Computer Vision basics (image processing, filtering, and OpenCV)",
    "AWS": "AWS Cloud fundamentals (EC2, S3, and basic cloud architecture)",
    "Azure": "Microsoft Azure fundamentals (cloud storage, VMs, and cloud concepts)",
    "Git": "Git & GitHub version control (branching, commits, pull requests)",
    "GitHub": "Git & GitHub version control (collaborative workflows, issues, PRs)",
    "FastAPI": "REST API development with FastAPI (endpoints, Pydantic validation)",
    "Flask": "Flask web microframework (routing, templates, and RESTful APIs)",
    "Pandas": "Data analysis and manipulation with Pandas (DataFrames and aggregations)",
    "NumPy": "Numerical computing with NumPy (arrays, matrix operations, broadcasting)",
    "Scikit-learn": "Practical Machine Learning with Scikit-learn (pipelines, classifiers, metrics)",
    "Python": "Python core programming (data structures, functions, OOP basics)",
    "Java": "Java programming fundamentals (object-oriented programming and syntax)",
    "C++": "C++ fundamentals (pointers, memory management, and OOP)",
    "HTML": "HTML5 semantic structure and web accessibility",
    "CSS": "CSS3 styling, Flexbox, and responsive grid layouts",
    "JavaScript": "JavaScript fundamentals (DOM manipulation, ES6 syntax, and async/await)",
    "React": "React frontend development (components, hooks, and state management)",
    "Angular": "Angular framework basics (components, services, and TypeScript)",
    "Node.js": "Node.js backend runtime and Express server development",
    "Power BI": "Power BI dashboarding, DAX expressions, and data visualization",
    "Excel": "Advanced Excel (VLOOKUP/XLOOKUP, Pivot Tables, and data analysis)",
    "Data Science": "Data Science methodology (EDA, hypothesis testing, and storytelling)",
    "Data Analysis": "Exploratory Data Analysis (EDA) and data visualization techniques",
}


def recommend_learning(missing_skills: List[str]) -> List[str]:
    """
    Generates actionable learning recommendations for a list of missing skills.

    Args:
        missing_skills (List[str]): List of skills that the student needs to acquire.

    Returns:
        List[str]: List of formatted recommendation strings.

    Example:
        >>> recommend_learning(["Docker", "PyTorch"])
        ['Learn Docker fundamentals (containers, images, and Dockerfiles)',
         'Learn PyTorch fundamentals (tensors, autograd, and building basic neural networks)']
    """
    if not missing_skills:
        return ["Great job! You have matched all the key skills required for this role."]

    recommendations: List[str] = []
    seen = set()

    for skill in missing_skills:
        if not skill or not isinstance(skill, str):
            continue
        cleaned_skill = skill.strip()
        if cleaned_skill in seen:
            continue
        seen.add(cleaned_skill)

        # Lookup in curated dictionary, or provide a clean standard fallback
        if cleaned_skill in SKILL_LEARNING_MAP:
            action = SKILL_LEARNING_MAP[cleaned_skill]
            recommendations.append(f"Learn {action}")
        else:
            recommendations.append(f"Learn {cleaned_skill} fundamentals and build a hands-on project")

    return recommendations
