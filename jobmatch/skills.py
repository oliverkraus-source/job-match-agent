"""A small synonym table so "k8s" finds "Kubernetes" and "ML" finds "machine learning".

Job ads and CVs rarely use the same words for the same skill. Plain keyword
search misses those matches; this table closes the most common gaps.
"""

from __future__ import annotations

import re

_GROUPS = [
    ["javascript", "js", "ecmascript"],
    ["typescript", "ts"],
    ["python", "py"],
    ["kubernetes", "k8s"],
    ["docker", "containers", "containerization", "containerisation"],
    ["postgresql", "postgres", "psql"],
    ["sql", "mysql", "postgresql", "sqlite"],
    ["machine learning", "ml"],
    ["deep learning", "neural networks", "dl"],
    ["natural language processing", "nlp"],
    ["large language models", "llm", "llms", "gpt", "claude"],
    ["retrieval-augmented generation", "retrieval augmented generation", "rag"],
    ["continuous integration", "ci", "ci/cd", "github actions", "gitlab ci"],
    ["amazon web services", "aws"],
    ["google cloud", "gcp", "google cloud platform"],
    ["microsoft azure", "azure"],
    ["cloud", "aws", "gcp", "azure"],
    ["scikit-learn", "sklearn"],
    ["pytorch", "torch"],
    ["react", "react.js", "reactjs"],
    ["node.js", "node", "nodejs"],
    ["rest", "rest api", "restful", "api"],
    ["testing", "unit tests", "pytest", "jest", "playwright", "tdd"],
    ["git", "github", "gitlab", "version control"],
    ["computer science", "informatik", "cs"],
    ["german", "deutsch"],
    ["english", "englisch"],
    ["apache airflow", "airflow"],
    ["apache spark", "spark", "pyspark"],
    ["apache kafka", "kafka"],
]

_INDEX: dict[str, set[str]] = {}
for group in _GROUPS:
    for term in group:
        _INDEX.setdefault(term, set()).update(group)


def expand(term: str) -> set[str]:
    """Return the term plus its known synonyms (all lower-case)."""
    term = term.lower().strip()
    return {term} | _INDEX.get(term, set())


def contains(text: str, term: str) -> bool:
    """Whole-word match, so "ts" does not match inside "results"."""
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text.lower()) is not None
