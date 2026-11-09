import pytest
from anthropic.lib.tools import ToolError

from jobmatch.documents import cv_lines, normalize
from jobmatch.session import MatchSession
from jobmatch.skills import contains, expand

CV = """# Jane Doe

## Skills
- **Python**, SQL, Docker
- English (C1)

## Experience
### Developer - Acme (2024)
- Built REST APIs with FastAPI
- Wrote the team’s deployment guide
"""


@pytest.fixture
def session():
    return MatchSession(job_text="job", cv_text=CV)


# ---------- documents and skills ----------

def test_cv_lines_track_the_section_and_strip_bullets():
    lines = cv_lines(CV)
    assert lines[0].section == "Skills" and lines[0].text == "**Python**, SQL, Docker"
    assert lines[-1].section == "Developer - Acme (2024)"


def test_normalize_ignores_markdown_case_and_curly_quotes():
    assert normalize("- **Python**, SQL") == normalize("python, sql")
    assert normalize("team’s guide.") == "team's guide"


def test_synonyms_and_whole_word_matching():
    assert "kubernetes" in expand("k8s")
    assert contains("Deployed on K8s", "k8s")
    assert not contains("great results", "ts")


# ---------- search_cv ----------

def test_search_finds_lines_via_synonyms(session):
    assert "Built REST APIs with FastAPI" in session.search_cv("RESTful services")


def test_search_reports_when_nothing_matches(session):
    assert session.search_cv("Kubernetes").startswith("No CV lines match")


# ---------- record_requirement: the anti-hallucination check ----------

def test_accepts_exact_quotes_even_with_formatting_differences(session):
    session.record_requirement("Python", "must", "met", "Python, SQL, Docker", "listed")
    session.record_requirement("Docs", "nice", "partial", "Wrote the team's deployment guide.", "close")
    assert len(session.assessments) == 2


def test_rejects_evidence_that_is_not_in_the_cv(session):
    with pytest.raises(ToolError, match="Evidence not found"):
        session.record_requirement("Kubernetes", "must", "met", "Ran production Kubernetes clusters", "x")
    assert session.evidence_rejections == 1
    assert session.assessments == {}


def test_met_without_evidence_is_rejected(session):
    with pytest.raises(ToolError, match="needs evidence"):
        session.record_requirement("Python", "must", "met", "", "x")


def test_missing_requirements_drop_any_evidence(session):
    session.record_requirement("Kubernetes", "must", "missing", "something invented", "not in CV")
    assert session.assessments["kubernetes"].evidence == ""


def test_recording_twice_updates_instead_of_duplicating(session):
    session.record_requirement("Python", "must", "partial", "Python, SQL, Docker", "x")
    assert "Updated" in session.record_requirement("python", "must", "met", "Python, SQL, Docker", "y")
    assert len(session.assessments) == 1 and session.assessments["python"].status == "met"


# ---------- scoring and report ----------

def test_score_weights_must_haves_three_times(session):
    session.record_requirement("Python", "must", "met", "Python, SQL, Docker", "")
    session.record_requirement("Kubernetes", "must", "missing", "", "")
    session.record_requirement("Docker", "nice", "partial", "Python, SQL, Docker", "")
    score = session.match_score()
    # earned = 3*1 + 3*0 + 1*0.5 = 3.5 of 7
    assert score["score"] == 50
    assert score["missing_must_haves"] == ["Kubernetes"]


def test_score_needs_at_least_one_requirement(session):
    with pytest.raises(ToolError):
        session.match_score()


def test_report_needs_three_requirements(session):
    session.record_requirement("Python", "must", "met", "Python, SQL, Docker", "")
    with pytest.raises(ToolError, match="at least 3"):
        session.submit_report("ok", [], [], ["point"])
