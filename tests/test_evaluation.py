import json
from pathlib import Path

from jobmatch.evaluation import grade
from jobmatch.session import MatchSession

CV = "## Skills\n- Python, Docker\n"
CASE = {
    "name": "demo", "score_range": [40, 80],
    "labels": [
        {"keywords": ["python"], "expect": ["met"]},
        {"keywords": ["kubernetes", "k8s"], "expect": ["missing", "partial"]},
        {"keywords": ["aws"], "expect": ["missing"]},
    ],
}


def test_grading_counts_correct_labels_overclaims_and_unassessed():
    s = MatchSession("job", CV)
    s.record_requirement("Python 3", "must", "met", "Python, Docker", "")
    s.record_requirement("Docker & Kubernetes", "must", "met", "Python, Docker", "")   # overclaim
    g = grade(CASE, s)
    assert [label.correct for label in g.labels] == [True, False, False]
    assert g.overclaims == 1          # Kubernetes marked met
    assert g.not_assessed == 1        # AWS never recorded
    assert g.score == 100 and not g.score_ok


def test_eval_cases_reference_existing_files_and_valid_statuses():
    root = Path(__file__).resolve().parent.parent
    for case in json.loads((root / "eval" / "cases.json").read_text()):
        assert (root / case["cv"]).exists() and (root / case["job"]).exists()
        for label in case["labels"]:
            assert set(label["expect"]) <= {"met", "partial", "missing"}
