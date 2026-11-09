"""The agent's workspace: the tools it can call and the state they build up.

Design rule: the model makes judgements, the code enforces facts.
  * Evidence for a requirement must be an exact quote from the CV - checked here,
    not trusted. A made-up quote is rejected and the model has to try again.
  * The match score is computed by code from the recorded assessments, so the
    number is reproducible and never "vibes-based".
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from anthropic.lib.tools import ToolError

from .documents import cv_lines, normalize
from .skills import contains, expand

Importance = Literal["must", "nice"]
Status = Literal["met", "partial", "missing"]

WEIGHTS = {"must": 3, "nice": 1}
POINTS = {"met": 1.0, "partial": 0.5, "missing": 0.0}
_STOPWORDS = frozenset(
    "a an and or the of in on with for to experience knowledge skills skill years year good strong".split()
)


@dataclass
class Assessment:
    requirement: str
    importance: Importance
    status: Status
    evidence: str
    reasoning: str


@dataclass
class Report:
    summary: str
    strengths: list[str]
    gaps_and_advice: list[str]
    cover_letter_outline: list[str]


@dataclass
class MatchSession:
    job_text: str
    cv_text: str
    assessments: dict[str, Assessment] = field(default_factory=dict)
    report: Report | None = None
    evidence_rejections: int = 0

    def __post_init__(self) -> None:
        self._lines = cv_lines(self.cv_text)
        self._cv_normalized = normalize(self.cv_text)

    # ------------------------------------------------------------------ tools

    def search_cv(self, query: str) -> str:
        terms: set[str] = set()
        for phrase in [query, *re.findall(r"[\w.+#/-]+", query)]:
            if phrase.lower() not in _STOPWORDS:
                terms |= expand(phrase)
        scored = []
        for line in self._lines:
            hits = sorted(t for t in terms if contains(line.text, t))
            if hits:
                scored.append((len(hits), line))
        if not scored:
            return (f"No CV lines match '{query}' (also tried synonyms). Try a broader term; "
                    "if nothing matches, treat the requirement as missing.")
        scored.sort(key=lambda item: item[0], reverse=True)
        return "\n".join(f"- [{line.section}] {line.text}" for _, line in scored[:6])

    def record_requirement(self, requirement: str, importance: Importance, status: Status,
                           evidence: str, reasoning: str) -> str:
        requirement = requirement.strip()
        if not requirement:
            raise ToolError("requirement must not be empty.")
        evidence = evidence.strip()
        if status == "missing":
            evidence = ""
        else:
            if not evidence:
                raise ToolError(f"Status '{status}' needs evidence: quote the CV line that supports it.")
            if normalize(evidence) not in self._cv_normalized:
                self.evidence_rejections += 1
                raise ToolError(
                    f"Evidence not found in the CV: '{evidence}'. Quote the CV word for word "
                    "(use search_cv to find the exact text), or record the requirement as 'missing'."
                )
        key = normalize(requirement)
        replaced = key in self.assessments
        self.assessments[key] = Assessment(requirement, importance, status, evidence, reasoning.strip())
        verb = "Updated" if replaced else "Recorded"
        return f"{verb} '{requirement}' as {status}. {len(self.assessments)} requirement(s) recorded so far."

    def match_score(self) -> dict:
        if not self.assessments:
            raise ToolError("No requirements recorded yet. Use record_requirement first.")
        items = list(self.assessments.values())
        total = sum(WEIGHTS[a.importance] for a in items)
        earned = sum(WEIGHTS[a.importance] * POINTS[a.status] for a in items)
        return {
            "score": round(100 * earned / total),
            "requirements": len(items),
            "met": sum(a.status == "met" for a in items),
            "partial": sum(a.status == "partial" for a in items),
            "missing": sum(a.status == "missing" for a in items),
            "missing_must_haves": [a.requirement for a in items if a.importance == "must" and a.status == "missing"],
        }

    def submit_report(self, summary: str, strengths: list[str], gaps_and_advice: list[str],
                      cover_letter_outline: list[str]) -> str:
        if len(self.assessments) < 3:
            raise ToolError("Record at least 3 requirements before submitting the report.")
        if not summary.strip() or not cover_letter_outline:
            raise ToolError("summary and cover_letter_outline must not be empty.")
        self.report = Report(summary.strip(), strengths, gaps_and_advice, cover_letter_outline)
        return "Report submitted. Reply with one short sentence to finish."

    # ------------------------------------------------------------------ helpers

    def score_json(self) -> str:
        return json.dumps(self.match_score(), ensure_ascii=False)

    def to_dict(self) -> dict:
        return {
            "assessments": [asdict(a) for a in self.assessments.values()],
            "score": self.match_score() if self.assessments else None,
            "report": asdict(self.report) if self.report else None,
            "evidence_rejections": self.evidence_rejections,
        }
