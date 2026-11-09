"""Grade an agent run against hand-written labels.

A label says: "the requirement mentioning any of these keywords should have one
of these statuses". The most important number is the overclaim count: the agent
saying a skill is "met" when the CV does not show it. That is the error that
would embarrass a candidate in an interview.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .session import MatchSession
from .skills import contains


@dataclass
class LabelResult:
    keywords: list[str]
    expect: list[str]
    got: list[str]          # statuses of all matching assessments ([] = not assessed)

    @property
    def correct(self) -> bool:
        return any(s in self.expect for s in self.got)

    @property
    def overclaim(self) -> bool:
        return "met" not in self.expect and "met" in self.got


@dataclass
class CaseGrade:
    name: str
    score: int
    score_range: tuple[int, int]
    labels: list[LabelResult] = field(default_factory=list)

    @property
    def score_ok(self) -> bool:
        return self.score_range[0] <= self.score <= self.score_range[1]

    @property
    def accuracy(self) -> float:
        return sum(label.correct for label in self.labels) / len(self.labels)

    @property
    def overclaims(self) -> int:
        return sum(label.overclaim for label in self.labels)

    @property
    def not_assessed(self) -> int:
        return sum(not label.got for label in self.labels)


def grade(case: dict, session: MatchSession) -> CaseGrade:
    result = CaseGrade(case["name"], session.match_score()["score"], tuple(case["score_range"]))
    for label in case["labels"]:
        got = [a.status for a in session.assessments.values()
               if any(contains(a.requirement, k) for k in label["keywords"])]
        result.labels.append(LabelResult(label["keywords"], label["expect"], got))
    return result
