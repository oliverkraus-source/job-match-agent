"""Render a finished run as a Markdown report."""

from __future__ import annotations

from .agent import RunResult

ICON = {"met": "✅ met", "partial": "🟡 partial", "missing": "❌ missing"}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ") or "-"


def render_markdown(result: RunResult, title: str = "Job match report") -> str:
    s = result.session
    score = s.match_score()
    report = s.report
    assert report is not None

    lines = [f"# {title}", "", f"**Match score: {score['score']}/100** "
             f"({score['met']} met, {score['partial']} partial, {score['missing']} missing)", ""]
    if score["missing_must_haves"]:
        lines += [f"**Missing must-haves:** {', '.join(score['missing_must_haves'])}", ""]
    lines += [report.summary, "", "## Requirements", "",
              "| Requirement | Importance | Status | Evidence from CV |", "|---|---|---|---|"]
    order = {"must": 0, "nice": 1}
    for a in sorted(s.assessments.values(), key=lambda a: order[a.importance]):
        evidence = f'"{a.evidence}"' if a.evidence else ""
        lines.append(f"| {_cell(a.requirement)} | {a.importance} | {ICON[a.status]} | {_cell(evidence)} |")

    for heading, items in (("Strengths", report.strengths),
                           ("Gaps and how to address them", report.gaps_and_advice),
                           ("Cover letter outline", report.cover_letter_outline)):
        lines += ["", f"## {heading}", ""] + [f"- {item}" for item in items]

    cost = f"${result.cost_usd:.3f}" if result.cost_usd is not None else "unknown (model not in price table)"
    lines += ["", "---", "",
              f"<sub>Model `{result.model}` · {result.turns} turns · {result.tool_calls} tool calls · "
              f"{s.evidence_rejections} evidence quotes rejected and retried · "
              f"{result.input_tokens + result.cache_read_tokens + result.cache_write_tokens:,} input tokens "
              f"({result.cache_read_tokens:,} from cache) · {result.output_tokens:,} output tokens · "
              f"estimated cost {cost}</sub>", ""]
    return "\n".join(lines)
