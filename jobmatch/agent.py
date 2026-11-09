"""The agent: Claude plus four tools, driven by the Anthropic SDK's tool runner.

The tool runner does the agent loop for us: send the conversation, run any tools
Claude asks for, send the results back, repeat until Claude stops calling tools.
If a tool raises ToolError, the runner sends the message back as an error result,
so Claude can correct itself (e.g. re-quote evidence it got wrong).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import anthropic
from anthropic import beta_tool
from anthropic.lib.tools import ToolError

from .pricing import cost_usd
from .session import MatchSession

MODEL = os.environ.get("JOBMATCH_MODEL", "claude-opus-5-5")
MAX_ITERATIONS = 30

SYSTEM_PROMPT = """\
You are a careful technical recruiter and career coach. You compare a job ad with a \
candidate's CV using the tools provided, and you are honest: an inflated match hurts \
the candidate later in interviews.

Work in this order:
1. Identify the job's concrete requirements (usually 6-12). Merge near-duplicates. Mark each \
"must" (required) or "nice" (nice to have, a plus, ideally).
2. For each requirement, call search_cv to find the supporting CV text. Run several searches \
at once when you can.
3. Call record_requirement once per requirement. "met": the CV clearly shows it. "partial": \
related or weaker experience (for example Docker but not Kubernetes, or coursework only). \
"missing": the CV does not show it. Evidence must be copied word for word from the CV.
4. Call get_match_score.
5. Call submit_report. The cover letter outline may only use experience from the recorded \
evidence: never invent experience, employers, numbers or degrees. For each gap give specific, \
realistic advice: a small project, a course, or how to frame transferable experience honestly.

Write the report in the language of the job ad. Refer to the candidate by name or as \
"the candidate"; do not assume their gender, age or background beyond what the CV states."""


class AgentError(RuntimeError):
    pass


Event = Callable[[str, dict, str, bool], None]   # (tool name, input, result text, is_error)


@dataclass
class RunResult:
    session: MatchSession
    model: str
    turns: int
    tool_calls: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = 0.0
    trace: list[dict] = field(default_factory=list)


def build_tools(session: MatchSession, on_event: Event | None = None) -> list:
    """Wrap the session's methods as tools. Docstrings become the tool descriptions Claude reads."""

    def traced(name: str, args: dict, fn: Callable[[], str]) -> str:
        try:
            result = fn()
        except ToolError as exc:
            if on_event:
                on_event(name, args, str(exc), True)
            raise
        if on_event:
            on_event(name, args, result, False)
        return result

    @beta_tool(strict=True)
    def search_cv(query: str) -> str:
        """Search the candidate's CV for lines about a skill, tool or requirement.

        Call this for every requirement before recording it, to find the exact CV text to quote
        as evidence. Common synonyms are searched automatically ("k8s" finds "Kubernetes").

        Args:
            query: A skill or requirement, e.g. "Kubernetes", "REST APIs" or "team lead".
        """
        return traced("search_cv", {"query": query}, lambda: session.search_cv(query))

    @beta_tool(strict=True)
    def record_requirement(requirement: str, importance: Literal["must", "nice"],
                           status: Literal["met", "partial", "missing"], evidence: str, reasoning: str) -> str:
        """Record how well the CV covers one requirement of the job ad.

        Call this once per requirement after searching the CV. Recording the same requirement
        again replaces the earlier entry. Rejected if the evidence is not an exact CV quote.

        Args:
            requirement: Short name of the requirement, e.g. "Kubernetes" or "3+ years Python".
            importance: "must" if the ad requires it, "nice" if it is optional or a plus.
            status: "met", "partial" (related or weaker experience) or "missing".
            evidence: A word-for-word quote from the CV supporting the status. Empty string if missing.
            reasoning: One sentence explaining the status.
        """
        args = {"requirement": requirement, "importance": importance, "status": status,
                "evidence": evidence, "reasoning": reasoning}
        return traced("record_requirement", args,
                      lambda: session.record_requirement(requirement, importance, status, evidence, reasoning))

    @beta_tool(strict=True)
    def get_match_score() -> str:
        """Compute the overall match score (0-100) from the recorded requirements.

        Call this after all requirements are recorded and before submit_report. Must-haves count
        three times as much as nice-to-haves; partial counts half.
        """
        return traced("get_match_score", {}, session.score_json)

    @beta_tool(strict=True)
    def submit_report(summary: str, strengths: list[str], gaps_and_advice: list[str],
                      cover_letter_outline: list[str]) -> str:
        """Submit the final report. Call this exactly once, as the last step.

        Args:
            summary: 2-3 sentences: overall fit, the score, and the biggest gap.
            strengths: The candidate's strongest points for this job, each backed by recorded evidence.
            gaps_and_advice: One item per important gap, with concrete advice on closing or framing it.
            cover_letter_outline: 3-6 bullet points for a tailored cover letter, using only recorded evidence.
        """
        args = {"summary": summary, "strengths": strengths, "gaps_and_advice": gaps_and_advice,
                "cover_letter_outline": cover_letter_outline}
        return traced("submit_report", args,
                      lambda: session.submit_report(summary, strengths, gaps_and_advice, cover_letter_outline))

    return [search_cv, record_requirement, get_match_score, submit_report]


def _has_credentials() -> bool:
    return any(os.environ.get(v) for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))


def run_agent(job_text: str, cv_text: str, *, client: anthropic.Anthropic | None = None, model: str = MODEL,
              on_event: Event | None = None, max_iterations: int = MAX_ITERATIONS) -> RunResult:
    session = MatchSession(job_text, cv_text)
    result = RunResult(session=session, model=model, turns=0, tool_calls=0)

    def record(name: str, args: dict, text: str, is_error: bool) -> None:
        result.tool_calls += 1
        result.trace.append({"tool": name, "input": args, "result": text, "is_error": is_error})
        if on_event:
            on_event(name, args, text, is_error)

    if client is None:
        if not _has_credentials():
            raise AgentError("Set ANTHROPIC_API_KEY to run the agent.")
        client = anthropic.Anthropic()

    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=build_tools(session, record),
        messages=[{"role": "user", "content": (
            f"<job_ad>\n{job_text.strip()}\n</job_ad>\n\n<cv>\n{cv_text.strip()}\n</cv>\n\n"
            "Analyse how well this CV matches the job ad."
        )}],
        max_iterations=max_iterations,
        output_config={"effort": "medium"},
        # Every turn re-sends the whole conversation; caching makes the repeated part ~20x cheaper.
        cache_control={"type": "ephemeral"},
        # If a safety classifier declines, the API retries on a recommended fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    last = None
    for message in runner:
        last = message
        result.turns += 1
        u = message.usage
        result.input_tokens += u.input_tokens or 0
        result.output_tokens += u.output_tokens or 0
        result.cache_read_tokens += u.cache_read_input_tokens or 0
        result.cache_write_tokens += u.cache_creation_input_tokens or 0
        turn_cost = cost_usd(message.model, u)
        result.cost_usd = None if (turn_cost is None or result.cost_usd is None) else result.cost_usd + turn_cost

    if last is None:
        raise AgentError("The model returned no response.")
    if last.stop_reason == "refusal":
        raise AgentError("The model declined this request.")
    if session.report is None:
        reason = "hit the turn limit" if result.turns >= max_iterations else f"stopped ({last.stop_reason})"
        raise AgentError(f"The agent {reason} after {result.turns} turns without submitting a report.")
    return result
