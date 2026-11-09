"""Run the real agent loop (the SDK's tool runner) against a scripted fake API.

Every request the SDK sends is captured, and the fake replies with pre-written
model turns. This tests the loop end to end - tool execution, error feedback,
the final report - without network access, an API key or any cost.
"""

import json

import anthropic
import httpx2
import pytest

from jobmatch.agent import MODEL, AgentError, run_agent
from jobmatch.report import render_markdown

JOB = "Requirements: Python, REST APIs, Kubernetes. Nice: Docker."
CV = "## Skills\n- Python, Docker\n\n## Experience\n- Built REST APIs with FastAPI\n"


def tool_use(id_, name, **inp):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


def reply(*content, stop="tool_use"):
    return {
        "id": "msg", "type": "message", "role": "assistant", "model": MODEL,
        "content": list(content), "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 200,
                  "cache_creation_input_tokens": 500, "cache_read_input_tokens": 2000},
    }


SCRIPT = [
    # Turn 1: two searches in parallel
    reply(tool_use("t1", "search_cv", query="Python"), tool_use("t2", "search_cv", query="Kubernetes")),
    # Turn 2: the model invents evidence -> must be rejected
    reply(tool_use("t3", "record_requirement", requirement="Kubernetes", importance="must", status="met",
                   evidence="Managed Kubernetes clusters", reasoning="x")),
    # Turn 3: corrected, plus the other requirements and the score
    reply(
        tool_use("t4", "record_requirement", requirement="Kubernetes", importance="must", status="missing",
                 evidence="", reasoning="Not in the CV."),
        tool_use("t5", "record_requirement", requirement="Python", importance="must", status="met",
                 evidence="Python, Docker", reasoning="Listed."),
        tool_use("t6", "record_requirement", requirement="REST APIs", importance="must", status="met",
                 evidence="Built REST APIs with FastAPI", reasoning="Direct experience."),
        tool_use("t7", "record_requirement", requirement="Docker", importance="nice", status="met",
                 evidence="Python, Docker", reasoning="Listed."),
        tool_use("t8", "get_match_score"),
    ),
    # Turn 4: the report
    reply(tool_use("t9", "submit_report", summary="Good fit, missing Kubernetes.",
                   strengths=["REST APIs with FastAPI"], gaps_and_advice=["Kubernetes: deploy a side project"],
                   cover_letter_outline=["Open with the FastAPI work"])),
    # Turn 5: done
    reply({"type": "text", "text": "Report submitted."}, stop="end_turn"),
]


def fake_client(script):
    requests = []
    replies = iter(script)

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=next(replies))

    client = anthropic.Anthropic(api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    return client, requests


def tool_results(request):
    last = request["messages"][-1]
    return {b["tool_use_id"]: b for b in last["content"] if b.get("type") == "tool_result"}


def test_full_agent_run_with_self_correction():
    client, requests = fake_client(SCRIPT)
    events = []
    result = run_agent(JOB, CV, client=client, on_event=lambda *e: events.append(e))

    # The loop ran all five turns and produced a report.
    assert len(requests) == 5 and result.turns == 5
    report = result.session.report
    assert report is not None and report.summary.startswith("Good fit")

    # Turn 2's invented evidence went back to the model as an error result...
    rejected = tool_results(requests[2])["t3"]
    assert rejected["is_error"] is True and "Evidence not found" in str(rejected["content"])
    assert result.session.evidence_rejections == 1
    # ...and both parallel searches were answered in one message.
    assert set(tool_results(requests[1])) == {"t1", "t2"}

    # Score: must 3+3+0, nice 1 -> 7 of 10
    assert result.session.match_score()["score"] == 70
    assert result.tool_calls == 9 and any(e[3] for e in events)

    # Request shape: strict tools, caching, fallbacks.
    first = requests[0]
    assert first["model"] == MODEL
    assert {t["name"] for t in first["tools"]} == {"search_cv", "record_requirement", "get_match_score",
                                                    "submit_report"}
    assert all(t["strict"] for t in first["tools"])
    assert first["cache_control"] == {"type": "ephemeral"} and first["fallbacks"] == "default"

    # Cost from usage: per turn (1000*4 + 200*20 + 500*5 + 2000*0.2) / 1e6 = 0.0109
    assert result.cost_usd == pytest.approx(5 * 0.0109)

    markdown = render_markdown(result)
    assert "**Match score: 70/100**" in markdown and "❌ missing" in markdown
    assert '"Built REST APIs with FastAPI"' in markdown


def test_error_when_agent_stops_without_a_report():
    client, _ = fake_client([reply({"type": "text", "text": "I'm done."}, stop="end_turn")])
    with pytest.raises(AgentError, match="without submitting a report"):
        run_agent(JOB, CV, client=client)


def test_error_on_refusal():
    client, _ = fake_client([reply(stop="refusal")])
    with pytest.raises(AgentError, match="declined"):
        run_agent(JOB, CV, client=client)


def test_turn_limit_stops_a_looping_agent():
    looping = [reply(tool_use(f"s{i}", "search_cv", query="Python")) for i in range(3)]
    client, requests = fake_client(looping)
    with pytest.raises(AgentError, match="turn limit"):
        run_agent(JOB, CV, client=client, max_iterations=3)
    assert len(requests) == 3


def test_clear_error_without_api_key(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(AgentError, match="ANTHROPIC_API_KEY"):
        run_agent(JOB, CV)
