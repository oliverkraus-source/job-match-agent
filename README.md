# Job-Match Agent

**An AI agent that compares a CV with a job ad and backs every claim with an exact quote from the CV.**

You give it a job ad and a CV (Markdown, text or PDF). The agent works out the job's
requirements, searches the CV for evidence, rates each requirement as met, partial or missing,
computes a match score, and writes a report with gaps, concrete advice and a tailored
cover-letter outline.

![CI](https://github.com/oliverkraus-source/job-match-agent/actions/workflows/ci.yml/badge.svg)

## The core idea: the model judges, the code enforces facts

LLMs are good at reading a job ad and judging whether "built REST APIs with FastAPI" covers
"experience with backend services". They're bad at being reliably honest about what's *not*
there. So the agent works through tools that enforce the facts:

| Tool | What it does | Guardrail |
|---|---|---|
| `search_cv` | Finds CV lines about a skill, with synonyms ("k8s" → Kubernetes, "LLM" → Claude/GPT) | Read-only |
| `record_requirement` | Records one requirement as met / partial / missing, with evidence | **The evidence must be a word-for-word quote from the CV.** Invented evidence is rejected with an error, and the agent has to search again or mark the requirement missing. |
| `get_match_score` | Computes the 0-100 score | **Computed by code** (must-haves weigh 3×, partial counts half), so it's reproducible |
| `submit_report` | Submits summary, strengths, gaps with advice, cover-letter outline | Rejected until at least 3 requirements are recorded |

The agent loop runs on the Anthropic SDK's **tool runner** (`client.beta.messages.tool_runner`).
When a tool raises `ToolError`, the runner sends the message back to Claude as an error result,
which is how the agent corrects itself.

```mermaid
sequenceDiagram
    participant U as You
    participant A as Agent (Claude)
    participant T as Tools (Python)
    U->>A: job ad + CV
    A->>T: search_cv("Kubernetes"), search_cv("Python") (parallel)
    T-->>A: matching CV lines
    A->>T: record_requirement(Kubernetes, met, "Managed K8s clusters")
    T-->>A: ❌ Evidence not found in the CV
    A->>T: record_requirement(Kubernetes, missing)
    A->>T: get_match_score()
    T-->>A: {"score": 72, ...}
    A->>T: submit_report(...)
    A-->>U: Markdown report
```

## Quick start

```bash
git clone https://github.com/oliverkraus-source/job-match-agent.git
cd job-match-agent
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...                    # Windows: set it in your environment variables

python -m jobmatch examples/jobs/junior_ai_engineer.md examples/cvs/alex_becker.md -o report.md
```

While it runs you see every tool call (rejected evidence shows up as `✗ record_requirement: …`).
A real run on the example files:

<img src="docs/agent_run.png" alt="Terminal output: 12 CV searches, 10 requirements recorded, score 91/100, 6 turns, 24 tool calls, cost $0.144" width="700">

And the report it wrote ([full size](docs/example_report.png)). Note the "He" in the summary:
this run predates the fix described under [Results](#results).

<img src="docs/example_report.png" alt="Markdown report: match score 91/100, requirement table with CV quotes, strengths, gaps with advice, cover letter outline" width="700">

Use your own files with `python -m jobmatch job_ad.pdf my_cv.pdf -o report.md`. Add
`--model claude-haiku-5-5` for a much cheaper run, and `--json run.json` to save the full
tool-call trace.

## Evaluation

`eval/cases.json` pairs the fictional example CVs with the example job ads (strong, medium,
weak and poor matches) and labels the clear-cut answers, e.g. "Sam's CV has no Python, so
Python must be *missing*". `python eval/run_eval.py` runs the agent on every case and reports:

- **Label accuracy:** did the agent rate the labelled requirements correctly?
- **Overclaims:** how often did it mark something *met* that the CV doesn't show? This is the
  most harmful error, because it sets the candidate up to fail in an interview.
- **Score in expected range,** plus evidence retries, turns and cost per run.

### Results

`claude-opus-5-5`, November 2026, commit `e535934`:

| Case | Score (expected range) | Label accuracy | Overclaims | Not assessed | Evidence retries | Turns | Cost |
|---|---|---|---|---|---|---|---|
| Alex × Junior AI Engineer (strong match) | 91 (65–95) ✅ | 100% | 0 | 0 | 0 | 6 | $0.134 |
| Sam × Frontend Developer (good match, junior) | 61 (45–90) ✅ | 86% | 0 | 0 | 0 | 5 | $0.117 |
| Alex × Data Engineer (weak match) | 35 (15–55) ✅ | 67% | 0 | 0 | 0 | 6 | $0.144 |
| Sam × Junior AI Engineer (poor match, career changer) | 33 (5–45) ✅ | 100% | 0 | 0 | 0 | 6 | $0.158 |

**24/27 labels correct, 0 overclaims, 4/4 scores in the expected range, about $0.14 per run.**
Prompt caching served about 75% of input tokens from cache.

**What the misses show.** All three misses are *partial* ratings where the label expected
something else, never a false *met*:

- **React for Sam** (label *met*, agent *partial*): the ad asks for professional experience, and
  Sam's React comes from a bootcamp and personal projects. The agent's call is defensible, and the
  label is arguably too strict. I left the label unchanged rather than editing labels after seeing
  results.
- **Airflow and Spark for Alex** (label *missing*, agent *partial*): the agent gave partial credit
  for adjacent experience (a Python ETL script). That's not an overclaim, but it errs in the
  generous direction. **Next iteration:** define *partial* more strictly in the system prompt
  (same kind of tool or a direct alternative, not just related work) and re-run the eval.
- **Evidence retries were 0:** the model always quoted the CV correctly, so on these cases the
  quote check acted as a safety net that was never needed. The tests exercise the rejection path.

The first real run also surfaced a fairness issue: the report called the gender-neutral fictional
candidate "he". Commit `f4e2197` added a rule to refer to the candidate by name and not assume
gender, age or background.

## Engineering notes

- **Prompt caching.** Every agent turn re-sends the whole conversation (job ad, CV, earlier tool
  calls). Top-level `cache_control` caches that prefix, so repeated input costs a fraction of the
  normal price. The cost estimate printed after each run comes from the token counts the API reports.
- **Strict tool schemas.** Tools are plain typed Python functions (`@beta_tool(strict=True)`). The
  SDK builds the JSON schemas, and strict mode guarantees the inputs match them, including enums
  like `met | partial | missing`.
- **Bounded loop.** At most 30 turns; the run fails with a clear error if the agent stops or loops
  without submitting a report, or if the model declines.
- **Tested without an API key.** `tests/test_agent.py` runs the real tool runner against a scripted
  fake API (`httpx2.MockTransport`). It covers parallel tool calls, rejected evidence and
  self-correction, the turn limit, refusals, and the exact request sent. 20 tests run in CI.

## Limitations

- Matching quality depends on how the CV is written; tables and multi-column PDFs extract poorly.
- The synonym table is small and hand-written. Embedding search would generalise better.
- The evaluation set is small (4 cases, about 27 labels) and its labels were written by the author.
  It catches regressions and overclaiming; it isn't a benchmark.
- The report is advice, not a hiring decision.

## Project structure

```
jobmatch/
  agent.py       tools + system prompt + tool runner, usage and cost tracking
  session.py     tool logic: CV search, evidence check, scoring, report
  skills.py      synonym table and whole-word matching
  documents.py   load .md/.txt/.pdf, split CV into sections and lines
  report.py      Markdown report
  evaluation.py  grading against labels (overclaims, accuracy, score range)
  pricing.py     cost estimate from token usage
eval/            labelled cases + run_eval.py
examples/        fictional CVs and job ads
tests/           20 tests, no network needed
```
