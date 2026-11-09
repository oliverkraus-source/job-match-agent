"""Command-line interface.

  python -m jobmatch examples/jobs/junior_ai_engineer.md examples/cvs/alex_becker.md
  python -m jobmatch job.md my_cv.pdf -o report.md --model claude-haiku-5-5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic

from .agent import MODEL, AgentError, run_agent
from .documents import load_text
from .report import render_markdown


def _print_event(name: str, args: dict, result: str, is_error: bool) -> None:
    if is_error:
        print(f"  ✗ {name}: {result}")
    elif name == "search_cv":
        n = result.count("\n- ") + 1 if result.startswith("- ") else 0
        print(f"  🔎 search_cv({args['query']!r}) -> {n} line(s)")
    elif name == "record_requirement":
        print(f"  📝 {args['requirement']} [{args['importance']}] -> {args['status']}")
    elif name == "get_match_score":
        print(f"  📊 score: {json.loads(result)['score']}/100")
    else:
        print(f"  📄 {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jobmatch", description="Match a CV against a job ad with an AI agent.")
    parser.add_argument("job", help="job ad (.md, .txt or .pdf)")
    parser.add_argument("cv", help="CV (.md, .txt or .pdf)")
    parser.add_argument("-o", "--output", help="write the Markdown report here (default: print it)")
    parser.add_argument("--model", default=MODEL, help=f"model to use (default: {MODEL})")
    parser.add_argument("--json", help="also save the raw results as JSON here")
    args = parser.parse_args(argv)

    print(f"Analysing {Path(args.cv).name} against {Path(args.job).name} with {args.model}...\n")
    try:
        result = run_agent(load_text(args.job), load_text(args.cv), model=args.model, on_event=_print_event)
    except AgentError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except anthropic.AuthenticationError:
        print("\nError: your ANTHROPIC_API_KEY was rejected - check that it is copied correctly.", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as exc:   # e.g. credit balance too low
        print(f"\nError from the API ({exc.status_code}): {exc.message}", file=sys.stderr)
        return 1

    markdown = render_markdown(result, title=f"{Path(args.cv).stem} × {Path(args.job).stem}")
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"\nReport written to {args.output}")
    else:
        print("\n" + markdown)
    if args.json:
        data = {"model": result.model, "turns": result.turns, "cost_usd": result.cost_usd,
                **result.session.to_dict(), "trace": result.trace}
        Path(args.json).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    cost = f"${result.cost_usd:.3f}" if result.cost_usd is not None else "unknown"
    print(f"Done: {result.turns} turns, {result.tool_calls} tool calls, estimated cost {cost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
