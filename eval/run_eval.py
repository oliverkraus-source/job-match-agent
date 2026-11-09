"""Run the agent on every labelled case and report how well it did.

Needs ANTHROPIC_API_KEY (each case is one real agent run; the script prints the cost).

  python eval/run_eval.py
  python eval/run_eval.py --model claude-haiku-5-5      # much cheaper
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jobmatch.agent import MODEL, AgentError, run_agent  # noqa: E402
from jobmatch.documents import load_text  # noqa: E402
from jobmatch.evaluation import grade  # noqa: E402
from jobmatch.report import render_markdown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    cases = json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8"))
    out_dir = ROOT / "eval" / "results" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, total_cost, failures = [], 0.0, 0
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['name']} ...", flush=True)
        try:
            result = run_agent(load_text(ROOT / case["job"]), load_text(ROOT / case["cv"]), model=args.model)
        except AgentError as exc:
            print(f"   FAILED: {exc}")
            failures += 1
            continue
        g = grade(case, result.session)
        total_cost += result.cost_usd or 0
        (out_dir / f"case{i}.md").write_text(render_markdown(result, title=case["name"]), encoding="utf-8")
        for label in g.labels:
            if not label.correct:
                print(f"   label miss: {label.keywords[0]} expected {label.expect}, got {label.got or 'not assessed'}")
        rows.append((case["name"], g, result))

    print(f"\n## Results ({args.model})\n")
    print("| Case | Score (expected) | Label accuracy | Overclaims | Not assessed | Evidence retries | Turns | Cost |")
    print("|---|---|---|---|---|---|---|---|")
    for name, g, r in rows:
        ok = "✅" if g.score_ok else "⚠️"
        print(f"| {name} | {g.score} ({g.score_range[0]}-{g.score_range[1]}) {ok} | {g.accuracy:.0%} | "
              f"{g.overclaims} | {g.not_assessed} | {r.session.evidence_rejections} | {r.turns} | "
              f"${r.cost_usd or 0:.3f} |")
    if rows:
        labels = [label for _, g, _ in rows for label in g.labels]
        print(f"\n**Overall:** {sum(lb.correct for lb in labels)}/{len(labels)} labels correct, "
              f"{sum(lb.overclaim for lb in labels)} overclaims, "
              f"{sum(g.score_ok for _, g, _ in rows)}/{len(rows)} scores in the expected range, "
              f"{failures} failed runs, total cost ${total_cost:.2f}")
    print(f"\nReports saved in {out_dir.relative_to(ROOT)}/")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
