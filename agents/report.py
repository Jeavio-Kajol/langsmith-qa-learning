"""Run the report agent from the command line.

    python -m agents.report --baseline-file old.json --candidate-file new.json
    python -m agents.report --baseline <experiment> --candidate <experiment> --out reports/compare.md

Compares per-example scores of two evaluation runs (LangSmith experiments or JSON files) and flags
regressions. Exit code is 1 if a regression, missing examples, or any error is found.
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents.report_agent import ReportAgent, render_markdown, results_from_file, results_from_langsmith
from ingest.telemetry import setup_tracing


def main(argv=None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Compare two evaluation runs and flag regressions.")
    parser.add_argument("--baseline", help="LangSmith experiment (project) name of the baseline")
    parser.add_argument("--candidate", help="LangSmith experiment (project) name of the candidate")
    parser.add_argument("--baseline-file", help="baseline results JSON")
    parser.add_argument("--candidate-file", help="candidate results JSON")
    parser.add_argument("--out", help="also write the Markdown report to this file")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM summary")
    args = parser.parse_args(argv)
    if bool(args.baseline) == bool(args.baseline_file) or bool(args.candidate) == bool(args.candidate_file):
        parser.error("give exactly one of --baseline/--baseline-file and one of --candidate/--candidate-file")

    setup_tracing("report-agent")
    try:
        client = None
        if args.baseline or args.candidate:
            from langsmith import Client

            client = Client()
        baseline = results_from_file(args.baseline_file) if args.baseline_file else results_from_langsmith(client, args.baseline)
        candidate = results_from_file(args.candidate_file) if args.candidate_file else results_from_langsmith(client, args.candidate)
    except Exception as exc:
        print(f"Could not load results: {exc}", file=sys.stderr)
        return 1

    report = ReportAgent(use_llm=not args.no_llm).run(baseline, candidate)
    text = render_markdown(report)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Report written to {path}")
    if report.error:
        print(f"FAIL: {report.error}", file=sys.stderr)
    if report.coverage_problem:
        print("FAIL: candidate is missing baseline examples or nothing could be compared", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
