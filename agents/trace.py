"""Run the trace agent from the command line.

    python -m agents.trace --file trace.json
    python -m agents.trace --langsmith-run <run-id> --publish-langsmith
    python -m agents.trace --langsmith-project <project> --limit 5       # recent RAG runs
    python -m agents.trace --phoenix-project <project> --trace-id <id> --publish-phoenix

Healthy traces are reported without calling the LLM. Exit code is 1 if any trace could not be
loaded, explained or published.
"""
import argparse
import sys

from dotenv import load_dotenv

from agents.publishers import TraceLangSmithPublisher, TracePhoenixPublisher
from agents.trace_agent import TraceAgent
from agents.trace_model import trace_from_file, trace_from_langsmith, trace_from_phoenix
from ingest.telemetry import setup_tracing

DEFAULT_RUN_NAME = "rag-question-answering"


def langsmith_run_ids(project: str, limit: int, run_name: str, client) -> list:
    runs = client.list_runs(project_name=project, is_root=True, filter=f'eq(name, "{run_name}")', limit=limit)
    return [str(run.id) for run in runs]


def main(argv=None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Explain what went wrong in a trace.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="trace JSON file")
    source.add_argument("--langsmith-run", help="a LangSmith run id")
    source.add_argument("--langsmith-project", help="analyze recent runs of this LangSmith project")
    source.add_argument("--phoenix-project", help="Phoenix project (use with --trace-id)")
    parser.add_argument("--trace-id", help="Phoenix trace id")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--publish-langsmith", action="store_true")
    parser.add_argument("--publish-phoenix", action="store_true")
    args = parser.parse_args(argv)
    if args.phoenix_project and not args.trace_id:
        parser.error("--phoenix-project requires --trace-id")

    setup_tracing("trace-agent")
    traces, load_failures = [], 0
    try:
        if args.file:
            traces.append(trace_from_file(args.file))
        elif args.phoenix_project:
            from phoenix.client import Client as PhoenixClient

            traces.append(trace_from_phoenix(PhoenixClient(), args.phoenix_project, args.trace_id))
        else:
            from langsmith import Client

            client = Client()
            run_ids = [args.langsmith_run] if args.langsmith_run else langsmith_run_ids(
                args.langsmith_project, args.limit, args.run_name, client
            )
            for run_id in run_ids:
                try:
                    traces.append(trace_from_langsmith(client, run_id))
                except Exception as exc:
                    load_failures += 1
                    print(f"FAIL {run_id}: could not load trace: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"Could not load trace: {exc}", file=sys.stderr)
        return 1
    if not traces:
        print("No traces to analyze.", file=sys.stderr)
        return 1

    publishers = []
    if args.publish_langsmith:
        publishers.append(TraceLangSmithPublisher())
    if args.publish_phoenix:
        publishers.append(TracePhoenixPublisher())

    agent = TraceAgent(publishers=publishers)
    failures = load_failures
    for trace in traces:
        diagnosis = agent.analyze(trace)
        if not diagnosis.ok:
            failures += 1
            print(f"FAIL {trace.trace_id}: {diagnosis.error or '; '.join(diagnosis.publish_errors)}", file=sys.stderr)
            continue
        if diagnosis.category == "healthy":
            print(f"OK   {trace.trace_id}: healthy, no problems detected")
        else:
            print(f"OK   {trace.trace_id}: {diagnosis.category} ({diagnosis.confidence} confidence)")
            print(f"     cause: {diagnosis.root_cause}\n     fix:   {diagnosis.suggested_fix}")
            print(f"     evidence: {', '.join(diagnosis.evidence)}")
        for finding in diagnosis.findings:
            print(f"     check: {finding.code} @ {finding.span}: {finding.detail}")
        for note in diagnosis.publish_notes:
            print(f"     note: {note}")
    print(f"\nAnalyzed {len(traces)} traces, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
