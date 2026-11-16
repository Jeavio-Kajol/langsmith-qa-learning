"""Run the judge agent from the command line.

    python -m agents.judge --file items.json                       # dry run, prints scores
    python -m agents.judge --langsmith-project my-project --limit 5 --publish-langsmith
    python -m agents.judge --file items.json --publish-phoenix     # needs PHOENIX_COLLECTOR_ENDPOINT

items.json: [{"question": "...", "answer": "...", "contexts": ["..."],
              "run_id": "<LangSmith run id, optional>", "span_id": "<Phoenix span id, optional>"}]
Exit code is 1 if any item could not be scored or published.
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents.judge_agent import JudgeAgent, JudgeItem
from agents.publishers import LangSmithPublisher, PhoenixPublisher
from ingest.telemetry import setup_tracing

DEFAULT_RUN_NAME = "rag-question-answering"


def items_from_file(path: str) -> list:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("input file must contain a JSON list")
    return [
        JudgeItem(
            question=entry.get("question", ""),
            answer=entry.get("answer", ""),
            contexts=entry.get("contexts", []),
            run_id=entry.get("run_id"),
            span_id=entry.get("span_id"),
            item_id=str(entry.get("id", index)),
        )
        for index, entry in enumerate(data)
    ]


def items_from_langsmith(project: str, limit: int, run_name: str, client=None) -> tuple[list, int]:
    """Return (items, skipped_count) from recent root runs of the RAG app."""
    if client is None:
        from langsmith import Client

        client = Client()
    items, skipped = [], 0
    for run in client.list_runs(project_name=project, is_root=True, filter=f'eq(name, "{run_name}")', limit=limit):
        inputs, outputs = run.inputs or {}, run.outputs or {}
        if not inputs.get("question") or not outputs.get("answer"):
            skipped += 1
            continue
        items.append(
            JudgeItem(
                question=inputs["question"],
                answer=outputs["answer"],
                contexts=outputs.get("retrieved_sources", []),
                run_id=str(run.id),
                item_id=str(run.id),
                session_id=str(run.session_id) if getattr(run, "session_id", None) else None,
            )
        )
    return items, skipped


def main(argv=None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Score RAG answers with the judge agent.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="JSON file of items to judge")
    source.add_argument("--langsmith-project", help="judge recent runs from this LangSmith project")
    parser.add_argument("--limit", type=int, default=5, help="max runs to fetch (default 5)")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--publish-langsmith", action="store_true", help="write scores as LangSmith feedback")
    parser.add_argument("--publish-phoenix", action="store_true", help="write scores as Phoenix annotations")
    args = parser.parse_args(argv)

    setup_tracing("judge-agent")
    skipped = 0
    try:
        if args.file:
            items = items_from_file(args.file)
        else:
            items, skipped = items_from_langsmith(args.langsmith_project, args.limit, args.run_name)
    except Exception as exc:
        print(f"Could not load items: {exc}", file=sys.stderr)
        return 1
    if not items:
        print("No items to judge.", file=sys.stderr)
        return 1

    publishers = []
    if args.publish_langsmith:
        publishers.append(LangSmithPublisher())
    if args.publish_phoenix:
        publishers.append(PhoenixPublisher())

    verdicts = JudgeAgent(publishers=publishers).evaluate_many(items)
    failures = 0
    for verdict in verdicts:
        if verdict.ok:
            scores = " ".join(f"{k}={v:.2f}" for k, v in verdict.scores.items())
            print(f"OK   {verdict.item_id}: {scores}\n     {verdict.reason}")
        else:
            failures += 1
            print(f"FAIL {verdict.item_id}: {verdict.error or '; '.join(verdict.publish_errors)}", file=sys.stderr)
        for note in verdict.publish_notes:
            print(f"     note: {note}")
    if skipped:
        print(f"Skipped {skipped} runs without question/answer.")
    print(f"\nJudged {len(verdicts)} items, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
