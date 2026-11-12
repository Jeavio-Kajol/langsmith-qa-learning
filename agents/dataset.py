"""Run the dataset agent from the command line.

    python -m agents.dataset --out datasets/candidates.json            # draft new questions for review
    python -m agents.dataset --upload datasets/candidates.json          # upload only approved=true examples

Drafting never uploads anything. Review the JSON, set "approved": true on the examples you accept, then
upload. Approved examples go to a separate dataset (default rag-golden-candidates-v1), not the
quality-gate dataset. Exit code is 1 if any section failed or nothing could be drafted.
"""
import argparse
import itertools
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents.dataset_agent import (
    DEFAULT_UPLOAD_DATASET, KNOWLEDGE_FILE, DatasetAgent, assign_ids, candidates_to_json, load_approved,
    load_sections, upload_approved,
)
from ingest.telemetry import setup_tracing

EXISTING_DATASETS = ("rag-golden-v1", DEFAULT_UPLOAD_DATASET)


def existing_questions(client) -> list:
    questions = []
    for name in EXISTING_DATASETS:
        if client.has_dataset(dataset_name=name):
            dataset = client.read_dataset(dataset_name=name)
            questions += [str((e.inputs or {}).get("question", "")) for e in client.list_examples(dataset_id=dataset.id)]
    return [q for q in questions if q]


def main(argv=None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Draft new RAG test questions for human review.")
    parser.add_argument("--out", help="write drafted candidates to this JSON file (must not exist)")
    parser.add_argument("--upload", metavar="FILE", help="upload approved=true examples from this file")
    parser.add_argument("--dataset", default=DEFAULT_UPLOAD_DATASET, help="dataset to upload to")
    parser.add_argument("--knowledge", default=KNOWLEDGE_FILE)
    parser.add_argument("--per-section", type=int, default=2, help="questions per section")
    parser.add_argument("--multi", type=int, default=2, help="multi-topic questions (from adjacent section pairs)")
    parser.add_argument("--offline", action="store_true", help="do not read existing questions from LangSmith")
    args = parser.parse_args(argv)
    if bool(args.out) == bool(args.upload):
        parser.error("give exactly one of --out or --upload")

    if args.upload:
        try:
            approved = load_approved(args.upload)
            if not approved:
                print("No examples have approved=true; nothing to upload.", file=sys.stderr)
                return 1
            from langsmith import Client

            uploaded, skipped = upload_approved(Client(), approved, args.dataset)
        except Exception as exc:
            print(f"Upload failed: {exc}", file=sys.stderr)
            return 1
        print(f"Uploaded {uploaded} approved examples to {args.dataset}, skipped {skipped} already present.")
        return 0

    path = Path(args.out)
    if path.exists():
        print(f"{path} already exists; refusing to overwrite possible approvals.", file=sys.stderr)
        return 1
    try:
        sections = load_sections(args.knowledge)
        existing = [] if args.offline else existing_questions(__import__("langsmith").Client())
    except Exception as exc:
        print(f"Could not prepare: {exc}", file=sys.stderr)
        return 1

    setup_tracing("dataset-agent")
    agent = DatasetAgent()
    jobs = [([s], args.per_section) for s in sections if args.per_section > 0]
    pairs = list(itertools.islice(zip(sections, sections[1:]), max(args.multi, 0)))
    jobs += [(list(pair), 1) for pair in pairs]
    candidates, failures = [], 0
    for sections_for_job, count in jobs:
        result = agent.generate_for(sections_for_job, count, existing + [c.question for c in candidates])
        label = " + ".join(result.sections)
        if not result.ok:
            failures += 1
            print(f"FAIL {label}: {result.error or 'nothing accepted'}", file=sys.stderr)
            if result.fatal:
                break
            continue
        candidates += result.accepted
        print(f"OK   {label}: {len(result.accepted)} accepted, {len(result.rejected)} rejected")
        for question, reason in result.rejected:
            print(f"     rejected: {question} -> {reason}")
    if not candidates:
        print("No candidates drafted.", file=sys.stderr)
        return 1
    assign_ids(candidates)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(candidates_to_json(candidates, args.knowledge), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDrafted {len(candidates)} candidates to {path}, {failures} sections failed. Review and set approved=true.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
