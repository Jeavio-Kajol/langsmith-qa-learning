# Copilot Instructions

QA chatbot + RAG app evaluated with LangSmith. All LLM calls use Groq (instructor-approved substitute for OpenAI; no OpenAI key needed). Embeddings come from Hugging Face (`sentence-transformers`).

## Commands

Run everything from the repo root with `python -m ...` (never `cd` into `chat/`, `rag/`, `tools/`). Use Python 3.12 (matches CI). Config comes from `.env` (see `.env.example`).

```powershell
python -m pip install -r requirements.txt
python -m unittest rag.test_rag_eval_runner -v                       # offline tests, no network
python -m unittest rag.test_rag_eval_runner.RAGEvalRunnerTests.test_strict_threshold_boundaries_fail -v   # single test
python -m rag.rag_eval_runner                # full quality gate (uses Groq quota, writes LangSmith experiment)
python -m rag.rag_eval_runner --limit 3      # diagnostic only; always exits 1, may fail with NO VALID DATA
python -m rag.upload_rag_dataset             # create rag-golden-v1 if missing
python -m chat.upload_chat_dataset           # create chat-golden-v1 if missing
python -m chat.chat_eval_runner              # chat evaluation (reports scores, not a gate)
python -m chat.chat_app / python -m rag.rag_app   # interactive apps; type `exit` to quit
python -m tools.check_langsmith_tracing      # standalone tracing check (no LLM)
```

There is no linter configured.

## Architecture

- Active RAG path: `rag/rag_eval_runner.py` → `rag/rag_app.py` → `rag/rag_index.py` → `documents/qa_knowledge.txt`.
- `chat/` is a separate, earlier exercise: `chat_eval_runner.py` calls the same `chat()` function as `chat_app.py`. Earlier exercises are kept intentionally; don't delete or merge them.
- Datasets (`chat-golden-v1`, 30 examples; `rag-golden-v1`, 56 examples) are defined in the `upload_*_dataset.py` files and live in LangSmith. Evaluators fetch them by name from the workspace of the API key.
- The app under test receives only the question; expected answers/keywords stay on the evaluator side.
- `RAGEvalRunner` / `EvalConfig` run one shared LLM-judge call per example (cached) that supplies four scores (recall, faithfulness+precision, hallucination, relevance), plus a refusal/no-answer callback.

## Quality gate conventions

- `THRESHOLDS` in `rag/rag_eval_runner.py` are strict: `>` for higher-is-better metrics, `<` for hallucination rate. Boundary values fail. Don't lower thresholds to hide provider errors.
- Missing scores, generation/judge/target errors, and quota exhaustion must fail the gate (exit code 1). Per-run state (`metric_scores`, error lists, caches) is module-level and must be reset between runs; tests cover this.
- Short-term rate limits retry; daily quota exhaustion stops further model calls.
- Inputs over `MAX_INPUT_CHARS` (10000) are rejected, never silently truncated.
- Never commit secrets; `.env` is gitignored, `.env.example` stays tracked.

## CI (`.github/workflows/rag-evaluation.yml`)

- `Offline Tests` runs on push/PR to `main`, manual, weekly. `RAG Quality Gate` (needs secrets `GROQ_API_KEY`, `LANGSMITH_API_KEY`) is skipped on push and fails its preflight on fork PRs. Do not switch to `pull_request_target`.
- CI installs CPU-only `torch` before `requirements.txt`.
- `README.md` "Phase 2 Requirement Check" tracks verification status; keep it accurate when behavior changes (don't claim live quality is verified without a real run).
