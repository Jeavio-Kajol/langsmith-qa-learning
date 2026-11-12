# LangSmith QA Learning

A QA chatbot and RAG application with a human-reviewed golden dataset and
a RAG quality gate using Groq, Hugging Face embeddings, and LangSmith.

## Phase 2 Requirement Check

Groq is an instructor-approved substitute because OpenAI API access is not
available for this project. The written learning plan specifies `gpt-4o-mini`;
the instructor approved using Groq instead, as confirmed on 4 November 2026.
Free ChatGPT access does not supply OpenAI API credits.
All active LLM calls remain on Groq; no OpenAI API key is required to run them.

| Requirement | Current Evidence / Remaining Work |
| --- | --- |
| traced chat | Implemented, including tags and session metadata. Verify UI filtering, run comparisons, and a human feedback score in LangSmith. |
| two free evaluators and an AI judge | Implemented; good/bad/edge keyword examples tested offline. Human review of evaluator logic and live experiment failure analysis still need evidence. |
| traced RAG and manual matrix | Indexing, retrieval, generation, input bounds, system instructions, and an interactive entry point are implemented. Run and record the six manual scenarios for Chat and RAG in your team's chosen test record. Oversized input is rejected safely, not silently truncated. |
| Chat 30+ and RAG 50+ | Local definitions contain 30 Chat and 56 RAG examples with uploaders. Confirm cloud counts, human review of every reference, and the approved v1 dataset in LangSmith. A verification flag alone is not proof of review. |
| reusable runner and core checks | `EvalConfig` and `RAGEvalRunner` are implemented. Four core callbacks cover recall, faithfulness, hallucination, and relevance, plus a refusal callback. Precision is returned with faithfulness; one shared AI judgment per example supplies four scores. |
| quality gate | Strict plan targets use `>` for higher-is-better metrics and `<` for hallucinations. Missing data, provider/judge/target errors, and diagnostic-only subsets cannot produce a full CI pass. 35 offline tests pass; live quality is not yet verified. |
| dashboards and PR blocking | Workflow exists. Configure/verify LangSmith dashboards and metric alerts, and require the GitHub checks in branch protection. A failing job alone does not block merging. |
| regression agreement | Proposed policy is below. Team approval, a passing baseline, and selection of the critical smoke cases remain outstanding. Automated relative-baseline gates are not implemented. |

The framework's core structure is in place, but the complete Phase 2 deliverables
cannot be marked verified from offline tests alone. No paid evaluations or
account-setting changes were made during this check. Earlier Groq experiments
and manual results are historical evidence, not a passing baseline for the
current implementation. Use this section for the current verification status.

## Phase 2 File Progression

The Python files are named by purpose. Keep the earlier exercises to demonstrate
how Phase 2 evolved; they are not all dependencies of the final RAG runner.

```text
chat/
	chat_app.py                 Interactive direct chat
	chat_eval_runner.py     Evaluate the real chat app on saved questions
	upload_chat_dataset.py     Upload 30 reviewed chat examples
rag/
	rag_app.py                  Interactive knowledge-grounded assistant
	rag_index.py                Read and search the QA notes
	rag_eval_runner.py          Full RAG evaluation and quality gate
	upload_rag_dataset.py       Upload 56 reviewed RAG examples
	test_rag_eval_runner.py     Offline regression tests
tools/
	check_langsmith_tracing.py  Shared LangSmith connection check
documents/
	qa_knowledge.txt            RAG reference notes
```

The small `__init__.py` files mark `chat/` and `rag/` as Python packages;
they do not start an application. Run commands from the project root using
`python -m ...`, not by changing into those folders.

| Learning Stage | Files | What It Adds |
| --- | --- | --- |
| direct chat and tracing | [chat/chat_app.py](chat/chat_app.py), [tools/check_langsmith_tracing.py](tools/check_langsmith_tracing.py) | Ask the LLM directly; inspect the request in LangSmith. |
| starter evaluation | [chat/chat_eval_runner.py](chat/chat_eval_runner.py) | Score chat answers with keyword checks and an AI judge. |
| knowledge-grounded answers | [rag/rag_index.py](rag/rag_index.py), [rag/rag_app.py](rag/rag_app.py) | Search our QA notes before generating an answer. |
| reviewed test questions | [chat/upload_chat_dataset.py](chat/upload_chat_dataset.py), [rag/upload_rag_dataset.py](rag/upload_rag_dataset.py) | Define and upload separate Chat and RAG datasets. |
| repeatable quality gate | [rag/rag_eval_runner.py](rag/rag_eval_runner.py), [rag/test_rag_eval_runner.py](rag/test_rag_eval_runner.py) | Run RAG evaluations, enforce thresholds, and test the checking code. |

Active path: `rag/rag_eval_runner.py` calls `rag/rag_app.py`, which uses `rag/rag_index.py`
to search [documents/qa_knowledge.txt](documents/qa_knowledge.txt).
The unused duplicate RAG backup has been removed.

### How the Chat Files Relate

- [chat/chat_app.py](chat/chat_app.py) is the interactive chat application.
- [chat/chat_eval_runner.py](chat/chat_eval_runner.py) reads saved
	questions from `chat-golden-v1` and calls the same `chat()` function as the
	interactive app. Python checks keywords and forbidden phrases; an AI judge
	scores relevance. The app receives only the question, never expected answers.
- [chat/upload_chat_dataset.py](chat/upload_chat_dataset.py) uploads the
	30-example `chat-golden-v1` dataset used by that evaluation. Upload once if it
	is missing; the evaluator does not create a separate starter question set.
- [tools/check_langsmith_tracing.py](tools/check_langsmith_tracing.py) checks
	the shared LangSmith connection by sending a small trace. It does not call an
	LLM, read QA notes, or need either application to be running. Apps do not
	import it; run it separately with `python -m tools.check_langsmith_tracing`.

Run the connected chat test flow from the project root:

```powershell
python -m chat.upload_chat_dataset
python -m chat.chat_eval_runner
```

The first command saves reviewed test cases; it does not generate AI answers.
The second calls Groq for actual chat answers and relevance judgments, so it
needs quota. In LangSmith, open Datasets > `chat-golden-v1` > Experiments and
select the new `chat-golden-eval` experiment, or follow the printed link. Inspect
each question, generated output, and evaluator feedback. These checks are
quality indicators, not proof that every answer is factually correct. The chat
runner reports scores; it is not the RAG CI threshold gate.

The test dataset is not a knowledge source: the question goes to the chatbot,
and the expected keywords stay on the checking side. RAG is different because
it additionally retrieves QA notes and sends them to the model before answering.

## Document Ingestion (`ingest/`)

Converts mixed-format files (resumes etc.) into one format: **Markdown + a JSON sidecar** (source, detected format, method, page count, SHA-256, metadata, warnings). Format is detected from file contents, not the extension.

```powershell
python -m ingest.convert resumes\ --out converted      # file or folder
python -m ingest.convert scan.heic --no-vision          # disable Groq vision fallback
python -m unittest ingest.test_ingest -v                # offline tests (incl. DeepEval fidelity metric)
```

- Supported: PDF, DOCX, XLSX, PPTX, ODT/ODS/ODP, RTF, HTML, CSV/TSV, EML, TXT, images (PNG, JPEG, HEIC, TIFF, WebP, GIF, BMP). Legacy `.doc/.xls/.ppt` need LibreOffice (`soffice`) on PATH; otherwise a clear error is raised.
- Images and scanned PDF pages: RapidOCR, then Tesseract, then a Groq vision model (`GROQ_VISION_MODEL`, needs `GROQ_API_KEY`). If no engine can read the image, the file converts with a warning and empty text rather than crashing.
- Tracing: OpenTelemetry spans (`ingest.convert_file`). Set `PHOENIX_COLLECTOR_ENDPOINT` (e.g. `http://localhost:6006`) to view them in Phoenix (Arize), `OTEL_EXPORTER_OTLP_ENDPOINT` for any OTLP collector, or `INGEST_OTEL_CONSOLE=1`.
- Verification status: offline tests pass with generated fixtures. Verified live: local OCR (Tesseract) on real PNG/JPEG/HEIC images, the Groq vision call (`qwen/qwen3.8-27b`), and trace export to a local Phoenix server. LibreOffice conversion of real `.doc` files was not exercised (LibreOffice not installed on the test machine).

## Manual Checks

Start each application separately to perform the manual checks:

```powershell
python -m chat.chat_app
python -m rag.rag_app
```

In each app, try a supported question, an out-of-domain question, an ambiguous
question, an adversarial instruction, empty input, and a long question. Type
`exit` to close each app. Write the input, expected behavior, observed answer,
and your pass/fail decision in your team's chosen test record. Model requests
use Groq quota. The evaluation runner saves automated results in LangSmith;
it does not generate a manual report.

## Local Setup

Use Python 3.12 for compatibility with the GitHub Actions environment.
Create a virtual environment, activate it, and install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Configure `.env` using `.env.example`: `GROQ_API_KEY`, `LANGSMITH_API_KEY`,
`LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT=qa-learning`.
Optional `GROQ_MODEL` and `GROQ_JUDGE_MODEL` select models with available quota.
Never commit API keys. `.gitignore` excludes `.env`, virtual environments,
and generated Python files; `.env.example` stays tracked.

The `rag-golden-v1` dataset must exist in the LangSmith workspace associated
with your API key. If it does not exist, upload the reviewed RAG examples:

```powershell
python -m rag.upload_rag_dataset
```

## Evaluation

```powershell
python -m unittest rag.test_rag_eval_runner -v
python -m rag.rag_eval_runner --limit 3
python -m rag.rag_eval_runner
```

Offline tests do not call external services. The limited run is diagnostic,
not a full CI pass: it may have no examples for a category and fail with
`NO VALID DATA`. Even a complete set of passing subset scores is diagnostic
only and returns exit code 1. The full run generates answers and uses an LLM judge, so
it consumes Groq tokens and writes a LangSmith experiment.

The pipeline reports context recall, context precision, faithfulness,
hallucination rate, no-answer rate, and answer relevance. Each metric must
meet its individual threshold in `rag/rag_eval_runner.py`. Missing scores,
generation errors, judge errors, and exhausted quota fail the gate with
exit code 1. Short-term rate limits retry; daily quota exhaustion stops
further model calls. Do not lower thresholds to hide provider errors.

`python -m rag.rag_app` starts the active interactive RAG application.

Check usage and model limits at https://console.groq.com/. A provider retry
delay indicates when that request may fit again, not when enough quota is
available for the full dataset. Generation and judging share quota when
they use the same model.

## GitHub Actions Setup

The repository is already cloned and connected to `origin`; no `git init`
is needed. The workflow is `.github/workflows/rag-evaluation.yml`.
Local files must be committed and pushed before GitHub can run it.

1. In GitHub repository Settings, open Secrets and variables > Actions.
	Add repository secrets `GROQ_API_KEY` and `LANGSMITH_API_KEY` directly
	in GitHub, never in source code or chat.
2. Optionally add repository variables `GROQ_MODEL` and `GROQ_JUDGE_MODEL`.
	The defaults use `openai/gpt-oss-20b`.
3. Ensure the secrets have access to the existing `rag-golden-v1` dataset.
4. Enable GitHub Actions and run RAG Evaluation from the Actions tab.
5. In Settings > Rules > Rulesets (or branch protection), protect `main`,
	require pull requests, and require both `Offline Tests` and
	`RAG Quality Gate` status checks. The checks may need to run once before
	they appear in the selector. A failed workflow alone does not prevent
	merging without these rules.

Offline tests run on pushes to `main`, PRs targeting `main`, manual runs,
and weekly runs. The full quality gate runs on PRs, manual runs, and Mondays
at 06:00 UTC. Pushes do not repeat the paid quality evaluation after merge.
Live jobs have a 30-minute timeout and preserve their output as an Actions
artifact; the console also links to the LangSmith experiment.

Fork PRs do not receive repository secrets: offline tests can run, but
the live gate fails its secret preflight. A maintainer must review the code
and evaluate it on a trusted same-repository branch. Do not switch to
`pull_request_target` to expose API keys to untrusted PR code.

## Regression Strategy

This is a proposed team policy, not a record of team approval. The current
CI workflow enforces absolute metric thresholds on the full dataset;
automatic baseline comparison and category-specific gates are not yet
implemented. `--limit` selects the first N returned examples and is not a
curated 20-case critical smoke suite.

| Change | Evaluation | Proposed Additional Acceptance |
| --- | --- | --- |
| Model version | Full golden dataset | No quality loss greater than 5 percentage points from the approved baseline |
| Prompt | Affected categories, then full CI gate | Target metric improves; unrelated metrics remain stable |
| New KB documents | Retrieval-focused cases, then full CI gate | Context recall remains above 0.85 |
| App refactor | Team-selected 20 critical cases, then full CI gate | Every critical case passes |
| Weekly | Full golden dataset | No quality loss greater than 3 percentage points week over week |

Approve and record a passing LangSmith experiment as the baseline, together
with the dataset version, model, prompt revision, and commit. Higher is
better for all metrics except hallucination rate; an increase in that rate
is a regression. Review the strategy and critical cases with the team.

## Monitoring and Completion

LangSmith experiments expose the evaluator feedback for comparisons.
Create dashboard charts for the metric keys reported by the runner and,
where supported by your LangSmith plan, configure alerts using the same
thresholds. Monitor provider failures separately from quality regressions.
Dashboards and alerts require configuration in your LangSmith account;
they are not provisioned by this repository.

Framework verification is complete only after the full evaluation meets its thresholds,


the workflow runs successfully on GitHub, required checks protect `main`,
monitoring is configured, and the team has reviewed the regression policy.