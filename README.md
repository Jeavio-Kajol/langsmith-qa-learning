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

## Resume-to-Template POC (`resume_template/`)

The dev team's converter (many resumes in different layouts -> one common template) is not available yet. This POC stands in for it so the **testing approach** can be built and proven now. The QA code is the deliverable; the converter is replaceable.

```powershell
python -m resume_template.converter resume_template\samples --out converted   # convert resumes (Groq)
python -m resume_template.evaluate                                             # convert + score every sample, all formats (live, uses Groq)
python -m resume_template.evaluate --judge                                     # also run the DeepEval LLM judge (more Groq calls)
python -m unittest resume_template.test_resume_template -v                     # offline, no network
python -m resume_template.make_samples                                         # regenerate .docx/.pdf samples from the .txt ones
```

- `loader.py`: reads `.pdf`, `.docx`, `.doc`, `.txt`, `.md`. `.doc` is converted through LibreOffice (`soffice` on PATH); without it a clear error is raised. Files with no extractable text (scans) are rejected, not silently converted to an empty resume.
- `template.py`: the target template (name, contact, summary, skills, experience, education), date rule (`YYYY-MM`, `YYYY`, or `Present`) and a deterministic Markdown renderer with fixed section order.
- `converter.py`: LLM conversion (Groq, JSON output, retries, >10000 chars rejected rather than truncated). Swap `convert_text` for the dev team's converter when available; the checks only need a `ResumeTemplate`.
- `checks.py`: deterministic verification. Universal (no expected answer needed): **structure**, **required_fields**, **formats**, **grounding** (every name/company/skill/email/phone must appear in the source, so nothing is invented), **completeness** (contact details not dropped), **experience_order**. Golden checks compare against `samples/golden.json`: contact, jobs (company/title/dates), skill recall >= 0.8, education.
- `judge.py` (DeepEval): a GEval LLM judge on Groq for the fuzzy part only, whether summary and bullets keep the source's meaning. Optional (`--judge`) because it costs extra calls; facts are checked deterministically instead.
- OpenTelemetry/Phoenix: `evaluate` emits one `resume_template.convert` span per file (format, pass/fail, failed checks). Set `PHOENIX_COLLECTOR_ENDPOINT` to view them in Phoenix, or `AGENTS_OTEL_CONSOLE=1` to print them.
- `samples/`: three resumes in different layouts (sectioned, one-line, narrative prose), each as `.txt`, `.docx` and `.pdf`, plus hand-written expected results.

How to grow the test set: add a resume file and a `golden.json` entry with the same stem; include hard cases (two-column layouts, tables, missing sections, non-English, current roles, employment gaps, very long resumes, scanned PDFs).

Verification status: offline tests pass (including real generated `.docx` and `.pdf` files). Live Groq run (`openai/gpt-oss-20b`) on all 9 sample files passed every deterministic check and the DeepEval judge; the judge also scored 0.0 on a deliberately tampered conversion. Spans were verified through the console exporter only; export to a Phoenix server was not exercised for this POC. `.doc` reading is not verified (LibreOffice is not installed on the test machine). The samples are machine-generated clean files, not real-world resumes, so this is not a quality baseline for the real converter.
## Agents (`agents/`)

### Judge agent

A LangGraph workflow (prepare Ã¢â€ â€™ score Ã¢â€ â€™ verify Ã¢â€ â€™ publish) that scores a RAG answer for faithfulness, hallucination rate, context precision and answer relevance. The Groq judge only labels each claim and context chunk; the numbers are computed in code. Invalid judge output is retried once, then reported as a failure. Inputs over the size limit are rejected, not truncated. Provider quota exhaustion stops further model calls. Scores are written as `agent_*` feedback so they never mix with the quality gate's own keys.

```powershell
python -m agents.judge --file items.json                                   # dry run, prints scores
python -m agents.judge --langsmith-project <project> --limit 5 --publish-langsmith
python -m agents.judge --file items.json --publish-phoenix                  # needs PHOENIX_COLLECTOR_ENDPOINT
python -m unittest agents.test_judge_agent -v                              # offline tests
```

`items.json` is a list of `{"question", "answer", "contexts": [...], "run_id"?, "span_id"?}`. `run_id` is a LangSmith run, `span_id` a Phoenix span. Exit code is 1 if any item fails to score or publish.

Verification status: offline tests pass. Verified live: Groq scoring (faithful vs. fabricated answers scored as expected), LangSmith feedback written to a real run, Phoenix annotations written to a real span on a local server. The judge is not part of the quality gate and its scores are not thresholded. `client.list_runs` is deprecated in LangSmith (removal after Jan 2027); migrate to `client.runs.query`.

### Trace agent

A LangGraph workflow (diagnose Ã¢â€ â€™ explain Ã¢â€ â€™ verify Ã¢â€ â€™ publish) that explains why a run went wrong. Deterministic checks find the problems (failed spans, empty retrieval or output, a leaf span using >= 60% of total time and >= 3 s, judge scores below 0.5 or hallucination above 0.5). The Groq LLM then explains the likely root cause. Its explanation is rejected unless the category is valid and every piece of evidence names a real span or check from that trace. Healthy traces are reported without any LLM call. The diagnosis is written as `agent_trace_diagnosis` feedback on the LangSmith root run, or as a note on the Phoenix root span.

```powershell
python -m agents.trace --langsmith-run <run-id> --publish-langsmith
python -m agents.trace --langsmith-project <project> --limit 5
python -m agents.trace --phoenix-project <project> --trace-id <id> --publish-phoenix
python -m agents.trace --file trace.json
python -m unittest agents.test_trace_agent -v
```

Verification status: offline tests pass. Verified live: LangSmith trace loaded, explained by Groq and written back as feedback; Phoenix trace loaded and explained by Groq, and the note insert was accepted by a local Phoenix server (Phoenix's API does not return notes, so the note text itself was not read back). Thresholds above are untuned defaults. The report agent is below; the dataset agent is not built yet.

### Report agent

Compares two evaluation runs (LangSmith experiments or JSON files) and flags regressions. All numbers and flags come from code: a metric is a regression if it gets worse by more than 0.05 (`hallucination_rate` is lower-is-better), or if it crosses the quality-gate threshold from pass to fail. Examples whose score worsened by 0.3 or more are listed, and baseline examples missing from the candidate are reported as a coverage problem. The Groq LLM only writes a short summary, rejected if it cites metrics or examples not in the comparison (use `--no-llm` to skip it). Exit code is 1 on any regression, coverage problem or error.

```powershell
python -m agents.report --baseline <experiment> --candidate <experiment> --out reports/compare.md
python -m agents.report --baseline-file old.json --candidate-file new.json
python -m unittest agents.test_report_agent -v
```

JSON format: `{"name": "v1", "examples": {"<example id>": {"faithfulness": 1.0, ...}}}`.

Verification status: offline tests pass. Verified live: Groq summary on sample files, and comparison of two real LangSmith experiments (scores read from feedback). It does not publish anywhere; it prints Markdown and optionally writes a file.

### Dataset agent

Drafts new RAG test questions from `documents/qa_knowledge.txt` for a human to review. The Groq LLM proposes questions; code keeps a candidate only if every keyword and context phrase appears in its source section, the answer's keywords appear in the answer, at least 80% of the answer's content words come from the source, and the question is not a near-duplicate of an existing one (read from `rag-golden-v1` and the candidates dataset). Drafts are written with `"approved": false`; nothing is uploaded until a human sets `approved` to `true` and runs the upload. Approved examples go to a separate dataset (`rag-golden-candidates-v1`, override with `--dataset`), marked `human_verified`, never directly into the quality-gate dataset. Only answerable categories are drafted (retrieval, context relevance, grounded, multi-topic); out-of-scope, ambiguous and adversarial cases still need to be written by hand. Drafting refuses to overwrite an existing file so approvals are not lost.

```powershell
python -m agents.dataset --out datasets/candidates.json          # draft; add --per-section N --multi N --offline
python -m agents.dataset --upload datasets/candidates.json       # upload approved=true examples only
python -m unittest agents.test_dataset_agent -v
```

Verification status: offline tests pass. Verified live: Groq drafted 11 examples that passed all checks, and approved examples were uploaded to LangSmith (human_verified metadata, re-upload skipped as duplicates). Answer quality is only checked mechanically, so a human must still read each draft.

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