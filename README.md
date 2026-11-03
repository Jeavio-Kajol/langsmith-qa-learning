# LangSmith QA Learning

A QA chatbot and RAG application with a human-reviewed golden dataset and
a Day 10 evaluation gate using Groq, Hugging Face embeddings, and LangSmith.

## Local Setup

Use Python 3.12 for compatibility with the GitHub Actions environment.
Create a virtual environment, activate it, and install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Configure `.env` using `.env.example`: `GROQ_API_KEY`, `LANGSMITH_API_KEY`,
`LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT=qa-learning-day10`.
Optional `GROQ_MODEL` and `GROQ_JUDGE_MODEL` select models with available quota.
Never commit API keys. `.gitignore` excludes `.env`, virtual environments,
and generated Python files; `.env.example` stays tracked.

The `rag-golden-v1` dataset must exist in the LangSmith workspace associated
with your API key. If it does not exist, upload the approved Day 9 examples:

```powershell
python day9_rag_golden_dataset.py
```

## Evaluation

```powershell
python -m unittest test_day10_eval_runner -v
python day10_eval_runner.py --limit 3
python day10_eval_runner.py
```

Offline tests do not call external services. The limited run is diagnostic,
not a full CI pass: it may have no examples for a category and fail with
`NO VALID DATA`. The full run generates answers and uses an LLM judge, so
it consumes Groq tokens and writes a LangSmith experiment.

The pipeline reports context recall, context precision, faithfulness,
hallucination rate, no-answer rate, and answer relevance. Each metric must
meet its individual threshold in `day10_eval_runner.py`. Missing scores,
generation errors, judge errors, and exhausted quota fail the gate with
exit code 1. Short-term rate limits retry; daily quota exhaustion stops
further model calls. Do not lower thresholds to hide provider errors.

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

Day 10 is complete only after the full evaluation meets its thresholds,
the workflow runs successfully on GitHub, required checks protect `main`,
monitoring is configured, and the team has reviewed the regression policy.