"""Live evaluation: convert every sample resume (every format) with the LLM and score it.

    python -m resume_template.evaluate            # exit 1 if any check fails
    python -m resume_template.evaluate --judge    # also run the DeepEval meaning-preservation judge

Each conversion is an OpenTelemetry span (`resume_template.convert`) carrying the file format and
check results. Set PHOENIX_COLLECTOR_ENDPOINT (or OTEL_EXPORTER_OTLP_ENDPOINT / AGENTS_OTEL_CONSOLE=1)
to see them in Phoenix; without it the spans are created but not exported.
"""
import argparse
import json
import sys
from pathlib import Path

from agents.telemetry import get_tracer, setup_tracing
from resume_template.checks import Check, run_checks
from resume_template.converter import build_llm, convert_text
from resume_template.loader import SUPPORTED, load_text

SAMPLES = Path(__file__).parent / "samples"


def evaluate(llm, samples_dir: Path = SAMPLES, judge=None) -> bool:
    tracer = get_tracer()
    golden = json.loads((samples_dir / "golden.json").read_text(encoding="utf-8"))
    all_passed = True
    for stem, expected in golden.items():
        for path in sorted(p for p in samples_dir.glob(f"{stem}.*") if p.suffix.lower() in SUPPORTED):
            label = f"{stem}{path.suffix}"
            with tracer.start_as_current_span("resume_template.convert") as span:
                span.set_attribute("resume.file", path.name)
                span.set_attribute("resume.format", path.suffix.lower().lstrip("."))
                try:
                    source = load_text(path)
                    template = convert_text(source, llm)
                    checks = run_checks(source, template, expected)
                    if judge is not None:
                        from resume_template.judge import judge_fidelity

                        ok, score, reason = judge_fidelity(source, template, judge)
                        checks.append(Check("judge_meaning_preserved", bool(ok), f"score {score:.2f}: {reason}"))
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_attribute("resume.passed", False)
                    print(f"{label:22} ERROR {exc}")
                    all_passed = False
                    continue
                failed = [c.name for c in checks if not c.passed]
                span.set_attribute("resume.passed", not failed)
                span.set_attribute("resume.failed_checks", ",".join(failed))
                for c in checks:
                    all_passed &= c.passed
                    print(f"{label:22} {'PASS' if c.passed else 'FAIL'} {c.name}" + (f"  -> {c.detail}" if c.detail else ""))
    print("RESULT:", "PASS" if all_passed else "FAIL")
    return all_passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="add the DeepEval LLM-judge check (extra Groq calls)")
    args = ap.parse_args()
    setup_tracing("resume-template")
    judge = None
    if args.judge:
        from resume_template.judge import GroqJudge

        judge = GroqJudge()
    ok = evaluate(build_llm(), judge=judge)
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
    sys.exit(0 if ok else 1)
