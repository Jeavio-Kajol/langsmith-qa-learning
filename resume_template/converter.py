"""POC stand-in for the dev team's converter: resume text -> ResumeTemplate via an LLM (Groq).

    python -m resume_template.converter resume_template\\samples --out converted
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from resume_template.loader import SUPPORTED, load_text
from resume_template.template import ResumeTemplate, render_markdown

MAX_INPUT_CHARS = 10000

PROMPT = """You convert a resume into a fixed JSON template. Return ONLY a JSON object.

Rules:
- Use only facts present in the resume. Never invent values; use "" or [] when missing.
- Dates: "YYYY-MM" when the month is known, "YYYY" when only the year is known, "Present" for current roles.
- Keep company names, institution names and the email exactly as written in the resume.
- Experience must be listed newest first. Skills are short items, one per list entry.

JSON shape:
{{"name": str, "email": str, "phone": str, "location": str, "summary": str,
 "skills": [str],
 "experience": [{{"company": str, "title": str, "start": str, "end": str, "bullets": [str]}}],
 "education": [{{"institution": str, "degree": str, "year": str or null}}]}}

Resume:
\"\"\"
{resume}
\"\"\"
"""


def build_llm():
    from dotenv import load_dotenv
    from langchain_groq import ChatGroq

    load_dotenv()
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def convert_text(resume_text: str, llm, attempts: int = 3, sleep=time.sleep) -> ResumeTemplate:
    if len(resume_text) > MAX_INPUT_CHARS:
        raise ValueError(f"Resume longer than {MAX_INPUT_CHARS} characters; refusing to truncate.")
    last_error = None
    for attempt in range(attempts):
        try:
            raw = llm.invoke(PROMPT.format(resume=resume_text)).content
            return ResumeTemplate.model_validate(_extract_json(raw))
        except Exception as exc:  # bad JSON, schema mismatch, or provider error: retry
            last_error = exc
            if attempt < attempts - 1:
                sleep(2 * (attempt + 1))
    raise RuntimeError(f"Conversion failed after {attempts} attempts: {last_error}")


def convert_file(path, llm) -> ResumeTemplate:
    return convert_text(load_text(path), llm)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert resumes into the common template")
    parser.add_argument("source", help="resume file or folder")
    parser.add_argument("--out", default="converted")
    args = parser.parse_args(argv)

    source = Path(args.source)
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in SUPPORTED) if source.is_dir() else [source]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = build_llm()
    failures = 0
    for f in files:
        try:
            t = convert_file(f, llm)
        except Exception as exc:
            failures += 1
            print(f"FAIL {f.name}: {exc}", file=sys.stderr)
            continue
        (out / f"{f.stem}.json").write_text(t.model_dump_json(indent=2), encoding="utf-8")
        (out / f"{f.stem}.md").write_text(render_markdown(t), encoding="utf-8")
        print(f"OK   {f.name} -> {out / (f.stem + '.md')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
