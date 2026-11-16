import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from resume_template.checks import _golden_checks, run_checks
from resume_template.converter import MAX_INPUT_CHARS, convert_text
from resume_template.loader import load_text
from resume_template.make_samples import write_docx, write_pdf
from resume_template.template import ResumeTemplate, render_markdown

SAMPLES = Path(__file__).parent / "samples"
GOLDEN = json.loads((SAMPLES / "golden.json").read_text(encoding="utf-8"))
SOURCE = (SAMPLES / "priya_sharma.txt").read_text(encoding="utf-8")


def good_output() -> dict:
    return {
        "name": "Priya Sharma",
        "email": "priya.sharma@example.com",
        "phone": "+91 98765 43210",
        "location": "Pune, India",
        "summary": "Backend engineer with 6 years of experience building payment APIs.",
        "skills": ["Python", "Django", "PostgreSQL", "Docker", "AWS"],
        "experience": [
            {"company": "Finlytics Pvt Ltd", "title": "Senior Software Engineer", "start": "2021-01", "end": "Present", "bullets": []},
            {"company": "CodeNest Technologies", "title": "Software Engineer", "start": "2018-06", "end": "2020-12", "bullets": []},
        ],
        "education": [{"institution": "College of Engineering Pune", "degree": "B.Tech in Computer Science", "year": "2018"}],
    }


def failed(output: dict) -> set:
    checks = run_checks(SOURCE, ResumeTemplate.model_validate(output), GOLDEN["priya_sharma"])
    return {c.name for c in checks if not c.passed}


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=self.replies.pop(0))


class CheckTests(unittest.TestCase):
    def test_correct_conversion_passes_every_check(self):
        self.assertEqual(failed(good_output()), set())

    def test_invented_company_is_flagged(self):
        o = good_output()
        o["experience"][1]["company"] = "Globex Corp"
        self.assertTrue({"grounding", "golden_experience"} <= failed(o))

    def test_dropped_contact_is_flagged(self):
        o = good_output()
        o["email"], o["phone"] = "", ""
        self.assertTrue({"required_fields", "completeness", "golden_contact"} <= failed(o))

    def test_unnormalised_date_is_flagged(self):
        o = good_output()
        o["experience"][0]["start"] = "Jan 2021"
        self.assertIn("formats", failed(o))

    def test_wrong_date_value_is_flagged(self):
        o = good_output()
        o["experience"][1]["start"] = "2017-06"
        self.assertEqual(failed(o), {"golden_experience"})

    def test_oldest_first_is_flagged(self):
        o = good_output()
        o["experience"].reverse()
        self.assertIn("experience_order", failed(o))

    def test_missing_skills_are_flagged(self):
        o = good_output()
        o["skills"] = ["Python"]
        self.assertEqual(failed(o), {"golden_skills"})

    def test_skill_with_qualifier_still_matches_golden(self):
        o = good_output()
        o["skills"][0] = "Python (pandas)"
        checks = _golden_checks(ResumeTemplate.model_validate(o), GOLDEN["priya_sharma"])
        self.assertTrue(next(c for c in checks if c.name == "golden_skills").passed)

    def test_all_resumes_render_with_identical_headings(self):
        a = render_markdown(ResumeTemplate.model_validate(good_output()))
        b = render_markdown(ResumeTemplate(name="X", email="x@y.com"))
        headings = lambda s: [l for l in s.splitlines() if l.startswith("## ")]
        self.assertEqual(headings(a), headings(b))


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _convert_roundtrip(self, suffix, writer):
        path = Path(self.tmp.name) / f"priya{suffix}"
        writer(SOURCE, path)
        text = load_text(path)
        t = convert_text(text, FakeLLM(json.dumps(good_output())))
        checks = run_checks(text, t, GOLDEN["priya_sharma"])
        self.assertEqual([c.name for c in checks if not c.passed], [])

    def test_docx_end_to_end(self):
        self._convert_roundtrip(".docx", write_docx)

    def test_pdf_end_to_end(self):
        self._convert_roundtrip(".pdf", write_pdf)

    def test_docx_tables_are_read(self):
        from docx import Document

        path = Path(self.tmp.name) / "t.docx"
        doc = Document()
        doc.add_table(rows=1, cols=2).rows[0].cells[0].text = "Skills"
        doc.save(str(path))
        self.assertIn("Skills", load_text(path))

    def test_file_without_text_is_rejected(self):
        path = Path(self.tmp.name) / "scan.pdf"
        write_pdf("", path)
        with self.assertRaises(ValueError):
            load_text(path)

    def test_legacy_doc_without_libreoffice_gives_clear_error(self):
        path = Path(self.tmp.name) / "old.doc"
        path.write_bytes(b"x")
        with mock.patch("resume_template.loader.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "LibreOffice"):
                load_text(path)

    def test_unsupported_format_is_rejected(self):
        with self.assertRaises(ValueError):
            load_text(Path(self.tmp.name) / "a.rtf")


class ConverterTests(unittest.TestCase):
    def test_parses_json_wrapped_in_text(self):
        reply = "Here you go:\n" + json.dumps(good_output())
        t = convert_text(SOURCE, FakeLLM(reply))
        self.assertEqual(t.name, "Priya Sharma")

    def test_retries_on_invalid_output_then_succeeds(self):
        llm = FakeLLM("not json", json.dumps({"name": "only name"}), json.dumps(good_output()))
        convert_text(SOURCE, llm, sleep=lambda s: None)
        self.assertEqual(llm.calls, 3)

    def test_fails_after_exhausting_attempts(self):
        with self.assertRaises(RuntimeError):
            convert_text(SOURCE, FakeLLM("x", "x", "x"), sleep=lambda s: None)

    def test_oversized_input_is_rejected_not_truncated(self):
        llm = FakeLLM()
        with self.assertRaises(ValueError):
            convert_text("a" * (MAX_INPUT_CHARS + 1), llm)
        self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
