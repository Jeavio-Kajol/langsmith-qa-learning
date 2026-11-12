"""Offline tests for the document converter (no network, no LLM calls).

    python -m unittest ingest.test_ingest -v
"""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from deepeval.test_case import LLMTestCase

from ingest import ocr
from ingest.convert import convert_file, write_document
from ingest.converters import ConversionError
from ingest.detect import detect_format
from ingest.fidelity import ContentFidelityMetric

FACTS = ["Jane Doe", "Senior Engineer", "Python"]


def minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref)
    return out


class IngestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def assertFidelity(self, document, facts=FACTS):
        case = LLMTestCase(input=document.source, actual_output=document.markdown, context=facts)
        metric = ContentFidelityMetric()
        metric.measure(case)
        self.assertTrue(metric.is_successful(), metric.reason)

    def test_pdf(self):
        path = self.dir / "resume.pdf"
        path.write_bytes(minimal_pdf("Jane Doe - Senior Engineer - Python"))
        document = convert_file(path)
        self.assertEqual(document.detected_format, "pdf")
        self.assertFidelity(document)

    def test_docx_with_table(self):
        import docx

        path = self.dir / "resume.docx"
        d = docx.Document()
        d.add_heading("Jane Doe", 1)
        d.add_paragraph("Senior Engineer")
        table = d.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Skill", "Python"
        d.save(path)
        document = convert_file(path)
        self.assertIn("| Skill | Python |", document.markdown)
        self.assertFidelity(document)

    def test_detection_ignores_wrong_extension(self):
        import docx

        real = self.dir / "r.docx"
        d = docx.Document()
        d.add_paragraph("Jane Doe Senior Engineer Python")
        d.save(real)
        disguised = self.dir / "resume.dat"
        disguised.write_bytes(real.read_bytes())
        self.assertEqual(detect_format(disguised), "docx")
        pdf = self.dir / "resume.txt"
        pdf.write_bytes(minimal_pdf("x"))
        self.assertEqual(detect_format(pdf), "pdf")

    def test_xlsx_pptx_odt_rtf_html_csv_eml_txt(self):
        from openpyxl import Workbook
        from pptx import Presentation

        wb = Workbook()
        wb.active.append(["Name", "Role", "Skill"])
        wb.active.append(["Jane Doe", "Senior Engineer", "Python"])
        wb.save(self.dir / "a.xlsx")
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "Jane Doe"
        slide.placeholders[1].text = "Senior Engineer Python"
        prs.save(self.dir / "a.pptx")
        with zipfile.ZipFile(self.dir / "a.odt", "w") as z:
            z.writestr("mimetype", "application/vnd.oasis.opendocument.text")
            z.writestr(
                "content.xml",
                '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
                'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text>'
                "<text:p>Jane Doe Senior Engineer Python</text:p></office:text></office:body></office:document-content>",
            )
        (self.dir / "a.rtf").write_text(r"{\rtf1\ansi Jane Doe Senior Engineer Python}")
        (self.dir / "a.html").write_text(
            "<html><head><title>CV</title></head><body><h1>Jane Doe</h1><script>x=1</script>"
            "<ul><li>Senior Engineer</li><li>Python</li></ul></body></html>"
        )
        (self.dir / "a.csv").write_text("Name,Role,Skill\nJane Doe,Senior Engineer,Python\n")
        (self.dir / "a.eml").write_text(
            "From: a@x.com\nTo: b@x.com\nSubject: CV\nContent-Type: text/plain\n\nJane Doe Senior Engineer Python\n"
        )
        (self.dir / "a.txt").write_text("Jane Doe\nSenior Engineer\nPython", encoding="utf-16")
        expected = {"xlsx": "xlsx", "pptx": "pptx", "odt": "odf", "rtf": "rtf", "html": "html",
                    "csv": "csv", "eml": "eml", "txt": "text"}
        for ext, kind in expected.items():
            with self.subTest(ext=ext):
                document = convert_file(self.dir / f"a.{ext}")
                self.assertEqual(document.detected_format, kind)
                self.assertNotIn("x=1", document.markdown)
                self.assertFidelity(document)

    def test_image_without_ocr_returns_warning_not_crash(self):
        from PIL import Image

        path = self.dir / "scan.png"
        Image.new("RGB", (40, 40), "white").save(path)
        with mock.patch.object(ocr, "ocr_rapid", side_effect=ocr.OCRUnavailable("off")), \
             mock.patch.object(ocr, "ocr_tesseract", side_effect=ocr.OCRUnavailable("off")), \
             mock.patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            document = convert_file(path)
        self.assertEqual(document.detected_format, "image")
        self.assertTrue(any("no text" in w for w in document.warnings))

    def test_image_uses_vision_fallback_when_ocr_empty(self):
        from PIL import Image

        path = self.dir / "scan.jpg"
        Image.new("RGB", (40, 40), "white").save(path)
        with mock.patch.object(ocr, "ocr_rapid", return_value=""), \
             mock.patch.object(ocr, "ocr_tesseract", side_effect=ocr.OCRUnavailable("off")), \
             mock.patch.object(ocr, "ocr_vision", return_value="Jane Doe Senior Engineer Python"):
            document = convert_file(path)
        self.assertEqual(document.method, "vision")
        self.assertFidelity(document)
        with mock.patch.object(ocr, "ocr_rapid", return_value=""), \
             mock.patch.object(ocr, "ocr_tesseract", side_effect=ocr.OCRUnavailable("off")), \
             mock.patch.object(ocr, "ocr_vision") as vision:
            convert_file(path, use_vision=False)
        vision.assert_not_called()

    def test_legacy_doc_without_libreoffice_fails_clearly(self):
        path = self.dir / "old.doc"
        path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100)
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(ConversionError, "LibreOffice"):
                convert_file(path)

    def test_rejects_empty_unknown_and_corrupt(self):
        (self.dir / "empty.txt").write_bytes(b"")
        (self.dir / "blob.bin").write_bytes(b"\x00\x01\x02\x03" * 50)
        (self.dir / "bad.pdf").write_bytes(b"%PDF-1.4 garbage")
        for name in ("empty.txt", "blob.bin", "bad.pdf"):
            with self.subTest(name=name), self.assertRaises(ConversionError):
                convert_file(self.dir / name)

    def test_write_document_outputs_md_and_json_without_collisions(self):
        a, b = self.dir / "x" / "cv.txt", self.dir / "y" / "cv.txt"
        a.parent.mkdir(), b.parent.mkdir()
        a.write_text("Jane Doe"), b.write_text("John Roe")
        out = self.dir / "out"
        first = write_document(convert_file(a), out)
        second = write_document(convert_file(b), out)
        self.assertNotEqual(first, second)
        self.assertEqual(len(list(out.glob("*.md"))), 2)
        self.assertEqual(len(list(out.glob("*.json"))), 2)

    def test_fidelity_metric_flags_missing_content(self):
        metric = ContentFidelityMetric()
        metric.measure(LLMTestCase(input="x", actual_output="Jane Doe", context=["Jane Doe", "Python"]))
        self.assertFalse(metric.is_successful())
        self.assertEqual(metric.score, 0.5)


if __name__ == "__main__":
    unittest.main()
