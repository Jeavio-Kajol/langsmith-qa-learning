"""Per-format converters. Each returns a ConvertedDocument body (Markdown) plus metadata."""
import csv
import email
import io
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from email import policy
from pathlib import Path
from typing import Optional

from ingest import ocr
from ingest.detect import LEGACY_EXTS


@dataclass
class Converted:
    markdown: str
    method: str
    metadata: dict = field(default_factory=dict)
    pages: int = 0
    warnings: list = field(default_factory=list)


class ConversionError(RuntimeError):
    pass


def md_table(rows: list[list[str]]) -> str:
    rows = [[(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in r] for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def convert_pdf(path: Path, use_vision: bool) -> Converted:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts, warnings, methods = [], [], {"text"}
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < ocr.MIN_OCR_CHARS:
            ocr_texts = []
            for image_file in page.images:
                try:
                    found, method, warning = ocr.extract_text(ocr.load_image(image_file.data), use_vision)
                except Exception as exc:
                    warnings.append(f"page {number}: image skipped ({type(exc).__name__})")
                    continue
                if found:
                    ocr_texts.append(found)
                    methods.add(method)
                if warning:
                    warnings.append(f"page {number}: {warning}")
            text = "\n\n".join(filter(None, [text, *ocr_texts]))
        if not text:
            warnings.append(f"page {number}: no text extracted")
        parts.append(f"<!-- page {number} -->\n{text}")
    info = reader.metadata or {}
    metadata = {k.lstrip("/").lower(): str(v) for k, v in info.items()}
    if reader.is_encrypted:
        warnings.append("PDF is encrypted")
    return Converted("\n\n".join(parts), "+".join(sorted(methods)), metadata, len(reader.pages), warnings)


def convert_docx(path: Path, use_vision: bool) -> Converted:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    blocks = []
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name or "") if paragraph.style is not None else ""
            if style == "Title":
                text = f"# {text}"
            elif style.startswith("Heading"):
                level = "".join(ch for ch in style if ch.isdigit()) or "1"
                text = f"{'#' * min(int(level) + 1, 6)} {text}"
            elif "List" in style:
                text = f"- {text}"
            blocks.append(text)
        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            blocks.append(md_table([[c.text for c in row.cells] for row in table.rows]))
    props = document.core_properties
    metadata = {"title": props.title, "author": props.author}
    return Converted("\n\n".join(b for b in blocks if b), "python-docx", {k: v for k, v in metadata.items() if v})


def convert_xlsx(path: Path, use_vision: bool) -> Converted:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), data_only=True, read_only=True)
    blocks = []
    for sheet in workbook.worksheets:
        rows = [["" if v is None else str(v) for v in row] for row in sheet.iter_rows(values_only=True)]
        table = md_table(rows)
        if table:
            blocks.append(f"## {sheet.title}\n\n{table}")
    sheetnames = workbook.sheetnames
    workbook.close()
    return Converted("\n\n".join(blocks), "openpyxl", {"sheets": sheetnames}, len(sheetnames))


def convert_pptx(path: Path, use_vision: bool) -> Converted:
    from pptx import Presentation

    presentation = Presentation(str(path))
    blocks = []
    for number, slide in enumerate(presentation.slides, start=1):
        lines = [f"## Slide {number}"]
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                lines.append(shape.text_frame.text.strip())
            if getattr(shape, "has_table", False) and shape.has_table:
                lines.append(md_table([[c.text for c in r.cells] for r in shape.table.rows]))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            lines.append(f"> Notes: {slide.notes_slide.notes_text_frame.text.strip()}")
        blocks.append("\n\n".join(lines))
    return Converted("\n\n".join(blocks), "python-pptx", pages=len(presentation.slides))


def convert_odf(path: Path, use_vision: bool) -> Converted:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("content.xml"))
    ns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
    blocks = []
    for element in root.iter():
        if element.tag in (ns + "p", ns + "h"):
            text = "".join(element.itertext()).strip()
            if text:
                blocks.append(f"## {text}" if element.tag == ns + "h" else text)
    return Converted("\n\n".join(blocks), "odf-xml")


def convert_rtf(path: Path, use_vision: bool) -> Converted:
    from striprtf.striprtf import rtf_to_text

    return Converted(rtf_to_text(read_text(path)).strip(), "striprtf")


def convert_html(path: Path, use_vision: bool) -> Converted:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(read_text(path), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for table in soup.find_all("table"):
        rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in table.find_all("tr")]
        table.replace_with(soup.new_string("\n\n" + md_table(rows) + "\n\n"))
    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            heading.replace_with(soup.new_string(f"\n\n{'#' * level} {heading.get_text(' ', strip=True)}\n\n"))
    for item in soup.find_all("li"):
        item.replace_with(soup.new_string(f"\n- {item.get_text(' ', strip=True)}"))
    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    title = soup.title.get_text(strip=True) if soup.title else None
    return Converted(cleaned.strip(), "beautifulsoup", {"title": title} if title else {})


def convert_csv(path: Path, use_vision: bool) -> Converted:
    text = read_text(path)
    dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
    return Converted(md_table(list(csv.reader(io.StringIO(text), dialect))), "csv")


def convert_eml(path: Path, use_vision: bool) -> Converted:
    message = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    headers = {h: str(message[h]) for h in ("From", "To", "Cc", "Date", "Subject") if message[h]}
    body = message.get_body(preferencelist=("plain", "html"))
    text = body.get_content() if body else ""
    if body is not None and body.get_content_type() == "text/html":
        from bs4 import BeautifulSoup

        text = BeautifulSoup(text, "html.parser").get_text("\n")
    attachments = [p.get_filename() for p in message.iter_attachments() if p.get_filename()]
    metadata = {k.lower(): v for k, v in headers.items()}
    if attachments:
        metadata["attachments"] = attachments
    return Converted(text.strip(), "email", metadata)


def convert_text(path: Path, use_vision: bool) -> Converted:
    return Converted(read_text(path).strip(), "text")


def convert_image(path: Path, use_vision: bool) -> Converted:
    image = ocr.load_image(path.read_bytes())
    text, method, warning = ocr.extract_text(image, use_vision)
    converted = Converted(text, method, {"width": image.width, "height": image.height}, 1)
    if warning:
        converted.warnings.append(warning)
    if not text:
        converted.warnings.append("no text extracted from image")
    return converted


def convert_legacy(path: Path, kind: str, use_vision: bool) -> Converted:
    """Convert .doc/.xls/.ppt through LibreOffice to the OOXML equivalent, then re-run."""
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if not office:
        default = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
        office = str(default) if default.exists() else None
    if not office:
        raise ConversionError(
            f"legacy .{kind} needs LibreOffice (soffice) on PATH; install it or save the file as .{LEGACY_EXTS[kind]}"
        )
    target = LEGACY_EXTS[kind]
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [office, "--headless", "--convert-to", target, "--outdir", tmp, str(path)],
            capture_output=True, text=True, timeout=180,
        )
        output = Path(tmp) / f"{path.stem}.{target}"
        if result.returncode != 0 or not output.exists():
            raise ConversionError(f"LibreOffice failed: {result.stderr.strip() or result.stdout.strip()}")
        converted = CONVERTERS[target](output, use_vision)
    converted.method = f"libreoffice+{converted.method}"
    return converted


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeError:
            continue
    return data.decode("latin-1", errors="replace")


CONVERTERS = {
    "pdf": convert_pdf, "docx": convert_docx, "xlsx": convert_xlsx, "pptx": convert_pptx,
    "odf": convert_odf, "rtf": convert_rtf, "html": convert_html, "csv": convert_csv,
    "eml": convert_eml, "text": convert_text, "image": convert_image,
}
