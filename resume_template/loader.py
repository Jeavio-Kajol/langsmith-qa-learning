"""Read plain text out of a resume file (.txt, .md, .doc, .docx, .pdf)."""
import shutil
import subprocess
import tempfile
from pathlib import Path

SUPPORTED = (".txt", ".md", ".doc", ".docx", ".pdf")


def _docx_text(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        parts += [" | ".join(c.text for c in row.cells) for row in table.rows]
    return "\n".join(parts)


def _doc_text(path: Path) -> str:
    """Legacy .doc has no pure-Python reader; convert to .docx with LibreOffice."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("Reading .doc needs LibreOffice ('soffice') on PATH; it was not found.")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([soffice, "--headless", "--convert-to", "docx", "--outdir", tmp, str(path)],
                       check=True, capture_output=True, timeout=120)
        return _docx_text(Path(tmp) / (path.stem + ".docx"))


def load_text(path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8")
    elif suffix == ".docx":
        text = _docx_text(path)
    elif suffix == ".doc":
        text = _doc_text(path)
    elif suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    else:
        raise ValueError(f"Unsupported resume format: {suffix}")
    if not text.strip():
        raise ValueError(f"No text could be extracted from {path.name} (scanned/image-only file?)")
    return text
