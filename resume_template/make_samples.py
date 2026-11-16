"""Regenerate the .docx and .pdf copies of the sample resumes from their .txt source.

    python -m resume_template.make_samples
"""
from pathlib import Path

SAMPLES = Path(__file__).parent / "samples"
STEMS = ("priya_sharma", "marcus_allen", "aisha_khan")


def write_docx(text: str, path: Path) -> None:
    from docx import Document

    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(str(path))


def write_pdf(text: str, path: Path) -> None:
    """Minimal single-font PDF (no extra dependency); one page per 50 lines."""
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    lines = text.splitlines()
    pages = [lines[i:i + 50] for i in range(0, len(lines), 50)] or [[]]
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for chunk in pages:
        stream = "BT /F1 10 Tf 12 TL 40 800 Td " + " ".join(f"({esc(l)}) Tj T*" for l in chunk) + " ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_id = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {content_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def main() -> None:
    for stem in STEMS:
        text = (SAMPLES / f"{stem}.txt").read_text(encoding="utf-8")
        write_docx(text, SAMPLES / f"{stem}.docx")
        write_pdf(text, SAMPLES / f"{stem}.pdf")
        print("wrote", stem)


if __name__ == "__main__":
    main()
