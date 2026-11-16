"""Content-based file format detection (falls back to the extension)."""
import zipfile
from pathlib import Path

OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"avif"}
IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp", "heic", "heif", "avif"}
LEGACY_EXTS = {"doc": "docx", "xls": "xlsx", "ppt": "pptx"}
TEXT_EXTS = {"txt", "md", "markdown", "log", "text"}


def detect_format(path: Path) -> str:
    """Return one of: pdf, docx, xlsx, pptx, odf, doc, xls, ppt, rtf, html, csv,
    eml, image, text, unknown."""
    ext = path.suffix.lower().lstrip(".")
    with path.open("rb") as handle:
        head = handle.read(4096)

    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8\xff") or head[:6] in (b"GIF87a", b"GIF89a"):
        return "image"
    if head[:2] == b"BM" and ext == "bmp":
        return "image"
    if head[:4] in (b"II*\x00", b"MM\x00*") or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        return "image"
    if head[4:8] == b"ftyp" and head[8:12] in HEIF_BRANDS:
        return "image"
    if head.startswith(OLE_MAGIC):
        return ext if ext in LEGACY_EXTS else "doc"
    if head.startswith(b"PK"):
        return _detect_zip(path)
    if head.lstrip().startswith(b"{\\rtf"):
        return "rtf"

    if ext in IMAGE_EXTS:
        return "image"
    if ext in ("htm", "html", "xhtml"):
        return "html"
    if ext in ("csv", "tsv"):
        return "csv"
    if ext == "eml":
        return "eml"
    if ext in TEXT_EXTS:
        return "text"

    sample = head.lstrip().lower()
    if sample.startswith((b"<!doctype html", b"<html")):
        return "html"
    if sample.startswith((b"from:", b"received:", b"mime-version:")):
        return "eml"
    if b"\x00" not in head:
        return "text"
    return "unknown"


def _detect_zip(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "mimetype" in names:
                mimetype = archive.read("mimetype").decode("ascii", "ignore")
                if mimetype.startswith("application/vnd.oasis.opendocument"):
                    return "odf"
    except zipfile.BadZipFile:
        return "unknown"
    if any(n.startswith("word/") for n in names):
        return "docx"
    if any(n.startswith("xl/") for n in names):
        return "xlsx"
    if any(n.startswith("ppt/") for n in names):
        return "pptx"
    return "unknown"
