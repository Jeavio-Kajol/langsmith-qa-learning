"""Image text extraction. Fallback chain: RapidOCR -> Tesseract -> Groq vision model."""
import base64
import io
import os
import shutil
from pathlib import Path
from typing import Optional

MIN_OCR_CHARS = 20
_rapid_engine = None
_heif_registered = False


class OCRUnavailable(RuntimeError):
    pass


def load_image(data: bytes):
    """Open image bytes with Pillow (HEIC/HEIF via pillow-heif) as upright RGB."""
    global _heif_registered
    from PIL import Image, ImageOps

    if not _heif_registered:
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except ImportError:
            pass
        _heif_registered = True
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def ocr_rapid(image) -> str:
    global _rapid_engine
    import numpy as np

    if _rapid_engine is None:
        try:
            from rapidocr import RapidOCR
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise OCRUnavailable("RapidOCR is not installed") from exc
        _rapid_engine = RapidOCR()
    result = _rapid_engine(np.array(image))
    if hasattr(result, "txts"):  # rapidocr >= 2
        return "\n".join(result.txts or [])
    lines = result[0] if isinstance(result, tuple) else result
    return "\n".join(item[1] for item in (lines or []))


def ocr_tesseract(image) -> str:
    try:
        import pytesseract

        if not shutil.which("tesseract"):
            candidates = [
                Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
            ]
            found = next((c for c in candidates if c.exists()), None)
            if found:
                pytesseract.pytesseract.tesseract_cmd = str(found)
        return pytesseract.image_to_string(image)
    except ImportError as exc:
        raise OCRUnavailable("pytesseract is not installed") from exc
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailable("tesseract binary not found") from exc


def ocr_vision(image) -> str:
    if not os.getenv("GROQ_API_KEY"):
        raise OCRUnavailable("GROQ_API_KEY is not set")
    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq

    model = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
    encoded = base64.b64encode(_png_bytes(image)).decode("ascii")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Transcribe all text in this image exactly as written, preserving reading order "
                "and structure. Use Markdown for headings, lists and tables. Output only the transcription.",
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ]
    )
    return str(ChatGroq(model=model, temperature=0).invoke([message]).content).strip()


def extract_text(image, use_vision_fallback: bool = True) -> tuple[str, str, Optional[str]]:
    """Return (text, method, warning). Local engines first; the vision model is used
    only when they are unavailable or yield too little text."""
    notes: list[str] = []
    best = ""
    for method, engine in (("rapidocr", ocr_rapid), ("tesseract", ocr_tesseract)):
        try:
            text = engine(image).strip()
        except OCRUnavailable as exc:
            notes.append(f"{method}: {exc}")
            continue
        except Exception as exc:
            notes.append(f"{method} failed: {type(exc).__name__}: {exc}")
            continue
        if len(text) >= MIN_OCR_CHARS:
            return text, method, None
        best = best or text
        notes.append(f"{method}: little or no text found")

    if use_vision_fallback:
        try:
            text = ocr_vision(image)
            if text:
                return text, "vision", "; ".join(notes) or None
        except OCRUnavailable as exc:
            notes.append(f"vision: {exc}")
        except Exception as exc:  # a provider error must not abort the whole batch
            notes.append(f"vision failed: {type(exc).__name__}: {exc}")
    return best, "none", "; ".join(notes) or None
