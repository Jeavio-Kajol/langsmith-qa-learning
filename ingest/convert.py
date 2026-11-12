"""Convert any supported document into a single format: Markdown + JSON sidecar.

    python -m ingest.convert <file-or-folder> [--out converted] [--no-vision]
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from ingest.converters import CONVERTERS, ConversionError, convert_legacy
from ingest.detect import LEGACY_EXTS, detect_format
from ingest.telemetry import get_tracer, setup_tracing

MAX_FILE_BYTES = 50 * 1024 * 1024
SCHEMA_VERSION = 1


@dataclass
class ConvertedDocument:
    source: str
    detected_format: str
    method: str
    markdown: str
    sha256: str
    pages: int = 0
    char_count: int = 0
    metadata: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION


def convert_file(path, use_vision: bool = True) -> ConvertedDocument:
    """Convert one file. Raises ConversionError for unsupported/unreadable input."""
    path = Path(path)
    if not path.is_file():
        raise ConversionError(f"not a file: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ConversionError("file is empty")
    if size > MAX_FILE_BYTES:
        raise ConversionError(f"file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB limit")

    with get_tracer().start_as_current_span("ingest.convert_file") as span:
        span.set_attribute("ingest.file_name", path.name)
        span.set_attribute("ingest.file_bytes", size)
        try:
            kind = detect_format(path)
            span.set_attribute("ingest.detected_format", kind)
            if kind == "unknown":
                raise ConversionError(f"unsupported or unrecognised file format: {path.name}")
            if kind in LEGACY_EXTS:
                result = convert_legacy(path, kind, use_vision)
            else:
                try:
                    result = CONVERTERS[kind](path, use_vision)
                except ConversionError:
                    raise
                except Exception as exc:
                    raise ConversionError(f"{kind} conversion failed: {type(exc).__name__}: {exc}") from exc
        except ConversionError as exc:
            span.record_exception(exc)
            span.set_attribute("ingest.success", False)
            raise

        markdown = result.markdown.strip()
        document = ConvertedDocument(
            source=path.name,
            detected_format=kind,
            method=result.method,
            markdown=markdown,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pages=result.pages,
            char_count=len(markdown),
            metadata=result.metadata,
            warnings=result.warnings,
        )
        if not markdown:
            document.warnings.append("conversion produced no text")
        span.set_attribute("ingest.method", document.method)
        span.set_attribute("ingest.char_count", document.char_count)
        span.set_attribute("ingest.warning_count", len(document.warnings))
        span.set_attribute("ingest.success", bool(markdown))
        return document


def write_document(document: ConvertedDocument, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(document.source).stem
    md_path = out_dir / f"{stem}.md"
    if md_path.exists() and json.loads((out_dir / f"{stem}.json").read_text("utf-8")).get("sha256") != document.sha256:
        stem = f"{stem}-{document.sha256[:8]}"
        md_path = out_dir / f"{stem}.md"
    md_path.write_text(document.markdown + "\n", encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(asdict(document), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return md_path


def main(argv=None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Convert documents/images to Markdown + JSON.")
    parser.add_argument("input", help="file or folder")
    parser.add_argument("--out", default="converted", help="output folder (default: converted)")
    parser.add_argument("--no-vision", action="store_true", help="disable Groq vision fallback for images")
    args = parser.parse_args(argv)

    setup_tracing()
    source = Path(args.input)
    files = sorted(p for p in source.rglob("*") if p.is_file()) if source.is_dir() else [source]
    failures = 0
    for file in files:
        try:
            document = convert_file(file, use_vision=not args.no_vision)
            target = write_document(document, Path(args.out))
            print(f"OK   {file.name} [{document.detected_format}/{document.method}] -> {target}")
            for warning in document.warnings:
                print(f"     warning: {warning}")
        except ConversionError as exc:
            failures += 1
            print(f"FAIL {file.name}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
