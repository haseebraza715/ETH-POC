"""Document parsing helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _read_pdf(path: Path) -> Optional[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("pypdf not available (%s); falling back to plain text read", exc)
        return None

    with path.open("rb") as fh:
        reader = PdfReader(fh)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text_parts.append(text.strip())
        return "\n".join(tp for tp in text_parts if tp)


def parse_document(path_str: str) -> str:
    """Return textual content of the provided document."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Document {path} not found")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        pdf_text = _read_pdf(path)
        if pdf_text:
            return pdf_text
        logger.info("Falling back to binary read for %s", path)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_bytes().decode("latin-1", errors="ignore")
    raise ValueError(f"Unsupported document type {suffix}")

