"""Read a CV from an uploaded file (PDF, DOCX or TXT) into plain text."""

import io
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedFileError(ValueError):
    pass


def _docx_text(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    # Many CV templates put content inside tables (e.g. two-column layouts).
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.extend(p.text for p in cell.paragraphs)
    return "\n".join(p for p in parts if p.strip())


def _pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text(filename: str, data: bytes) -> str:
    """Return the text content of a CV file."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        text = _pdf_text(data)
    elif ext == ".docx":
        text = _docx_text(data)
    elif ext in {".txt", ".md"}:
        text = data.decode("utf-8", errors="replace")
    else:
        raise UnsupportedFileError(f"Unsupported file type '{ext}'. Use PDF, DOCX or TXT.")
    return text.strip()
