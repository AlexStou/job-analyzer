import io

import pytest
from docx import Document

from job_analyzer.file_loader import UnsupportedFileError, extract_text


def test_txt():
    assert extract_text("cv.txt", "Python, MATLAB".encode("utf-8")) == "Python, MATLAB"


def test_docx_reads_paragraphs_and_tables():
    doc = Document()
    doc.add_paragraph("Profile paragraph")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Skills in a table"
    table.cell(0, 1).text = "Ελληνικά"
    buffer = io.BytesIO()
    doc.save(buffer)

    text = extract_text("cv.docx", buffer.getvalue())
    assert "Profile paragraph" in text
    assert "Skills in a table" in text
    assert "Ελληνικά" in text


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileError):
        extract_text("cv.png", b"")
