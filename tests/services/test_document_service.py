import json
from pathlib import Path

from app.services.document_service import DocumentService


def test_chunk_text_fixture_case_1():
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures" / "document_service"
    input_path = fixtures_dir / "chunk_text_input_1.txt"
    expected_path = fixtures_dir / "chunk_text_expected_1.json"

    text = input_path.read_text(encoding="utf-8")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    service = DocumentService()
    chunks = service.chunk_text(text=text, chunk_size=expected["chunk_size"], chunk_overlap=expected["chunk_overlap"])

    assert {
        "count": len(chunks),
        "chunk_size": expected["chunk_size"],
        "chunk_overlap": expected["chunk_overlap"],
        "chunks": chunks,
    } == expected
