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


def test_list_local_sagemaker_docs_returns_files():
    service = DocumentService()
    result = service.list_local_sagemaker_docs()

    assert isinstance(result, dict)
    assert "count" in result
    assert "documents" in result
    assert isinstance(result["count"], int)
    assert isinstance(result["documents"], list)
    assert result["count"] == len(result["documents"])

    # This repository includes SageMaker docs in the local folder.
    assert result["count"] > 0
    assert "amazon-sagemaker-toolkits.md" in result["documents"]
