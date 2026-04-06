"""Count total chunks to estimate build time."""
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_dir = Path("sagemaker-docs")
total_chunks = 0
total_docs = 0
for p in sorted(docs_dir.glob("*.md")):
    text = p.read_text(encoding="utf-8")
    if text.strip():
        total_docs += 1
        total_chunks += len(splitter.split_text(text))
print(f"Docs: {total_docs}, Chunks: {total_chunks}")
print(f"Estimated time at 1s/chunk: {total_chunks // 60}m {total_chunks % 60}s")
