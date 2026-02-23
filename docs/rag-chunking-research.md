# RAG Chunking Research Notes

## Overview

Summary of research findings from a discussion about chunking strategies, text splitters, and graph-based RAG approaches, with an assessment of their applicability to this project's AWS SageMaker documentation pipeline.

---

## Chunk Size Research

### Power-of-2 Chunk Sizes: Debunked

A common claim in the RAG community is that chunk sizes should be powers of 2 (256, 512, 1024, 2048) because "chunking algorithms use recursion and this saves computation." **This is not backed by any research.**

- LangChain's `RecursiveCharacterTextSplitter` iterates over separators sequentially and checks `len(chunk) <= chunk_size`. There is no binary recursion that benefits from power-of-2 alignment.
- No academic paper, LangChain documentation, or RAG benchmark recommends power-of-2 chunk sizes for text splitting.
- The confusion likely stems from low-level computing, where powers of 2 genuinely matter for memory alignment, GPU tensor dimensions, and buffer allocation — concepts that do not apply to text chunking.

**Takeaway**: Choose chunk sizes based on content characteristics and empirical testing, not arbitrary powers of 2.

### Optimal Chunk Size Findings

Key findings from published research and evaluations:

| Source | Finding |
|--------|---------|
| [LlamaIndex evaluation](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5) | Chunk sizes of 512-1024 performed best across question-answering tasks. 1024 struck the best balance of faithfulness and relevancy. |
| [Rethinking Chunk Size for Long-Document Retrieval (2025)](https://arxiv.org/html/2505.21700v2) | Optimal chunk size is highly task-dependent. Smaller chunks (64-128 tokens) suit fact-based lookups; larger chunks (512-1024) suit contextual understanding. |
| [NVIDIA Technical Blog](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/) | No single chunk size works for all document types. Recommends empirical testing with your specific corpus and queries. |
| [Context Window Utilization (2024)](https://arxiv.org/html/2407.19794v2) | Chunk size of 512 provides good results while minimizing latency. Going to 1024 yields marginal improvement with significantly higher context usage. |

### Chunk Overlap

Research suggests 10-20% overlap is generally effective:
- Too little overlap risks losing context at chunk boundaries.
- Too much overlap creates redundancy and inflates index size without proportional quality gains.
- Overlap should scale with chunk size: e.g., 50 chars for 500-char chunks, 100 chars for 1000-char chunks.

---

## Alternative Splitter Strategies

### For Markdown Documents

Since all sagemaker-docs are markdown files, markdown-aware splitters may produce better chunks than the generic `RecursiveCharacterTextSplitter`.

| Splitter | How It Works | Best For |
|----------|-------------|----------|
| `RecursiveCharacterTextSplitter` | Tries separators in order: `\n\n`, `\n`, ` `, `""` | General-purpose, any text format |
| `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)` | Uses markdown-specific separators: headers, code fences, horizontal rules, lists | Markdown docs where structure should guide split decisions |
| `MarkdownHeaderTextSplitter` | Splits on `#`/`##`/`###` headers, preserves hierarchy as metadata | Docs with strong header structure (like AWS reference docs) |
| `TokenTextSplitter` | Splits by token count (tiktoken) instead of character count | When aligning with LLM token limits matters |

The `MarkdownHeaderTextSplitter` is particularly interesting for this project because the sagemaker-docs have consistent header structures (CloudFormation property names as `#` headers, sub-properties as `##`/`###`). Preserving the header hierarchy as a prefix on each chunk could improve retrieval quality by providing additional context.

---

## Graph-RAG Assessment

### What Is Graph-RAG?

Graph-RAG treats documents as structured data — extracting entities and relationships into a knowledge graph. Queries traverse the graph rather than performing pure vector similarity search. Key implementations:

- **Microsoft GraphRAG** (open-source): LLM extracts entities/relationships → Hierarchical Leiden community detection → community summarization → dual-mode querying (global search via summaries, local search via entity neighbors).
- **Neo4j + LangChain/LlamaIndex**: Popular integration path using Neo4j as the graph database with LangChain for entity extraction and Cypher query generation.

### Suitability for This Project

Graph-RAG is **not suitable** for the current AWS SageMaker documentation corpus:

- The 200+ markdown files are **flat reference documents** — CloudFormation property definitions, how-to guides, and service documentation.
- Only one file (`sagemaker-roles.md`) has rich entity-relationship structure (API operations → IAM actions → AWS services) that would benefit from graph representation.
- The documents are relatively independent — each describes a single resource type or topic with minimal cross-referencing.
- Standard hybrid search (BM25 + neural) with good chunking is well-suited for this corpus type.

**When Graph-RAG would be appropriate**: Legal documents with citations and cross-references, medical records with patient-condition-treatment relationships, or large codebases with dependency graphs — documents where entities have rich, interconnected relationships.

### Infrastructure Requirements

Graph-RAG requires significantly more infrastructure than standard vector RAG:

| Component | Purpose |
|-----------|---------|
| Graph database (Neo4j, Amazon Neptune, FalkorDB) | Stores entities and relationships |
| LLM-driven entity extraction | Processes every chunk at index time (expensive) |
| Community detection algorithm | Clusters entities into groups |
| LLM-driven community summarization | Generates summaries at index time (additional LLM cost) |
| Graph query layer | Translates natural language to graph queries (e.g., Cypher) |

The overhead of adding and maintaining a graph database alongside OpenSearch, plus the cost of LLM-driven entity extraction for 200+ documents, makes this approach disproportionate for the current use case.

### LangGraph vs Graph-RAG

These are **different concepts** despite sharing the word "graph":

| Concept | What It Is |
|---------|------------|
| **Graph-RAG** | A retrieval strategy that builds and queries a knowledge graph from documents |
| **LangGraph** | A framework for building stateful, multi-step LLM agent workflows as directed graphs (nodes = processing steps, edges = transitions) |

LangGraph is an orchestration tool — similar to this project's use of Google ADK for agent workflows. It does **not** perform graph-based retrieval. You could use LangGraph to orchestrate an agent that *calls* a Graph-RAG system, but LangGraph itself provides zero Graph-RAG functionality.

---

## Experiments

Based on this research, two new experiments have been added to `experiments/`:

| Notebook | What It Tests |
|----------|--------------|
| `chunking_strategy_comparison.ipynb` | Chunk sizes (256 vs 512 vs 1024) with proportional overlap, plus overlap sensitivity |
| `text_splitter_comparison.ipynb` | RecursiveCharacterTextSplitter vs markdown-aware splitters |

See the [Experiments wiki page](../wiki/Experiments.md) for detailed documentation of each experiment's methodology, metrics, and key questions.
