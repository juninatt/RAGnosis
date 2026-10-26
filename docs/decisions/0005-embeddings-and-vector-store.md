# 0005: Embeddings and a numpy vector store

**Status:** accepted. Replaces the ChromaDB choice in 0001.

## Context

Chunks need to be embedded and searched by similarity. 0001 planned ChromaDB
for storage. On the development machine, Windows Smart App Control blocks
unsigned native code, and one of ChromaDB's many dependencies (`grpc`) was
blocked. The first corpus is small: about 3,400 chunks of 384 numbers each.

## Decision

- **Embedding model:** `intfloat/multilingual-e5-small` through
  `sentence-transformers`, on the CPU. It is small enough for a laptop, MIT
  licensed, and matches questions to passages across languages, so Swedish
  questions find English text. E5 models expect `query: ` and `passage: `
  prefixes, which the embedder adds.
- **Vector store:** our own, built on numpy. Vectors are kept in one matrix and
  search compares the query with all of them (exact search). The index is three
  files: `vectors.npy`, `chunks.jsonl` and `manifest.json`.
- **One model per index.** The manifest records the model, and indexing or
  searching with a different model is refused until `index --rebuild`, because
  vectors from different models cannot be compared.
- **Incremental indexing:** only articles whose revision changed are embedded
  again, and articles removed from the title list leave the index.
- **CPU-only PyTorch** from PyTorch's own package index, so installs (and CI)
  do not download GPU libraries that cannot be used.

## Consequences

- About 60 fewer packages than with ChromaDB, and no native code beyond numpy
  and PyTorch.
- Exact search stays fast for tens of thousands of chunks. At far larger sizes
  an approximate index (ChromaDB, LanceDB, FAISS) is worth the dependencies;
  only `vector_store.py` would change.
- The whole index is loaded into memory. At this size that is a few MB.
- Retrieval quality is checked by opt-in tests (`tests/test_retrieval_quality.py`)
  that ask known questions against the real index.
