---
name: artemis-rag
description: Use the Artemis rag engine — local hybrid retrieval with citations.
---

# RAG engine

When to use: whenever Artemis needs to index docs + problem sets by section or bm25 + dense hybrid retrieval with rerank.

Steps:
1. index docs + problem sets by section
2. BM25 + dense hybrid retrieval with rerank
3. extractive answers with path#section citations
4. say I don't know below confidence
5. dictionary lookups for define-X
6. incremental refresh on doc changes
7. per-caller path scoping

Safety: local files only, no network; scoped by allowed_prefixes.

Full doc: [RAG.md](../../RAG.md)
