# RAG engine (retrieval-augmented generation)

**Status:** built, local only. Retrieval is BM25-led: dense embeddings come from the untrained
model and carry little meaning yet.

`engines/rag/` is its own engine.

## Index
- Sources: every `.md` under `artemis_dense/` except `everything.md`, plus self-authored docs
  that the registry marks approved and in the corpus. Also the generated problem sets
  (`data/problems/*.jsonl`).
- Chunks: markdown sections (split at headings, then 900-character windows). Each chunk carries
  `{path, section, sha256 of the file, text}`. The citation format is `path#section`.
- The dictionary (292,125 records) is **not** chunked. "define X" and "what does X mean" go
  straight to `skills.define(X)`, an exact headword lookup that cites `dictionary:<source>:<word>`.
- Store: `chunks.jsonl`, `files.json` (sha256 per file) and `emb.npy` in the index dir.
- Incremental: `refresh()` re-chunks only new or changed files and drops deleted ones. Hook it to
  self_author with `author.on_write.append(rag.refresh_file)`.

## Retrieval
BM25 (k1 1.5, b 0.75), plus dense cosine on `transformer.embed` (weight 0.1, because the model is
untrained and no offline sentence-embedding model is installed on this box), plus rerank bonuses
for heading hits and exact phrases. `allowed_prefixes` limits which paths a caller can see (for
example an external agent's world). `answer()` is extractive: the best matching sentences, with
citations. It says **"I don't know"** when the normalised BM25 confidence is under 0.25.

## Connections
Tool `rag.answer`. Conversation grounding (see [Conversation.md](Conversation.md)).
