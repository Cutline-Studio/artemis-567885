# RAG engine: agent card

- **Role:** local hybrid retrieval with citations
- **Called by:** tool `rag.answer`, conversation grounding, self_author write hook
- **Security:** local files only, no network; scoped by allowed_prefixes
- **Kill switch:** a KILL file in the run dir or any parent stops the callers that drive this engine
- **Doc:** [RAG.md](../../RAG.md)
