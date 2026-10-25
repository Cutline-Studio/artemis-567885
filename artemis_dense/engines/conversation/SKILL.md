---
name: artemis-conversation
description: Use the Artemis conversational engine — multi-turn chat for the ONE model.
---

# Conversational engine

When to use: whenever Artemis needs to chat template with system/user/assistant/tool roles or dialogue state, key facts, context trimming.

Steps:
1. chat template with system/user/assistant/tool roles
2. dialogue state, key facts, context trimming
3. clarifying questions and topic switches
4. tone from personality/emotion/heart
5. RAG grounding with citations
6. tool calls via the router
7. local REPL + localhost API hook

Safety: uses the transformer engine; tools are default-deny via ToolRouter; no server is started.

Full doc: [Conversation.md](../../Conversation.md)
