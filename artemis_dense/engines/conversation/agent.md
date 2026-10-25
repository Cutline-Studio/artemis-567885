# Conversational engine: agent card

- **Role:** multi-turn chat for the ONE model
- **Called by:** chat REPL (`python -m artemis_dense.chat`), planned localhost API, Stage 2 SFT data
- **Security:** uses the transformer engine; tools are default-deny via ToolRouter; no server is started
- **Kill switch:** a KILL file in the run dir or any parent stops the callers that drive this engine
- **Doc:** [Conversation.md](../../Conversation.md)
