# Transformer engine: agent card

- **Role:** the single entry point to the ONE dense model
- **Called by:** self_train, watch, rag, scaling, conversation, chat REPL
- **Security:** local only; never loads a second model
- **Kill switch:** a KILL file in the run dir or any parent stops the callers that drive this engine
- **Doc:** [Transformer.md](../../Transformer.md)
