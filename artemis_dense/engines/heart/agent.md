# Heart engine: agent card

- **Role:** applies the ranked values from Heart.md
- **Called by:** conversation (candidate choice), self_train (signals)
- **Security:** heuristic scorer; cannot override the constitution or approvals
- **Kill switch:** a KILL file in the run dir or any parent stops the callers that drive this engine
- **Doc:** [Heart.md](../../Heart.md)
