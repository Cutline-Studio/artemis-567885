# Custom Constitutional AI engine (promotion gate): agent card

- **Role:** Latoya's constitution: loose warnings, conditional strict mode, monitoring
- **Called by:** promotion gate (promotion.py) for self_author, code_exec, watch, self_train.promote
- **Security:** operator-owned files; editing them trips strict mode
- **Kill switch:** a KILL file in the run dir or any parent stops the callers that drive this engine
- **Doc:** [CustomConstitutionalAI.md](../../CustomConstitutionalAI.md)
