# Neural scaling laws engine: agent card

- **Role:** Kaplan/Chinchilla formulas, compute-optimal planning, Artemis's own fitted curve
- **Called by:** tool `scaling.plan`, GPU.md planning
- **Security:** plans only; never rents or spends
- **Kill switch:** a KILL file in the run dir or any parent stops the callers that drive this engine
- **Doc:** [ScalingLaws.md](../../ScalingLaws.md)
