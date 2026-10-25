# Physics engine: agent card

- **Role:** unit-checked physics: constants, kinematics, ODE simulation
- **Called by:** tool head `physics.compute`, autonomous `solve_problem`, problem-set builder
- **Security:** pure computation, no I/O; raises DimensionError on unit mismatch
- **Kill switch:** a KILL file in the run dir or any parent stops the callers that drive this engine
- **Doc:** [Physics.md](../../Physics.md)
