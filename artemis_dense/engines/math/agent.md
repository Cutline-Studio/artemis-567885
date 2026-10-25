# Mathematics engine: agent card

- **Role:** exact, symbolic, numeric and arbitrary-precision math with verification
- **Called by:** tool head `math.compute` (always sandboxed), autonomous `solve_problem`, physics engine, problem-set builder
- **Security:** SymPy parses strings, so tool calls run inside the code_exec sandbox
- **Kill switch:** a KILL file in the run dir or any parent stops the callers that drive this engine
- **Doc:** [Mathematics.md](../../Mathematics.md)
