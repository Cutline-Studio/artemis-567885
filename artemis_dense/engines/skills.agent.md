# agent: skills
- **Does:** keeps a JSON registry of named procedures `{name, description, trigger, kind, body, stats}`
  and handles acquire/store, lookup (keyword overlap, then score), reuse, and a Laplace-smoothed score update.
- **Kinds:** `steps` (whitelisted engine actions), `code` (re-run through code_exec), and
  `callable` (registered in-process, not persisted as code).
- **Writes only:** its `skills.json`.
- **External access:** read-only `artemis.skills.lookup`. `skills.write` and `skills.acquire` are protected.
- **No secrets** in skill bodies (see [../GOVERNANCE.md](../GOVERNANCE.md)). Full list: [../skills.md](../skills.md).
