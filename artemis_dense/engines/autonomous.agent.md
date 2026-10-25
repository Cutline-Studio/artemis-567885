# agent: autonomous (orchestrator)
- **Does:** loops decide → act → log, with a policy that does a queued code task first, stops
  after 2 consecutive rollbacks, records a skill after 2 clean cycles, reuses a good skill, and
  otherwise trains a cycle.
- **Safety bounds (required):**
  - It can only call whitelisted local engine actions.
  - `max_steps` and `max_seconds` limit each run.
  - It has no network, money or GPU actions. `costs_money` actions are refused unless
    `approve_spend=True` (default False) and a human-supplied handler exists, and none ships.
  - It deletes nothing, and every engine dir must sit inside its run_dir.
  - The kill switch stops it.
- **Writes only:** `run_dir/decisions.jsonl`, plus whatever the engines write under run_dir.
- See [../GOVERNANCE.md](../GOVERNANCE.md).
