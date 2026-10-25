# agent: code_exec
- **Does:** runs a write → run → observe → revise loop (`iterate`, with `max_iters`) on untrusted
  code in a sandbox (`run`).
- **Sandbox:** subprocess + rlimits + empty env + `python -I -S` + timeout + output cap are always
  on. bubblewrap (no network, private filesystem view) is used when available. Without it, the
  socket guard and cwd are best-effort only. Details are in [../README.md](../README.md).
- **Writes only:** `run_dir/exec/task-*` and `run_dir/code_exec_log.jsonl`.
- **Connects:** saves passing code as a `code` skill and sends every result to the trainer's
  `exec_signal.jsonl` (a stub). A2A environment `execute` routes here.
- **Never:** passes host env vars or secrets, allows network, or reads outside its task dir (when bwrap is in use).
