# agent: self_train
- **Does:** runs training cycles on the dense model, evaluates each one, checkpoints, rolls back
  on a NaN/inf loss or an eval regression past `regression_tol`, and grows when loss stalls
  (`add_layer` up to `num_layers_max`, then `widen_ffn` by `d_ff_step` up to `d_ff_max`).
- **Inputs:** `run_dir`, `DenseConfig` (its caps), `TrainConfig`, `batch_fn`.
- **Writes only:** `run_dir/{state.json, ckpt.pt, ckpt_prev.pt, self_train_log.jsonl, exec_signal.jsonl}`.
- **Never:** network, GPU rental, or raising its own caps (caps come from config, edited by a human).
- **Hook:** `add_execution_signal(record)` stores code_exec results. It is a stub with no training.
- **Stops on:** kill switch (`KILL` in run_dir or any parent dir).
- **Not reachable** from A2A. See [../GOVERNANCE.md](../GOVERNANCE.md).
