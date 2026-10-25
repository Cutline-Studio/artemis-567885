# Sandbox: where untrusted Artemis output is tested

**Status:** built. The `sandbox.env` loader is wired into code_exec, self_author, self_train and
the watch practice loop, and tested.

Anything Artemis generates or modifies (code, docs, skills, checkpoints) is **untrusted output
under test** until it passes the [promotion gate](CustomConstitutionalAI.md). `sandbox.env` holds
the isolated environment's settings. `artemis_dense/sandbox.py` loads it and enforces it.

## What sandbox.env sets
- Markers: `ARTEMIS_ENV=sandbox`, `ARTEMIS_UNTRUSTED=1`, and a label that reads "UNTRUSTED OUTPUT
  UNDER TEST - NOT PRODUCTION".
- Paths: everything is under `/tmp/artemis-sandbox` (work, checkpoints, data, logs, tmp, home).
  `/workspace`, `/home/box` and `/cursor/stores` are denied.
- No live services: network off, the A2A engine in loopback, and push, payments, email, deploy
  and GPU rental all off. Proxies point at a dead port.
- Runtime: the tiny model on CPU, seed 1234, deterministic.
- Limits: 300 s, 2 threads, 2048 MB of memory, 1024 MB of disk per file, 32 processes, 256 open
  files and 1 MB of output.
- Audit: `logs/audit.jsonl`. The whole root is wiped on exit.

## How it is enforced (`sandbox.py`)
| setting | enforcement |
|---|---|
| paths | `require_inside()`: run dirs and writes must be under the sandbox root, or `SandboxViolation` is raised. `check_path()` refuses denied production paths. The root itself must be under `/tmp`. |
| denied paths in children | code_exec never bind-mounts a denied path into bubblewrap, even when `ro_paths` asks for one |
| network | under `ARTEMIS_UNTRUSTED=1` / network off, code_exec **requires** bubblewrap (`--unshare-all`: no network namespace). The best-effort non-bwrap fallback is refused. |
| limits | `SandboxEnv.limits()` clamps `SandboxLimits` (timeout, CPU, address space, file size, open files, output) and adds `RLIMIT_NPROC` (counted per uid, so the cap is current threads + 32) |
| child env | only the sandbox markers plus PATH/HOME/TMPDIR/LANG. Nothing is inherited from the parent (no keys or tokens). |
| push / payments / email / deploy / GPU rental | `deny("push")` etc. raise unless the env allows them |
| audit | every load, session, code_exec init/run, denial and promotion check goes to `ARTEMIS_AUDIT_LOG` |
| wipe | `with sb.session():` creates the dirs, then removes the root on exit (`ARTEMIS_WIPE_ON_EXIT=1`) |

Libraries in user site-packages (SymPy, mpmath) live under the denied `/home/box`. For sandboxed
math, `stage_libs()` copies them into `<root>/libs`, which is bound read-only. The math engine's
code is shipped into the task dir as a file, so the repo stays invisible too.

## Who loads it
- **code_exec**: `CodeExecEngine(run_dir, sandbox=sb)`.
- **self_author**: stages every write in `<root>/work/self_author/` before the promotion gate.
- **self_train** (the self-improvement engine; there is no separate `self_improvement` module):
  runs train in a sandbox run dir. `promote()` only succeeds from there.
- **watch** practice: runs commands through a router whose code_exec has `sandbox=sb`.

```python
from artemis_dense.sandbox import SandboxEnv
sb = SandboxEnv.load()                       # or SandboxEnv.load(ARTEMIS_SANDBOX_ROOT="/tmp/x")
with sb.session():
    ex = CodeExecEngine(sb.work_dir("exec"), sandbox=sb)
    ex.run("print('hi')")
```

## Tested (`tests/test_sandbox_promotion.py`)
A sandboxed run sees no `/workspace`, `/home/box` or `/cursor`. A raw `_socket` connect to
1.1.1.1:80 fails. The child sees `ARTEMIS_UNTRUSTED=1` and `ARTEMIS_NETWORK=off`. The other tests
cover the limits and denials, run dirs outside the root being refused, and wipe-on-exit.

## Limits of the sandbox
No cgroup CPU or memory controller and no seccomp filter. bwrap plus rlimits is the boundary. The
disk limit is per file (`RLIMIT_FSIZE`), not a quota.
