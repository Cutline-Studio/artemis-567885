# Governance — Artemis 567885 dense rebuild

Applies to everything under `artemis_dense/`. Items marked **(planned)** are not implemented yet.
Everything else describes what the code does now.

## 1. Authority
The human owner, **Latoya (Cutline Industries LLC)**, is the final approver. No engine, agent,
orchestrator, A2A peer or assistant can approve on her behalf. An agent saying something is
approved is not approval.

## 2. Approval matrix
| action | engines / agents may do freely | needs explicit human approval | how the code enforces it |
|---|---|---|---|
| Train a cycle on CPU inside a run dir | yes | — | `SelfTrainer.run_cycle` |
| Grow within existing caps | yes | — | `plan_growth` reads caps from config |
| **Raise caps** (`num_layers_max`, `d_ff_max`) or change policy | no | yes, as a human edit of config | caps are never written by engines |
| **Spend money** of any kind | no | yes | `costs_money` actions refused unless `approve_spend=True` and a human handler exists; none ships |
| **GPU rental** | no | yes | no GPU action exists; `gpu.*` is COSTLY in the A2A policy |
| **External network access** | no | yes | code_exec has no network (bwrap netns; socket guard as fallback); the A2A prototype opens no sockets |
| **Push code / open PRs** | no | yes | no engine runs git; assistants follow §8 |
| **Merge** | no | yes, always by the human | — |
| **Modify model weights** outside the trainer's own checkpoints | no | yes | `model_weights` worlds are protected; `model.weights*` is PROTECTED over A2A |
| **Grant write/execute in the training world** (`self_train`) | no | yes | `grant()` requires `human_approved=True` and `allow_protected=True` |
| Grant read/write/execute in non-protected worlds | operator only | operator decides | `EnvironmentManager.grant` is a local call, not exposed over A2A |
| **Delete files** | only an engine's own temp capture dirs under its run dir | yes, for anything else | the orchestrator deletes nothing; code_exec only removes its own `out-*` dir |
| Look up skills, echo, text stats over A2A | allowlisted orgs | — | facade + scoped token |

## 3. Hard limits (no agent can override)
- An external A2A party can never assert human approval: the node always passes `human_approved=False`.
- Protected capabilities (`self_train.*`, `skills.write`, `skills.acquire`, `model.weights*`,
  `code_exec.*`, `orchestrator.*`, `autonomous.*`) are refused unless the org has an explicit
  protected grant and a local human approves.
- Sandboxed code gets an empty environment. No host env vars, tokens or keys are passed.
- The orchestrator has fixed step and time budgets and a fixed action whitelist.
- The kill switch overrides everything (§6).

## 4. Changing caps and policies
Caps live in `DenseConfig.caps` (saved in the run's `state.json` and checkpoint). Policy lives in
`engines/a2a/policy.py` (`FACADE`, `PROTECTED`, `COSTLY`, `TRUST_SCOPES`). Both change only
through a human edit committed to git, which is the version history. No engine writes them.
**(planned)** a signed, versioned `policy.json` with a change log entry per edit.

## 5. Audit log
- **A2A + worlds:** `<run_dir>/a2a_audit.jsonl` (node) and the environment manager's log file.
  These are append-only and hash-chained (`AuditLog.verify_chain()`), so they are tamper-evident
  but not tamper-proof. They record every policy decision (principal, org, capability, reason,
  task id), handshake, task outcome, world definition, grant (scopes, jti, expiry, paths),
  revocation, and every enter/read/write/execute/leave with allow or deny and the reason.
- **Orchestrator:** `<run_dir>/decisions.jsonl`, one record per step with the action and reason.
- **Trainer:** `self_train_log.jsonl`. **code_exec:** `code_exec_log.jsonl`.
- **Not recorded:** file contents, tokens' signatures, keys.
- **Retention:** logs are kept indefinitely until a human archives them. **(planned)**
  rotation and an explicit retention period set by the owner.

## 6. Incidents and the kill switch
- **Kill switch (implemented):** create a file named `KILL` in the run dir or any parent of it
  (for example `touch <run_root>/KILL`, or `artemis_dense.killswitch.engage(dir, reason)`).
  - The trainer raises `Killed` before and during every step, so nothing is checkpointed.
  - The orchestrator logs and stops before the next step.
  - The A2A node rejects every request with `UNAVAILABLE`.
  - The environment manager denies every operation.
  - Running sandboxed children end at their timeout.
- **Resuming:** a human deletes the `KILL` file.
- **Incident steps:**
  1. Engage the kill switch.
  2. Revoke the affected grants (`revoke(jti)`) and remove the org from the allowlist.
  3. Run `verify_chain()` on the audit logs.
  4. Tell the owner.
  5. Fix it on a local branch and review it in chat (§8).
- **(planned)** a single CLI and stop button.

## 7. Trust levels for A2A agents
| level | who | can reach |
|---|---|---|
| untrusted | not on the allowlist | nothing (`PERMISSION_DENIED`) |
| basic | allowlisted, pinned Ed25519 key | `artemis.echo`, `artemis.text_stats` |
| partner | allowlisted, pinned key | basic + read-only skill lookup + `env.*` (each world still needs its own grant) |
| protected grant | partner, plus a named capability in `protected_grants` | only with local human approval per call; the wire can't supply it, so in this prototype it is effectively unreachable |

Proximity channels ([NFC.md](NFC.md), [RadioBand.md](RadioBand.md),
[UltraWideband.md](UltraWideband.md), [FreeSpaceOptical.md](FreeSpaceOptical.md)) never raise
trust on their own. **(planned)**

## 8. Data and secrets
- No secrets in logs, skills, worlds, cards, tokens or test fixtures.
- Sandboxed code runs with a minimal env, and bwrap hides `~/.ssh` and home.
- Tokens carry only identities, scopes, expiry and ids.
- Agent private keys are generated in memory in the prototype. **(planned)** OS keystore or HSM storage.

## 9. Change review
1. All work happens on a local branch (`dense-model`).
2. The owner reviews it in chat.
3. A push or PR happens only after the owner explicitly says so.
4. Merging is always the owner's call.

## Links
[skills.md](skills.md) · [worlds.md](worlds.md) · [README.md](README.md) ·
agents: [self_train](engines/self_train.agent.md), [skills](engines/skills.agent.md),
[code_exec](engines/code_exec.agent.md), [autonomous](engines/autonomous.agent.md),
[a2a](engines/a2a/agent.md) · [HTTP_HTTPS.md](HTTP_HTTPS.md)
