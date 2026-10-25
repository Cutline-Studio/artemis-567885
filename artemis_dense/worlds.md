# Worlds (environments)

A "world" is what the code calls an **Environment**
(`engines/a2a/environments.py`): a bounded place an agent can enter and act in, only under a
grant. This file is the reference. [skills.md](skills.md) has the summary and
[GOVERNANCE.md](GOVERNANCE.md) has the approval rules.

## Definition
```python
Environment(id, type, root, resources=[], default_deny=True)
```
| type | meaning | protected |
|---|---|---|
| `workspace` | a directory of files to read/write | no |
| `sandbox` | scratch area whose scripts are run via code_exec | no |
| `runtime` | an execution context (e.g. code_exec's sandbox) | no |
| `model_weights` | checkpoints / weights | **yes** |
| `self_train` | the training world (self_train run dir) | **yes** |

Default-deny: with no grant, nothing is allowed, including `enter`.

## Grants
`EnvironmentManager.grant(principal, env_id, scopes, ttl_s=600, paths=None, rate_per_min=60,
human_approved=False, allow_protected=False)` returns a signed token (Ed25519, issuer = this org)
with claims `{iss, sub=principal, aud="<org>:env:<env_id>", kind="env_grant", scopes, iat, exp,
jti, env, paths, rate_per_min}`.
- Scopes are `read`, `write` and `execute`, and they are independent: `write` does not imply
  `execute`.
- `paths` limits access to relative prefixes. The default `[""]` means the whole root.
- `revoke(jti)` takes effect on the very next operation, including inside an open session.
- A protected world refuses the grant unless both flags are set. That is a local operator action;
  nothing on the A2A wire can set them.

## Operations and enforcement
`enter → read | write | execute → leave`. Every operation re-checks the grant, so expiry and
revocation apply mid-session. Denials raise `EnvDenied` and are audited with the reason. Over A2A
they show up as `env.*` facade capabilities and a denied operation returns `TASK_STATE_REJECTED`.

## Tested behaviour (`tests/test_a2a.py`)
- A read-only agent can read, but its write is denied.
- An agent with write but not execute cannot run code.
- `../`, absolute paths and symlink escapes are blocked.
- Path-prefix limits are enforced.
- Expired and revoked grants are denied.
- A grant works only for its own agent and its own world.
- The same agent with different grants behaves differently in two worlds.
- Protected worlds need human approval.
- Execute runs in the sandbox.

## Planned
Persisting grants and revocations across restarts (today they are in memory; the audit log is on
disk). Grant issuance by an approval UI. Proximity-confirmed grants
([NFC.md](NFC.md), [UltraWideband.md](UltraWideband.md)).
