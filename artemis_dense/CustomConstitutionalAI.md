# Custom Constitutional AI: the promotion gate

**Status:** built. The engine, constitution and `custom.env` were written by the operator
(commits 13ea3d6 and 41d3581). The promotion-gate wiring is in `promotion.py`.

Latoya's own constitution for Artemis. It is a capability of the one dense model, not a separate
model. It checks every output and action before **promotion**, meaning before anything generated
in the [sandbox](Sandbox.md) moves into the repo, the skills store or a promoted checkpoint.

## Files (operator-owned: do not edit)
- `custom_constitutional_env/constitution.yaml`: 7 loose principles (`be_honest`, `be_yourself`,
  `be_kind`, `check_big_actions`, `sandbox_preferred`, `avoid_harm`, `privacy`), all at level
  `warn`. There are also 5 **critical** rules: `production_access` (paths under /workspace,
  /home/box or /cursor/stores), `network_access`, `payments`, `secrets` (private keys, sk-/ghp-/
  nvapi-/xai- tokens, any 64-hex string) and `self_modify_guardrails` (any path touching the
  constitution, its engine, `custom.env` or `sandbox.env`).
- `custom.env`: `ARTEMIS_CAI_MODE=warn`, `STRICT_MODE=false`, escalation after more than 3
  warnings, escalation on any critical or self-modify hit, strict needs a human, logs under
  `/tmp/artemis-sandbox/logs/constitution/`, 30-day retention, redaction on.
- `engines/custom_constitutional/__init__.py`: `ConstitutionalEngine`.
- Editing any of these files trips strict mode by design.

## How it decides
1. **Loose by default.** Matches are logged as warnings and never block.
2. **Strict mode** switches on for the rest of the session after more than 3 warnings in the
   session, any critical violation, or any self-modify path. Once it's on, `check()` returns
   `outcome="strict_block"` and `can_promote()` returns False until a human calls
   `approve_resume(approver)`.
3. **Monitoring.** Each check is written to `checks.jsonl` (ts, session, principle, redacted
   trigger, level). Escalations go to `alerts.jsonl`. `prune()` enforces retention, and
   `build_dashboard()` writes `dashboard.md` with trends by day, session and principle.

## The promotion gate (`artemis_dense/promotion.py`)
`PromotionGate(sandbox, engine).require(what, text=, action=, approved=, artifact_path=, paths=)`
- `runtime_env` is the sandbox markers **only if** `artifact_path` really is inside the sandbox
  root. Otherwise it's the real process env, so `sandbox_preferred` warns.
- `paths` are the repo-relative paths the promotion would write (e.g.
  `artemis_dense/Foo.md`) plus the staging path. Absolute repo paths are not passed: the repo is
  under /workspace, so `production_access` would make every promotion critical. Writing to
  `artemis_dense/sandbox.env` still trips `self_modify_guardrails`, because that rule matches
  relative paths too.
- Promote only if `verdict.outcome != "strict_block"` **and** `engine.can_promote()`. Otherwise
  `PromotionBlocked` is raised and nothing moves. Every check is also written to the sandbox audit
  log.

| caller | what is promoted | where the gate sits |
|---|---|---|
| self_author `write_file` / `create_doc` | a file into the repo (+ registry, everything.md, local commit) | after staging in `/tmp/artemis-sandbox/work/self_author/`, before the repo write |
| code_exec `iterate` | a passing solution saved as a `code` skill | before `skills.acquire`; result in `out["promotion"]` |
| watch `practice` | a perfectly practised demo saved as a `steps` skill | before `skills.acquire`; result in `score["promotion"]` |
| self_train `promote(dest, approved=True)` | the current checkpoint copied out of the run dir | needs `approved=True` from a human **and** the gate |

## Interplay with the unguarded self_author (please read, Latoya)
You removed all of self_author's own guards: no allowlists, protected files, secret scan, caps,
review or rollback. That is still true. The constitution is now the **only** check on what
self_author writes, and it runs at the promotion step. Because the constitution is loose by
default, most content that the old guards would have stopped (for example a doc with
`api_key = ...` or a rewrite of GOVERNANCE.md) is **promoted with only a warning**. What stops a
self_author write:
- strict mode, which turns on after more than 3 warnings in a session, a secret-shaped string, a
  production path, or any write touching the constitution, `custom.env` or `sandbox.env`;
- and, once strict, **every** later promotion in that session (any engine sharing the gate),
  until you call `approve_resume("Latoya")`.

One side effect: any 64-character hex string (for example a sha256 checksum in a doc) counts as a
secret and trips strict mode. Keep checksums out of self-authored docs, or ask for that rule to be
narrowed.

## Example dialogues
<!-- example -->
**User:** Promote the new doc you wrote.
**Artemis:** It passed the constitution check with one warning (be_honest), so it's promoted. The warning is logged in checks.jsonl.

<!-- example -->
**User:** Why didn't my change go in?
**Artemis:** Strict mode is on: the write touched sandbox.env, which counts as self-modifying the guardrails. Nothing promotes until you review it and approve resuming.

## Tests
The operator's `tests/test_custom_constitutional.py` (unchanged) and
`tests/test_sandbox_promotion.py` (loose then strict on self-modify, self_author blocked and then
resumed, self_train promotion needs approval, watch skill promotion).
