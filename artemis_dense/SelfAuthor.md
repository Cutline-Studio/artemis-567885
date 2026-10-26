# SelfAuthor: Artemis writing its own files

**Status:** built. `engines/self_author.py` is the **unguarded** version Latoya chose on
2026-10-25. Its only content check is the custom constitution at the promotion gate.

Artemis is one dense LLM. Self-authoring is that model writing docs and code into its own repo.

## What it does
- `write_file(rel_path, content)`: writes **any** file under the repo root (code, engines, tests,
  core docs), then makes one local git commit.
- `create_doc(name, content, title=None, subdir="docs/generated")`: writes
  `artemis_dense/<subdir>/<name>.md`, registers it in `docs_index.json` as `review="approved"`
  and `in_corpus=true` (so it goes straight into the training corpus), rebuilds everything.md
  and commits.
- Commits are local only. It never pushes and never opens a pull request.

## What was removed, at Latoya's choice
The earlier guarded version had allowlisted folders, protected files, a secret scan, content
checks, size and rate caps, test-gated rollback, a pending-review queue, and corpus entry only
after approval. All of those were removed (commit d6acca6). The guarded version is kept at git tag
**`self-author-guarded`** and can be restored with:
```
git checkout self-author-guarded -- artemis_dense/engines/self_author.py tests/test_self_author.py
```

## The one check left: the promotion gate
Every write is first staged in the sandbox (`/tmp/artemis-sandbox/work/self_author/`) and checked
by `PromotionGate` with Latoya's custom constitution ([CustomConstitutionalAI.md](CustomConstitutionalAI.md)).
- **Loose by default.** Principle matches are logged as warnings and the write goes through.
- **Conditional strict mode.** Strict turns on for the rest of the session after more than 3
  warnings, any critical hit (a secret-shaped string, a production path, network, payments) or
  any write touching the constitution, its engine, `custom.env` or `sandbox.env`. In strict mode
  the write raises `PromotionBlocked` and nothing reaches the repo, and every later promotion in
  that session is blocked until Latoya calls `approve_resume`.
- Also kept: paths must resolve inside the repo, and the kill switch stops it.

## Risks, stated plainly
- A doc containing something like `api_key = ...` or a full rewrite of GOVERNANCE.md is
  **promoted with only a warning** unless strict mode is already on.
- A bad edit to code or tests can be committed. Nothing runs the tests or rolls back.
- Self-written docs go straight into the training corpus with no human review, so a mistake can
  be learned by the next training run.
- False alarms: any 64-character hex string (such as a sha256 checksum) counts as a secret and
  trips strict mode, which then blocks the whole session.
- Mitigations that still hold: the sandbox staging, the constitution's strict mode, local-only
  commits that Latoya can review or revert, and the kill switch.

## Example dialogues (training)
<!-- example -->
**User:** Write yourself a new doc about your skills.
**Artemis:** I'll draft it in the sandbox first. The constitution checks it; if it passes, it's added to the repo with a local commit and goes into my training data. Nothing is pushed.

<!-- example -->
**User:** Is self-authoring safe now?
**Artemis:** Less safe than before. The old guards are gone, so the constitution is the only check, and it's loose by default. It blocks only in strict mode, for example on secrets or edits to its own rules.

## Related
[CustomConstitutionalAI.md](CustomConstitutionalAI.md) · [Sandbox.md](Sandbox.md) ·
[CreateMdFile.md](CreateMdFile.md) · [GOVERNANCE.md](GOVERNANCE.md) · [everything.md](everything.md)
