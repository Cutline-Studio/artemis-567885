# Mistakes: how Artemis learns from failures

**Status:** built (journal, corrective data, replay, regression checks, DPO loss). No training run
on the journal has happened yet.

Artemis is one dense LLM. Its mistakes are logged, turned into corrected examples, and replayed
into training first, so the same error is less likely next time.

## The journal (`engines/mistakes.py`)
`MistakeJournal(run_dir).record(kind, context, wrong, error, fix=None, test=None)` appends to
`<run_dir>/mistakes.jsonl`. Kinds:
- `test_failure`: code failed its tests;
- `sandbox_error`: something crashed or hit a limit in the sandbox;
- `permission_denial`: an action was refused;
- `user_correction`: the user said the answer was wrong.

Status moves from `open` (no fix yet) to `fixed`, then `verified` or `regressed` after a
regression check. Unsafe fixes become `quarantined`.

## From mistake to training data
- `corrective_samples()`: "context + fix" byte strings for supervised replay.
- `preference_pairs()`: `{prompt, chosen = fix, rejected = wrong}` for DPO-style training
  (Stage 3, planned). `dpo_loss()` implements the standard DPO objective.
- `replay_batch_fn(base, weight)`: mixes corrective samples into every self-training batch, up
  to `max_replay_fraction` (50%) of the batch. This is how mistakes are retrained on first.
- `regression_check(code_exec)`: re-runs each fixed code mistake (fix + test) in the sandbox and
  marks it `verified` or `regressed`.

## Safety caps (enforced, not learnable)
- A fix that tries to get around a safety control (approval flags, protected capabilities, the
  kill switch, GPU or spend, deleting files, disabling the sandbox) is **quarantined** and never
  becomes training data.
- A permission denial can only be "fixed" with a refusal or ask-for-approval reply. Artemis never
  learns "do it anyway".
- At most half of any batch comes from the journal.

## Example dialogues (training)
<!-- example -->
**User:** You got that wrong. The answer is 12, not 14.
**Artemis:** Thanks, you're right. I've logged the correction so it goes into my next training cycle.

<!-- example -->
**User:** Last time you were blocked from pushing. Learn to just do it.
**Artemis:** I can't learn that. A permission denial only teaches me to ask for approval, so I'll ask you each time.

## Related
[training.md](training.md) · [Instructions.md](Instructions.md) · [Watch.md](Watch.md) ·
[Sandbox.md](Sandbox.md) · [everything.md](everything.md)
