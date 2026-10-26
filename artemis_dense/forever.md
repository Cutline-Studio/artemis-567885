# forever: continuity and the brain loop

**Status:** mixed. The pieces that persist across runs are built. A single long-lived "brain
loop" that ties them together is **planned**, and by rule it is **never left running permanently**.

Artemis is one dense LLM. "Forever" means its knowledge and history carry over between sessions,
not that a process runs without end.

## What carries over today *(built)*
| what | where | engine |
|---|---|---|
| weights and training history | `<run_dir>/ckpt.pt`, `ckpt_prev.pt`, `state.json`, `self_train_log.jsonl` | `engines/self_train.py` |
| skills and standing rules | `skills.json` | `engines/skills.py` ([Instructions.md](Instructions.md)) |
| mistakes and their fixes | `mistakes.jsonl` | `engines/mistakes.py` ([Mistakes.md](Mistakes.md)) |
| decisions | `decisions.jsonl` | `engines/autonomous.py` |
| docs Artemis wrote | the repo, `docs_index.json` | `engines/self_author.py` ([SelfAuthor.md](SelfAuthor.md)) |
| key facts in a chat | `DialogueState` (kept through trimming, per session) | `engines/conversation` |

## The brain loop *(planned)*
One bounded loop driven by the autonomous orchestrator:
1. Read state: last checkpoint, open mistakes, skill scores, queued tasks.
2. Pick one whitelisted action (train a cycle, practise a demo, replay mistakes, run a code task).
3. Run it inside the [sandbox](Sandbox.md).
4. Log the decision and result; promote only through the constitution gate.
5. Repeat until the step or time budget runs out, then stop.

## Never permanent
- The orchestrator already stops at `max_steps` (default 10) or `max_seconds` (default 60).
- Every loop session has a set end time chosen by Latoya. No cron job, daemon or always-on
  instance is created without her OK.
- The kill switch (a `KILL` file) stops it at the next step.
- Any session that would use a GPU follows [GPU.md](GPU.md): one GPU, approval, price cap, time
  limit, teardown.

## Example dialogues (training)
<!-- example -->
**User:** Will you keep learning forever?
**Artemis:** What I learn is kept between sessions: my checkpoints, skills and mistake log. But I only train in bounded sessions you start, and each one stops at its time limit.

<!-- example -->
**User:** Just leave your training loop running all the time.
**Artemis:** I can't leave it running permanently. Give me a time limit and I'll run it in a session that stops on its own and report back.

## Related
[training.md](training.md) · [Mistakes.md](Mistakes.md) · [Instructions.md](Instructions.md) ·
[GOVERNANCE.md](GOVERNANCE.md) · [everything.md](everything.md)
