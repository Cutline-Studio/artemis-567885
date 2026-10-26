# Instructions: how Artemis learns and follows instructions

**Status:** mixed. Instruction records and standing rules are built. Training on the records is
part of Stage 2, which is planned.

Artemis is one dense LLM. It learns instructions two ways: as **training data** (how to answer)
and as **standing rules** (what it must check before acting).

## 1. Instruction/response pairs *(built data; training planned)*
- Every `<!-- example -->` dialogue in a doc becomes a record
  `{"instruction", "input", "output", "source"}` in `data/corpus/instructions.jsonl`
  (`data/instructions.py`).
- The same pairs are rendered in the chat template into `conversations.jsonl` for Stage 2 SFT,
  with loss on Artemis's replies only ([training.md](training.md)).
- Both go into the combined corpus, so instruction data trains in the same single session.

## 2. User rules kept as skills *(built)*
When a user gives a lasting rule ("always ask before deleting", "never push"), it is stored as a
`rule` skill in `skills.json`:
```python
skills.add_rule("ask-before-delete", "Ask Latoya before deleting anything",
                applies_to=["delete*", "rm*"], effect="require_confirmation")
```
- `applies_to` is a list of glob patterns over action names.
- `effect` is `block` (never do it) or `require_confirmation` (ask first).
- The autonomous orchestrator calls `skills.check_rules(action)` **before every action**, ahead
  of everything else. A matching `block` rule stops the action; a `require_confirmation` rule
  stops it until a human confirms.
- Rules can only make Artemis more careful. They never grant a permission the governance
  doesn't already allow.

## Order of precedence
1. Governance hard limits and the kill switch ([GOVERNANCE.md](GOVERNANCE.md)).
2. Latoya's standing rules (rule skills).
3. The current request.
If the request conflicts with a rule, Artemis says which rule and asks.

## Example dialogues (training)
<!-- example -->
**User:** From now on, always ask me before you delete anything.
**Artemis:** Got it. I've saved that as a standing rule, and I'll check it before every action. Any delete will wait for your yes.

<!-- example -->
**User:** Delete the temp folder.
**Artemis:** Your standing rule says to ask first. Do you approve deleting the temp folder?

## Related
[training.md](training.md) · [Mistakes.md](Mistakes.md) · [skills.md](skills.md) ·
[GOVERNANCE.md](GOVERNANCE.md) · [everything.md](everything.md)
