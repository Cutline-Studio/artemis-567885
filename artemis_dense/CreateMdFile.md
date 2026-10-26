# CreateMdFile: how to add a new Artemis doc

**Status:** built. `scripts/new_md.py` creates, registers and links a new doc.

Artemis is one dense LLM. Its docs are part of what it trains on, so every new doc follows the same
shape and is registered in one place.

## Command
```
python3 scripts/new_md.py <Name> [--title T] [--status planned|built|mixed|reference] [--purpose P] [--rebuild]
```
What it does:
1. Checks the name: letters, digits, `_` and `-` only, starting with a letter, up to 64 characters.
2. Refuses to overwrite an existing file.
3. Writes `artemis_dense/<Name>.md` from the template below.
4. Registers it in `artemis_dense/docs_index.json` (`file`, `title`, `status`, `review`,
   `in_corpus`) via `scripts/docs_registry.py`.
5. Rebuilds `everything.md` (all registered docs in one file).
6. With `--rebuild`, also rebuilds the training corpus, then calls `scripts/build_glossary.py`.
   That glossary script is not in this branch yet, so the `--rebuild` step currently ends in an
   error after the corpus is rebuilt. Don't use `--rebuild` while a training run reads the corpus.

To rebuild only everything.md after editing docs by hand: `python3 scripts/build_everything.py`.

## The template
Every doc has: a `# Title`; a **Status** line (built, planned, mixed or reference) with each
section marked *(built)* or *(planned)*; **Purpose**; **What it is**; **How it works in
Artemis**; **Safety and governance** (link to GOVERNANCE.md, no secrets); **Example dialogues**;
**Related** links.

## Example dialogues feed training
Write each training dialogue like this, with a blank line after it:
```
<!-- example -->
**User:** a question
**Artemis:** a short, honest answer in Artemis's voice
```
`data/instructions.py` turns these into instruction records and `data/conversations.py` into
chat-template SFT data. Template placeholders in parentheses are skipped.

## Rules for doc content
- Real, checkable facts only; mark planned parts as planned; label estimates.
- No secrets, keys, tokens, passwords or card numbers.
- Avoid 64-character hex strings (for example sha256 checksums): the constitution treats them as
  secrets and trips strict mode on self-authored docs.
- Write in Artemis's voice ([personality.md](personality.md)).
- Docs Artemis writes itself go through [SelfAuthor.md](SelfAuthor.md) and the promotion gate
  instead of this script.

## Example dialogues (training)
<!-- example -->
**User:** How do I add a new doc for you?
**Artemis:** Run python3 scripts/new_md.py with the doc's name. It creates the file from the template, registers it and rebuilds everything.md. Then fill in each section and mark what's built or planned.

## Related
[everything.md](everything.md) · [training.md](training.md) · [SelfAuthor.md](SelfAuthor.md) ·
[Dictionary.md](Dictionary.md) · [GOVERNANCE.md](GOVERNANCE.md)
