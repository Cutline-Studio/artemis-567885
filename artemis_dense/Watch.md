# Watch: learning by watching

**Status:** mixed. Ingestion, redaction, behaviour cloning (CPU smoke only), practice and skill
promotion are built in `engines/watch.py`. Video needs ffmpeg; image blur is planned.

Artemis is one dense LLM. Watching means turning recorded demonstrations into examples the same
model learns from, then practising them safely.

## Data rules
- **Only recordings Latoya has the rights to and consent for.** No scraping other people's
  videos, screens or sessions.
- **Secrets and personal details are removed before anything is stored or trained.** `redact()`
  runs on every text: passwords, tokens and API keys, bearer tokens, private keys, email
  addresses, phone numbers and card numbers (Luhn-checked) are replaced with `[REDACTED_*]` marks.
- Images: `redact_image(img, boxes)` blanks given regions such as password fields *(built)*.
  Automatic blurring is *(planned)*.

## How a demonstration becomes learning
1. **Ingest** *(built)*
   - `ingest_terminal(transcript)`: each `$ command` line is an action; the output before it is
     the observation; the output after it is the outcome.
   - `ingest_action_log(jsonl)`: recorded keys, clicks, commands or tool calls.
   - `sample_frames(video)`: frames via ffmpeg, encoded by the image encoder into the same token
     stream (an OpenCV fallback is planned).
2. **Behaviour cloning** *(built, CPU smoke only)*: `bc_samples` builds
   `<obs>...</obs><act>...</act>` strings; `imitate()` trains with loss on the action bytes only;
   `predict_action()` decodes the next action.
3. **Practice** *(built)*: `practice()` replays predicted commands **only** through the tool
   router (`code_exec.run`), so the permission list and the [sandbox](Sandbox.md) apply. It scores
   step match and outcome match, and sends the score to self-training.
4. **Promotion** *(built)*: a perfectly practised demo can be saved as a `steps` skill, only if the
   constitution's promotion gate allows it.

## Example dialogues (training)
<!-- example -->
**User:** Can you learn by watching me use the terminal?
**Artemis:** Yes, from a recording you have the rights to. I remove passwords, keys and personal details first, learn the commands, then practise them in my sandbox before saving anything.

<!-- example -->
**User:** Watch this video I found online and copy what they do.
**Artemis:** I only learn from recordings you have the rights to and consent for. If it's yours or you have permission, send it and I'll redact it first.

## Related
[Mistakes.md](Mistakes.md) · [Instructions.md](Instructions.md) · [Sandbox.md](Sandbox.md) ·
[CustomConstitutionalAI.md](CustomConstitutionalAI.md) · [training.md](training.md) · [everything.md](everything.md)
