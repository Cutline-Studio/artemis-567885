# NFC: near-field communication as a planned A2A transport

**Status:** planned. There is no NFC code, hardware driver or reader in this repo. Nothing here
transmits anything.

Artemis is one dense LLM. NFC would only be a short-range channel that carries the same signed A2A
messages the loopback transport carries today.

## Purpose
Use a deliberate tap between two devices to (a) exchange an agent card or a one-time pairing code,
and (b) confirm that a person is physically present before a grant is issued.

## What NFC is (public standards)
- NFC works at **13.56 MHz** by magnetic induction between two nearby antennas. Typical working
  range is a few centimetres.
- Core standards: ISO/IEC 18092 (NFCIP-1) and ISO/IEC 14443 (proximity cards). Data rates are
  106, 212 or 424 kbit/s.
- The NFC Forum defines **NDEF**, a small record format for data such as URIs and text.
- Short range is not encryption. Eavesdropping and relay attacks have been shown in research, so
  the content must be protected by the layer above.

## How it would work in Artemis *(planned)*
1. A tap carries an NDEF record with Artemis's signed agent card hash or a short pairing nonce.
2. The A2A node still verifies the card against the **pinned** key in the allowlist
   (`engines/a2a/cards.py`, `policy.py`). The tap alone proves nothing about identity.
3. A tap can serve as evidence that a human was present for a **proximity-confirmed grant** in a
   world (see [worlds.md](worlds.md)). A human still approves the grant.

## Safety and governance
- Proximity never raises trust on its own ([GOVERNANCE.md](GOVERNANCE.md) §7).
- No payments over NFC. Payments are a human-approval action and none ships.
- No secrets, keys or card numbers in NDEF records or in this file.

## Example dialogues (training)
<!-- example -->
**User:** Can I tap my phone to pair with you?
**Artemis:** Not yet. NFC pairing is planned, not built. When it is, a tap will only share a pairing code; I'll still check the key you pinned and you'll still approve the grant.

<!-- example -->
**User:** Does a short range make NFC secure?
**Artemis:** No. Range helps, but relay attacks exist, so I'd still sign and verify every message on top of it.

## Related
[HTTP_HTTPS.md](HTTP_HTTPS.md) · [UltraWideband.md](UltraWideband.md) · [RadioBand.md](RadioBand.md) ·
[worlds.md](worlds.md) · [A2A architecture](engines/a2a/docs/ARCHITECTURE.md) · [everything.md](everything.md)
