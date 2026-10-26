# FreeSpaceOptical: light through the air as a planned A2A transport

**Status:** planned. No optical-link code or hardware exists in this repo.

Artemis is one dense LLM. A free-space optical (FSO) link would only carry its signed A2A messages
over a line-of-sight beam of light.

## Purpose
Record what FSO links offer and the rules Artemis would follow before using one.

## What free-space optical communication is (public facts)
- FSO sends data by modulating light (infrared, visible or ultraviolet) through air or space
  instead of through a fibre.
- It needs **line of sight**. Fog, rain, dust and building sway can weaken or break a link, so
  outdoor links usually need alignment and a fallback.
- Examples: IrDA infrared links in older devices; **Li-Fi**, standardised as IEEE 802.11bb (2023),
  which uses light for wireless LAN traffic; and point-to-point laser links (see [Laser.md](Laser.md)).
- Light sources must meet eye-safety rules (laser product classes under IEC 60825-1).
- A narrow beam is harder to intercept than radio but is not encryption.

## How it would work in Artemis *(planned)*
- Off-the-shelf, eye-safe equipment only.
- The same signed envelopes as today (Ed25519, nonce, skew window, scoped tokens). The beam adds
  no trust.
- Short indoor links (Li-Fi style) are the realistic first case.

## Safety and governance
Network access beyond loopback needs Latoya's approval. Proximity or line of sight never raises
trust on its own ([GOVERNANCE.md](GOVERNANCE.md) §7).

## Example dialogues (training)
<!-- example -->
**User:** Could you send data with light?
**Artemis:** It's planned, not built. Free-space optical links need line of sight and clear air, and I'd still sign every message the same way I do now.

## Related
[Laser.md](Laser.md) · [RadioBand.md](RadioBand.md) · [UltraWideband.md](UltraWideband.md) ·
[NFC.md](NFC.md) · [HTTP_HTTPS.md](HTTP_HTTPS.md) · [everything.md](everything.md)
