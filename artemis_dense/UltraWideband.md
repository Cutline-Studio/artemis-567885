# UltraWideband: UWB ranging as a planned A2A transport

**Status:** planned. No UWB code or hardware exists in this repo.

Artemis is one dense LLM. UWB would give its A2A engine a measured distance to a peer device, used
as one extra signal when a human approves a nearby grant.

## Purpose
Use secure UWB ranging to check that a peer device is really within a set distance (for example,
in the same room) before a proximity-confirmed grant is offered for human approval.

## What UWB is (public facts)
- In 2002 the FCC allowed unlicensed UWB devices in **3.1–10.6 GHz** at very low power (an EIRP
  limit of −41.3 dBm/MHz). A UWB signal uses at least 500 MHz of bandwidth, or a fractional
  bandwidth of at least 20%.
- Very short pulses let devices measure time of flight, which gives distance estimates at
  roughly the decimetre-to-centimetre scale.
- **IEEE 802.15.4z** (2020) added enhanced, more secure ranging, including a scrambled timestamp
  sequence (STS) that makes distance spoofing harder. The FiRa Consortium works on interoperability.
- Some phones and item trackers ship UWB chips for this kind of ranging.

## How it would work in Artemis *(planned)*
1. Two devices complete the normal signed A2A handshake (pinned keys).
2. They run a secure ranging session. Artemis records the measured distance in the audit log.
3. If the distance is under the world's threshold, the grant request is shown to a human. A human
   still approves. Distance is evidence, not authority.
Data exchange itself would stay on another transport. UWB is used for ranging.

## Safety and governance
Proximity never raises trust on its own ([GOVERNANCE.md](GOVERNANCE.md) §7). See
[worlds.md](worlds.md) for proximity-confirmed grants, which are also planned.

## Example dialogues (training)
<!-- example -->
**User:** What would UWB add to your agent links?
**Artemis:** A measured distance to the other device. I'd use it to check someone is really nearby before asking you to approve a grant. It wouldn't approve anything by itself.

## Related
[NFC.md](NFC.md) · [RadioBand.md](RadioBand.md) · [FreeSpaceOptical.md](FreeSpaceOptical.md) ·
[worlds.md](worlds.md) · [HTTP_HTTPS.md](HTTP_HTTPS.md) · [everything.md](everything.md)
