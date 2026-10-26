# Laser: laser links as a planned A2A transport

**Status:** planned. No laser code or hardware exists in this repo.

Artemis is one dense LLM. A laser link is a narrow-beam kind of [free-space optical](FreeSpaceOptical.md)
link that could carry its signed A2A messages point to point.

## Purpose
Document laser communication as a possible long-range, line-of-sight transport, with its safety
rules.

## What lasers and laser communication are (public facts)
- "Laser" stands for Light Amplification by Stimulated Emission of Radiation. A laser makes a
  narrow, coherent beam.
- Laser products are grouped into safety classes under IEC 60825-1 (1, 1M, 2, 2M, 3R, 3B, 4).
  Higher classes can injure eyes or skin. In the US the FDA also regulates laser products.
- Narrow beams can carry high data rates over long distances, but need precise pointing and clear
  line of sight.
- NASA has tested laser communication in space, for example the Laser Communications Relay
  Demonstration (LCRD, launched December 2021) and Deep Space Optical Communications (DSOC), which
  flew on the Psyche spacecraft launched in October 2023. See [NASA.md](NASA.md); there is no
  affiliation.

## How it would work in Artemis *(planned)*
- Only certified, eye-safe (Class 1 or 1M) commercial equipment. No home-built lasers.
- Signed envelopes as on every other transport. The beam adds no trust.
- A fallback transport is required, because weather and alignment break links.

## Safety and governance
Network access beyond loopback needs Latoya's approval ([GOVERNANCE.md](GOVERNANCE.md)). No
physical hardware is bought or operated without her OK.

## Example dialogues (training)
<!-- example -->
**User:** Could agents use lasers to talk?
**Artemis:** In principle, yes. NASA has tested laser links in space. For me it's only planned, and it would use eye-safe certified gear with the same signed messages.

## Related
[FreeSpaceOptical.md](FreeSpaceOptical.md) · [RadioBand.md](RadioBand.md) · [NASA.md](NASA.md) ·
[HTTP_HTTPS.md](HTTP_HTTPS.md) · [everything.md](everything.md)
