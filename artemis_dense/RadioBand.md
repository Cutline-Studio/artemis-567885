# RadioBand: radio links as a planned A2A transport

**Status:** planned. There is no radio code or hardware in this repo, and Artemis transmits nothing.

Artemis is one dense LLM. A radio link would only be another way to carry its signed A2A messages
between agents when there is no wired network.

## Purpose
Describe which radio options could carry A2A envelopes, and the rules any such link must follow.

## What radio bands are (public facts)
- Radio uses part of the electromagnetic spectrum. In the US the **FCC** decides who may transmit
  where. Many consumer devices use **unlicensed** bands under FCC Part 15 rules, with power and
  equipment limits.
- Common unlicensed (ISM) bands in the US include 902–928 MHz and 2.400–2.4835 GHz. Wi-Fi also uses
  the 5 GHz and 6 GHz bands.
- Typical technologies: Wi-Fi (IEEE 802.11), Bluetooth Low Energy (2.4 GHz), and long-range,
  low-rate links such as LoRa in sub-GHz bands.
- Licensed bands (for example amateur radio) need a licence. US amateur rules also prohibit
  messages encoded to hide their meaning, which conflicts with encrypted traffic.
- Trade-off: lower frequencies reach farther at lower data rates; higher frequencies carry more
  data over shorter distances.

## How it would work in Artemis *(planned)*
- Only certified off-the-shelf radios in unlicensed bands. No custom transmitters.
- The radio carries the same signed envelopes as `LoopbackTransport` (Ed25519 signature, nonce,
  ±60 s skew, scoped tokens). The link adds no trust.
- Long-range low-rate links fit small messages (status, task ids), not model weights or files.

## Safety and governance
- Network access of any kind beyond loopback needs Latoya's approval ([GOVERNANCE.md](GOVERNANCE.md)).
- Proximity or reception never raises trust on its own (§7).
- Inside the [sandbox](Sandbox.md) all networking is off.

## Example dialogues (training)
<!-- example -->
**User:** Can you talk to other agents over radio?
**Artemis:** Not today. Radio is a planned transport. If we add it, it would use certified unlicensed-band radios and the same signed messages I use now.

<!-- example -->
**User:** Could we use ham radio for encrypted agent traffic?
**Artemis:** US amateur rules don't allow messages encoded to hide their meaning, so that's a poor fit. An unlicensed-band link with signed, encrypted messages would be the better plan.

## Related
[NFC.md](NFC.md) · [UltraWideband.md](UltraWideband.md) · [FreeSpaceOptical.md](FreeSpaceOptical.md) ·
[Laser.md](Laser.md) · [HTTP_HTTPS.md](HTTP_HTTPS.md) · [everything.md](everything.md)
