# Anthropic: public reference

**Status:** reference.

**No affiliation.** Artemis 567885 and Cutline Industries LLC are not affiliated with, endorsed by
or sponsored by Anthropic. Artemis is not Claude and is not built on Anthropic's models: it is its
own small dense model trained from scratch in this repo. This page lists public facts because some
Artemis ideas draw on Anthropic's published research.

## Public facts
- Anthropic is an AI safety and research company founded in 2021. It develops the **Claude**
  family of AI models.
- **Constitutional AI**: Bai et al., "Constitutional AI: Harmlessness from AI Feedback"
  (arXiv:2212.08073, December 2022). The paper trains a model to critique and revise its own
  outputs against a written list of principles, and uses AI feedback for preference training.
- **Model Context Protocol (MCP)**: an open protocol introduced by Anthropic in November 2024 for
  connecting AI applications to tools and data. Its specification lives at modelcontextprotocol.io.

## How this relates to Artemis
- Artemis's [custom constitution](CustomConstitutionalAI.md) borrows the idea of a written list of
  principles. It is Latoya's own design and works differently: a rule-based promotion gate,
  loose by default with conditional strict mode, not the paper's training method.
- The A2A architecture notes compare Artemis's protocol with MCP
  ([engines/a2a/docs/ARCHITECTURE.md](engines/a2a/docs/ARCHITECTURE.md)).

## Example dialogues (training)
<!-- example -->
**User:** Are you Claude?
**Artemis:** No. I'm Artemis 567885, a separate model built in this repo. I have no affiliation with Anthropic, though my constitution borrows the general idea of written principles from their published research.

## Related
[CustomConstitutionalAI.md](CustomConstitutionalAI.md) · [NASA.md](NASA.md) · [FBI.md](FBI.md) · [everything.md](everything.md)
