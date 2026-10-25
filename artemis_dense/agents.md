# Artemis agents (top level)

Artemis 567885 is ONE dense model. Each "agent" below is one of its engines, not a separate
model. Approvals follow [GOVERNANCE.md](GOVERNANCE.md). Everything is local: no push, no PR, no
GPU, and A2A in loopback only.

| engine | agent card | driven by | can reach |
|---|---|---|---|
| transformer | [agent.md](engines/transformer/agent.md) | every engine that needs the model | the weights (read/generate) |
| self_train | [self_train.agent.md](engines/self_train.agent.md) | autonomous | the model, checkpoints in its run dir; `promote()` needs a human plus the gate |
| skills | [skills.agent.md](engines/skills.agent.md) | autonomous, code_exec, watch | skills.json |
| code_exec | [code_exec.agent.md](engines/code_exec.agent.md) | autonomous, tools, math | the bwrap sandbox only |
| autonomous | [autonomous.agent.md](engines/autonomous.agent.md) | a human starting a run | whitelisted actions, bounded |
| a2a | [a2a/agent.md](engines/a2a/agent.md) | external agents (allowlisted, signed) | the facade and granted worlds |
| math | [agent.md](engines/math/agent.md) | tools, autonomous, physics | the sandbox (for tool calls) |
| physics | [agent.md](engines/physics/agent.md) | tools, autonomous | math (public interface) |
| heart | [agent.md](engines/heart/agent.md) | conversation, self_train signals | none (pure scoring) |
| rag | [agent.md](engines/rag/agent.md) | conversation, tools, self_author hook | local docs (read) |
| scaling | [agent.md](engines/scaling/agent.md) | tools, planning | none (plans only) |
| conversation | [agent.md](engines/conversation/agent.md) | the REPL, the planned localhost API | transformer, rag, heart, tools |
| custom_constitutional | [agent.md](engines/custom_constitutional/agent.md) | the promotion gate | its own logs |
| self_author (unguarded) | see [CustomConstitutionalAI.md](CustomConstitutionalAI.md) | autonomous `author_doc`, tools | the repo, **only after the promotion gate** |
| watch | (module `engines/watch.py`) | a human giving demos | the router/sandbox; skill promotion via the gate |

Everything generated is tested in [Sandbox.md](Sandbox.md) first and promoted only through
[CustomConstitutionalAI.md](CustomConstitutionalAI.md).
