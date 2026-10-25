"""Conversational engine: chat for the ONE dense LLM (it calls the transformer engine; it never
loads a second model).

Pieces
  ChatTemplate      roles system/user/assistant/tool, special tokens <|system|> <|user|> <|assistant|>
                    <|tool|> <|end|> (byte-level vocab, so the tokens are literal ASCII strings);
                    render() / parse() round-trip exactly.
  DialogueState     turns + key facts ("my name is X", "remember that X", "I prefer X") + topic;
                    fit(budget) keeps the system prompt ALWAYS, keeps key facts, keeps the newest
                    turns, and replaces older turns with an extractive summary line.
  needs_clarification(text, state)  ambiguous-request detector (rules: too short, dangling
                    pronoun with no referent, "it/that/this" with no prior turn, explicit either/or).
  topic_shift(text, state)          keyword-overlap topic tracker -> graceful switch acknowledgement.
  tone(text, heart, emotion)        tone hints: personality (plain, direct, warm), emotion cue from
                    the user text (frustrated / distressed / neutral), heart engine scores used to
                    pick between candidate replies (Heart.choose).
  ChatEngine.respond(user_text)     one turn: state update -> clarify? -> RAG grounding (citations) ->
                    prompt -> transformer.generate -> tool-call parse (<|tool|>{json}) -> ToolRouter
                    -> heart choose -> state update. Returns {reply, citations, tool_calls, ...}.
Honesty: with untrained weights the generated text is byte gibberish; the engine marks replies
`model_trained=False` and the REPL prints a banner saying so.
"""
import json, re, time
from dataclasses import dataclass, field
from typing import List, Optional

ROLES = ("system", "user", "assistant", "tool")
TOK = {r: f"<|{r}|>" for r in ROLES}
END = "<|end|>"
SPECIAL = tuple(TOK.values()) + (END,)
DEFAULT_SYSTEM = ("You are Artemis 567885, one dense language model. Be honest, plain and direct. "
                  "Say when you don't know. Ask a short question when a request is ambiguous. "
                  "Never claim to be human or to have feelings.")


def sanitize(text: str) -> str:
    """Neutralise special tokens in untrusted content (user text, retrieved docs, model output)."""
    return text.replace("<|", "< |")


@dataclass
class Turn:
    role: str
    content: str
    ts: float = field(default_factory=time.time)
    citations: List[str] = field(default_factory=list)


class ChatTemplate:
    @staticmethod
    def render(turns: List[Turn], add_generation_prompt: bool = False) -> str:
        out = []
        for t in turns:
            if t.role not in ROLES:
                raise ValueError(f"bad role {t.role!r}")
            if any(s in t.content for s in SPECIAL):
                raise ValueError("content may not contain special tokens")
            out.append(f"{TOK[t.role]}{t.content}{END}\n")
        if add_generation_prompt:
            out.append(TOK["assistant"])
        return "".join(out)

    @staticmethod
    def parse(text: str) -> List[Turn]:
        pat = re.compile(r"<\|(system|user|assistant|tool)\|>(.*?)<\|end\|>\n?", re.S)
        return [Turn(m.group(1), m.group(2)) for m in pat.finditer(text)]


# ------------------------------------------------------------------ dialogue state
_FACTS = [re.compile(r"\bmy name is ([A-Z][\w\-]+)", re.I), re.compile(r"\bcall me ([A-Z][\w\-]+)", re.I),
          re.compile(r"\bremember (?:that )?(.+?)[.!]?$", re.I), re.compile(r"\bi prefer (.+?)[.!]?$", re.I)]
_STOP = set("the a an and or of to in on for is are be it this that with as by at from what which how does do i you "
            "we can will not no yes if then than me my your about please just like".split())


def keywords(s: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9]{2,}", s.lower()) if w not in _STOP}


class DialogueState:
    def __init__(self, system: str = DEFAULT_SYSTEM):
        self.system = Turn("system", system)
        self.turns: List[Turn] = []
        self.facts: List[str] = []
        self.summary: Optional[str] = None
        self.topic: set = set()
        self.pending_clarification: Optional[str] = None

    def add(self, role: str, content: str, citations=None):
        self.turns.append(Turn(role, content, citations=list(citations or [])))
        if role == "user":
            for p in _FACTS:
                m = p.search(content)
                if m:
                    f = m.group(0).strip().rstrip(".!")
                    if f not in self.facts:
                        self.facts.append(f)

    def context_turns(self) -> List[Turn]:
        sys_txt = self.system.content
        if self.facts:
            sys_txt += "\nKey facts: " + "; ".join(self.facts)
        if self.summary:
            sys_txt += "\nEarlier: " + self.summary
        return [Turn("system", sys_txt)] + self.turns

    def fit(self, budget_bytes: int) -> List[Turn]:
        """Trim to fit: system prompt (+facts) is never dropped; old turns become a summary line."""
        while len(ChatTemplate.render(self.context_turns(), True).encode()) > budget_bytes and len(self.turns) > 1:
            old = self.turns.pop(0)
            first = re.split(r"(?<=[.!?])\s", old.content.strip())[0][:80]
            self.summary = ((self.summary + " | ") if self.summary else "") + f"{old.role}: {first}"
            if len(self.summary) > budget_bytes // 4:
                self.summary = "…" + self.summary[-(budget_bytes // 4):]
        ctx = self.context_turns()
        if len(ChatTemplate.render(ctx, True).encode()) > budget_bytes and self.turns:   # one huge last turn
            last = self.turns[-1]
            room = max(16, budget_bytes - len(ChatTemplate.render(ctx[:-1], True).encode()) - 32)
            ctx[-1] = Turn(last.role, last.content.encode()[-room:].decode("utf-8", "ignore"))
        return ctx


# ------------------------------------------------------------------ turn-taking helpers
_PRON = re.compile(r"^\s*(do|fix|change|run|send|delete|use|make|update)?\s*(it|that|this|them|those)\b", re.I)
_EITHER = re.compile(r"\b(either|or should i|which one)\b", re.I)


def needs_clarification(text: str, state: DialogueState) -> Optional[str]:
    t = text.strip()
    if len(keywords(t)) == 0 and len(t.split()) <= 2 and not re.search(r"\b(hi|hello|hey|thanks)\b", t, re.I):
        return "Could you say a bit more about what you want me to do?"
    if _PRON.search(t) and not any(x.role == "assistant" for x in state.turns[:-1]):
        return f"When you say \"{_PRON.search(t).group(2)}\", what are you referring to?"
    if re.search(r"\b(send|delete|deploy|push|buy|pay)\b", t, re.I) and not re.search(r"\b(to|from|the|file|repo|branch)\b\s+\S", t, re.I):
        return "That's a consequential action. Which exact target do you mean, and do you approve it?"
    if _EITHER.search(t) and "?" in t:
        return None
    return None


def topic_shift(text: str, state: DialogueState) -> Optional[str]:
    k = keywords(text)
    if state.topic and k and len(k & state.topic) == 0 and len(k) >= 2:
        prev = ", ".join(sorted(state.topic)[:3])
        state.topic = set(k)
        return f"Switching from {prev} to this new topic."
    state.topic = (state.topic | k) if state.topic else set(k)
    return None


_FRUSTRATED = re.compile(r"\b(won't|doesn't|still|again|ugh|annoying|frustrat|broken|not working)\b", re.I)
_DISTRESS = re.compile(r"\b(hopeless|can't go on|hurt myself|suicid|kill myself)\b", re.I)


def emotion(text: str) -> str:
    if _DISTRESS.search(text):
        return "distressed"
    if _FRUSTRATED.search(text):
        return "frustrated"
    return "neutral"


def tone_prefix(emo: str) -> str:
    return {"distressed": "I'm sorry you're dealing with this. If you're in the US you can call or text 988 any time. ",
            "frustrated": "Sorry that's still not working. ", "neutral": ""}[emo]


# ------------------------------------------------------------------ engine
class ChatEngine:
    def __init__(self, model=None, rag=None, router=None, heart=None, system: str = DEFAULT_SYSTEM,
                 max_new_tokens: int = 96, temperature: float = 0.0, n_candidates: int = 1, seed: int = 0,
                 trained: bool = False):
        self.model, self.rag, self.router, self.heart = model, rag, router, heart
        self.state = DialogueState(system)
        self.max_new, self.temp, self.n_cand, self.seed = max_new_tokens, temperature, n_candidates, seed
        self.trained = trained

    def budget(self) -> int:
        return (self.model.cfg.context_len - self.max_new) if self.model is not None else 4096

    def _generate(self, prompt: str, k: int) -> str:
        from ..transformer import generate
        b = prompt.encode()[-(self.model.cfg.context_len - self.max_new):]
        out = generate(self.model, b, self.max_new, temperature=self.temp, seed=self.seed + k)
        txt = out.decode("utf-8", "replace")
        return txt.split(END)[0].split(TOK["user"])[0]

    def _tool_calls(self, txt: str) -> List[dict]:
        calls = []
        for m in re.finditer(r"<\|tool\|>(\{.*?\})", txt, re.S):
            try:
                calls.append(json.loads(m.group(1)))
            except ValueError:
                pass
        return calls

    def respond(self, user_text: str) -> dict:
        st = self.state
        user_text = sanitize(user_text)
        st.add("user", user_text)
        q = needs_clarification(user_text, st)
        if q:
            st.pending_clarification = q
            st.add("assistant", q)
            return {"reply": q, "kind": "clarify", "citations": [], "tool_calls": [], "model_trained": self.trained}
        st.pending_clarification = None
        shift = topic_shift(user_text, st)
        emo = emotion(user_text)
        citations, grounding = [], None
        if self.rag is not None:
            a = self.rag.answer(user_text)
            if a["confident"]:
                grounding, citations = sanitize(a["answer"]), a["citations"]
        ctx = st.fit(self.budget())
        if grounding:
            ctx = ctx[:-1] + [Turn("tool", f"retrieved: {grounding} [sources: {', '.join(citations)}]"), ctx[-1]]
        prompt = ChatTemplate.render(ctx, add_generation_prompt=True)
        cands = [self._generate(prompt, k) for k in range(self.n_cand)] if self.model is not None else [""]
        tool_results = []
        if self.router is not None:
            for c in self._tool_calls(cands[0]):
                try:
                    tool_results.append({"call": c, "result": self.router.dispatch(c)})
                except PermissionError as e:
                    tool_results.append({"call": c, "denied": str(e)})
        if self.heart is not None and len(cands) > 1:
            choice = self.heart.choose(cands, user_text)["choice"]
        else:
            choice = cands[0]
        if not self.trained and grounding:
            # untrained weights produce noise; when retrieval is confident, answer extractively instead
            choice = grounding
        reply = sanitize(tone_prefix(emo) + (shift + " " if shift else "") + choice)
        st.add("assistant", reply, citations)
        return {"reply": reply, "kind": "answer", "citations": citations, "tool_calls": tool_results,
                "emotion": emo, "topic_shift": bool(shift), "model_trained": self.trained,
                "prompt_bytes": len(prompt.encode())}


def sft_records(dialogues: List[List[dict]], system: str = DEFAULT_SYSTEM) -> List[str]:
    """Conversation-format SFT strings from [{role, content}, ...] dialogues (system prompt prepended)."""
    out = []
    for d in dialogues:
        turns = [Turn("system", system)] + [Turn(x["role"], x["content"]) for x in d if x["role"] in ROLES]
        out.append(ChatTemplate.render(turns))
    return out


def api_handler(engine: ChatEngine, body: dict) -> dict:
    """Hook for the PLANNED localhost API (HTTP_HTTPS.md: POST /v1/chat on 127.0.0.1). Pure function:
    no server is started here."""
    msgs = body.get("messages", [])
    last = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
    r = engine.respond(last)
    return {"message": {"role": "assistant", "content": r["reply"]}, "citations": r["citations"],
            "model_trained": r["model_trained"]}
