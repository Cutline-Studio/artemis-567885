"""Watch engine: learning by watching (imitation / behaviour cloning).

Demonstration = ordered steps {"observation": {"text": str, "image": tensor|None}, "action":
{"type": "command"|"key"|"click"|"tool", "value": str}}.
  ingest_terminal(transcript)   "$ cmd" lines are actions, the output before each is its observation
  ingest_action_log(jsonl)      recorded keys / clicks / commands / tool calls
  sample_frames(video)          ffmpeg (if installed) -> frames -> ImagePatchEncoder (same token stream)
  bc_samples(demo)              byte strings "<obs>..</obs><act>..</act>" for next-action prediction
  imitate(model, demo, steps)   CPU smoke behaviour cloning (tests only; no real training runs)
  predict_action(model, obs)    greedy decode of the action after "<act>"
  practice(demo, model)         replay predicted actions through the ToolRouter (sandbox + permissions),
                                compare to the demo (step match + outcome) and send scores to self_train
DATA RULES: only recordings the owner has rights to and consent for; redact() runs on every text
before storage or training; redact_image() blanks given regions (blur = planned).
"""
import json, os, re, shutil, subprocess, tempfile
from typing import List, Optional

import torch

# ---------------------------------------------------------------- redaction
_PATTERNS = [
    (re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9\-_]{16,}|nvapi-[A-Za-z0-9\-_]{16,}|"
                r"xox[abpr]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+)"),
     "[REDACTED_TOKEN]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_\.=]{8,}"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED_KEY]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"), "[REDACTED_PHONE]"),
]
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn(digits: str) -> bool:
    s, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        s, alt = s + d, not alt
    return s % 10 == 0


def redact(text: str) -> str:
    text = _CARD.sub(lambda m: "[REDACTED_CARD]" if _luhn(re.sub(r"\D", "", m.group())) else m.group(), text)
    for pat, rep in _PATTERNS:
        text = pat.sub(rep, text)
    return text


def redact_image(img: torch.Tensor, boxes) -> torch.Tensor:
    """Blank (zero) regions [x0, y0, x1, y1] of a [C,H,W] or [B,C,H,W] image, e.g. password fields."""
    out = img.clone()
    for x0, y0, x1, y1 in boxes:
        out[..., y0:y1, x0:x1] = 0
    return out


# ---------------------------------------------------------------- ingestion
def ingest_terminal(transcript: str, prompt: str = "$ ") -> List[dict]:
    steps, obs = [], []
    for line in redact(transcript).splitlines():
        if line.startswith(prompt):
            steps.append({"observation": {"text": "\n".join(obs)[-500:]},
                          "action": {"type": "command", "value": line[len(prompt):].strip()}})
            obs = []
        else:
            obs.append(line)
    for i, s in enumerate(steps):              # record each command's observed outcome (next obs)
        s["outcome"] = steps[i + 1]["observation"]["text"] if i + 1 < len(steps) else "\n".join(obs)
    return steps


def ingest_action_log(jsonl: str) -> List[dict]:
    steps = []
    for line in jsonl.strip().splitlines():
        r = json.loads(line)
        steps.append({"observation": {"text": redact(r.get("observation", ""))},
                      "action": {"type": r["type"], "value": redact(str(r["value"]))},
                      "outcome": redact(r.get("outcome", ""))})
    return steps


def sample_frames(video_path: str, every_s: float = 1.0, size: int = 32, max_frames: int = 16) -> torch.Tensor:
    """[N, 3, size, size] float frames in [0,1] via ffmpeg. Raises NotImplementedError if ffmpeg is absent
    (opencv path = planned)."""
    ff = shutil.which("ffmpeg")
    if not ff:
        raise NotImplementedError("video frontend needs ffmpeg (opencv fallback is planned)")
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([ff, "-loglevel", "error", "-i", video_path, "-vf",
                        f"fps=1/{every_s},scale={size}:{size}", "-frames:v", str(max_frames),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", os.path.join(d, "out.raw")],
                       check=True, timeout=60, env={"PATH": "/usr/bin:/bin"})
        raw = open(os.path.join(d, "out.raw"), "rb").read()
    n = len(raw) // (size * size * 3)
    t = torch.frombuffer(bytearray(raw[:n * size * size * 3]), dtype=torch.uint8)
    return t.view(n, size, size, 3).permute(0, 3, 1, 2).float() / 255.0


def frames_to_tokens(frames: torch.Tensor, image_encoder) -> torch.Tensor:
    """[N,3,H,W] -> [1, N*patches, d_model] prefix tokens for ArtemisDense.forward(prefix=...)."""
    return image_encoder(frames).reshape(1, -1, image_encoder.d_model)


# ---------------------------------------------------------------- behaviour cloning
def fmt_obs(step) -> bytes:
    return f"<obs>{step['observation']['text'][-120:]}</obs><act>".encode()


def bc_samples(demo: List[dict]) -> List[bytes]:
    return [fmt_obs(s) + f"{s['action']['value']}</act>".encode() for s in demo]


def imitate(model, demo: List[dict], steps: int = 200, lr: float = 3e-3, seed: int = 0) -> List[float]:
    """CPU smoke behaviour cloning: loss only on the action bytes. Returns per-step losses."""
    torch.manual_seed(seed)
    samples = bc_samples(demo)
    T = max(len(s) for s in samples)
    X = torch.zeros(len(samples), T, dtype=torch.long)
    M = torch.zeros(len(samples), T - 1)
    for i, s in enumerate(samples):
        X[i, :len(s)] = torch.tensor(list(s))
        start = len(fmt_obs(demo[i]))
        M[i, start - 1:len(s) - 1] = 1                         # predict action bytes (+ </act>)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        logits, _ = model(X[:, :-1])
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), X[:, 1:].reshape(-1),
                                               reduction="none").view_as(M)
        loss = (ce * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(round(loss.item(), 4))
    model.eval()
    return losses


@torch.no_grad()
def predict_action(model, step, max_len: int = 64) -> str:
    from . import transformer as tx
    out = tx.generate(model, fmt_obs(step), max_len)
    return out.split(b"</act>")[0].decode("utf-8", "replace")


def match_rate(model, demo) -> float:
    return sum(predict_action(model, s) == s["action"]["value"] for s in demo) / max(1, len(demo))


# ---------------------------------------------------------------- practice-and-check
def practice(demo: List[dict], policy, router, trainer=None, skills=None, skill_name=None, gate=None) -> dict:
    """policy(step) -> predicted action value. Commands are executed ONLY via router.dispatch
    ("code_exec.run"), i.e. the router's permission list and the code_exec sandbox apply."""
    results = []
    for s in demo:
        pred = policy(s)
        rec = {"expected": s["action"]["value"], "predicted": pred, "step_match": pred == s["action"]["value"]}
        if s["action"]["type"] == "command":
            code = ("import subprocess\nr = subprocess.run(" + json.dumps(pred) +
                    ", shell=True, capture_output=True, text=True)\nprint(r.stdout, end='')")
            try:
                r = router.dispatch({"name": "code_exec.run", "args": {"code": code}})
                rec["outcome_match"] = r["stdout"].strip() == s.get("outcome", "").strip()
                rec["ran"] = True
            except PermissionError as e:
                rec.update({"ran": False, "denied": str(e), "outcome_match": False})
        results.append(rec)
    n = max(1, len(results))
    score = {"step_match_rate": sum(r["step_match"] for r in results) / n,
             "outcome_match_rate": sum(r.get("outcome_match", False) for r in results) / n,
             "steps": results}
    if skills is not None and skill_name and score["step_match_rate"] == 1.0:
        # promotion: a perfectly practised demo becomes a stored skill, only through the constitution gate
        from ..promotion import PromotionGate
        ce = getattr(router, "code_exec", None)
        g = gate or PromotionGate(sandbox=getattr(ce, "sandbox", None) or None)
        body = [{"obs": s.get("obs"), "action": s["action"]} for s in demo]
        v = g.check(f"watched skill {skill_name}", text=json.dumps(body), action="watch.promote_skill",
                    artifact_path=getattr(ce, "run_dir", None))
        score["promotion"] = {"allowed": g.promotable(v), "outcome": v.outcome,
                              "violations": [x.principle for x in v.violations]}
        if g.promotable(v):
            skills.acquire(skill_name, f"Learned by watching ({len(demo)} steps)", skill_name, "steps",
                           body, source="watch", overwrite=True)
    if trainer is not None:
        trainer.add_execution_signal({"task": "watch.practice", "passed": score["step_match_rate"] == 1.0,
                                      "stdout": json.dumps({k: score[k] for k in ("step_match_rate", "outcome_match_rate")}),
                                      "iterations": len(results)})
    return score
