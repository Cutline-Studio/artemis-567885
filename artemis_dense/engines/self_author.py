"""Self-author engine (GUARDED version): Artemis proposes and creates new .md docs and skill files.

Flow for create_doc(name, ...):
  1. name/path checks: only inside ALLOWED_DIRS (docs/generated/, skills/generated/), never a
     protected file, never outside the repo
  2. content checks: <= MAX_BYTES, must carry a "Status:" line (built/planned), no secrets
     (watch.redact must leave the text unchanged)
  3. rate cap: <= MAX_PER_HOUR writes
  4. write through the EnvironmentManager (signed grant, scope, path containment, audit log)
  5. register in docs_index.json as review="pending", in_corpus=False; rebuild everything.md
  6. run the test command; on failure ROLL BACK (remove file, restore registry + everything.md)
  7. git commit on the local branch (never push)
approve(rel) / reject(rel) are the owner's review actions; only approved docs enter the corpus.
"""
import json, os, re, shutil, subprocess, sys, time
from typing import List, Optional

from .watch import redact
from .a2a import Identity, AuditLog, EnvironmentManager

ALLOWED_DIRS = ("docs/generated", "skills/generated")
PROTECTED = ("GOVERNANCE.md", "heart.md", "personality.md", "care.md", "emotion.md", "forever.md",
             "killswitch.py", "model.py", "config.py")
MAX_BYTES = 32_000
MAX_PER_HOUR = 20
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")


class AuthorDenied(PermissionError):
    pass


def _git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=60)


class SelfAuthor:
    mode = "guarded"

    def __init__(self, repo_root: str, run_dir: str, test_cmd: Optional[List[str]] = None,
                 principal: str = "local/artemis", commit: bool = True):
        self.root = os.path.realpath(repo_root)
        self.pkg = os.path.join(self.root, "artemis_dense")
        self.run_dir = os.path.abspath(run_dir)
        self.test_cmd = test_cmd
        self.commit = commit
        self.principal = principal
        self.audit = AuditLog(os.path.join(self.run_dir, "self_author_audit.jsonl"))
        ident = Identity.generate("cutline", "self_author")
        self.envs = EnvironmentManager(ident, self.run_dir, self.audit)
        self.sessions = {}
        for d in ALLOWED_DIRS:
            env_id = d.replace("/", "-")
            self.envs.define(env_id, "workspace", os.path.join(self.pkg, d))
            g = self.envs.grant(principal, env_id, ["read", "write"], ttl_s=24 * 3600, rate_per_min=MAX_PER_HOUR)
            self.sessions[d] = self.envs.enter(principal, env_id, g)
        self.writes: List[float] = []
        sys.path.insert(0, os.path.join(self.root, "scripts"))
        import docs_registry
        self.reg = docs_registry

    # ------------------------------------------------------------ checks
    def _check(self, subdir: str, name: str, content: str):
        if subdir not in ALLOWED_DIRS:
            raise AuthorDenied(f"{subdir!r} is not an allowed directory")
        if not _NAME.match(name) or name + ".md" in PROTECTED:
            raise AuthorDenied(f"name {name!r} is not allowed")
        if len(content.encode()) > MAX_BYTES:
            raise AuthorDenied("content too large")
        if not re.search(r"Status:\**\s*(built|planned|mixed|reference)", content, re.I):
            raise AuthorDenied("content must declare Status: built/planned")
        if redact(content) != content:
            raise AuthorDenied("content contains a secret or personal data")
        now = time.time()
        self.writes = [t for t in self.writes if now - t < 3600]
        if len(self.writes) >= MAX_PER_HOUR:
            raise AuthorDenied("rate cap reached")

    # ------------------------------------------------------------ create
    def create_doc(self, name: str, content: str, title: Optional[str] = None,
                   subdir: str = "docs/generated") -> dict:
        try:
            self._check(subdir, name, content)
        except AuthorDenied as e:
            self.audit.record("self_author.create", False, name=name, subdir=subdir, reason=str(e))
            raise
        rel = f"{subdir}/{name}.md"
        idx_before = json.dumps(self.reg.load(self.root))
        ev = os.path.join(self.pkg, "everything.md")
        ev_before = open(ev).read() if os.path.exists(ev) else None
        if os.path.exists(os.path.join(self.pkg, rel)):
            raise AuthorDenied(f"{rel} exists")
        self.envs.write(self.sessions[subdir], self.principal, f"{name}.md", content)   # gated + audited
        self.writes.append(time.time())
        status = re.search(r"Status:\**\s*(built|planned|mixed|reference)", content, re.I).group(1).lower()
        self.reg.register(rel, title or name, status, review="pending", in_corpus=False, root=self.root)
        self.reg.build_everything(self.root)
        ok, out = True, ""
        if self.test_cmd:
            r = subprocess.run(self.test_cmd, cwd=self.root, capture_output=True, text=True, timeout=900)
            ok, out = r.returncode == 0, (r.stdout + r.stderr)[-2000:]
        if not ok:                                          # roll back (Heal.md)
            os.remove(os.path.join(self.pkg, rel))
            self.reg.save(json.loads(idx_before), self.root)
            if ev_before is None:
                os.path.exists(ev) and os.remove(ev)
            else:
                open(ev, "w").write(ev_before)
            self.audit.record("self_author.rollback", True, file=rel, reason="tests failed")
            return {"file": rel, "created": False, "rolled_back": True, "test_output": out}
        commit = None
        if self.commit:
            _git(self.root, "add", os.path.join("artemis_dense", rel), "artemis_dense/docs_index.json",
                 "artemis_dense/everything.md")
            c = _git(self.root, "commit", "-q", "-m", f"self_author: add {rel} (pending review)")
            commit = _git(self.root, "rev-parse", "--short", "HEAD").stdout.strip() if c.returncode == 0 else None
        self.audit.record("self_author.create", True, file=rel, commit=commit)
        return {"file": rel, "created": True, "rolled_back": False, "commit": commit, "review": "pending"}

    # ------------------------------------------------------------ owner review
    def pending(self) -> List[dict]:
        return [d for d in self.reg.load(self.root)["docs"] if d.get("review") == "pending"]

    def approve(self, rel: str):
        for d in self.reg.load(self.root)["docs"]:
            if d["file"] == rel:
                self.reg.register(rel, d["title"], d["status"], review="approved", in_corpus=True, root=self.root)
                self.audit.record("self_author.approve", True, file=rel)
                return
        raise KeyError(rel)
