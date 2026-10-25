"""Self-author engine (UNGUARDED version, chosen by the owner on 2026-10-25).

The guarded version (allowlisted dirs, protected files, secret scan, content checks, size/rate caps,
test-gated rollback, pending-review list, corpus only after approval) is preserved at git tag
`self-author-guarded`; `git checkout self-author-guarded -- artemis_dense/engines/self_author.py
tests/test_self_author.py` restores it.

What this version does:
  write_file(rel_path, content)  writes ANY file under the repo root (code, engines, tests, core docs),
                                 creating directories as needed, then makes one local git commit.
  create_doc(name, content, title=None, subdir="docs/generated")
                                 writes artemis_dense/<subdir>/<name>.md, registers it in docs_index.json
                                 as review="approved", in_corpus=True (goes straight into the corpus),
                                 rebuilds everything.md, commits.
What it does NOT do any more: allowlists, protected files, secret scanning, content checks,
size/rate caps, running tests or rolling back, pending review, environment-permission checks.
PROMOTION GATE (added after, at the owner's request): every write is first staged inside sandbox.env
(/tmp/artemis-sandbox/work/self_author/...), then checked by the custom Constitutional AI engine via
artemis_dense.promotion.PromotionGate; only if the verdict allows it is the file copied into the repo,
registered and committed. A blocked verdict raises PromotionBlocked and nothing reaches the repo.
This is the only content check left; the old self_author guards stay removed.
The engine is loose by default (warnings never block); it blocks only in strict mode (see promotion.py).
What remains: paths are resolved inside the repo root (the scope of the request, "any file in the
repo"); the global kill switch still stops it; it never pushes or opens PRs (commits are local only).
"""
import os, re, subprocess, sys
from typing import Optional

from .. import killswitch
from ..promotion import PromotionGate, PromotionBlocked  # noqa: F401  (re-exported)


class AuthorDenied(PermissionError):
    """Raised only for paths outside the repo or when the kill switch is engaged."""


def _git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=60)


class SelfAuthor:
    mode = "unguarded"

    def __init__(self, repo_root: str, run_dir: str, test_cmd=None, principal: str = "local/artemis",
                 commit: bool = True, gate: Optional[PromotionGate] = None):
        self.root = os.path.realpath(repo_root)
        self.pkg = os.path.join(self.root, "artemis_dense")
        self.run_dir = os.path.abspath(run_dir)
        self.commit = commit
        sys.path.insert(0, os.path.join(self.root, "scripts"))
        import docs_registry
        self.reg = docs_registry
        self.on_write = []                      # hooks, e.g. RAG re-indexing: fn(abs_path)
        self.gate = gate or PromotionGate()

    def _stage_and_gate(self, rel: str, content: str, what: str):
        sb = self.gate.sandbox
        staged = os.path.join(sb.work_dir("self_author"), os.path.basename(rel))
        with open(staged, "w", encoding="utf-8") as f:
            f.write(content)
        try:
            self.gate.require(what, text=content, action="self_author.write", artifact_path=staged,
                              paths=[rel])
        finally:
            os.remove(staged)

    def _resolve(self, rel_path: str) -> str:
        full = os.path.realpath(os.path.join(self.root, rel_path))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise AuthorDenied(f"{rel_path!r} is outside the repo")
        return full

    def _commit(self, paths, msg) -> Optional[str]:
        if not self.commit:
            return None
        _git(self.root, "add", "--", *paths)
        c = _git(self.root, "commit", "-q", "-m", msg)
        return _git(self.root, "rev-parse", "--short", "HEAD").stdout.strip() if c.returncode == 0 else None

    def write_file(self, rel_path: str, content: str) -> dict:
        killswitch.check(self.run_dir)
        full = self._resolve(rel_path)
        self._stage_and_gate(os.path.relpath(full, self.root), content, f"file {rel_path}")  # repo-relative path
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        for h in self.on_write:
            h(full)
        rel = os.path.relpath(full, self.root)
        return {"file": rel, "written": True, "commit": self._commit([rel], f"self_author: write {rel}")}

    def create_doc(self, name: str, content: str, title: Optional[str] = None,
                   subdir: str = "docs/generated") -> dict:
        killswitch.check(self.run_dir)
        rel = f"{subdir}/{name}.md" if subdir else f"{name}.md"
        full = self._resolve(os.path.join("artemis_dense", rel))
        self._stage_and_gate(os.path.join("artemis_dense", rel), content, f"doc {rel}")
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        m = re.search(r"Status:\**\s*(built|planned|mixed|reference)", content, re.I)
        self.reg.register(rel, title or name, m.group(1).lower() if m else "unspecified",
                          review="approved", in_corpus=True, root=self.root)
        self.reg.build_everything(self.root)
        for h in self.on_write:
            h(full)
        commit = self._commit([os.path.join("artemis_dense", rel), "artemis_dense/docs_index.json",
                               "artemis_dense/everything.md"], f"self_author: add {rel}")
        return {"file": rel, "created": True, "rolled_back": False, "commit": commit, "review": "approved"}

    def pending(self):
        return []                               # no review queue in the unguarded version

    def approve(self, rel: str):
        pass
