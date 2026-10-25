import json, os, shutil, subprocess, sys
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import new_md, docs_registry  # noqa: E402
from artemis_dense.engines.self_author import SelfAuthor, AuthorDenied  # noqa: E402

DOC = "# Generated\n\n**Status:** planned.\n\nA small generated note.\n"


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "artemis_dense").mkdir(parents=True)
    shutil.copytree(os.path.join(REPO, "scripts"), r / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (r / "artemis_dense" / "GOVERNANCE.md").write_text("# Governance\n")
    docs_registry.register("GOVERNANCE.md", "Governance", "built", root=str(r))
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.email", "t@t"], check=True)
    return str(r)


def test_new_md_scaffolds_and_registers(repo):
    p = new_md.create("Example", title="Example doc", root=repo)
    text = open(p).read()
    assert "**Status:** planned" in text and "<!-- example -->" in text
    idx = docs_registry.load(repo)
    assert any(d["file"] == "Example.md" for d in idx["docs"])
    assert "<!-- file: Example.md -->" in open(os.path.join(repo, "artemis_dense", "everything.md")).read()
    with pytest.raises(FileExistsError):
        new_md.create("Example", root=repo)
    with pytest.raises(ValueError):
        new_md.create("../evil", root=repo)


def test_allowed_write_registers_pending_and_commits(repo, tmp_path):
    a = SelfAuthor(repo, str(tmp_path / "run"))
    r = a.create_doc("Note1", DOC, title="Note 1")
    assert r["created"] and r["commit"]
    assert os.path.exists(os.path.join(repo, "artemis_dense", "docs", "generated", "Note1.md"))
    assert [d["file"] for d in a.pending()] == ["docs/generated/Note1.md"]
    entry = next(d for d in docs_registry.load(repo)["docs"] if d["file"] == "docs/generated/Note1.md")
    assert entry["in_corpus"] is False                       # not in corpus until approved
    a.approve("docs/generated/Note1.md")
    entry = next(d for d in docs_registry.load(repo)["docs"] if d["file"] == "docs/generated/Note1.md")
    assert entry["in_corpus"] is True and entry["review"] == "approved"
    log = subprocess.run(["git", "-C", repo, "log", "--oneline"], capture_output=True, text=True).stdout
    assert "self_author: add docs/generated/Note1.md" in log


def test_protected_and_outside_writes_blocked(repo, tmp_path):
    a = SelfAuthor(repo, str(tmp_path / "run"))
    with pytest.raises(AuthorDenied):
        a.create_doc("GOVERNANCE", DOC)
    with pytest.raises(AuthorDenied):
        a.create_doc("x", DOC, subdir="engines")
    with pytest.raises(AuthorDenied):
        a.create_doc("x", "# no status line\n")


def test_secret_in_content_blocked(repo, tmp_path):
    a = SelfAuthor(repo, str(tmp_path / "run"))
    with pytest.raises(AuthorDenied, match="secret"):
        a.create_doc("Leaky", DOC + "\napi_key = abcdef123456\n")


def test_rollback_on_test_failure(repo, tmp_path):
    a = SelfAuthor(repo, str(tmp_path / "run"), test_cmd=[sys.executable, "-c", "raise SystemExit(1)"])
    before = docs_registry.load(repo)
    r = a.create_doc("Broken", DOC)
    assert r["rolled_back"] and not r["created"]
    assert not os.path.exists(os.path.join(repo, "artemis_dense", "docs", "generated", "Broken.md"))
    assert docs_registry.load(repo) == before
