"""Tests for the UNGUARDED self_author (the guarded tests are at tag self-author-guarded)."""
import os, shutil, subprocess, sys
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import new_md, docs_registry  # noqa: E402
from artemis_dense.engines.self_author import SelfAuthor, AuthorDenied  # noqa: E402
from artemis_dense import killswitch  # noqa: E402

DOC = "# Generated\n\n**Status:** planned.\n\nA small generated note.\n"


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "artemis_dense").mkdir(parents=True)
    shutil.copytree(os.path.join(REPO, "scripts"), r / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (r / "artemis_dense" / "GOVERNANCE.md").write_text("# Governance\n")
    docs_registry.register("GOVERNANCE.md", "Governance", "built", root=str(r))
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(r), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(r), "commit", "-qm", "init"], check=True)
    return str(r)


def _log(repo):
    return subprocess.run(["git", "-C", repo, "log", "--oneline"], capture_output=True, text=True).stdout


def test_new_md_scaffolds_and_registers(repo):
    p = new_md.create("Example", title="Example doc", root=repo)
    assert "**Status:** planned" in open(p).read()
    assert any(d["file"] == "Example.md" for d in docs_registry.load(repo)["docs"])
    assert "<!-- file: Example.md -->" in open(os.path.join(repo, "artemis_dense", "everything.md")).read()
    with pytest.raises(FileExistsError):
        new_md.create("Example", root=repo)


def test_create_doc_goes_straight_into_corpus_and_commits(repo, tmp_path):
    a = SelfAuthor(repo, str(tmp_path / "run"))
    assert a.mode == "unguarded"
    r = a.create_doc("Note1", "no status line, api_key = not-checked-anymore", title="Note 1")
    entry = next(d for d in docs_registry.load(repo)["docs"] if d["file"] == "docs/generated/Note1.md")
    assert r["created"] and r["commit"] and entry["in_corpus"] is True and entry["review"] == "approved"
    assert a.pending() == []
    assert "self_author: add docs/generated/Note1.md" in _log(repo)


def test_can_overwrite_core_docs_and_code(repo, tmp_path):
    a = SelfAuthor(repo, str(tmp_path / "run"))
    r1 = a.write_file("artemis_dense/GOVERNANCE.md", "# replaced\n")
    r2 = a.write_file("artemis_dense/engines/new_engine.py", "X = 1\n")
    r3 = a.create_doc("GOVERNANCE", "# also replaced\n", subdir="")
    assert open(os.path.join(repo, "artemis_dense", "GOVERNANCE.md")).read() == "# also replaced\n"
    assert all(r["commit"] for r in (r1, r2, r3))
    assert "self_author: write artemis_dense/engines/new_engine.py" in _log(repo)


def test_large_content_and_many_writes_allowed(repo, tmp_path):
    a = SelfAuthor(repo, str(tmp_path / "run"), commit=False)
    for i in range(30):                                   # no rate cap
        a.create_doc(f"N{i}", "x" * 100_000)              # no size cap


def test_still_confined_to_repo_and_kill_switch(repo, tmp_path):
    a = SelfAuthor(repo, str(tmp_path / "run"))
    with pytest.raises(AuthorDenied):
        a.write_file("../outside.txt", "x")
    killswitch.engage(str(tmp_path / "run"))
    with pytest.raises(killswitch.Killed):
        a.write_file("artemis_dense/x.md", "x")


def test_write_hook_called(repo, tmp_path):
    seen = []
    a = SelfAuthor(repo, str(tmp_path / "run"), commit=False)
    a.on_write.append(seen.append)
    a.create_doc("Hooked", DOC)
    assert seen and seen[0].endswith("Hooked.md")
