"""Scaffold a new Artemis doc from the CreateMdFile.md template and register it.

  python3 scripts/new_md.py <Name> [--title T] [--status planned|built|mixed|reference] [--purpose P]

Creates artemis_dense/<Name>.md (refuses to overwrite), registers it in artemis_dense/docs_index.json,
rebuilds everything.md, and rebuilds the corpus manifest + glossary if --rebuild is given.
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docs_registry

TEMPLATE = """# {title}

**Status:** {status}. Mark each section *(built)* or *(planned)*. Verify facts or mark them *(verify)*.

## Purpose
{purpose}

## What it is
*(planned)* Plain explanation.

## How it works in Artemis
*(planned)* Which engines and files are involved, with links.

## Safety and governance
Follows [GOVERNANCE.md](GOVERNANCE.md). No secrets, keys, tokens or card numbers in this file.

## Example dialogues (training)
<!-- example -->
**User:** (a question about {title})
**Artemis:** (a short, honest answer in Artemis's voice)

## Related
[everything.md](everything.md) · [CreateMdFile.md](CreateMdFile.md) · [training.md](training.md) · [Dictionary.md](Dictionary.md)
"""
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")


def create(name: str, title: str = None, status: str = "planned", purpose: str = "(one or two sentences)",
           root: str = docs_registry.ROOT, subdir: str = "", review: str = "approved", in_corpus: bool = True,
           body: str = None) -> str:
    if not NAME.match(name):
        raise ValueError(f"bad doc name {name!r}: letters, digits, _ and - only")
    rel = os.path.join(subdir, name + ".md") if subdir else name + ".md"
    path = os.path.join(docs_registry.pkg(root), rel)
    if os.path.exists(path):
        raise FileExistsError(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    title = title or name
    text = body if body is not None else TEMPLATE.format(title=title, status=status, purpose=purpose)
    open(path, "w", encoding="utf-8").write(text)
    docs_registry.register(rel, title, status, review, in_corpus, root)
    docs_registry.build_everything(root)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--title"); ap.add_argument("--status", default="planned")
    ap.add_argument("--purpose", default="(one or two sentences)")
    ap.add_argument("--rebuild", action="store_true", help="also rebuild corpus manifest and glossary")
    a = ap.parse_args()
    p = create(a.name, a.title, a.status, a.purpose)
    print("created", p)
    if a.rebuild:
        import subprocess
        subprocess.run([sys.executable, "-m", "artemis_dense.data.build_corpus"], cwd=docs_registry.ROOT, check=True)
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "build_glossary.py")], check=True)
