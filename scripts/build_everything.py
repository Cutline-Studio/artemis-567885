"""Rebuild artemis_dense/everything.md from artemis_dense/docs_index.json."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docs_registry

if __name__ == "__main__":
    print(docs_registry.build_everything())
