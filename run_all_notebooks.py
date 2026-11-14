"""
Rebuild the whole RAG on the real NASA PSI-159 data by running every notebook in order.

    python run_all_notebooks.py            # run all, stop at the first failing notebook
    python run_all_notebooks.py --from 6   # resume from notebook #6
    python run_all_notebooks.py --list     # show the order

Each notebook runs inside its own folder (like opening it in Jupyter) and is saved
with its outputs, so you can open it afterwards and see the results.
When everything passes, start the API server:

    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
RAG = ROOT / "rag"

ORDER = [
    ("ingestion", "document_loader.ipynb", "real data -> knowledge_base/*.md"),
    ("ingestion", "Chunker.ipynb", "knowledge base -> chunks.json"),
    ("embeddings", " embedder.ipynb", "chunks -> embeddings.npy"),
    ("vector_store", "index_builder.ipynb", "embeddings -> faiss.index"),
    ("vector_store", "Vector_store.ipynb", "FAISS search test"),
    ("retrieval", "Test.ipynb", "dense + BM25 + hybrid + reranker tests"),
    ("context", "test.ipynb", "context selection / building test"),
    ("answer", "test.ipynb", "answerability gate test"),
    ("ml", "ML_test.ipynb", "retrain + test the real-data ML model"),
    ("llm", "test.ipynb", "OpenRouter LLM test"),
    (".", "pipeline_test.ipynb", "full RAG pipeline test"),
    ("Test", "test_retrieval_accuracy.ipynb", "30-question retrieval test"),
    ("Test", "final testing.ipynb", "final testing"),
]


def run(index, folder, name):
    path = (RAG / folder / name).resolve()
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=1800, kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    t0 = time.time()
    try:
        client.execute()
        ok, err = True, ""
    except Exception as exc:  # keep the partial outputs for debugging
        ok, err = False, str(exc)
    nbformat.write(nb, path)
    return ok, err, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--only", type=int)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for i, (folder, name, what) in enumerate(ORDER, 1):
            print(f"{i:2}. rag/{folder}/{name.strip()}  — {what}")
        return 0

    for i, (folder, name, what) in enumerate(ORDER, 1):
        if i < args.start or (args.only and i != args.only):
            continue
        print(f"[{i:2}/{len(ORDER)}] rag/{folder}/{name.strip()}  ({what}) ...", flush=True)
        ok, err, secs = run(i, folder, name)
        if not ok:
            print(f"      FAILED after {secs:.0f}s\n")
            print(err[-3000:])
            print(f"\nFix the error, then resume with:  python run_all_notebooks.py --from {i}")
            return 1
        print(f"      OK ({secs:.0f}s)", flush=True)

    print("\nAll notebooks passed. Start the server with:")
    print("  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
