import json
from pathlib import Path

from rank_bm25 import BM25Okapi


# rag/retrieval/bm25_retriever.py -> rag/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_FILE = (
    PROJECT_ROOT
    / "embeddings"
    / "processed"
    / "chunks"
    / "chunks.json"
)


class BM25Retriever:

    def __init__(
        self,
        chunks_file=CHUNKS_FILE
    ):

        self.chunks_file = Path(
            chunks_file
        )

        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_file}"
            )

        with open(
            self.chunks_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.chunks = json.load(f)

        self.tokenized_corpus = [
            self.tokenize(
                chunk.get("text", "")
            )
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    @staticmethod
    def tokenize(text):

        # Split on anything that is not a letter, digit, dot, % or hyphen,
        # so "**Fuel:** butane" and "CFI-21088-A1," tokenize cleanly.
        import re
        return re.findall(r"[a-z0-9][a-z0-9.%-]*", str(text).lower())

    def search(self, query, top_k=10):

        query_tokens = self.tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1
        ):

            results.append({
                "chunk_id": int(index),
                "score": float(scores[index]),
                "rank": rank
            })

        return results