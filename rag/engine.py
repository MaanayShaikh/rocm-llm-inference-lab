"""Minimal RAG engine: sentence-transformers embeddings (CPU) + cosine search + vLLM generation.

The LLM is served by vLLM on an AMD Instinct GPU (ROCm); embeddings run on CPU so the
GPU stays dedicated to generation.
"""
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from rag.chunking import chunk_text

INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", "rag/index"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "Cite sources like [source:chunk]. If the context does not contain the answer, "
    "say you do not know."
)


def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def build_index(docs_dir: str = "data/docs") -> int:
    chunks: List[Dict] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        for i, c in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            chunks.append({"source": path.name, "chunk": i, "text": c})
    if not chunks:
        raise SystemExit(f"No .md/.txt files found in {docs_dir}")
    vectors = _embedder().encode([c["text"] for c in chunks], normalize_embeddings=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks))
    return len(chunks)


class RagEngine:
    def __init__(self, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key="EMPTY")
        self.model = model
        self.vectors = np.load(INDEX_DIR / "vectors.npy")
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text())
        self.embedder = _embedder()

    def retrieve(self, question: str, k: int = 4) -> List[Dict]:
        q = self.embedder.encode([question], normalize_embeddings=True)[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def answer(self, question: str, k: int = 4):
        hits = self.retrieve(question, k)
        context = "\n\n".join(f"[{h['source']}:{h['chunk']}]\n{h['text']}" for h in hits)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=400,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content, hits
