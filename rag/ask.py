"""CLI:  python -m rag.ask "your question" """
import argparse
import os
from rag.engine import RagEngine

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--base-url", default=f"http://localhost:{os.environ.get('PORT', '8000')}")
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-8B"))
    a = ap.parse_args()
    answer, hits = RagEngine(a.base_url, a.model).answer(a.question)
    print(answer, "\n\nSources:")
    for h in hits:
        print(f"  [{h['source']}:{h['chunk']}] score={h['score']:.3f}")
