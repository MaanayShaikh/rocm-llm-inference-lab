"""Build the vector index from data/docs:  python -m rag.ingest"""
import argparse
from rag.engine import build_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs")
    n = build_index(ap.parse_args().docs)
    print(f"Indexed {n} chunks.")
