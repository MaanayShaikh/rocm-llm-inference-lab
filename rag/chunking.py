"""Paragraph-aware text chunking (no heavy dependencies, easy to unit test)."""
from typing import List


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """Split text into chunks of at most max_chars, preferring paragraph boundaries.

    Consecutive chunks share `overlap` trailing characters so context is not cut mid-idea.
    """
    if max_chars <= overlap:
        raise ValueError("max_chars must be larger than overlap")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        while len(para) > max_chars:  # hard-split very long paragraphs
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:max_chars])
            para = para[max_chars - overlap:]
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current and overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
    if current:
        chunks.append(current)
    return chunks
