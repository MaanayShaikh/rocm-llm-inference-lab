import pytest
from rag.chunking import chunk_text


def test_chunks_respect_max_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(10))
    chunks = chunk_text(text, max_chars=500, overlap=50)
    assert chunks and all(len(c) <= 500 + 60 for c in chunks)


def test_short_text_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_long_paragraph_is_split():
    assert len(chunk_text("x" * 2000, max_chars=500, overlap=50)) >= 4


def test_bad_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", max_chars=10, overlap=10)
