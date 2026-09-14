"""MemPalace 向量写入 PostgreSQL + pgvector。"""

from __future__ import annotations

from app.config import settings
from app.mempalace import chroma_store


def test_pgvector_add_search_delete_by_user(monkeypatch):
    dim = settings.embedding_dimensions

    def fake_embed(text: str) -> list[float]:
        vec = [0.0] * dim
        if "cat" in text or "猫" in text:
            vec[0] = 1.0
        else:
            vec[1] = 1.0
        return vec

    monkeypatch.setattr(chroma_store, "_embed", fake_embed)
    chroma_store.setup()
    chroma_store.delete_by_user("pgvec-test")

    chroma_store.add("m-cat", "the cat is gray", {"user_id": "pgvec-test", "session_id": "s1"})
    chroma_store.add("m-other", "unrelated note", {"user_id": "pgvec-test", "session_id": "s1"})

    hits = chroma_store.search("about the cat", k=1, where={"user_id": "pgvec-test"})
    assert hits
    assert hits[0][0] == "m-cat"
    assert hits[0][2]["user_id"] == "pgvec-test"

    chroma_store.delete_by_user("pgvec-test")
    empty = chroma_store.search("about the cat", k=1, where={"user_id": "pgvec-test"})
    assert empty == []
