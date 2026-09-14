from __future__ import annotations

import httpx
import psycopg

from app.config import settings

_schema_ready = False


def _connect() -> psycopg.Connection:
    return psycopg.connect(settings.pg_dsn)


def _embed_url() -> str:
    base = settings.embedding_base_url.strip().rstrip("/")
    if not base:
        raise ValueError("请在 .env 中配置 EMBEDDING_BASE_URL。")
    if base.endswith("/embeddings"):
        return base
    return f"{base}/embeddings"


def _embed(text: str) -> list[float]:
    key = settings.embedding_api_key.strip()
    model = settings.embedding_model
    if not key:
        raise ValueError("请在 .env 中配置 EMBEDDING_API_KEY。")
    if not model:
        raise ValueError("请在 .env 中配置 EMBEDDING_MODEL_NAME。")
    payload: dict = {"model": model, "input": text}
    if settings.embedding_dimensions:
        payload["dimensions"] = settings.embedding_dimensions
    resp = httpx.post(
        _embed_url(),
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    vec = data["data"][0]["embedding"]
    return [float(x) for x in vec]


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def setup() -> None:
    """创建 pgvector 扩展与 mempalace 表（与 Checkpoint 共用 PG_DSN）。"""
    global _schema_ready
    dim = int(settings.embedding_dimensions)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS mempalace (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    embedding vector({dim}) NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS mempalace_user_id_idx ON mempalace (user_id)"
            )
        conn.commit()
    _schema_ready = True


def _ensure_schema() -> None:
    global _schema_ready
    if not _schema_ready:
        setup()


def add(memory_id: str, text: str, metadata: dict) -> None:
    _ensure_schema()
    user_id = str(metadata.get("user_id") or "local")
    session_id = str(metadata.get("session_id") or "")
    lit = _vector_literal(_embed(text))
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mempalace (id, content, user_id, session_id, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    user_id = EXCLUDED.user_id,
                    session_id = EXCLUDED.session_id,
                    embedding = EXCLUDED.embedding
                """,
                (memory_id, text, user_id, session_id, lit),
            )
        conn.commit()


def search(query: str, k: int = 3, where: dict | None = None):
    _ensure_schema()
    user_id = (where or {}).get("user_id")
    lit = _vector_literal(_embed(query))
    with _connect() as conn:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    """
                    SELECT id, content, user_id, session_id
                    FROM mempalace
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (str(user_id), lit, int(k)),
                )
            else:
                cur.execute(
                    """
                    SELECT id, content, user_id, session_id
                    FROM mempalace
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (lit, int(k)),
                )
            rows = cur.fetchall()
    return [
        (rid, content, {"user_id": uid, "session_id": sid})
        for rid, content, uid, sid in rows
    ]


def delete_by_user(user_id: str) -> None:
    _ensure_schema()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM mempalace WHERE user_id = %s", (user_id,))
        conn.commit()
