from __future__ import annotations

from app.mempalace import chroma_store


def test_delete_by_user_deletes_rows_for_user(monkeypatch):
    executed: list[tuple] = []

    class FakeCursor:
        def execute(self, sql, params=None):
            executed.append((sql, params))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(chroma_store, "_schema_ready", True)
    monkeypatch.setattr(chroma_store, "_connect", lambda: FakeConn())

    chroma_store.delete_by_user("Sancho")

    sql, params = executed[0]
    assert "DELETE FROM mempalace" in " ".join(sql.split())
    assert params == ("Sancho",)
