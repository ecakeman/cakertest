"""M10: AsyncPostgresSaver persists LangGraph state across compiles."""

from __future__ import annotations

import asyncio
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings
from app.runtime.graph import compile_graph, get_graph


def test_postgres_checkpoint_restores_messages(monkeypatch):
    async def fake_llm_node(state, config):
        prior = state.get("messages") or []
        humans = [m.content for m in prior if isinstance(m, HumanMessage)]
        if len(humans) >= 2 and "张三" in str(humans[0]):
            text = "你叫张三"
        else:
            text = "好的，记住了"
        return {"messages": [AIMessage(content=text)]}

    async def fake_mempalace(state, config):
        return {}

    monkeypatch.setattr("app.runtime.nodes.llm_node", fake_llm_node)
    monkeypatch.setattr("app.runtime.nodes.mempalace_inject_node", fake_mempalace)

    thread_id = f"m10-{uuid.uuid4().hex[:12]}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "session_id": thread_id,
            "user_id": "local",
        }
    }

    async def _run():
        async with AsyncPostgresSaver.from_conn_string(settings.pg_dsn) as checkpointer:
            await checkpointer.setup()
            compile_graph(checkpointer)
            graph = get_graph()
            await graph.ainvoke(
                {
                    "messages": [],
                    "input": "我叫张三",
                    "result": "",
                    "skip_inject_system": False,
                    "result_set_handled": False,
                    "streaming": False,
                    "sandbox_context": "",
                },
                config=config,
            )

        async with AsyncPostgresSaver.from_conn_string(settings.pg_dsn) as checkpointer:
            await checkpointer.setup()
            compile_graph(checkpointer)
            graph = get_graph()
            return await graph.ainvoke(
                {
                    "messages": [],
                    "input": "我叫什么？",
                    "result": "",
                    "skip_inject_system": False,
                    "result_set_handled": False,
                    "streaming": False,
                    "sandbox_context": "",
                },
                config=config,
            )

    out = asyncio.run(_run())
    texts = [str(m.content) for m in (out.get("messages") or []) if isinstance(m, AIMessage)]
    assert out.get("result") == "你叫张三" or any("张三" in t for t in texts)
