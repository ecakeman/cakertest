from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api.admin import router as admin_router
from app.api.chat import router as chat_router
from app.api.terminal import router as terminal_router
from app.api.web_data import router as web_data_router
from app.config import settings
from app.execution.cleanup import cleanup_orphan_containers
from app.mempalace import chroma_store
from app.runtime.graph import compile_graph
from app.web_store.store import store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("var").mkdir(parents=True, exist_ok=True)
    store.ensure_dirs()
    store.ensure_default_user()
    cleanup_orphan_containers()
    chroma_store.setup()
    async with AsyncPostgresSaver.from_conn_string(settings.pg_dsn) as checkpointer:
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        compile_graph(checkpointer)
        yield


app = FastAPI(title="caker", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(web_data_router)
app.include_router(terminal_router)


@app.get("/health")
async def health():
    return {"ok": True}


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

# TODO: 你加 request_id 中间件、structlog 日志
