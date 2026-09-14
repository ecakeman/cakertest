from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Qwen（阿里云百炼 / DashScope，OpenAI 兼容模式）
    # BASE_URL 说明：https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
    llm_model_name: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    embedding_model_name: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_dimensions: int = Field(default=1024, ge=1)

    @property
    def embedding_model(self) -> str:
        """DashScope 兼容模式模型名为 text-embedding-v4，非 openai/ 前缀。"""
        name = self.embedding_model_name.strip()
        if name.startswith("openai/"):
            return name.removeprefix("openai/")
        return name

    workspace_root: str = Field(default="./var/workspace")
    # LangGraph Checkpoint（AsyncPostgresSaver）；本地开发默认见 .env.example / docker compose
    pg_dsn: str = Field(
        default="postgresql://agent:agent@127.0.0.1:5432/agent_skills",
    )
    # 本地路线可留空；对象存储不在跟写范围
    s3_endpoint: str = Field(default="http://localhost:9000")
    s3_bucket: str = Field(default="agent-skills-state")
    upload_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1)

    # Qwen3.5-397B-A17B 原生 262K；官方建议部署上下文至少 128K 以保留长程推理能力。
    # 压缩在 estimate_tokens >= max_input_tokens * compact_ratio 时触发（默认约 111K）。
    max_input_tokens: int = Field(default=131_072, ge=1000)
    compact_ratio: float = Field(default=0.85, ge=0.1, le=1.0)
    graph_recursion_limit: int = Field(default=50, ge=1)
    mempalace_auto_inject: bool = Field(default=False)
    stream_emit_tool_status: bool = Field(default=True)

    # CEER V2 — Sandbox terminal + venue shell
    sandbox_terminal_enabled: bool = Field(default=True)
    sandbox_docker_bin: str = Field(default="docker")
    sandbox_venue_image: str = Field(default="python:3.12-slim")
    sandbox_venue_mount: str = Field(default="/workspace")
    docker_pull_mirror_prefix: str = Field(
        default="docker.m.daocloud.io",
        description="Empty to disable; prepended for venue image pull only",
    )

    @field_validator("pg_dsn")
    @classmethod
    def pg_dsn_required(cls, v: str) -> str:
        dsn = (v or "").strip()
        if not dsn:
            raise ValueError(
                "PG_DSN is required (LangGraph AsyncPostgresSaver). "
                "Start postgres: docker compose up -d postgres"
            )
        return dsn


settings = Settings()
