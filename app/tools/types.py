from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolContext:
    user_id: str = "local"
    session_id: str = "demo"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    title: str | None = None


@dataclass
class ToolCallResult:
    text: str
    is_error: bool = False


ToolHandler = Callable[[dict[str, Any], ToolContext], ToolCallResult | Awaitable[ToolCallResult]]
