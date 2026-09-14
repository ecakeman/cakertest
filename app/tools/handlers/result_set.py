from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools.schema import pydantic_input_schema
from app.tools.types import ToolDefinition, ToolCallResult, ToolContext, ToolHandler


class ResultSetArgs(BaseModel):
    text: str = Field(..., description="Final answer text for the user")


def handle_result_set(args: dict, ctx: ToolContext) -> ToolCallResult:
    _ = ctx
    parsed = ResultSetArgs.model_validate(args)
    return ToolCallResult(text=parsed.text)


DEFINITION = ToolDefinition(
    name="result_set",
    description="Submit the final answer to the user (non-streaming chat only). Call last.",
    input_schema=pydantic_input_schema(ResultSetArgs),
)
HANDLER: ToolHandler = handle_result_set
