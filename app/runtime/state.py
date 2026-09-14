from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    # Checkpoint 核心：checkpointer 按 thread_id 读写；nodes 追加，routes 读取最后一条
    messages: Annotated[list[BaseMessage], add_messages]
    # 本轮用户原文：API 写入 → inject_user_node 读入并变成 HumanMessage
    input: str
    # 本轮最终答复：end_node / apply_result_set_node 写入 → HTTP 响应读取
    result: str
    # start_node 根据是否已有 messages 设置；route_after_start 决定是否走 inject_system
    skip_inject_system: bool
    # apply_result_set_node 在 result_set 工具成功后置 True；route_after_tools 可直接 end
    result_set_handled: bool
    # API 区分 /chat-graph 与 /stream；影响 llm 绑定的工具集（是否含 result_set）
    streaming: bool
    # 可选沙箱说明：API 注入 → inject_system_node 拼进 system prompt
    sandbox_context: str
