"""
ov-cli server: Pydantic 请求模型。
"""

from typing import Any, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    stream: bool = True
    max_tokens: int = 32768
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    presence_penalty: Optional[float] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[str] = None  # "none", "auto", "required"
    parallel_tool_calls: Optional[bool] = None



