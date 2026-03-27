from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(default=3, ge=1, le=10, description="检索文档数量")
    conversation_id: Optional[str] = Field(None, description="会话ID")
    stream: bool = Field(default=False, description="是否流式输出")

class SourceDocument(BaseModel):
    content: str
    metadata: dict
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = []
    conversation_id: Optional[str] = None
    latency_ms: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
