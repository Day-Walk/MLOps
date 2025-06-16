"""
챗봇 API 스키마
OpenAI 챗봇용 커스텀 응답 형식
"""
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatResponse(BaseModel):
    """챗봇 응답 스키마 - 커스텀 형식"""
    str1: str = Field(..., description="필수 응답 텍스트 (GPT 답변)")
    placeid: Optional[List[str]] = Field(None, description="선택적 장소 ID(UUID) 배열")
    str2: Optional[str] = Field(None, description="선택적 추가 텍스트")

class ChatStats(BaseModel):
    """챗봇 통계"""
    active_sessions: int
    total_sync_sessions: int
    total_stream_sessions: int
