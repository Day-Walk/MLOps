"""
챗봇 API 스키마
OpenAI 챗봇용 커스텀 응답 형식
"""
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatResponse(BaseModel):
    """챗봇 응답 스키마 - 커스텀 형식"""
    str1: Optional[str] = Field(None, description="챗봇의 주요 텍스트 응답")
    placeid: Optional[List[str]] = Field(None, description="추천 장소의 UUID 목록")
    str2: Optional[str] = Field(None, description="추가적인 텍스트 응답")

class AgentChatRequest(BaseModel):
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID")

class ChatStats(BaseModel):
    """챗봇 통계"""
    active_sessions: int
    sessions_detail: dict
    total_sync_sessions: int
    total_stream_sessions: int
