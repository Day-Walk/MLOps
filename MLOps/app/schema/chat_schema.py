"""
챗봇 API 스키마
OpenAI 챗봇용 커스텀 응답 형식
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any
import json
import re

class ChatResponse(BaseModel):
    """챗봇 응답 스키마 - 커스텀 형식"""
    str1: str = Field(default="", description="챗봇의 주요 텍스트 응답")
    placeid: List[str] = Field(default_factory=list, description="추천 장소의 UUID 목록")
    str2: str = Field(default="", description="추가적인 텍스트 응답")

    @validator('placeid', pre=True)
    def validate_placeid(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        if isinstance(v, list):
            return [str(item) for item in v if item]
        return []

    @validator('str1', 'str2', pre=True)
    def validate_strings(cls, v):
        return str(v) if v is not None else ""

class AgentChatRequest(BaseModel):
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID")

class ChatStats(BaseModel):
    """챗봇 통계"""
    active_sessions: int
    sessions_detail: dict
    total_sync_sessions: int
    total_stream_sessions: int

def normalize_agent_response(response: Union[str, Dict, Any]) -> ChatResponse:
    """
    에이전트 응답을 ChatResponse 형식으로 정규화합니다.
    """
    # 기본값 설정
    default_response = {"str1": "", "placeid": [], "str2": ""}
    
    if isinstance(response, str):
        # 문자열에서 JSON 추출 시도
        json_response = extract_json_from_string(response)
        if json_response:
            response = json_response
        else:
            # JSON이 없으면 str1에 전체 텍스트를 넣음
            default_response["str1"] = response
            return ChatResponse(**default_response)
    
    if isinstance(response, dict):
        # 딕셔너리인 경우 필요한 키들을 확인하고 기본값으로 채움
        normalized = {}
        normalized["str1"] = response.get("str1", response.get("답변", response.get("response", "")))
        normalized["placeid"] = response.get("placeid", response.get("place_ids", response.get("uuids", [])))
        normalized["str2"] = response.get("str2", response.get("코스", response.get("course", "")))
        
        return ChatResponse(**normalized)
    
    # 기타 경우
    default_response["str1"] = str(response) if response else "응답을 생성하지 못했습니다."
    return ChatResponse(**default_response)

def extract_json_from_string(text: str) -> Optional[Dict]:
    """
    문자열에서 JSON 객체를 추출합니다.
    """
    if not text or not isinstance(text, str):
        return None
    
    # 여러 JSON 추출 패턴 시도
    patterns = [
        # 완전한 JSON 객체
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        # 중괄호로 시작하고 끝나는 패턴
        r'\{.*?\}',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # 마지막 시도: 첫 번째 {부터 마지막 }까지
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        try:
            return json.loads(text[start_idx:end_idx+1])
        except json.JSONDecodeError:
            pass
    
    return None
