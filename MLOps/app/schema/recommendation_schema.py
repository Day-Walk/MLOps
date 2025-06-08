from fastapi import Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ReccomendRequest(BaseModel):
    """추천 요청 스키마 - place ID만 반환"""
    user_id: str = Query(..., description="사용자 ID")
    query: str = Query(..., description="검색어")

class ReccomendResponse(BaseModel):
    """추천 응답 스키마 - place ID만 반환"""
    success: bool = Field(..., description="성공 여부")
    top_3_place_ids: List[str] = Field(..., description="상위 3개 장소 ID")
    other_place_ids: List[str] = Field(..., description="나머지 장소 ID")
