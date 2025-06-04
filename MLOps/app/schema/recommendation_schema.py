from pydantic import BaseModel, Field
from typing import List, Dict, Any

class RecommendationRequest(BaseModel):
    """추천 요청 스키마"""
    user_id: str = Field(..., description="사용자 ID")
    search_query: str = Field(..., description="검색어")
    user_features: Dict[str, Any] = Field(default_factory=dict, description="사용자 특징")

class RecommendationResponse(BaseModel):
    """추천 응답 스키마 - place ID만 반환"""
    success: bool = Field(..., description="성공 여부")
    top_3_place_ids: List[str] = Field(..., description="상위 3개 추천 장소 ID")
    other_place_ids: List[str] = Field(..., description="나머지 장소 ID 리스트")
    total: int = Field(..., description="전체 장소 수")
