# MLOps/app/schemas/recommendation_schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class RecommendationRequest(BaseModel):
    """추천 요청 스키마"""
    user_id: str = Field(..., description="사용자 ID")
    search_query: str = Field(..., description="검색어")
    user_features: Dict[str, Any] = Field(..., description="사용자 특징")

class PlaceResult(BaseModel):
    """장소 결과 스키마"""
    place_id: str
    category: str
    sub_category: str
    ctr_score: Optional[float] = None
    rank: Optional[int] = None

class RecommendationResponse(BaseModel):
    """추천 응답 스키마"""
    success: bool
    top_3_recommendations: List[PlaceResult]
    other_places: List[PlaceResult]
    total: int
