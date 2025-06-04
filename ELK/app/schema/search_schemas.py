# ELK/app/schemas/search_schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    """검색 요청 스키마"""
    query: str = Field(..., description="검색어")
    max_results: int = Field(default=23, description="최대 결과 수")

class PlaceInfo(BaseModel):
    """장소 정보 스키마"""
    place_id: str = Field(alias="HEX(id)", description="장소 ID")
    category: str = Field(..., description="카테고리")
    sub_category: str = Field(..., description="서브 카테고리")
    user_id: Optional[str] = Field(None, description="사용자 ID")
    search_score: Optional[float] = Field(None, description="검색 점수")
    
    class Config:
        allow_population_by_field_name = True

class SearchResponse(BaseModel):
    """검색 결과 응답 스키마"""
    success: bool
    places: List[PlaceInfo]
    total: int
