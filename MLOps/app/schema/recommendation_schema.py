from fastapi import Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ReccomendRequest(BaseModel):
    """추천 요청 스키마 - place ID만 반환"""
    userid: str = Query(..., description="사용자 ID")
    query: str = Query(..., description="검색어")

class ReccomendResponse(BaseModel):
    """추천 응답 스키마 - place ID만 반환"""
    success: bool = Field(..., description="성공 여부")
    places: Dict[str, List[Dict[str, str]]] = Field(
        ...,
        description="추천 장소 목록",
        example={
            "recommend": [{"id": "place_id_1"}, {"id": "place_id_2"}, {"id": "place_id_3"}],
            "normal": [{"id": "place_id_4"}, {"id": "place_id_5"}]
        }
    )