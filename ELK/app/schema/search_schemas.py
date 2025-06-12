from pydantic import BaseModel
from typing import List

class Place(BaseModel):
    place_id: str
    name: str
    category: str
    sub_category: str

class SearchResponse(BaseModel):
    """검색 결과 응답 스키마"""
    success: bool
    places: List[Place]
    total: int
