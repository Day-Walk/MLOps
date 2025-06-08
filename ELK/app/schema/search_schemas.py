from pydantic import BaseModel
from typing import List

class SearchResponse(BaseModel):
    """검색 결과 응답 스키마"""
    success: bool
    places: List[str]
    total: int
