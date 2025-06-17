from pydantic import BaseModel
from typing import List

class Place(BaseModel):
    uuid: str
    name: str
    category: str
    subcategory: str

class SearchResponse(BaseModel):
    """검색 결과 응답 스키마"""
    success: bool
    places: List[Place]
    total: int

class LLMToolResponse(BaseModel):
    success: bool
    uuids: List[str]
    total: int
