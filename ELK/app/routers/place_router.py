from fastapi import APIRouter, HTTPException, Query
from typing import List

from app.schema.search_schemas import SearchResponse, LLMToolResponse, Place
from app.services.elasticsearch_service import ElasticsearchService

router = APIRouter(prefix="/api/place", tags=["place-search"])

elasticsearch_service = ElasticsearchService()

@router.get("/search", response_model=SearchResponse)
async def search_places(query: str, user_id: str, max_results: int = 23):
    """장소 검색 API"""
    try:
        if not elasticsearch_service.is_connected():
            raise HTTPException(status_code=503, detail="Elasticsearch 연결 실패")
        
        places = elasticsearch_service.search_places(
            query=query,
            max_results=max_results,
            user_id=user_id
        )
        
        # 딕셔너리를 Place 객체로 변환
        place_objects = [Place(**place) for place in places]
        
        return SearchResponse(
            success=True,
            places=place_objects,
            total=len(place_objects)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/llm-tool", response_model=LLMToolResponse)
async def search_places_for_llm_tool(region: str, categories: List[str] = Query(..., min_length=1), user_id: str | None = None):
    """LLM 도구를 위한 장소 검색 API"""
    try:
        if not elasticsearch_service.is_connected():
            raise HTTPException(status_code=503, detail="Elasticsearch 연결 실패")
        
        uuids, total = elasticsearch_service.search_places_for_llm_tool(
            region=region,
            categories=categories,
            user_id=user_id
        )
        
        return LLMToolResponse(
            success=True,
            uuids=uuids,
            total=total
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 