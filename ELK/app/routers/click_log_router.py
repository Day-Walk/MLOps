from fastapi import APIRouter, HTTPException, Query
from typing import List

from app.schema.log_schemas import ClickLogRequest, ClickLogResponse
from app.schema.search_schemas import MostClickedPlace
from app.services.elasticsearch_service import ElasticsearchService

router = APIRouter(prefix="/api", tags=["클릭 로그"])

elasticsearch_service = ElasticsearchService()

@router.post("/click-log", response_model=ClickLogResponse)
async def insert_click_log(click_log: ClickLogRequest):
    """클릭 로그 저장 API"""
    try:
        if not elasticsearch_service.is_connected():
            return ClickLogResponse(
                success=False,
                message="Elasticsearch 연결 실패"
            )
        
        log_dict = click_log.dict()
        
        success, log_id = elasticsearch_service.insert_click_log(log_dict)
        
        if success:
            return ClickLogResponse(
                success=True,
                message="클릭 로그 저장 완료!",
            )
        else:
            return ClickLogResponse(
                success=False,
                message="클릭 로그 저장 실패!"
            )
            
    except Exception as e:
        print(f"클릭 로그 저장 오류: {e}")
        return ClickLogResponse(
            success=False,
            message=str(e)
        )

@router.get("/click-log/user/{user_id}")
async def get_user_click_logs(user_id: str, days: int = Query(default=30, ge=1, le=365)):
    """사용자의 클릭 로그 조회 API"""
    try:
        if not elasticsearch_service.is_connected():
            raise HTTPException(status_code=503, detail="Elasticsearch 연결 실패")
        
        logs = elasticsearch_service.get_click_logs_by_user(user_id, days)
        
        return {
            "success": True,
            "userId": user_id,
            "logs": logs,
            "count": len(logs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/click-log/place/{place_id}/count")
async def get_place_click_count(place_id: str):
    """특정 장소의 클릭 수 조회 API"""
    try:
        if not elasticsearch_service.is_connected():
            raise HTTPException(status_code=503, detail="Elasticsearch 연결 실패")
        
        count = elasticsearch_service.get_click_count_by_place(place_id)
        
        return {
            "success": True,
            "placeId": place_id,
            "clickCount": count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/most-click-place", response_model=List[MostClickedPlace])
async def get_most_clicked_places():
    """오늘 새벽 5시부터 가장 많이 클릭된 장소 UUID 목록 조회 API"""
    try:
        if not elasticsearch_service.is_connected():
            raise HTTPException(status_code=503, detail="Elasticsearch 연결 실패")
        
        places_data = elasticsearch_service.get_most_clicked_places_today()
        
        return places_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 