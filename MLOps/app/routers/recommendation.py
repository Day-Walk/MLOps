from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import traceback
from app.schema.recommendation_schema import ReccomendRequest, ReccomendResponse
from app.services.elk_client import ELKClient
from app.services.deepctr_service import DeepCTRService

router = APIRouter(prefix="/api", tags=["recommendation"])

# 서비스 초기화
elk_client = ELKClient()
deepctr_service = DeepCTRService()

@router.get("/recommend", response_model=ReccomendResponse)
async def recommend_places(userid: str, query: str):
    """Place ID 리스트 기반 추천 API"""
    try:
        top_3_place_ids, other_places_data = await deepctr_service.rank_places_by_ctr(userid, query)
        
        # 스키마에 맞게 데이터 형식 변환
        recommend_places = [{"id": place_id} for place_id in top_3_place_ids]
        normal_places = [{"id": place['place_id']} for place in other_places_data]
        
        return ReccomendResponse(
            success=True,
            places={
                "recommend": recommend_places,
                "normal": normal_places
            }
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/recommend/health")
async def recommendation_health():
    """추천 시스템 헬스체크"""
    return {
        "status": "healthy" if (deepctr_service and elk_client) else "unhealthy",
        "elk_client": "ready" if elk_client else "not ready",
        "deepctr_service": "ready" if deepctr_service else "not ready",
        "service": "recommendation"
    }