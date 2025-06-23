from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import traceback
from app.schema.recommendation_schema import ReccomendRequest, ReccomendResponse
from app.services.elk_client import ELKClient
from app.services.deepctr_service import DeepCTRService

router = APIRouter(prefix="/api", tags=["recommendation"])

def services_provider():
    """ELKClient, DeepCTRService 를 최초 호출 시 생성해 캐시합니다.
    생성 과정에서 예외가 나면 None 을 반환해 API 가 503 을 응답하도록 합니다."""
    if not hasattr(services_provider, "cache"):
        try:
            services_provider.cache = (ELKClient(), DeepCTRService())
        except Exception as e:
            print("[recommendation] 서비스 초기화 실패:", e)
            services_provider.cache = None
    return services_provider.cache

@router.get("/recommend", response_model=ReccomendResponse)
async def recommend_places(
    userid: str,
    query: str,
    services=Depends(services_provider)
):
    """Place ID 리스트 기반 추천 API"""
    try:
        if services is None:
            raise HTTPException(status_code=503, detail="Recommendation service not available")

        elk_client, deepctr_service = services

        top_places_data, other_places_data = await deepctr_service.rank_places_by_ctr(userid, query)
        
        # 스키마에 맞게 데이터 형식 변환
        recommend_places = [
            {k: v for k, v in {"id": p.get("id"), "category": p.get("category"), "subcategory": p.get("subcategory")}.items() if v is not None}
            for p in top_places_data
        ]
        normal_places = [
            {k: v for k, v in {"id": p.get("id"), "category": p.get("category"), "subcategory": p.get("subcategory")}.items() if v is not None}
            for p in other_places_data
        ]
        
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
    services = services_provider()
    status = "healthy" if services else "unhealthy"
    elk_ready = deepctr_ready = "not ready"
    if services:
        elk_ready = "ready"
        deepctr_ready = "ready"

    return {
        "status": status,
        "elk_client": elk_ready,
        "deepctr_service": deepctr_ready,
        "service": "recommendation"
    }