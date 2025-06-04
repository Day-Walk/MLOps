from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import traceback
from .schema.recommendation_schema import RecommendationRequest, RecommendationResponse
from .services.elk_client import ELKClient
from .services.deepctr_service import DeepCTRService

app = FastAPI(
    title="MLOps Recommendation API - Simple", 
    version="1.0.0",
    description="Place ID 기반 단순 추천 시스템"
)

# 서비스 초기화
elk_client = ELKClient()
deepctr_service = DeepCTRService()

@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend_places(request: RecommendationRequest):
    """Place ID 리스트 기반 추천 API"""
    try:
        # 1. ELK 서버에서 장소 검색
        places = await elk_client.search_places(
            query=request.search_query,
            max_results=23
        )
        
        if not places:
            return RecommendationResponse(
                success=True,
                top_3_place_ids=[],
                other_place_ids=[],
                total=0
            )
        
        # 2. DeepCTR 모델로 개인화 추천 (순위 매기기)
        ranked_places = deepctr_service.rank_places_by_ctr(
            user_id=request.user_id,
            places=places
        )
        
        # 3. place_id만 추출
        place_ids = []
        for place in ranked_places:
            place_id = place.get('place_id', place.get('HEX(id)', 'unknown'))
            if place_id != 'unknown':
                place_ids.append(place_id)
        
        # 4. 상위 3개와 나머지 분리
        top_3_ids = place_ids[:3]
        other_ids = place_ids[3:]
        
        return RecommendationResponse(
            success=True,
            top_3_place_ids=top_3_ids,
            other_place_ids=other_ids,
            total=len(place_ids)
        )
        
    except Exception as e:
        print(f"추천 API 오류: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"추천 처리 중 오류: {str(e)}")

@app.get("/api/users")
async def get_available_users():
    """사용 가능한 사용자 목록 조회"""
    try:
        users = deepctr_service.get_available_users()
        return {
            "success": True,
            "users": users[:10],
            "total_users": len(users)
        }
    except Exception as e:
        return {
            "success": False,
            "users": [],
            "error": str(e)
        }

@app.get("/api/users/{user_id}")
async def get_user_info(user_id: str):
    """특정 사용자 정보 조회"""
    try:
        user_info = deepctr_service.get_user_info(user_id)
        return {"success": True, "user": user_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"사용자 정보 조회 중 오류: {str(e)}")

@app.get("/")
async def root():
    """API 기본 정보"""
    try:
        available_users = deepctr_service.get_available_users()
        return {
            "name": "MLOps Recommendation API - Simple",
            "status": "running",
            "available_users": len(available_users),
            "sample_users": available_users[:3] if available_users else [],
            "response_format": "place_id_lists_only"
        }
    except Exception as e:
        return {
            "name": "MLOps Recommendation API - Simple",
            "status": "running with errors",
            "error": str(e)
        }
