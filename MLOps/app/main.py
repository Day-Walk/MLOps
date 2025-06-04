# app/main.py
from fastapi import FastAPI, HTTPException
from .schema.recommendation_schema import RecommendationRequest, RecommendationResponse, PlaceResult
from .services.elk_client import ELKClient
from .services.deepctr_service import DeepCTRService

app = FastAPI(
    title="MLOps Recommendation API with DeepCTR", 
    version="1.0.0",
    description="DeepCTR 기반 개인화 추천 시스템"
)

# 서비스 초기화
elk_client = ELKClient()
deepctr_service = DeepCTRService()

@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend_places(request: RecommendationRequest):
    """DeepCTR 기반 장소 추천 API"""
    try:
        # 1. ELK 서버에서 장소 검색
        places = await elk_client.search_places(
            query=request.search_query,
            max_results=23
        )
        
        if not places:
            return RecommendationResponse(
                success=False,
                top_3_recommendations=[],
                other_places=[],
                total=0
            )
        
        # 2. DeepCTR 모델로 개인화 추천
        ranked_places = deepctr_service.rank_places_by_ctr(
            user_id=request.user_id,
            places=places
        )
        
        # 3. 상위 3개와 나머지 분리
        top_3 = ranked_places[:3]
        others = ranked_places[3:]
        
        # PlaceResult 객체로 변환
        top_3_results = [PlaceResult(**place) for place in top_3]
        other_results = [PlaceResult(**place) for place in others]
        
        return RecommendationResponse(
            success=True,
            top_3_recommendations=top_3_results,
            other_places=other_results,
            total=len(ranked_places)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users")
async def get_available_users():
    """사용 가능한 사용자 목록 조회"""
    try:
        users = deepctr_service.get_available_users()
        return {"users": users[:10]}  # 상위 10명만 반환
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}")
async def get_user_info(user_id: str):
    """특정 사용자 정보 조회"""
    try:
        user_info = deepctr_service.get_user_info(user_id)
        if user_info is None:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
        return user_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "name": "MLOps Recommendation API with DeepCTR",
        "status": "running",
        "model_loaded": deepctr_service.model is not None,
        "available_users": len(deepctr_service.get_available_users())
    }
