from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import traceback
from .schema.recommendation_schema import ReccomendRequest, ReccomendResponse
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

@app.get("/api/recommend", response_model=ReccomendResponse)
async def recommend_places(request: ReccomendRequest):
    """Place ID 리스트 기반 추천 API"""
    try:
        top_3_place_ids, other_places = deepctr_service.rank_places_by_ctr(request.user_id, request.query)
        return JSONResponse(content={"top_3_place_ids": top_3_place_ids, "other_places": other_places})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))