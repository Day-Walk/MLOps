from fastapi import APIRouter, HTTPException

from app.services.elasticsearch_service import ElasticsearchService

router = APIRouter(prefix="/api/training-data", tags=["학습 데이터"])

elasticsearch_service = ElasticsearchService()

@router.get("/{user_id}")
async def get_training_data(user_id: str):
    """DeepCTR 모델 학습을 위한 데이터 생성 API"""
    try:
        if not elasticsearch_service.is_connected():
            raise HTTPException(status_code=503, detail="Elasticsearch 연결 실패")
        
        data = elasticsearch_service.get_search_click_data_for_user(user_id)
        
        return {
            "success": True,
            "userId": user_id,
            "data": data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 