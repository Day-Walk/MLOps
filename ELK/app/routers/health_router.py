from fastapi import APIRouter

from app.services.elasticsearch_service import ElasticsearchService

router = APIRouter(tags=["health-check"])

elasticsearch_service = ElasticsearchService()

@router.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    try:
        es_connected = elasticsearch_service.is_connected()
        return {
            "status": "healthy" if es_connected else "unhealthy",
            "elasticsearch": "connected" if es_connected else "disconnected"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)} 