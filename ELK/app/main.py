from fastapi import FastAPI, HTTPException
from schema.search_schemas import SearchResponse
from services.elasticsearch_service import ElasticsearchService

app = FastAPI(title="ELK Search API", version="1.0.0")

# Elasticsearch 서비스 초기화
elasticsearch_service = ElasticsearchService()

@app.get("/api/place/search", response_model=SearchResponse)
async def search_places(query: str, max_results: int = 23):
    """장소 검색 API"""
    try:
        if not elasticsearch_service.is_connected():
            raise HTTPException(status_code=503, detail="Elasticsearch 연결 실패")
        
        places = elasticsearch_service.search_places(
            query=query,
            max_results=max_results
        )
        
        return SearchResponse(
            success=True,
            places=places,
            total=len(places)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"name": "ELK Search API", "status": "running"}

@app.get("/health")
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