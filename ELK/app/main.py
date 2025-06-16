from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from datetime import datetime

from app.schema.search_schemas import SearchResponse
from app.services.elasticsearch_service import ElasticsearchService
from app.schema.log_schemas import LogRequest, LogResponse

# Elasticsearch 서비스 초기화
elasticsearch_service = ElasticsearchService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작 시 실행"""
    elasticsearch_service.create_log_index_if_not_exists()
    yield

app = FastAPI(title="ELK Search API", version="1.0.0", lifespan=lifespan)

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
    
@app.post("/api/chatbot")
async def insert_chatbot_log(log_data: LogRequest):
    """챗봇 로그 삽입"""
    try:
        if not elasticsearch_service.is_connected():
            return LogResponse(
                isSuccess=False,
                message="Elasticsearch 연결 실패"
            )
        
        log_dict = log_data.dict()
        log_dict['createAt'] = datetime.now().isoformat()
        success = elasticsearch_service.insert_chatbot_log(log_dict)
        
        if success:
            return LogResponse(
                isSuccess=True,
                message="로그 삽입 완료"
            )
        else:
            return LogResponse(
                isSuccess=False,
                message="로그 삽입 실패"
            )
            
    except Exception as e:
        print(f"챗봇 로그 삽입 오류: {e}")
        return LogResponse(
            isSuccess=False,
            message=str(e)
        )
        
@app.get("/api/chatbot/log/{user_id}")
async def get_chatbot_log(user_id: str):
    """챗봇 로그 조회"""
    try:
        if not elasticsearch_service.is_connected():
            raise HTTPException(status_code=503, detail="Elasticsearch 연결 실패")
        
        logs = elasticsearch_service.search_logs_by_user(user_id)
        return logs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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