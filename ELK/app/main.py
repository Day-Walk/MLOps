from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from app.schema.search_schemas import SearchResponse, LLMToolResponse, Place
from app.services.elasticsearch_service import ElasticsearchService
from app.schema.log_schemas import LogRequest, LogResponse, ClickLogRequest, ClickLogResponse

# Elasticsearch 서비스 초기화
elasticsearch_service = ElasticsearchService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작 시 실행"""
    elasticsearch_service.create_log_index_if_not_exists()
    elasticsearch_service.create_click_log_index_if_not_exists()
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
        
        # 딕셔너리를 Place 객체로 변환
        place_objects = [Place(**place) for place in places]
        
        return SearchResponse(
            success=True,
            places=place_objects,
            total=len(place_objects)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/place/search/llm-tool", response_model=LLMToolResponse)
async def search_places_for_llm_tool(region: str, categories: List[str] = Query(..., min_length=1, max_length=3)):
    """LLM 도구를 위한 장소 검색 API"""
    try:
        if not elasticsearch_service.is_connected():
            raise HTTPException(status_code=503, detail="Elasticsearch 연결 실패")
        
        uuids, total = elasticsearch_service.search_places_for_llm_tool(
            region=region,
            categories=categories
        )
        
        return LLMToolResponse(
            success=True,
            uuids=uuids,
            total=total
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

@app.post("/api/click-log", response_model=ClickLogResponse)
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

@app.get("/api/click-log/user/{user_id}")
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

@app.get("/api/click-log/place/{place_id}/count")
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