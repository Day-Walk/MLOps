from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone, timedelta

from app.schema.log_schemas import LogRequest, LogResponse
from app.services.elasticsearch_service import ElasticsearchService

router = APIRouter(prefix="/api/chatbot", tags=["챗봇"])

elasticsearch_service = ElasticsearchService()

@router.post("", response_model=LogResponse)
async def insert_chatbot_log(log_data: LogRequest):
    """챗봇 로그 삽입"""
    try:
        if not elasticsearch_service.is_connected():
            return LogResponse(
                isSuccess=False,
                message="Elasticsearch 연결 실패"
            )
        
        log_dict = log_data.dict()
        log_dict['createAt'] = datetime.now(timezone(timedelta(hours=9))).isoformat()
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

@router.get("/log/{user_id}")
async def get_chatbot_log(user_id: str):
    """챗봇 로그 조회"""
    try:
        if not elasticsearch_service.is_connected():
            raise HTTPException(status_code=503, detail="Elasticsearch 연결 실패")
        
        logs = elasticsearch_service.search_logs_by_user(user_id)
        return logs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 