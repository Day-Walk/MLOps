"""
OpenAI 연결 챗봇 라우터
실제 GPT API와 연결, GET 방식 + SSE 스트리밍 지원
"""
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import asyncio
import json
import uuid
from datetime import datetime
import traceback
import os

from app.services.openai_service import OpenAIService
from app.services.place_extractor import PlaceExtractor
from app.schema.chat_schema import ChatResponse, ChatStats

router = APIRouter(prefix="/api", tags=["chatbot"])

# 서비스 초기화 (기존과 동일)
openai_service = OpenAIService(os.getenv("OPENAI_API_KEY"))
place_extractor = PlaceExtractor()
active_sessions = {}

@router.get("/chat", response_model=ChatResponse)
async def chat_endpoint(
    message: str = Query(..., description="사용자 메시지"),
    session_id: Optional[str] = Query(None, description="세션 ID")
):
    """
    OpenAI 챗봇 API (GET 방식)
    응답: {"str1": "GPT 답변", "placeid": ["uuid1", "uuid2"], "str2": null}
    """
    if not openai_service:
        return JSONResponse(
            status_code=503,
            content={
                "str1": "죄송합니다. 챗봇 서비스가 준비되지 않았습니다.",
                "placeid": None,
                "str2": None
            }
        )
    
    try:
        session_id = session_id or str(uuid.uuid4())
        active_sessions[session_id] = {
            "start_time": datetime.now(), 
            "type": "sync"
        }
        
        print(f"📝 채팅 요청: {message} (세션: {session_id})")
        print(f"🔄 활성 세션 수: {len(active_sessions)}")
        
        # OpenAI API 호출
        gpt_response = await openai_service.get_chat_completion(message)
        
        # 장소 ID 추출
        place_ids = place_extractor.extract_place_ids_from_text(gpt_response)
        
        # 커스텀 응답 형식
        response = {
            "str1": gpt_response,
            "placeid": place_ids if place_ids else None,
            "str2": None
        }
        
        # 세션 정리
        active_sessions.pop(session_id, None)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"❌ 채팅 처리 오류: {e}")
        active_sessions.pop(session_id, None)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "str1": "죄송합니다. 일시적인 오류가 발생했습니다.",
                "placeid": None,
                "str2": None
            }
        )

@router.get("/chat/stream")
async def chat_stream_endpoint(
    message: str = Query(..., description="사용자 메시지"),
    session_id: Optional[str] = Query(None, description="세션 ID")
):

    """
    정해진 형식 SSE 스트리밍 챗봇 API
    완성된 메시지를 {"str1": "", "placeid": [], "str2": ""} 형식으로 전송
    """
    session_id = session_id or str(uuid.uuid4())
    active_sessions[session_id] = {
        "start_time": datetime.now(), 
        "type": "structured_stream"
    }
    
    print(f"🔄 구조화 스트림 요청: {message} (세션: {session_id})")
    
    async def generate_structured_stream():
        try:
            # 1. OpenAI에서 완전한 응답 받기 (내부적으로는 일반 호출)
            gpt_response = await openai_service.get_chat_completion(message)
            
            # 2. 장소 ID 추출
            place_ids = place_extractor.extract_place_ids_from_text(gpt_response)
            
            # 3. 시작 메시지 (처리 시작 알림)
            start_message = {
                "type": "start",
                "str1": "답변을 준비하고 있습니다...",
                "placeid": None,
                "str2": None,
                "session_id": session_id
            }
            yield f"data: {json.dumps(start_message, ensure_ascii=False)}\n\n"
            
            # 4. 약간의 지연 (실제 처리 시간 시뮬레이션)
            await asyncio.sleep(1)
            
            # 5. 진행 상황 메시지 (선택사항)
            progress_message = {
                "type": "progress", 
                "str1": "장소 정보를 분석 중입니다...",
                "placeid": None,
                "str2": None,
                "session_id": session_id
            }
            yield f"data: {json.dumps(progress_message, ensure_ascii=False)}\n\n"
            
            await asyncio.sleep(1)
            
            # 6. 최종 완성된 응답 (정해진 형식)
            final_message = {
                "type": "complete",
                "str1": gpt_response,
                "placeid": place_ids if place_ids else None,
                "str2": None,  # 필요시 추가 정보
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(final_message, ensure_ascii=False)}\n\n"
            
            # 7. 스트림 종료 신호
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"❌ 구조화 스트림 처리 오류: {e}")
            error_message = {
                "type": "error",
                "str1": f"처리 중 오류가 발생했습니다: {str(e)}",
                "placeid": None,
                "str2": None,
                "session_id": session_id
            }
            yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        
        finally:
            # 세션 정리
            active_sessions.pop(session_id, None)
    
    return StreamingResponse(
        generate_structured_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Session-ID": session_id,
            "X-Stream-Type": "structured"
        }
    )
@router.get("/chat/clear")
async def clear_chat(session_id: str = Query(..., description="초기화할 세션 ID")):
    """대화 기록 초기화 (GET 요청)"""
    try:
        # 활성 세션에서 제거
        removed = active_sessions.pop(session_id, None)
        
        return {
            "success": True, 
            "message": f"세션 {session_id} 대화가 초기화되었습니다.",
            "was_active": removed is not None
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/health")
async def chat_health_check():
    """챗봇 헬스체크"""
    try:
        if openai_service:
            openai_connected = await openai_service.test_connection()
        else:
            openai_connected = False
        
        return {
            "status": "healthy" if openai_connected else "unhealthy",
            "openai_connected": openai_connected,
            "service": "chatbot",
            "active_sessions": len(active_sessions)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "openai_connected": False
        }

@router.get("/chat/stats")
async def get_chat_stats():
    """실시간 통계 (동시 접속 모니터링)"""
    return {
        "active_sessions": len(active_sessions),
        "sessions_detail": {
            session_id: {
                "start_time": session_data["start_time"].isoformat(),
                "type": session_data["type"],
                "duration_seconds": (datetime.now() - session_data["start_time"]).total_seconds()
            }
            for session_id, session_data in active_sessions.items()
        },
        "total_sync_sessions": len([s for s in active_sessions.values() if s["type"] == "sync"]),
        "total_stream_sessions": len([s for s in active_sessions.values() if s["type"] == "stream"])
    }
