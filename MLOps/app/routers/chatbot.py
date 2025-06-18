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
from app.schema.chat_schema import ChatResponse, ChatStats, AgentChatRequest
from app.services.chatbot_agent_service import ChatbotAgentService

router = APIRouter(prefix="/api", tags=["chatbot"])

# 서비스 초기화
openai_service = OpenAIService(os.getenv("OPENAI_API_KEY"))
place_extractor = PlaceExtractor()
# Langchain Agent 서비스 초기화
try:
    chatbot_agent_service = ChatbotAgentService()
except Exception as e:
    print(f"Error initializing ChatbotAgentService: {e}")
    chatbot_agent_service = None

active_sessions = {}

@router.post("/chat/agent")
async def chat_agent_endpoint(request: AgentChatRequest):
    """
    LangChain 에이전트 기반 챗봇 API
    """
    if not chatbot_agent_service:
        raise HTTPException(status_code=503, detail="Chatbot agent service is not available.")

    try:
        current_session_id = request.session_id or str(uuid.uuid4())
        print(f"Agent chat request: {request.message} (Session: {current_session_id})")
        
        # 에이전트 호출
        agent_response = await asyncio.to_thread(
            chatbot_agent_service.get_response, 
            user_message=request.message, 
            session_id=current_session_id
        )

        return JSONResponse(content={"session_id": current_session_id, "response": agent_response})

    except Exception as e:
        print(f"Agent chat processing error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred.")

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
        
        print(f"채팅 요청: {message} (세션: {session_id})")
        print(f"활성 세션 수: {len(active_sessions)}")
        
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
        print(f"채팅 처리 오류: {e}")
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
    userid: Optional[str] = Query(None, description="사용자 ID")
):
    """
    LangChain 에이전트 기반 SSE 스트리밍 챗봇 API
    완성된 메시지를 {"str1": "", "placeid": [], "str2": ""} 형식으로 전송
    """
    if not chatbot_agent_service:
        async def error_stream():
            error_message = {
                "type": "error",
                "str1": "죄송합니다. 챗봇 서비스가 준비되지 않았습니다.",
                "placeid": None, "str2": None, "userid": userid
            }
            yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    session_id = userid or str(uuid.uuid4())
    active_sessions[session_id] = {"start_time": datetime.now(), "type": "agent_stream"}
    print(f"Agent 스트림 요청: {message} (사용자: {session_id})")

    async def generate_agent_stream():
        try:
            # 1. 에이전트를 별도 스레드에서 실행
            agent_response = await asyncio.to_thread(
                chatbot_agent_service.get_response,
                user_message=message,
                session_id=session_id
            )
            
            # 2. 결과 추출
            final_text = agent_response.get("final_answer", "")
            place_ids = agent_response.get("place_uuids", [])
            
            # 3. 최종 메시지 포맷팅 및 전송
            final_message = {
                "type": "complete",
                "str1": final_text,
                "placeid": place_ids if place_ids else None,
                "str2": None,
                "userid": session_id,
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(final_message, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Agent 스트림 처리 오류: {e}")
            traceback.print_exc()
            error_message = {
                "type": "error",
                "str1": f"처리 중 오류가 발생했습니다: {str(e)}",
                "placeid": None, "str2": None, "userid": session_id
            }
            yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        
        finally:
            active_sessions.pop(session_id, None)
    
    return StreamingResponse(
        generate_agent_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-User-ID": session_id,
            "X-Stream-Type": "structured_agent"
        }
    )

@router.get("/chat/clear")
async def clear_chat(session_id: str = Query(..., description="초기화할 세션 ID")):
    """대화 기록 초기화 (GET 요청)"""
    try:
        # 에이전트 세션 기록 삭제
        if chatbot_agent_service:
            chatbot_agent_service.clear_session(session_id)
        
        # 활성 세션 추적기에서 제거
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
