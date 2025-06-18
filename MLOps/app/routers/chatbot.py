"""
Langchain Agent 기반 챗봇 라우터
"""
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import asyncio
import json
import uuid
from datetime import datetime
import traceback
import os

from app.services.langchain_agent_service import LangchainAgentService

router = APIRouter(prefix="/api", tags=["chatbot"])

# Langchain Agent 서비스 초기화
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        langchain_agent_service = None
    else:
        langchain_agent_service = LangchainAgentService(openai_api_key=openai_api_key)
except Exception as e:
    print(f"Error initializing LangchainAgentService: {e}")
    traceback.print_exc()
    langchain_agent_service = None

# 활성 세션을 추적하기 위한 간단한 딕셔너리
active_sessions = {}

@router.get("/chat/stream")
async def chat_stream_endpoint(
    query: str = Query(..., description="사용자 질문"),
    userid: Optional[str] = Query(None, description="사용자 ID (세션 유지를 위해 사용)")
):
    """
    LangChain 에이전트 기반 SSE 스트리밍 챗봇 API
    - GET 방식으로 `userid`와 `query`를 받습니다.
    - 에이전트가 생성한 최종 응답을 JSON 형식으로 스트리밍합니다.
    - 출력 형식: {"placeid": ["장소UUID", ...], "str1": "추천 코스"}
    """
    if not langchain_agent_service:
        async def error_stream():
            error_message = {
                "placeid": [],
                "str1": "죄송합니다. 챗봇 서비스가 준비되지 않았습니다. 서비스 초기화에 실패했습니다."
            }
            yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    session_id = userid or str(uuid.uuid4())
    active_sessions[session_id] = {"start_time": datetime.now(), "query": query}
    print(f"Agent 스트림 요청: {query} (사용자: {session_id})")

    async def generate_agent_stream():
        try:
            # 1. 에이전트를 별도 스레드에서 실행
            agent_response = await asyncio.to_thread(
                langchain_agent_service.get_response,
                user_message=query,
                user_id=session_id
            )

            # 2. 응답을 요청된 형식으로 변환
            # agent_response: {"place_uuids": list, "course": str}
            final_response = {
                "placeid": agent_response.get("place_uuids", []),
                "str1": agent_response.get("course", "추천 코스를 생성하지 못했습니다.")
            }

            # 3. 최종 메시지 전송
            yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"Agent 스트림 처리 오류: {e}")
            traceback.print_exc()
            
            error_message = {
                "placeid": [],
                "str1": f"처리 중 오류가 발생했습니다: {str(e)}"
            }
            yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        
        finally:
            active_sessions.pop(session_id, None)
            print(f"Agent 스트림 종료 (사용자: {session_id})")

    return StreamingResponse(
        generate_agent_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@router.get("/chat/clear")
async def clear_chat_history(
    userid: str = Query(..., description="초기화할 사용자 ID")
):
    """지정된 사용자의 대화 기록을 초기화합니다."""
    if not langchain_agent_service:
        raise HTTPException(status_code=503, detail="Chatbot service is not available.")
    
    try:
        langchain_agent_service.clear_session(userid)
        return {"success": True, "message": f"User {userid}'s chat history has been cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/health")
async def chat_health_check():
    """챗봇 서비스 상태 확인"""
    return {
        "status": "healthy" if langchain_agent_service else "unhealthy",
        "service": "langchain_agent_chatbot",
        "active_sessions": len(active_sessions)
    }
