"""
MLOps 통합 API - 추천 시스템 + OpenAI 챗봇
기존 추천 시스템과 새로운 OpenAI 챗봇을 하나의 API로 통합
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback
import os
from datetime import datetime

# 라우터 임포트
from app.routers import recommendation, chatbot, crowd

# 환경 변수 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(
    title="MLOps 통합 API",
    version="3.0.0",
    description="추천 시스템 + OpenAI 데이트 코스 챗봇 통합 서비스"
)

# CORS 설정 (백엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 예외 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("=================== 에러 발생 ===================")
    print(f"Request URL: {request.url}")
    print(f"에러 타입: {type(exc).__name__}")
    print(f"에러 메시지: {str(exc)}")
    traceback.print_exc()
    print("=============================================")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

# 라우터 등록
app.include_router(recommendation.router)
app.include_router(chatbot.router)
app.include_router(crowd.router)

@app.get("/")
async def root():
    """API 기본 정보"""
    recommendation_status = "active" if recommendation and hasattr(recommendation, 'deepctr_service') and recommendation.deepctr_service else "inactive"
    chatbot_status = "active" if chatbot and hasattr(chatbot, 'langchain_agent_service') and chatbot.langchain_agent_service else "inactive"
    crowd_status = "active" if crowd and hasattr(crowd, 'router') else "inactive"
    
    return {
        "service": "MLOps 통합 API",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "recommendation": {
                "endpoint": "/api/recommend",
                "description": "ELK + DeepCTR 기반 장소 추천",
                "status": recommendation_status
            },
            "chatbot": {
                "endpoints": {
                    "chat": "/api/chat",
                    "stream": "/api/chat/stream",
                    "stats": "/api/chat/stats"
                },
                "description": "OpenAI GPT 기반 데이트 코스 추천 챗봇",
                "status": chatbot_status
            },
            "crowd": {
                "endpoint": "/api/crowd",
                "description": "혼잡도 예측 API",
                "status": crowd_status
            }
        },
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """전체 서비스 헬스체크"""
    # 추천 시스템 상태 확인
    recommendation_status = "active" if recommendation and hasattr(recommendation, 'deepctr_service') and recommendation.deepctr_service else "inactive"
    
    # 챗봇 상태 확인
    chatbot_status = "active" if chatbot and hasattr(chatbot, 'langchain_agent_service') and chatbot.langchain_agent_service else "inactive"
    
    # 혼잡도 서비스 상태
    crowd_status = "active" if crowd and hasattr(crowd, 'router') else "inactive"

    # 전체 상태 결정
    overall_status = "healthy" if (recommendation_status == "active" or chatbot_status == "active") else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "recommendation": recommendation_status,
            "chatbot": chatbot_status,
            "crowd": crowd_status
        },
        "active_chat_sessions": len(getattr(chatbot, 'active_sessions', {}) if chatbot else {}),
        "version": "3.0.0"
    }

@app.get("/stats")
async def get_overall_stats():
    """전체 서비스 통계"""
    recommendation_status = "active" if recommendation and hasattr(recommendation, 'deepctr_service') and recommendation.deepctr_service else "inactive"
    chatbot_status = "active" if chatbot and hasattr(chatbot, 'langchain_agent_service') and chatbot.langchain_agent_service else "inactive"
    crowd_status = "active" if crowd and hasattr(crowd, 'router') else "inactive"

    return {
        "api_version": "3.0.0",
        "services": {
            "recommendation": {
                "status": recommendation_status,
                "type": "ELK + DeepCTR"
            },
            "chatbot": {
                "status": chatbot_status,
                "type": "OpenAI GPT",
                "active_sessions": len(getattr(chatbot, 'active_sessions', {}) if chatbot else {})
            },
            "crowd": {
                "status": crowd_status,
                "type": "Congestion Prediction"
            }
        },
        "endpoints": {
            "recommendation": ["/api/recommend", "/api/recommend/health"],
            "chatbot": ["/api/chat", "/api/chat/stream", "/api/chat/stats", "/api/chat/health"],
            "crowd": ["/api/crowd"]
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
