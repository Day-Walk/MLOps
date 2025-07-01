from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.services.elasticsearch_service import ElasticsearchService
from app.routers import place_router, chatbot_router, click_log_router, training_router, health_router

# Elasticsearch 서비스 초기화
elasticsearch_service = ElasticsearchService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작 시 실행"""
    elasticsearch_service.create_log_index_if_not_exists()
    elasticsearch_service.create_click_log_index_if_not_exists()
    elasticsearch_service.create_search_log_index_if_not_exists()
    yield

app = FastAPI(title="ELK Search API", version="1.0.0", lifespan=lifespan)

# 라우터 등록
app.include_router(place_router.router)
app.include_router(chatbot_router.router)
app.include_router(click_log_router.router)
app.include_router(training_router.router)
app.include_router(health_router.router)