# ELK/app/main.py
from fastapi import FastAPI, HTTPException
from .schema.search_schemas import SearchRequest, SearchResponse, PlaceInfo
from .services.elasticsearch_service import ElasticsearchService

app = FastAPI(title="ELK Search API", version="1.0.0")

# Elasticsearch 서비스 초기화
elasticsearch_service = ElasticsearchService()

@app.post("/api/search", response_model=SearchResponse)
async def search_places(request: SearchRequest):
    """장소 검색 API"""
    try:
        if not elasticsearch_service.is_connected():
            raise HTTPException(status_code=503, detail="Elasticsearch 연결 실패")
        
        places = await elasticsearch_service.search_places(
            query=request.query,
            max_results=request.max_results
        )
        
        # PlaceInfo 객체로 변환
        place_objects = []
        for place_data in places:
            place_data['HEX(id)'] = place_data.pop('place_id', '')
            place_obj = PlaceInfo(**place_data)
            place_objects.append(place_obj)
        
        return SearchResponse(
            success=True,
            places=place_objects,
            total=len(place_objects)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"name": "ELK Search API", "status": "running"}
