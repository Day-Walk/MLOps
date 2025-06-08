from elasticsearch import Elasticsearch
from typing import List, Dict, Any

class ElasticsearchService:
    """Elasticsearch 서비스"""
    
    def __init__(self, host: str = "54.180.24.146", port: int = 9200):
        self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
        self.index_name = "place_data_v3"

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        try:
            return self.es.ping()
        except:
            return False

    async def search_places(self, query: str, max_results: int = 23) -> List[str]:
        """장소 검색"""
        search_body = {
            "query": {
                "match": {
                    "name": {
                        "query": query,
                        "fuzziness": "AUTO"
                    }
                }
            },
            "sort": [{"_score": {"order": "desc"}}],
            "size": max_results,
            "_source": ["place_id", "name", "category", "sub_category"]
        }
        
        response = self.es.search(index=self.index_name, body=search_body)
        
        hits = response['hits']['hits']
        places = [
            {
                'place_id': hit['_source']['place_id'],
                'name': hit['_source']['name'], 
                'category': hit['_source']['category'],
                'sub_category': hit['_source']['sub_category']
            }
            for hit in hits
        ]
        
        return places
