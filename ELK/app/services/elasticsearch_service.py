# ELK/app/services/elasticsearch_service.py
from elasticsearch import Elasticsearch
from typing import List, Dict, Any

class ElasticsearchService:
    """Elasticsearch 서비스"""
    
    def __init__(self, host: str = "elasticsearch", port: int = 9200):
        self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
        self.index_name = "place_data_v3"

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        try:
            return self.es.ping()
        except:
            return False

    async def search_places(self, query: str, max_results: int = 23) -> List[Dict[str, Any]]:
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
            "_source": ["HEX(id)", "category", "sub_category", "user_id"]
        }
        
        response = self.es.search(index=self.index_name, body=search_body)
        
        places = []
        for hit in response['hits']['hits']:
            place = {
                'place_id': hit['_source'].get('HEX(id)', ''),
                'category': hit['_source'].get('category', ''),
                'sub_category': hit['_source'].get('sub_category', ''),
                'user_id': hit['_source'].get('user_id'),
                'search_score': hit['_score']
            }
            places.append(place)
        
        return places
