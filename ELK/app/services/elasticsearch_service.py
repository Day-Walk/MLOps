from elasticsearch import Elasticsearch
from typing import List, Dict, Any
from datetime import datetime


class ElasticsearchService:
    """Elasticsearch 서비스"""
    
    def __init__(self, host: str = "elasticsearch", port: int = 9200):
        self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
        self.index_name = "place_data_v3"
        self.log_index_name = "chatbot_log"
        
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        try:
            return self.es.ping()
        except:
            return False

    def search_places(self, query: str, max_results: int = 23) -> List[str]:
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
            "_source": ["uuid", "name", "category", "subcategory"]
        }
        
        response = self.es.search(index=self.index_name, body=search_body)
        
        hits = response['hits']['hits']
        places = [
            {
                'uuid': hit['_source']['uuid'],
                'name': hit['_source']['name'], 
                'category': hit['_source']['category'],
                'subcategory': hit['_source']['subcategory']
            }
            for hit in hits
        ]
        
        return places

    def create_log_index_if_not_exists(self):
        """로그 인덱스가 없으면 생성"""
        try:
            if not self.es.indices.exists(index=self.log_index_name):
                # 로그 인덱스 매핑 설정
                mapping = {
                    "mappings": {
                        "properties": {
                                "userId": {"type": "keyword"},
                                "question": {"type": "text", "analyzer": "nori"},
                                "answer": {
                                    "properties": {
                                        "title": {"type": "text"},
                                        "placeList": {
                                            "properties": {
                                                "placeId": {"type": "keyword"},
                                                "name": {"type": "text", "analyzer": "nori"},
                                                "address": {"type": "text", "analyzer": "nori"},
                                                "imgUrl": {"type": "keyword"}
                                            }
                                        },
                                        "detail": {"type": "text", "analyzer": "nori"}
                                    }
                                },
                                "createAt": {"type": "date"}
                            }
                        }
                    }
                self.es.indices.create(index=self.log_index_name, body=mapping)
                print(f"로그 인덱스 '{self.log_index_name}' 생성 완료")
        except Exception as e:
            raise Exception(f"로그 인덱스 생성 오류: {e}")

    def insert_chatbot_log(self, log_data: dict) -> bool:
        """챗봇 로그 데이터를 Elasticsearch에 삽입"""
        try:
            # 문서 ID 생성 (타임스탬프 + userId 조합)
            doc_id = f"{log_data['userId']}_{int(datetime.now().timestamp())}"
            
            # Elasticsearch에 문서 삽입
            response = self.es.index(
                index=self.log_index_name,
                id=doc_id,
                body=log_data
            )
            
            # 삽입 성공 여부 확인
            if response.get('result') in ['created', 'updated']:
                return True
            else:
                return False
                
        except Exception as e:
            print(f"로그 삽입 오류: {e}")
            return False
        
    def search_logs_by_user(self, user_id: str, limit: int = 50) -> List[Dict]:
        """사용자별 로그 검색"""
        try:
            query = {
                "query": {
                    "term": {
                        "userId": user_id
                    }
                },
                "sort": [
                    {"createAt": {"order": "desc"}}
                ],
                "size": limit
            }
            
            response = self.es.search(index=self.log_index_name, body=query)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            print(f"로그 검색 오류: {e}")
            return []