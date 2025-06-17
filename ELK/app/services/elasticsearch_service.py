from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Tuple
from datetime import datetime


class ElasticsearchService:
    """Elasticsearch 서비스"""
    
    def __init__(self, host: str = "elasticsearch", port: int = 9200):
        self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
        self.index_name = "place_data"
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
    
    def search_places_chatbot(self, query: str, max_results: int = 100):
        """챗봇 장소 검색"""
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
            "_source": ["uuid", "name", "category", "subcategory", "gu", "dong", "ro", "station", "location", "opentime", "breaktime", "closedate", "phone", "alias", "address", "content"]
        }
        
        response = self.es.search(index=self.index_name, body=search_body)
        
        hits = response['hits']['hits']
        places = [
            {
                'uuid': hit['_source']['uuid'],
                'name': hit['_source']['name'], 
                'category': hit['_source']['category'],
                'subcategory': hit['_source']['subcategory'],
                'gu': hit['_source']['gu'],
                'dong': hit['_source']['dong'],
                'ro': hit['_source']['ro'],
                'station': hit['_source']['station'],
                'location': hit['_source']['location'],
                'opentime': hit['_source']['opentime'],
                'breaktime': hit['_source']['breaktime'],
                'closedate': hit['_source']['closedate'],
                'phone': hit['_source']['phone'],
                'alias': hit['_source']['alias'],
                'address': hit['_source']['address'],
                'content': hit['_source']['content']
            }
            for hit in hits
        ]
        return places

    def search_places_for_llm_tool(self, region: str, categories: List[str]) -> Tuple[List[str], int]:
        """
        LLM 도구를 위한 장소 검색.
        지역과 카테고리 정보를 바탕으로 장소 uuid 목록과 총 개수를 반환합니다.
        """
        query_body = {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "size": 10000,
            "_source": ["uuid"],
            "track_total_hits": True
        }

        if region:
            query_body["query"]["bool"]["must"].append({
                "multi_match": {
                    "query": region,
                    "fields": ["gu", "dong", "ro", "station", "address"]
                }
            })

        if categories:
            query_body["query"]["bool"]["filter"].append({
                "bool": {
                    "should": [
                        {"terms": {"category.keyword": categories}},
                        {"terms": {"subcategory.keyword": categories}}
                    ],
                    "minimum_should_match": 1
                }
            })
            
        response = self.es.search(index=self.index_name, body=query_body)
        
        uuids = [hit['_source']['uuid'] for hit in response['hits']['hits']]
        total = response['hits']['total']['value']
        
        return uuids, total

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
                                            "imgUrl": {"type": "keyword"},
                                            "location": {
                                                "lat": {"type": "float"},
                                                "lng": {"type": "float"}
                                            }
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
        
    def search_logs_by_user(self, user_id: str) -> List[Dict]:
        """사용자별 로그 검색 (최근 7일)"""
        try:
            query = {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"userId": user_id}},
                            {
                                "range": {
                                    "createAt": {
                                        "gte": "now-7d/d"
                                    }
                                }
                            }
                        ]
                    }
                },
                "sort": [
                    {"createAt": {"order": "desc"}}
                ]
            }
            
            response = self.es.search(index=self.log_index_name, body=query)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            print(f"로그 검색 오류: {e}")
            return []