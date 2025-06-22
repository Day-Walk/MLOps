import pandas as pd
from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone


class ElasticsearchService:
    """Elasticsearch 서비스"""
    
    def __init__(self, host: str = "elasticsearch", port: int = 9200):
        self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
        self.index_name = "place_data"
        self.log_index_name = "chatbot_log"
        self.click_log_index_name = "click_log"
        self.search_log_index_name = "search_log"
        
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        try:
            return self.es.ping()
        except:
            return False

    def search_places(self, query: str, max_results: int = 23, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
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
        
        if user_id:
            place_ids = [place['uuid'] for place in places]
            log_data = {
                "userId": user_id,
                "query": query,
                "placeIds": place_ids,
                "timestamp": datetime.now(timezone.utc)
            }
            self.insert_search_log(log_data)
            
        return places

    def search_places_for_llm_tool(self, region: str, categories: List[str], user_id: Optional[str] = None) -> Tuple[List[str], int]:
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
            "size": 100,
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
                    {"createAt": {"order": "asc"}}
                ]
            }
            
            response = self.es.search(index=self.log_index_name, body=query)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            print(f"로그 검색 오류: {e}")
            return []

    def create_click_log_index_if_not_exists(self):
        """클릭 로그 인덱스가 없으면 생성"""
        try:
            if not self.es.indices.exists(index=self.click_log_index_name):
                # 클릭 로그 인덱스 매핑 설정
                mapping = {
                    "mappings": {
                        "properties": {
                            "userId": {"type": "keyword"},
                            "placeId": {"type": "keyword"},
                            "timestamp": {"type": "date"}
                        }
                    }
                }
                self.es.indices.create(index=self.click_log_index_name, body=mapping)
                print(f"클릭 로그 인덱스 '{self.click_log_index_name}' 생성 완료")
        except Exception as e:
            raise Exception(f"클릭 로그 인덱스 생성 오류: {e}")

    def insert_click_log(self, log_data: dict) -> Tuple[bool, Optional[str]]:
        """클릭 로그 데이터를 Elasticsearch에 삽입"""
        try:
            # 문서 ID 생성 (타임스탬프 + userId + placeId 조합)
            doc_id = f"{log_data['userId']}_{log_data['placeId']}_{int(datetime.now().timestamp())}"
            
            # Elasticsearch에 문서 삽입
            response = self.es.index(
                index=self.click_log_index_name,
                id=doc_id,
                body=log_data
            )
            
            # 삽입 성공 여부 확인
            if response.get('result') in ['created', 'updated']:
                return True, doc_id
            else:
                return False, None
                
        except Exception as e:
            print(f"클릭 로그 삽입 오류: {e}")
            return False, None

    def get_click_logs_by_user(self, user_id: str, days: int = 30) -> List[Dict]:
        """사용자의 클릭 로그 조회"""
        try:
            query = {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"userId": user_id}},
                            {
                                "range": {
                                    "timestamp": {
                                        "gte": f"now-{days}d/d"
                                    }
                                }
                            }
                        ]
                    }
                },
                "sort": [
                    {"timestamp": {"order": "desc"}}
                ]
            }
            
            response = self.es.search(index=self.click_log_index_name, body=query)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            print(f"클릭 로그 검색 오류: {e}")
            return []

    def get_click_count_by_place(self, place_id: str) -> int:
        """특정 장소의 클릭 수 조회"""
        try:
            query = {
                "query": {
                    "term": {"placeId": place_id}
                }
            }
            
            response = self.es.count(index=self.click_log_index_name, body=query)
            return response.get('count', 0)
        except Exception as e:
            print(f"클릭 수 조회 오류: {e}")
            return 0

    def create_search_log_index_if_not_exists(self):
        """검색 로그 인덱스가 없으면 생성"""
        try:
            if not self.es.indices.exists(index=self.search_log_index_name):
                # 검색 로그 인덱스 매핑 설정
                mapping = {
                    "mappings": {
                        "properties": {
                            "userId": {"type": "keyword"},
                            "query": {"type": "text"},
                            "placeIds": {"type": "keyword"},
                            "timestamp": {"type": "date"}
                        }
                    }
                }
                self.es.indices.create(index=self.search_log_index_name, body=mapping)
                print(f"검색 로그 인덱스 '{self.search_log_index_name}' 생성 완료")
        except Exception as e:
            raise Exception(f"검색 로그 인덱스 생성 오류: {e}")

    def insert_search_log(self, log_data: dict) -> Tuple[bool, Optional[str]]:
        """검색 로그 데이터를 Elasticsearch에 삽입"""
        try:
            # 문서 ID 생성 (타임스탬프 + userId 조합)
            doc_id = f"{log_data['userId']}_{int(datetime.now().timestamp())}"
            
            # Elasticsearch에 문서 삽입
            response = self.es.index(
                index=self.search_log_index_name,
                id=doc_id,
                body=log_data
            )
            
            # 삽입 성공 여부 확인
            if response.get('result') in ['created', 'updated']:
                return True, doc_id
            else:
                return False, None
                
        except Exception as e:
            print(f"검색 로그 삽입 오류: {e}")
            return False, None

    def get_all_search_logs_by_user(self, user_id: str) -> List[Dict]:
        """사용자의 모든 검색 로그를 시간순으로 조회"""
        try:
            query = {
                "query": {"term": {"userId": user_id}},
                "sort": [{"timestamp": {"order": "asc"}}],
                "size": 1000
            }
            response = self.es.search(index=self.search_log_index_name, body=query)
            return response["hits"]["hits"]
        except Exception as e:
            print(f"사용자 검색 로그 조회 오류: {e}")
            return []

    def get_all_click_logs_by_user(self, user_id: str) -> List[Dict]:
        """사용자의 모든 클릭 로그를 시간순으로 조회"""
        try:
            query = {
                "query": {"term": {"userId": user_id}},
                "sort": [{"timestamp": {"order": "asc"}}],
                "size": 10000
            }
            response = self.es.search(index=self.click_log_index_name, body=query)
            return response["hits"]["hits"]
        except Exception as e:
            print(f"사용자 클릭 로그 조회 오류: {e}")
            return []

    def get_search_click_data_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """사용자 검색 및 클릭 데이터를 기반으로 학습 데이터 생성"""
        
        def _ensure_utc_aware(dt_str: str) -> datetime:
            """datetime 문자열을 UTC-aware datetime 객체로 변환"""
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                # Naive datetime은 UTC로 간주
                return dt.replace(tzinfo=timezone.utc)
            return dt

        search_logs = self.get_all_search_logs_by_user(user_id)
        click_logs = self.get_all_click_logs_by_user(user_id)

        if not search_logs:
            return []

        training_data = []
        click_iterator = iter(click_logs)
        current_click = next(click_iterator, None)

        for i, search_hit in enumerate(search_logs):
            search_log = search_hit["_source"]
            search_time = _ensure_utc_aware(search_log["timestamp"])

            next_search_time = datetime.now(timezone.utc)
            if i + 1 < len(search_logs):
                next_search_log = search_logs[i + 1]["_source"]
                next_search_time = _ensure_utc_aware(next_search_log["timestamp"])

            clicked_in_window = set()
            while current_click:
                click_log = current_click["_source"]
                click_time = _ensure_utc_aware(click_log["timestamp"])

                if search_time <= click_time < next_search_time:
                    clicked_in_window.add(click_log["placeId"])
                    current_click = next(click_iterator, None)
                elif click_time >= next_search_time:
                    break
                else: # click_time < search_time
                    current_click = next(click_iterator, None)

            for place_id in search_log.get("placeIds", []):
                training_data.append({
                    "userid": user_id,
                    "place_id": place_id,
                    "yn": 1 if place_id in clicked_in_window else 0
                })

        return training_data