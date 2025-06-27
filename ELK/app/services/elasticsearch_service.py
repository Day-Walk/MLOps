import pandas as pd
from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
import os

load_dotenv()

def _ensure_utc_aware(dt_str: str) -> datetime:
    """datetime 문자열을 UTC-aware datetime 객체로 변환"""
    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def _process_user_log_data(args: Tuple[str, List[Dict], List[Dict]]) -> List[Dict]:
    """단일 사용자의 검색 및 클릭 로그를 처리하여 학습 데이터를 생성합니다."""
    user_id, user_search_logs, user_click_logs = args
    
    training_data = []
    click_iterator = iter(user_click_logs)
    current_click = next(click_iterator, None)

    for i, search_log in enumerate(user_search_logs):
        search_time = _ensure_utc_aware(search_log["timestamp"])

        next_search_time = datetime.now(timezone.utc)
        if i + 1 < len(user_search_logs):
            next_search_log = user_search_logs[i + 1]
            next_search_time = _ensure_utc_aware(next_search_log["timestamp"])

        clicked_in_window = set()
        while current_click:
            click_time = _ensure_utc_aware(current_click["timestamp"])

            if search_time <= click_time < next_search_time:
                clicked_in_window.add(current_click["placeId"])
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

class ElasticsearchService:
    """Elasticsearch 서비스"""
    def __init__(self):
        host = os.getenv('ES_HOST')
        port_str = os.getenv('ES_PORT')
        
        if not host or not port_str:
            raise ValueError("환경 변수 ES_HOST와 ES_PORT가 설정되어야 합니다.")
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError("ES_PORT 환경 변수는 반드시 숫자여야 합니다.")

        self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
        self.index_name = "place_data_v2"
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
        """장소 검색 (copy_to 필드와 multi_match로 고도화)"""
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "name^2",
                        "alias^2",
                        "categories^2",
                        "addresses^2",
                        "content"
                    ],
                    "fuzziness": "AUTO"
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
                "timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat()
            }
            self.insert_search_log(log_data)
            
        return places

    def search_places_for_llm_tool(self, region: str, categories: List[str], user_id: Optional[str] = None) -> Tuple[List[str], int]:
        """
        LLM 도구를 위한 장소 검색.
        지역과 카테고리 정보를 바탕으로 장소 uuid 목록과 총 개수를 반환합니다.
        """
        # 1. bool 쿼리를 기본 쿼리로 정의
        base_query = {
            "bool": {
                "must": [],
                "filter": []
            }
        }

        if region:
            base_query["bool"]["must"].append({
                "multi_match": {
                    "query": region,
                    "fields": ["gu", "dong", "ro", "station", "address"]
                }
            })

        if categories:
            base_query["bool"]["filter"].append({
                "bool": {
                    "should": [
                        {"terms": {"category.keyword": categories}},
                        {"terms": {"subcategory.keyword": categories}}
                    ],
                    "minimum_should_match": 1
                }
            })
            
        # 2. function_score 쿼리로 기본 쿼리를 감싸고, random_score 함수를 추가
        query_body = {
            "query": {
                "function_score": {
                    "query": base_query,
                    "functions": [
                        {
                            "random_score": {}
                        }
                    ],
                    "boost_mode": "multiply" # 원래 점수와 랜덤 점수를 곱하여 자연스럽게 섞음
                }
            },
            "size": 100,
            "_source": ["uuid"],
            "track_total_hits": True
        }
            
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
            doc_id = f"{log_data['userId']}_{int(datetime.now(timezone(timedelta(hours=9))).timestamp())}"
            
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
                ],
                "size": 10000  # Elasticsearch의 기본 최대 결과창 크기(10000)로 설정
            }

            response = self.es.search(index=self.log_index_name, body=query)
            return [hit['_source'] for hit in response['hits']['hits']]

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
            doc_id = f"{log_data['userId']}_{log_data['placeId']}_{int(datetime.now(timezone(timedelta(hours=9))).timestamp())}"
            
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

    def get_most_clicked_places_today(self) -> List[Dict[str, Any]]:
        """
        오늘/어제 새벽 5시부터 현재까지 가장 많이 클릭된 장소 4개의 UUID와 클릭 수를 반환합니다.
        """
        try:
            # KST 시간대 정의
            kst = timezone(timedelta(hours=9))
            
            # 현재 KST 시간
            now_kst = datetime.now(kst)
            
            # 오늘 새벽 5시 KST
            today_5am_kst = now_kst.replace(hour=5, minute=0, second=0, microsecond=0)

            # 만약 현재 시간이 오늘 새벽 5시 이전이라면, 시작 시간은 어제 새벽 5시로 설정
            if now_kst < today_5am_kst:
                start_time_kst = today_5am_kst - timedelta(days=1)
            else:  # 그렇지 않다면, 시작 시간은 오늘 새벽 5시
                start_time_kst = today_5am_kst

            query = {
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "range": {
                                    "timestamp": {
                                        "gte": start_time_kst.isoformat(),
                                        "lte": now_kst.isoformat()
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": 0,
                "aggs": {
                    "top_places": {
                        "terms": {
                            "field": "placeId",
                            "size": 4,
                            "order": {
                                "_count": "desc"
                            }
                        }
                    }
                }
            }
            
            response = self.es.search(index=self.click_log_index_name, body=query)
            
            buckets = response.get('aggregations', {}).get('top_places', {}).get('buckets', [])
            
            result = [
                {"uuid": bucket['key'], "clicks": bucket['doc_count']} 
                for bucket in buckets
            ]
            
            return result
            
        except Exception as e:
            print(f"가장 많이 클릭된 장소 조회 오류: {e}")
            return []

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
            doc_id = f"{log_data['userId']}_{int(datetime.now(timezone(timedelta(hours=9))).timestamp())}"
            
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

    def _get_all_logs(self, index_name: str) -> List[Dict]:
        """Scroll API를 사용하여 인덱스의 모든 문서를 가져옵니다."""
        logs = []
        try:
            # 검색 시점에 timestamp 기준 오름차순 정렬
            scroll = self.es.search(
                index=index_name,
                body={"query": {"match_all": {}}, "sort": [{"timestamp": {"order": "asc"}}]},
                scroll="2m"
            )
            sid = scroll['_scroll_id']
            scroll_size = len(scroll['hits']['hits'])
            
            pbar = tqdm(total=scroll['hits']['total']['value'], desc=f"Scrolling {index_name}")

            while scroll_size > 0:
                # 결과 저장
                hits = scroll['hits']['hits']
                logs.extend(hit['_source'] for hit in hits)
                pbar.update(len(hits))
                
                # 다음 배치의 결과 가져오기
                scroll = self.es.scroll(scroll_id=sid, scroll='2m')
                sid = scroll['_scroll_id']
                scroll_size = len(scroll['hits']['hits'])

            pbar.close()
            # 스크롤 컨텍스트 클리어
            self.es.clear_scroll(scroll_id=sid)
        except Exception as e:
            if 'pbar' in locals() and pbar: pbar.close()
            print(f"{index_name}에서 모든 로그를 가져오는 중 오류 발생: {e}")

        return logs

    def get_training_data_for_all_users(self) -> pd.DataFrame:
        """모든 사용자의 검색 및 클릭 데이터를 기반으로 병렬 처리를 통해 학습 데이터 생성"""

        all_search_logs = self._get_all_logs(self.search_log_index_name)
        all_click_logs = self._get_all_logs(self.click_log_index_name)

        if not all_search_logs:
            print("검색 로그가 없어 학습 데이터를 생성할 수 없습니다.")
            return pd.DataFrame()

        search_df = pd.DataFrame(all_search_logs)
        click_df = pd.DataFrame(all_click_logs)

        # userId로 데이터 그룹화
        search_logs_by_user = search_df.groupby('userId')
        click_logs_by_user = click_df.groupby('userId')

        unique_users = search_df['userId'].unique()
        
        print("Pandas apply를 사용하여 태스크를 효율적으로 준비합니다...")
        # apply를 사용하여 각 그룹을 딕셔너리 리스트로 변환 (이 과정이 훨씬 빠름)
        search_log_dicts = search_logs_by_user.apply(lambda g: g.to_dict(orient='records'))
        click_log_dicts = click_logs_by_user.apply(lambda g: g.to_dict(orient='records'))

        tasks = []
        for user_id in tqdm(unique_users, desc="태스크 리스트 생성 중"):
            user_search_logs = search_log_dicts.get(user_id, [])
            user_click_logs = click_log_dicts.get(user_id, [])
            tasks.append((user_id, user_search_logs, user_click_logs))

        training_data_list = []
        print("병렬 처리를 시작합니다...")
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(_process_user_log_data, tasks), total=len(tasks), desc="사용자별 데이터 병렬 처리 중"))

        # 결과 리스트를 평탄화
        training_data = [item for sublist in results for item in sublist]
        
        if not training_data:
            return pd.DataFrame()

        print(f"총 {len(training_data)}개의 학습 데이터 생성 완료.")
        return pd.DataFrame(training_data)