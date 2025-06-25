from elasticsearch import Elasticsearch, helpers
import csv, json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KoreanContentElasticsearch:    
    def __init__(self, host: str = "15.164.50.188", port: int = 9200):
        """
        Elasticsearch 클라이언트 초기화
        
        Args:
            host: Elasticsearch 호스트 주소
            port: Elasticsearch 포트 번호
        """
        try:
            # 클라이언트 설정
            client_url = f"http://{host}:{port}"
            
            self.es = Elasticsearch(client_url)
            
            # 연결 테스트
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Elasticsearch 연결 실패: {e}")
            raise
    
    def _test_connection(self) -> None:
        """연결 상태 확인"""
        try:
            info = self.es.info()
            logger.info(f"Elasticsearch 연결 성공!")
            logger.info(f"클러스터: {info['cluster_name']}")
            logger.info(f"버전: {info['version']['number']}")
            print(f"Elasticsearch 연결 성공!")
            print(f"클러스터: {info['cluster_name']}")
            print(f"버전: {info['version']['number']}")
        except Exception as e:
            logger.error(f"연결 테스트 실패: {e}")
            raise
    
    def create_korean_content_index(self, index_name: str) -> bool:
        """
        한국어 콘텐츠를 위한 인덱스 생성 (nori_tokenizer 사용)
        """
        try:
            # 인덱스가 이미 존재하는지 확인
            if self.es.indices.exists(index=index_name):
                logger.warning(f"인덱스 '{index_name}'이 이미 존재합니다.")
                print(f"경고: 인덱스 '{index_name}'이 이미 존재합니다.")
                return True
            
            # 한국어 분석을 위한 설정
            settings = {
                "index": {
                    "analysis": {
                        "tokenizer": {
                            "my_nori_tokenizer": {
                                "type": "nori_tokenizer",
                                "decompound_mode": "mixed"
                            }
                        },
                        "analyzer": {
                            "my_nori_analyzer": {
                                "type": "custom",
                                "tokenizer": "my_nori_tokenizer",
                                "filter": ["lowercase"]
                            }
                        }
                    }
                }
            }
            
            # 필드 매핑 정의
            mappings = {
                "properties": {
                    "uuid": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "category": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                        "copy_to": "categories",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "subcategory": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                        "copy_to": "categories",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "gu": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                        "copy_to": "addresses",
                    },
                    "dong": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                        "copy_to": "addresses",
                    },
                    "ro": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                        "copy_to": "addresses",
                    },
                    "station": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                        "copy_to": "addresses",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "location": {
                        "type": "geo_point"
                    },
                    "opentime": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                    },
                    "breaktime": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                    },
                    "closedate": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                    },
                    "phone": {
                        "type": "text"
                    },
                    "name": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "alias": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                    },
                    "address": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "my_nori_analyzer",
                        "search_analyzer": "my_nori_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    }
                }
            }
            
            index_body = {
                "mappings": mappings,
                "settings": settings
            }
            
            # 인덱스 생성
            logger.info(f"인덱스 '{index_name}' 생성 중...")
            print(f"인덱스 '{index_name}' 생성 중...")
            
            response = self.es.indices.create(index=index_name, body=index_body)
            
            if response.get('acknowledged', False):
                logger.info(f"인덱스 '{index_name}' 생성 완료")
                return True
            else:
                logger.error(f"인덱스 생성 응답 확인 실패")
                print(f"오류: 인덱스 생성 응답 확인 실패")
                return False
                
        except Exception as e:
            logger.error(f"인덱스 생성 실패: {e}")
            print(f"오류: 인덱스 생성 실패 - {e}")
            return False

    def insert_data_from_json(self, index_name: str, json_file_path: str) -> bool:
        """
        JSON 파일에서 데이터를 읽어서 벌크 삽입

        Args:
            index_name: 대상 인덱스 이름
            json_file_path: JSON 파일 경로

        Returns:
            삽입 성공 여부
        """
        try:
            logger.info(f"JSON 파일 읽는 중: {json_file_path}")
            print(f"JSON 파일 읽는 중: {json_file_path}")

            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"읽어온 데이터 개수: {len(data)}")
            print(f"읽어온 데이터 개수: {len(data)}")

            actions = []
            for doc in data:
                # 필드 이름 매핑 및 전처리
                location_str = doc.get("location", "")
                lat, lon = None, None
                if location_str:
                    try:
                        lat, lon = map(float, location_str.split())
                    except ValueError:
                        logger.warning(f"잘못된 location 형식: {location_str}")
                
                es_doc = {
                    "_index": index_name,
                    "_id": doc.get("uuid"),
                    "_source": {
                        "uuid": doc.get("uuid"),
                        "name": doc.get("name"),
                        "category": doc.get("category"),
                        "subcategory": doc.get("subCategory"),
                        "alias": doc.get("alias"),
                        "address": doc.get("address"),
                        "content": doc.get("content"),
                        "gu": doc.get("gu"),
                        "dong": doc.get("dong"),
                        "ro": doc.get("ro"),
                        "station": doc.get("station"),
                        "location": {"lat": lat, "lon": lon} if lat and lon else None,
                        "opentime": doc.get("opentime"),
                        "breaktime": doc.get("breaktime"),
                        "closedate": doc.get("closeDate"),
                        "phone": doc.get("phoneNum")
                    }
                }
                # None 값을 가진 필드 제거
                es_doc["_source"] = {k: v for k, v in es_doc["_source"].items() if v is not None}
                actions.append(es_doc)
            
            logger.info("벌크 삽입 실행 중...")
            print("벌크 삽입 실행 중...")
            
            success, failed = helpers.bulk(self.es, actions, stats_only=True, raise_on_error=False)

            logger.info(f"벌크 삽입 완료 - 성공: {success}개, 실패: {failed}개")
            print(f"벌크 삽입 완료 - 성공: {success}개, 실패: {failed}개")
            
            return failed == 0

        except FileNotFoundError:
            logger.error(f"파일을 찾을 수 없습니다: {json_file_path}")
            print(f"오류: 파일을 찾을 수 없습니다 - {json_file_path}")
            return False
        except json.JSONDecodeError:
            logger.error(f"JSON 파일 형식이 잘못되었습니다: {json_file_path}")
            print(f"오류: JSON 파일 형식이 잘못되었습니다 - {json_file_path}")
            return False
        except Exception as e:
            logger.error(f"데이터 삽입 실패: {e}")
            print(f"오류: 데이터 삽입 실패 - {e}")
            return False


if __name__ == '__main__':
    es_client = KoreanContentElasticsearch()

    # 인덱스 이름 설정
    INDEX_NAME = "place_data_v2"
    
    # JSON 파일 경로 설정
    JSON_FILE_PATH = "data/place_json_preprocessing.json"

    if es_client.create_korean_content_index(INDEX_NAME):
        es_client.insert_data_from_json(INDEX_NAME, JSON_FILE_PATH)