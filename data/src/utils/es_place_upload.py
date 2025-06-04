from elasticsearch import Elasticsearch
import csv, json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KoreanContentElasticsearch:    
    def __init__(self, host: str = "localhost", port: int = 9200):
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
        
        Args:
            index_name: 생성할 인덱스 이름
            
        Returns:
            생성 성공 여부
        """
        try:
            # 인덱스가 이미 존재하는지 확인
            if self.es.indices.exists(index=index_name):
                logger.warning(f"인덱스 '{index_name}'이 이미 존재합니다.")
                print(f"경고: 인덱스 '{index_name}'이 이미 존재합니다.")
                return True
            
            # 한국어 분석을 위한 설정
            index_settings = {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "tokenizer": {
                        "nori_user_dict": {
                            "type": "nori_tokenizer",
                            "decompound_mode": "mixed",
                            "discard_punctuation": "true"
                        }
                    },
                    "analyzer": {
                        "korean_analyzer": {
                            "type": "custom",
                            "tokenizer": "nori_user_dict",
                            "filter": [
                                "nori_part_of_speech",
                                "nori_readingform",
                                "lowercase"
                            ]
                        },
                        "korean_search_analyzer": {
                            "type": "custom", 
                            "tokenizer": "nori_user_dict",
                            "filter": [
                                "nori_part_of_speech",
                                "lowercase"
                            ]
                        }
                    },
                    "filter": {
                        "nori_part_of_speech": {
                            "type": "nori_part_of_speech",
                            "stoptags": [
                                "E", "IC", "J", "MAG", "MAJ", "MM", 
                                "SP", "SSC", "SSO", "SC", "SE", "XPN", "XSA", "XSN", "XSV"
                            ]
                        }
                    }
                }
            }
            
            # 필드 매핑 정의
            field_mappings = {
                "HEX(id)": {
                    "type": "keyword"
                },
                "name": {
                    "type": "text", 
                    "analyzer": "korean_analyzer",
                    "search_analyzer": "korean_search_analyzer",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "category": {
                    "type": "text",
                    "analyzer": "korean_analyzer",
                    "search_analyzer": "korean_search_analyzer", 
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    }
                },
                "subCategory": {
                    "type": "text",
                    "analyzer": "korean_analyzer",
                    "search_analyzer": "korean_search_analyzer",
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    }
                },
                "address": {
                    "type": "text",
                    "analyzer": "korean_analyzer",
                    "search_analyzer": "korean_search_analyzer",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "content": {
                    "type": "text",
                    "analyzer": "korean_analyzer",
                    "search_analyzer": "korean_search_analyzer"
                }
            }
            
            # 인덱스 생성 요청 바디
            index_body = {
                "settings": index_settings,
                "mappings": {
                    "properties": field_mappings
                }
            }
            
            # 인덱스 생성
            logger.info(f"인덱스 '{index_name}' 생성 중...")
            print(f"인덱스 '{index_name}' 생성 중...")
            
            response = self.es.indices.create(index=index_name, body=index_body)
            
            if response.get('acknowledged', False):
                logger.info(f"인덱스 '{index_name}' 생성 완료")
                print(f"인덱스 '{index_name}' 생성 완료")
                print("매핑 정보:")
                print("- name: text (한국어 분석)")
                print("- category: text (한국어 분석)")  
                print("- subCategory: text (한국어 분석)")
                print("- address: text (한국어 분석)")
                print("- content: text (한국어 분석)")
                print("- timestamp: date (자동 생성)")
                print(f"사용 분석기: nori_tokenizer 기반 korean_analyzer")
                return True
            else:
                logger.error(f"인덱스 생성 응답 확인 실패")
                print(f"오류: 인덱스 생성 응답 확인 실패")
                return False
                
        except Exception as e:
            logger.error(f"인덱스 생성 실패: {e}")
            print(f"오류: 인덱스 생성 실패 - {e}")
            return False
    
    def insert_data_from_csv(self, index_name: str, csv_file_path: str, id_field: str = None) -> bool:
        """
        JSON 파일에서 데이터를 읽어서 벌크 삽입
        
        Args:
            index_name: 대상 인덱스 이름
            json_file_path: JSON 파일 경로
            id_field: 문서 ID로 사용할 필드명 (없으면 자동 생성)
            
        Returns:
            삽입 성공 여부
        """
        try:
            logger.info(f"CSV 파일 읽는 중: {csv_file_path}")
            print(f"CSV 파일 읽는 중: {csv_file_path}")
            
            # CSV 파일을 데이터프레임으로 읽기
            df = pd.read_csv(csv_file_path)
            data = df.to_dict('records')
            
            logger.info(f"읽어온 데이터 개수: {len(data)}")
            print(f"읽어온 데이터 개수: {len(data)}")
            
            # 필수 필드 확인
            required_fields = ["HEX(id)", "name", "category", "sub_category", "address", "content"]
            processed_data = []
            
            for i, doc in enumerate(data):
                # 필수 필드 존재 확인
                missing_fields = [field for field in required_fields if field not in doc]
                if missing_fields:
                    logger.warning(f"문서 {i+1}에서 누락된 필드: {missing_fields}")
                    print(f"경고: 문서 {i+1}에서 누락된 필드: {missing_fields}")
                    continue
                
                processed_data.append(doc)
            
            if not processed_data:
                logger.error("삽입할 유효한 데이터가 없습니다.")
                print("오류: 삽입할 유효한 데이터가 없습니다.")
                return False
            
            logger.info(f"유효한 데이터 개수: {len(processed_data)}")
            print(f"유효한 데이터 개수: {len(processed_data)}")
            
            # 벌크 삽입 준비
            bulk_body = []
            for i, doc in enumerate(processed_data):
                # 문서 ID 결정
                doc_id = doc.get(id_field) if id_field else i + 1
                
                # 벌크 작업 정의
                bulk_body.append({
                    "index": {
                        "_index": index_name,
                        "_id": doc_id
                    }
                })
                bulk_body.append(doc)
            
            logger.info("벌크 삽입 실행 중...")
            print("벌크 삽입 실행 중...")
            
            # 벌크 삽입 결과 추적
            success_count = 0
            errors = []
            
            try:
                # 벌크 삽입 실행 및 응답 처리
                response = self.es.bulk(body=bulk_body)
                
                # 응답에서 성공/실패 항목 처리
                if not response.get('errors'):
                    success_count = len(processed_data)
                else:
                    for item in response['items']:
                        if 'index' in item and item['index'].get('status') == 201:
                            success_count += 1
                        else:
                            error_reason = item.get('index', {}).get('error', {}).get('reason', '알 수 없는 오류')
                            errors.append(error_reason)
                            
            except Exception as e:
                logger.error(f"벌크 삽입 중 오류 발생: {str(e)}")
                print(f"오류: 벌크 삽입 중 오류 발생 - {str(e)}")
                return False
            
            # 결과 출력
            logger.info(f"벌크 삽입 완료 - 성공: {success_count}개")
            print(f"벌크 삽입 완료 - 성공: {success_count}개")
            
            if errors:
                logger.warning(f"삽입 실패: {len(errors)}개")
                print(f"경고: 삽입 실패 {len(errors)}개")
                
                # 처음 3개 에러만 상세 출력
                for i, error in enumerate(errors[:3]):
                    logger.warning(f"에러 {i+1}: {error}")
                    print(f"에러 {i+1}: {error}")
                
                if len(errors) > 3:
                    logger.warning(f"추가로 {len(errors)-3}개의 에러가 더 있습니다.")
                    print(f"추가로 {len(errors)-3}개의 에러가 더 있습니다.")
        except FileNotFoundError:
            logger.error(f"파일을 찾을 수 없습니다: {json_file_path}")
            print(f"오류: 파일을 찾을 수 없습니다 - {json_file_path}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            print(f"오류: JSON 파싱 실패 - {e}")
            return False
        except Exception as e:
            logger.error(f"데이터 삽입 실패: {e}")
            print(f"오류: 데이터 삽입 실패 - {e}")
            return False