import sys
import os
from datetime import datetime, timezone, timedelta
from elasticsearch.helpers import bulk
from tqdm import tqdm

# Add project root to path so we can import from 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.elasticsearch_service import ElasticsearchService, _ensure_utc_aware

def migrate_timestamps(es_service: ElasticsearchService, index_name: str, timestamp_field: str):
    """
    Migrates timestamps in a given index from UTC to KST.
    Uses scroll API to fetch all documents and bulk API to update them.
    """
    print(f"'{index_name}' 인덱스의 타임스탬프 마이그레이션을 시작합니다...")
    
    es = es_service.es
    
    # KST timezone
    kst = timezone(timedelta(hours=9))
    
    # 1. Scroll API를 사용하여 모든 문서 가져오기
    try:
        scroll = es.search(
            index=index_name,
            body={"query": {"match_all": {}}},
            scroll="5m",
            size=1000  # 한 번에 가져올 문서 수
        )
        sid = scroll['_scroll_id']
        scroll_size = len(scroll['hits']['hits'])
        total_docs = scroll['hits']['total']['value']
        
        if total_docs == 0:
            print(f"'{index_name}' 인덱스에 문서가 없습니다. 다음으로 넘어갑니다.")
            return

        pbar = tqdm(total=total_docs, desc=f"'{index_name}' 인덱스 처리 중")
        
        while scroll_size > 0:
            hits = scroll['hits']['hits']
            
            # 2. Bulk-update를 위한 액션 리스트 생성
            actions = []
            for hit in hits:
                doc_id = hit['_id']
                source = hit['_source']
                
                if timestamp_field not in source or not source.get(timestamp_field):
                    continue
                
                timestamp_str = source[timestamp_field]

                try:
                    # UTC-aware datetime 객체로 변환
                    utc_dt = _ensure_utc_aware(str(timestamp_str))
                    
                    # KST로 변환
                    kst_dt = utc_dt.astimezone(kst)
                    
                    # 업데이트할 필드
                    update_doc = {
                        timestamp_field: kst_dt.isoformat()
                    }
                    
                    action = {
                        "_op_type": "update",
                        "_index": index_name,
                        "_id": doc_id,
                        "doc": update_doc
                    }
                    actions.append(action)
                except (ValueError, TypeError) as e:
                    print(f"문서 ID {doc_id}의 타임스탬프('{timestamp_str}') 처리 중 오류: {e}")
                    continue

            # 3. Bulk API로 업데이트 실행
            if actions:
                try:
                    bulk(es, actions)
                except Exception as e:
                    print(f"Bulk 업데이트 중 오류 발생: {e}")

            pbar.update(len(hits))
            
            # 다음 배치의 결과 가져오기
            scroll = es.scroll(scroll_id=sid, scroll='5m')
            sid = scroll['_scroll_id']
            scroll_size = len(scroll['hits']['hits'])

        pbar.close()
        es.clear_scroll(scroll_id=sid)
        print(f"'{index_name}' 인덱스 마이그레이션 완료.")
        
    except Exception as e:
        if 'pbar' in locals() and pbar: pbar.close()
        print(f"'{index_name}' 인덱스 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    print("Elasticsearch 서비스 초기화 중...")
    try:
        es_service = ElasticsearchService()
        
        if not es_service.is_connected():
            print("Elasticsearch에 연결할 수 없습니다. 스크립트를 종료합니다.")
            sys.exit(1)
        
        print("연결 성공. 마이그레이션을 시작합니다.")
        
        # 마이그레이션할 인덱스와 타임스탬프 필드 목록
        indices_to_migrate = [
            (es_service.search_log_index_name, "timestamp"),
            (es_service.click_log_index_name, "timestamp"),
            (es_service.log_index_name, "createAt")
        ]
        
        for index, field in indices_to_migrate:
            migrate_timestamps(es_service, index, field)
            
        print("\n모든 마이그레이션 작업이 완료되었습니다.")

    except Exception as e:
        print(f"스크립트 실행 중 오류가 발생했습니다: {e}")
        sys.exit(1)
