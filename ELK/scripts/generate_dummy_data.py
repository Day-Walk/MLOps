import sys
import os
import asyncio
import pandas as pd
import random
from datetime import datetime, timedelta, timezone

# 프로젝트 루트 경로를 sys.path에 추가
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT_PATH, 'ELK'))

from app.services.elasticsearch_service import ElasticsearchService

# --- 설정 ---
# 이 스크립트는 Docker 외부에서 실행되므로, 명시된 IP로 접속합니다.
ES_HOST = "15.164.50.188"
USER_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data", "user_db.json")
NUM_SEARCHES_PER_USER = 2
NUM_CLICKS_PER_SEARCH = (3, 4)

# 현실적인 검색어 목록
SEARCH_QUERIES = ["강남", "홍대", "맛집", "카페", "성수", "잠실", "이태원", "한남동"]

class DummyDataGenerator:
    def __init__(self, es_host: str, user_data_path: str):
        print("Elasticsearch 서비스에 연결 중...")
        self.es_service = ElasticsearchService(host=es_host)
        if not self.es_service.is_connected():
            print("Elasticsearch 연결 실패. 스크립트를 종료합니다.")
            sys.exit(1)
        
        print(f"사용자 데이터 로드 중: {user_data_path}")
        try:
            # JSON 파일을 DataFrame으로 읽습니다.
            self.users_df = pd.read_json(user_data_path)
        except FileNotFoundError:
            print(f"오류: 사용자 데이터 파일({user_data_path})을 찾을 수 없습니다.")
            print("프로젝트 루트에서 'dvc pull data/user_db.json.dvc'를 실행했는지 확인하세요.")
            sys.exit(1)
        except Exception as e:
            print(f"오류: JSON 파일 파싱 중 문제 발생 - {e}")
            sys.exit(1)
            
        self.search_results_pool = {}

    async def _prepare_search_pool(self):
        """미리 정의된 검색어에 대한 실제 장소 목록을 가져와 풀을 만듭니다."""
        print("현실적인 데이터 생성을 위해 검색어 풀을 준비합니다...")
        for query in SEARCH_QUERIES:
            print(f"  '{query}'에 대한 장소 검색 중...")
            # max_results를 기본값(23)으로 사용
            places = self.es_service.search_places(query=query)
            if places:
                self.search_results_pool[query] = [p['uuid'] for p in places]
        
        if not self.search_results_pool:
            print("오류: 검색 결과 풀을 생성할 수 없습니다. place_data 인덱스에 데이터가 있는지 확인하세요.")
            sys.exit(1)
        print("검색어 풀 준비 완료.")

    async def generate(self):
        """전체 더미 데이터 생성 프로세스를 실행합니다."""
        await self._prepare_search_pool()
        
        user_ids = self.users_df['id'].unique()
        total_users = len(user_ids)
        print(f"\n총 {total_users}명의 사용자에 대해 더미 데이터 생성을 시작합니다.")

        for i, user_id in enumerate(user_ids):
            print(f"--- [{i+1}/{total_users}] 사용자 '{user_id}' 작업 시작 ---")
            
            # 각 사용자의 시간은 과거의 특정 시점부터 시작
            current_time = datetime.now(timezone.utc) - timedelta(days=random.randint(1, 30))

            for j in range(NUM_SEARCHES_PER_USER):
                # 1. 검색 로그 생성
                search_query = random.choice(list(self.search_results_pool.keys()))
                place_ids = self.search_results_pool[search_query]
                
                # 검색 시간은 현재 시간
                search_time = current_time
                search_log = {
                    "userId": str(user_id),
                    "query": search_query,
                    "placeIds": place_ids,
                    "timestamp": search_time
                }
                success, _ = self.es_service.insert_search_log(search_log)
                if success:
                    print(f"  ({j+1}) 검색 로그 저장: '{search_query}' (결과 {len(place_ids)}개)")
                else:
                    print(f"  ({j+1}) 검색 로그 저장 실패")

                # 2. 클릭 로그 생성 (현재 검색과 다음 검색 사이)
                
                # 다음 검색을 위한 시간 간격 미리 설정
                time_to_next_search = timedelta(minutes=random.randint(5, 120))
                
                num_clicks = random.randint(*NUM_CLICKS_PER_SEARCH)
                # 검색 결과 수보다 많이 클릭할 수 없음
                num_clicks = min(num_clicks, len(place_ids))
                
                clicked_places = random.sample(place_ids, num_clicks)

                print(f"    -> {num_clicks}개의 클릭 로그 생성 시뮬레이션...")
                for k, place_id in enumerate(clicked_places):
                    # 클릭 시간: 검색 시간 이후, 다음 검색 시간 이전
                    # 클릭 사이의 시간 간격을 랜덤하게 부여
                    time_after_search = timedelta(seconds=random.randint(10, 200))
                    click_time = search_time + time_after_search
                    
                    # 현재 시간을 클릭 시간 이후로 업데이트
                    current_time = click_time 

                    click_log = {
                        "userId": str(user_id),
                        "placeId": place_id,
                        "timestamp": click_time
                    }
                    success, _ = self.es_service.insert_click_log(click_log)

                # 3. 다음 검색을 위해 시간 점프
                current_time += time_to_next_search
            
            print(f"--- 사용자 '{user_id}' 작업 완료 ---\n")
        print("모든 더미 데이터 생성이 완료되었습니다.")

async def main():
    generator = DummyDataGenerator(es_host=ES_HOST, user_data_path=USER_DATA_PATH)
    await generator.generate()

if __name__ == "__main__":
    asyncio.run(main()) 