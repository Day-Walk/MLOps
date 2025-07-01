import requests
from typing import List, Dict, Any
import pandas as pd

class ELKClient:
    """ELK 서버 클라이언트"""
    
    def __init__(self, elk_url: str = "http://15.164.50.188:9201"):
        self.elk_url = elk_url
    
    async def search_places(self, query: str, user_id: str, max_results: int = 23) -> List[Dict[str, Any]]:
        """ELK 서버에서 장소 검색"""
        try:
            response = requests.get(
                f"{self.elk_url}/api/place/search",
                params={
                    "query": query,
                    "user_id": user_id,
                    "max_results": max_results
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("places", [])
            else:
                return []
                
        except Exception as e:
            print(f"ELK 서버 호출 실패: {e}")
            return []

    def load_user_click_log(self, user_id: str, days: int = 30):
        """유저 데이터 로드"""
        try:
            response = requests.get(
                f"{self.elk_url}/api/click-log/user/{user_id}",
                params={'days': days}
            )
            
            if response.status_code == 200:
                df = pd.DataFrame(response.json().get("logs", []))
                df.columns = df.columns.str.lower()
                return df
            else:
                return None
        except Exception as e:
            print(f"ELK 서버 호출 실패: {e}")
            return None