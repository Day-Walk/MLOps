import requests
from typing import List, Dict, Any

class ELKClient:
    """ELK 서버 클라이언트"""
    
    def __init__(self, elk_url: str = "http://15.164.50.188:9201"):
        self.elk_url = elk_url
    
    async def search_places(self, query: str, max_results: int = 23) -> List[Dict[str, Any]]:
        """ELK 서버에서 장소 검색"""
        try:
            response = requests.get(
                f"{self.elk_url}/api/place/search",
                params={
                    "query": query,
                    "max_results": max_results
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("data", [])
            else:
                return []
                
        except Exception as e:
            print(f"ELK 서버 호출 실패: {e}")
            return []
