"""
장소 정보 추출기
GPT 응답에서 장소명을 추출하여 UUID로 변환
"""
import re
from typing import List, Optional

class PlaceExtractor:
    """장소 정보 추출 클래스"""
    
    def __init__(self):
        # 더미 장소 매핑 (실제 DB 연동 시 교체)
        self.place_mapping = {
            "홍대입구역": "place-uuid-001",
            "홍대 카페거리": "place-uuid-002", 
            "성수동 카페": "place-uuid-005",
            "성수동 감성카페": "place-uuid-006",
            "강남역": "place-uuid-008",
            "강남 맛집": "place-uuid-009",
            "이태원": "place-uuid-011",
            "명동": "place-uuid-012",
            "잠실": "place-uuid-013"
        }
        
        self.place_patterns = [
            r'\d+\.\s*([^-\n]+)\s*-',  # "1. 장소명 -" 패턴
            r'【([^】]+)】',            # "【장소명】" 패턴  
            r'\*\*([^*]+)\*\*',        # "**장소명**" 패턴
        ]
    
    def extract_place_ids_from_text(self, text: str) -> Optional[List[str]]:
        """GPT 응답 텍스트에서 장소 ID 추출"""
        try:
            if not self._contains_recommendations(text):
                return None
            
            place_ids = []
            extracted_names = self._extract_place_names(text)
            
            for name in extracted_names:
                uuid = self._find_matching_uuid(name)
                if uuid and uuid not in place_ids:
                    place_ids.append(uuid)
            
            return place_ids if place_ids else None
            
        except Exception as e:
            print(f"❌ 장소 ID 추출 실패: {e}")
            return None
    
    def _extract_place_names(self, text: str) -> List[str]:
        """텍스트에서 장소명 패턴 추출"""
        extracted_names = []
        
        for pattern in self.place_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                place_name = match.strip()
                if place_name and len(place_name) > 1:
                    extracted_names.append(place_name)
        
        return list(set(extracted_names))
    
    def _find_matching_uuid(self, place_name: str) -> Optional[str]:
        """장소명에 해당하는 UUID 찾기"""
        place_name_lower = place_name.lower()
        
        for mapped_name, uuid in self.place_mapping.items():
            if mapped_name.lower() in place_name_lower or place_name_lower in mapped_name.lower():
                return uuid
        
        return None
    
    def _contains_recommendations(self, text: str) -> bool:
        """추천 내용이 포함된 응답인지 확인"""
        keywords = ["추천", "코스", "어떠세요", "1.", "2.", "3."]
        return any(keyword in text for keyword in keywords)
