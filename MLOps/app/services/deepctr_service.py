# app/services/deepctr_service.py
import numpy as np
from typing import List, Dict, Any
from .user_data_service import UserDataService

class DeepCTRService:
    """DeepCTR 모델 서비스 - 개선된 버전"""
    
    def __init__(self):
        print("DeepCTR 서비스 초기화 중...")
        self.model = None
        self.user_service = UserDataService()
        print(f"사용자 서비스 초기화 완료: {self.user_service.get_user_count()}명의 사용자")
    
    def predict_ctr(self, user_id: str, places: List[Dict[str, Any]]) -> List[float]:
        """CTR 예측"""
        try:
            user_features = self.user_service.get_user_features(user_id)
            
            if user_features:
                base_score = hash(user_id) % 100 / 100
                ctr_scores = []
                
                for i, place in enumerate(places):
                    place_id = place.get('place_id', place.get('HEX(id)', str(i)))
                    place_hash = hash(str(place_id)) % 100 / 100
                    score = (base_score + place_hash + np.random.random() * 0.3) / 2.3
                    ctr_scores.append(min(max(score, 0), 1))
                
                return ctr_scores
            else:
                return [np.random.random() for _ in places]
                
        except Exception as e:
            print(f"CTR 예측 중 오류: {e}")
            return [np.random.random() for _ in places]
    
    def rank_places_by_ctr(self, user_id: str, places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """CTR 점수로 장소 순위 매기기 - place_id 정규화"""
        try:
            # 데이터 정규화 - place_id 추출 로직 개선
            normalized_places = []
            for place in places:
                # HEX(id) 또는 place_id 필드에서 ID 추출
                place_id = place.get('place_id') or place.get('HEX(id)') or 'unknown'
                
                normalized_place = {
                    'place_id': place_id,
                    'category': place.get('category', 'unknown'),
                    'sub_category': place.get('sub_category', 'unknown'),
                    'search_score': place.get('search_score', 0.0)
                }
                normalized_places.append(normalized_place)
            
            # CTR 예측
            ctr_scores = self.predict_ctr(user_id, normalized_places)
            
            # 점수 추가
            for i, place in enumerate(normalized_places):
                place['ctr_score'] = round(ctr_scores[i], 4)
            
            # CTR 점수로 정렬
            sorted_places = sorted(normalized_places, key=lambda x: x['ctr_score'], reverse=True)
            
            # 순위 추가
            for i, place in enumerate(sorted_places):
                place['rank'] = i + 1
            
            return sorted_places
            
        except Exception as e:
            print(f"장소 순위 매기기 중 오류: {e}")
            return []

    
    def get_user_info(self, user_id: str) -> Dict:
        """사용자 정보 조회 - 안전한 처리"""
        try:
            user_info = self.user_service.get_user_features(user_id)
            return user_info if user_info else {
                'user_id': user_id,
                'user_name': 'unknown',
                'age': 25,
                'gender': 'unknown',
                'like_list': []
            }
        except Exception as e:
            print(f"사용자 정보 조회 중 오류: {e}")
            return {
                'user_id': user_id,
                'user_name': 'unknown',
                'age': 25,
                'gender': 'unknown',
                'like_list': []
            }
    
    def get_available_users(self) -> List[str]:
        """사용 가능한 사용자 목록 반환"""
        try:
            return self.user_service.get_user_list()
        except Exception as e:
            print(f"사용자 목록 조회 중 오류: {e}")
            return []
