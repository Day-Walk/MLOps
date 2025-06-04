# app/services/deepctr_service.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from ..model.deepfm_model import DeepFMClickPredictor
from .user_data_service import UserDataService

class DeepCTRService:
    """DeepCTR 모델 서비스"""
    
    def __init__(self, model_path: str = "app/models/deepfm_model.pth"):
        """
        DeepCTR 서비스 초기화
        
        Args:
            model_path: 학습된 모델 파일 경로
        """
        self.model_path = model_path
        self.model = None
        self.user_service = UserDataService()
        
        # 모델 피처 정의
        self.sparse_features = [
            'user_id', 'place_id', 'place_name', 
            'category', 'sub_category', 'user_name', 'age', 'gender'
        ]
        self.sequence_feature = 'like_list'
        
        # 모델 로드 시도
        self.load_model()
    
    def load_model(self):
        """학습된 모델 로드"""
        try:
            if os.path.exists(self.model_path):
                # 더미 데이터로 모델 인스턴스 생성
                dummy_df = pd.DataFrame({
                    'user_id': ['dummy'],
                    'place_id': ['dummy'],
                    'place_name': ['dummy'],
                    'category': ['dummy'],
                    'sub_category': ['dummy'],
                    'user_name': ['dummy'],
                    'age': [25],
                    'gender': [1],
                    'like_list': [[]],
                    'yn': [0]
                })
                
                self.model = DeepFMClickPredictor(
                    dataframe=dummy_df,
                    sparse_features=self.sparse_features,
                    target='yn',
                    sequence_feature=self.sequence_feature,
                    max_len=50,  # 기본값
                    vocab_size=100  # 기본값
                )
                
                # 저장된 모델 가중치 로드
                self.model.load_model(self.model_path)
                print("DeepCTR 모델 로드 완료")
            else:
                print(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                print("더미 모델을 사용합니다.")
                self.model = None
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            self.model = None
    
    def predict_ctr(self, user_id: str, places: List[Dict[str, Any]]) -> List[float]:
        """
        사용자와 장소 리스트에 대한 CTR 예측
        
        Args:
            user_id: 사용자 ID
            places: 장소 정보 리스트
            
        Returns:
            List[float]: 각 장소에 대한 CTR 예측값 리스트
        """
        if self.model is None:
            # 모델이 없으면 더미 점수 반환
            return [np.random.random() for _ in places]
        
        # 사용자 특징 조회
        user_features = self.user_service.get_user_features(user_id)
        if user_features is None:
            # 사용자를 찾을 수 없으면 기본값 사용
            user_features = {
                'user_id': user_id,
                'user_name': 'unknown',
                'age': 25,
                'gender': 1,
                'like_list': []
            }
        
        ctr_scores = []
        
        for place in places:
            try:
                # 장소 특징 준비
                place_features = {
                    'place_id': place.get('place_id', 'unknown'),
                    'place_name': place.get('name', 'unknown'),
                    'category': place.get('category', 'unknown'),
                    'sub_category': place.get('sub_category', 'unknown')
                }
                
                # CTR 예측
                ctr_score = self.model.predict_single(user_features, place_features)
                ctr_scores.append(ctr_score)
                
            except Exception as e:
                print(f"CTR 예측 중 오류 발생: {e}")
                # 오류 시 랜덤 점수
                ctr_scores.append(np.random.random())
        
        return ctr_scores
    
    def rank_places_by_ctr(self, user_id: str, places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        CTR 점수로 장소 순위 매기기
        
        Args:
            user_id: 사용자 ID
            places: 장소 정보 리스트
            
        Returns:
            List[Dict]: CTR 점수와 순위가 추가된 장소 리스트
        """
        # CTR 예측
        ctr_scores = self.predict_ctr(user_id, places)
        
        # 장소에 CTR 점수 추가
        for i, place in enumerate(places):
            place['ctr_score'] = ctr_scores[i]
        
        # CTR 점수로 정렬
        sorted_places = sorted(places, key=lambda x: x['ctr_score'], reverse=True)
        
        # 순위 추가
        for i, place in enumerate(sorted_places):
            place['rank'] = i + 1
        
        return sorted_places
    
    def get_user_info(self, user_id: str) -> Dict:
        """사용자 정보 조회"""
        return self.user_service.get_user_features(user_id)
    
    def get_available_users(self) -> List[str]:
        """사용 가능한 사용자 ID 목록 반환"""
        return self.user_service.get_user_list()
