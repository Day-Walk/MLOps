import numpy as np
from typing import List, Dict, Any
from app.model.deepfm_train import DeepFMModdelTrain
from app.services.elk_client import ELKClient
from app.services.user_data_service import UserDataService
import pandas as pd

class DeepCTRService:
    """DeepCTR 모델 서비스"""
    
    def __init__(self):
        self.model = DeepFMModdelTrain("/home/ubuntu/working/MLOps/data/final_click_log.csv")
        self.elk_client = ELKClient()
        self.user_data_service = UserDataService()
        
    def rank_places_by_ctr(self, user_id: str, query: str):
        # 사용자 정보 조회
        user_info = self.user_data_service.get_user_info(user_id)
        
        # 장소 정보 - elk 클라이언트 사용
        places = self.elk_client.search_places(query=query, max_results=23)
        places_data = pd.DataFrame(places)
        
        # 두 데이터프레임 합치기
        input_data = pd.concat([places_data, user_info], axis=1)
        
        # 모델 예측
        predictions = self.model.predict(input_data)
        
        # input data에 예측값 추가
        input_data['ctr'] = predictions
        
        # 예측값 기준 정렬
        input_data = input_data.sort_values(by='ctr', ascending=False)
        
        # 상위 3개 장소 반환, 나머지도 반환
        top_places = input_data.head(3)['place_id'].tolist()
        return top_places, input_data[4:].to_list()
    