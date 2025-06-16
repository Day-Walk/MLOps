import numpy as np
from typing import List, Dict, Any
from app.model.deepfm_train import DeepFMModdelTrain
from app.services.elk_client import ELKClient
from app.services.user_data_service import UserDataService
import pandas as pd

class DeepCTRService:
    """DeepCTR 모델 서비스"""
    
    def __init__(self):
        self.model = DeepFMModdelTrain("../data/final_click_log.csv")
        self.elk_client = ELKClient()
        self.user_data_service = UserDataService()
        
    async def rank_places_by_ctr(self, userid: str, query: str):
        # 사용자 정보 조회
        user_info_df = self.user_data_service.get_user_info(userid)
        
        # 장소 정보
        places_list = await self.elk_client.search_places(query=query, max_results=23)
        if not places_list:
            # 검색 결과가 없으면 빈 리스트 반환
            return [], []
            
        places_df = pd.DataFrame(places_list)
        
        # DataFrame의 컬럼 이름 변경
        places_df = places_df.rename(columns={'uuid': 'place_id', 'name': 'place_name'})
        
        # 사용자 정보를 DataFrame의 모든 행에 추가
        if not user_info_df.empty:
            for col, val in user_info_df.items():
                places_df[col] = val.iloc[0]
        else:
            # 사용자가 존재하지 않을 경우, 기본값으로 채움
            places_df['userid'] = userid
            places_df['age'] = 30 # 평균 또는 기본값
            places_df['gender'] = 1 # 남성을 기본값으로 가정
            places_df['like_list'] = "[1,2,3,4,5]"
        
        input_data = places_df
                
        # 모델 예측
        predictions = self.model.predict(input_data)
        
        # input data에 예측값 추가
        input_data['ctr'] = predictions
        
        # 예측값 기준 정렬
        input_data = input_data.sort_values(by='ctr', ascending=False)
        
        # 상위 3개 장소 반환, 나머지도 반환
        top_places = input_data.head(3)['place_id'].tolist()
        other_places = input_data.iloc[3:].to_dict('records')
        return top_places, other_places
    