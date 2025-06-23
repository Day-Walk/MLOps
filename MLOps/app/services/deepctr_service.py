import numpy as np
from typing import List, Dict, Any
from app.model.deepfm_train import DeepFMModdelTrain
from app.services.elk_client import ELKClient
from app.services.db_connection import DatabaseService
import pandas as pd
import os
import torch

class DeepCTRService:
    """DeepCTR 모델 서비스"""
    
    def __init__(self):
        """DeepCTR 모델 서비스 초기화"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepFMModdelTrain()
        self.model_path = os.environ.get("DEEPFM_TRAIN_MODEL_PATH", "")
        self.elk_client = ELKClient()
        self.db_service = DatabaseService()
        
    async def rank_places_by_ctr(self, userid: str, query: str):
        # 사용자 정보 조회
        user_info_df = self.db_service.get_user_info_by_user_id(userid)
        
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
            user_info = user_info_df.iloc[0].to_dict()
            for col, val in user_info.items():
                places_df[col] = val
        else:
            # 사용자가 존재하지 않을 경우, 기본값으로 채움
            print("사용자가 존재하지 않습니다.")
            places_df['userid'] = userid
            places_df['name'] = 'unknown'
            places_df['age'] = 30 # 평균 또는 기본값
            places_df['gender'] = 1 # 남성을 기본값으로 가정
            places_df['like_list'] = "[]"
        
        # 원본 데이터에서 응답에 필요한 컬럼 보존
        output_columns = ['place_id', 'place_name', 'category', 'subcategory']
        preserved_df = places_df[output_columns]

        # 모델 예측
        predictions = self.model.predict(places_df)
        
        # 예측 결과를 원본 데이터에 추가
        preserved_df['ctr'] = predictions
        
        # 예측값 기준 정렬
        df_to_sort = pd.DataFrame(preserved_df)
        sorted_df = df_to_sort.sort_values(by='ctr', ascending=False)
        
        # 'place_id'를 'id'로 다시 변경
        sorted_df = sorted_df.rename(columns={'place_id': 'id'})
        
        # 상위 3개 장소 반환, 나머지도 반환
        top_places = sorted_df.head(3).to_dict(orient='records')
        other_places = sorted_df.iloc[3:].to_dict(orient='records')
        return top_places, other_places
    