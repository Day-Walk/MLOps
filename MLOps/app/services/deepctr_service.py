import numpy as np
from typing import List, Dict, Any
from app.model.deepfm_train import DeepFMModdelTrain
from app.services.elk_client import ELKClient
from app.services.db_connection import DatabaseService
import pandas as pd
import os
import torch
import json

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
        places_list = await self.elk_client.search_places(query=query, user_id=userid, max_results=23)
        if not places_list:
            # 검색 결과가 없으면 빈 리스트 반환
            return [], []
            
        places_df = pd.DataFrame(places_list)
        
        # DataFrame의 컬럼 이름 변경
        places_df = places_df.rename(columns={'uuid': 'place_id', 'name': 'place_name'})
        
        # 사용자 정보를 DataFrame의 모든 행에 추가
        if not user_info_df.empty:
            # 기본 정보는 첫 행에서 추출 (userid, name, gender, age)
            base_cols = ['userid', 'name', 'gender', 'age']
            user_info = user_info_df.iloc[0][base_cols].to_dict()

            # like_list 가 누적될 리스트
            combined_likes: list[str] = []

            for _, row in user_info_df.iterrows():
                category = row.get('category_name', '')
                tags_raw = row.get('like_list', '[]')

                # like_list 컬럼이 문자열(JSON)일 수도 있고 list 객체일 수도 있음
                if isinstance(tags_raw, str):
                    try:
                        tags_list = json.loads(tags_raw)
                    except json.JSONDecodeError:
                        tags_list = []
                else:
                    tags_list = tags_raw

                if tags_list:
                    for tag in tags_list:
                        combined_likes.append(f"{category}_{tag}")

            # userid, name, gender, age 컬럼 채우기
            for col, val in user_info.items():
                places_df[col] = val

            # like_list 는 JSON 문자열로 저장 (모델 입력 형식에 맞게 조정 가능)
            places_df['like_list'] = json.dumps(combined_likes, ensure_ascii=False)
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
        
        # CTR 0.3 이상인 장소만 필터링
        recommended_df = sorted_df[sorted_df['ctr'] >= 0.3].copy()
        normal_df = sorted_df[sorted_df['ctr'] < 0.3].copy()

        # 'place_id'를 'id'로 변경
        if not recommended_df.empty:
            recommended_df['id'] = recommended_df['place_id']
            recommended_df = recommended_df.drop(columns=['place_id'])
        if not normal_df.empty:
            normal_df['id'] = normal_df['place_id']
            normal_df = normal_df.drop(columns=['place_id'])
        
        # 상위 3개 장소 반환, 나머지도 반환
        top_places = recommended_df.to_dict(orient='records')
        other_places = normal_df.to_dict(orient='records')
        return top_places, other_places
    