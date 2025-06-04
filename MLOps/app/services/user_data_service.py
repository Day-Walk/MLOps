# app/services/user_data_service.py
import pandas as pd
import ast
from typing import Dict, Optional
import os

class UserDataService:
    """사용자 데이터 관리 서비스"""
    
    def __init__(self, csv_path: str = "app/data/final_click_log.csv"):
        """
        사용자 데이터 서비스 초기화
        
        Args:
            csv_path: 사용자 데이터 CSV 파일 경로
        """
        self.csv_path = csv_path
        self.user_data = None
        self.load_user_data()
    
    def load_user_data(self):
        """CSV 파일에서 사용자 데이터 로드"""
        try:
            if os.path.exists(self.csv_path):
                self.user_data = pd.read_csv(self.csv_path)
                
                # like_list 컬럼 파싱 (문자열을 리스트로 변환)
                if 'like_list' in self.user_data.columns:
                    self.user_data['like_list'] = self.user_data['like_list'].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
                    )
                
                print(f"사용자 데이터 로드 완료: {len(self.user_data)} 개 레코드")
            else:
                print(f"사용자 데이터 파일을 찾을 수 없습니다: {self.csv_path}")
                self.user_data = pd.DataFrame()
        except Exception as e:
            print(f"사용자 데이터 로드 중 오류 발생: {e}")
            self.user_data = pd.DataFrame()
    
    def get_user_features(self, user_id: str) -> Optional[Dict]:
        """
        사용자 ID로 사용자 특징 조회
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            Dict: 사용자 특징 딕셔너리 또는 None
        """
        if self.user_data is None or self.user_data.empty:
            return None
        
        # 해당 사용자의 가장 최근 레코드 조회
        user_records = self.user_data[self.user_data['user_id'] == user_id]
        
        if user_records.empty:
            return None
        
        # 가장 최근 레코드 선택 (첫 번째 레코드)
        user_record = user_records.iloc[0]
        
        return {
            'user_id': user_record['user_id'],
            'user_name': user_record['user_name'],
            'age': user_record['age'],
            'gender': user_record['gender'],
            'like_list': user_record['like_list']
        }
    
    def get_all_users(self) -> pd.DataFrame:
        """모든 사용자 데이터 반환"""
        return self.user_data
    
    def get_user_list(self) -> list:
        """사용자 ID 목록 반환"""
        if self.user_data is None or self.user_data.empty:
            return []
        return self.user_data['user_id'].unique().tolist()
