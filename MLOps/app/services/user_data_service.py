# app/services/user_data_service.py
import pandas as pd
import os
from typing import Dict, Optional, List

class UserDataService:
    """사용자 데이터 관리 서비스"""
    
    def __init__(self, csv_path: str = None):
        """
        사용자 데이터 서비스 초기화
        
        Args:
            csv_path: 사용자 데이터 CSV 파일 경로
        """
        # 여러 가능한 경로 시도
        possible_paths = [
            csv_path,
            "app/data/final_click_log.csv",
            "./app/data/final_click_log.csv",
            "/app/app/data/final_click_log.csv",
            "data/final_click_log.csv",
            "./data/final_click_log.csv"
        ]
        
        self.csv_path = None
        self.user_data = None
        
        # 파일이 존재하는 경로 찾기
        for path in possible_paths:
            if path and os.path.exists(path):
                self.csv_path = path
                break
        
        self.load_user_data()
    
    def load_user_data(self):
        """CSV 파일에서 사용자 데이터 로드 또는 더미 데이터 생성"""
        try:
            if self.csv_path and os.path.exists(self.csv_path):
                self.user_data = pd.read_csv(self.csv_path)
                print(f"사용자 데이터 로드 완료: {len(self.user_data)} 개 레코드 ({self.csv_path})")
            else:
                print("CSV 파일을 찾을 수 없습니다. 더미 데이터를 생성합니다.")
                self._create_dummy_data()
                
        except Exception as e:
            print(f"사용자 데이터 로드 중 오류 발생: {e}")
            print("더미 데이터를 생성합니다.")
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """더미 사용자 데이터 생성"""
        dummy_data = {
            'user_id': [
                '0x06fa1ba7a7e44621a2338e6093e53341',
                '0x1234567890abcdef1234567890abcdef',
                '0x9876543210fedcba9876543210fedcba',
                '0xabcdef1234567890abcdef1234567890',
                '0xfedcba0987654321fedcba0987654321'
            ],
            'user_name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 22, 28, 35],
            'gender': ['F', 'M', 'M', 'F', 'F'],
            'like_list_padded': [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [1, 3, 5, 7, 9],
                [2, 4, 6, 8, 10],
                [1, 4, 7, 10, 13]
            ]
        }
        self.user_data = pd.DataFrame(dummy_data)
        print(f"더미 사용자 데이터 생성 완료: {len(self.user_data)} 개 레코드")
    
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
        
        # 해당 사용자의 레코드 조회
        user_records = self.user_data[self.user_data['user_id'] == user_id]
        
        if user_records.empty:
            return None
        
        # 첫 번째 레코드 선택
        user_record = user_records.iloc[0]
        
        return {
            'user_id': user_record['user_id'],
            'user_name': user_record['user_name'],
            'age': user_record['age'],
            'gender': user_record['gender'],
            'like_list': user_record.get('like_list_padded', [])
        }
    
    def get_all_users(self) -> pd.DataFrame:
        """모든 사용자 데이터 반환"""
        return self.user_data if self.user_data is not None else pd.DataFrame()
    
    def get_user_list(self) -> List[str]:
        """사용자 ID 목록 반환"""
        if self.user_data is None or self.user_data.empty:
            return []
        return self.user_data['user_id'].unique().tolist()
    
    def get_user_count(self) -> int:
        """사용자 수 반환"""
        if self.user_data is None or self.user_data.empty:
            return 0
        return len(self.user_data['user_id'].unique())
