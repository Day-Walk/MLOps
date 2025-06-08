import pandas as pd
from typing import Dict

class UserDataService:
    """사용자 데이터 관리 서비스"""
    
    def __init__(self):
        self.data_path = "/home/ubuntu/working/MLOps/data/user.csv"
        self.user_data = pd.read_csv(self.data_path)
        
    def get_user_info(self, user_id: str) -> Dict:
        """사용자 정보 조회"""
        return self.user_data[self.user_data['user_id'] == user_id]