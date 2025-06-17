import pandas as pd
from typing import Dict

class UserDataService:
    """사용자 데이터 관리 서비스"""
    
    def __init__(self):
        self.data_path = "/app/data/final_click_log.csv"
        self.user_data = pd.read_csv(self.data_path)
        
    def get_user_info(self, userid: str) -> Dict:
        """사용자 정보 조회"""
        user_df = self.user_data[self.user_data['userid'] == userid][["userid", "name", "age", "gender", "like_list"]]
        return user_df.iloc[0:1]