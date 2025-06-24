import json
import pandas as pd
from datetime import datetime

class CrowdPreprocessor:
    def __init__(self):
        self.df = None

    def safe_float_convert(self, value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def process_json(self, json_data):
        try:
            ppltn = json_data['CITYDATA']['LIVE_PPLTN_STTS'][0]
            road = json_data['CITYDATA']['ROAD_TRAFFIC_STTS']['AVG_ROAD_DATA']
            weather = json_data['CITYDATA']['WEATHER_STTS'][0]
        except (KeyError, IndexError, TypeError):
            return None

        # 기본 데이터 추출
        data = {
            'AREA_NM': ppltn.get('AREA_NM'),
            'AREA_CONGEST_LVL': ppltn.get('AREA_CONGEST_LVL'),
            'ROAD_TRAFFIC_IDX': road.get('ROAD_TRAFFIC_IDX'),
            'PPLTN_RATE_20': self.safe_float_convert(ppltn.get('PPLTN_RATE_20')),
            'PPLTN_RATE_30': self.safe_float_convert(ppltn.get('PPLTN_RATE_30')),
            'PPLTN_RATE_40': self.safe_float_convert(ppltn.get('PPLTN_RATE_40')),
            'TEMP': self.safe_float_convert(weather.get('TEMP')),
            'HUMIDITY': self.safe_float_convert(weather.get('HUMIDITY')),
            'ROAD_TRAFFIC_SPD': self.safe_float_convert(road.get('ROAD_TRAFFIC_SPD')),
            'CALL_API_TIME': datetime.now()  
        }
        
        # 데이터프레임 생성
        self.df = pd.DataFrame([data])
        
        # 시간 특성 생성
        self.df['hour'] = self.df['CALL_API_TIME'].dt.hour
        self.df['day_of_week'] = self.df['CALL_API_TIME'].dt.dayofweek
        self.df['is_rush_hour'] = ((self.df['hour'].between(7, 9)) | 
                                  (self.df['hour'].between(17, 19))).astype(int)
        
        return self.df

# 사용 예시
if __name__ == "__main__":
    # JSON 파일 읽기
    with open('서울대공원_20250520_000003.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 데이터 처리
    preprocessor = CrowdPreprocessor()
    processed_data = preprocessor.process_json(json_data)
    print(processed_data)
