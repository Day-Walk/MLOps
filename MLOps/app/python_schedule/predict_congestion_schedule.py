import os
import pandas as pd
import requests
from datetime import datetime
import schedule
import time
import numpy as np
from dotenv import load_dotenv
import joblib
from pathlib import Path
from collections import defaultdict
import urllib.parse 

class CongestionDataPreprocessor:
    def __init__(self):
        self.numeric_cols = [
            'MALE_PPLTN_RATE', 'FEMALE_PPLTN_RATE', 'PPLTN_RATE_0', 'PPLTN_RATE_10', 
            'PPLTN_RATE_20', 'PPLTN_RATE_30', 'PPLTN_RATE_40', 'PPLTN_RATE_50', 
            'PPLTN_RATE_60', 'PPLTN_RATE_70', 'RESNT_PPLTN_RATE', 'NON_RESNT_PPLTN_RATE',
            'ROAD_TRAFFIC_SPD', 'TEMP', 'SENSIBLE_TEMP', 'HUMIDITY', 'WIND_SPD', 
            'UV_INDEX_LVL', 'PM25', 'PM10', 'AIR_IDX_MVL', 'EVENT_COUNT'
        ]
        self.non_numeric_cols = ['AREA_CONGEST_LVL', 'ROAD_TRAFFIC_IDX', 'PRECIPITATION']

    def _safe_float_convert(self, value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def process_initial_json(self, json_data):
        try:
            ppltn = json_data['CITYDATA']['LIVE_PPLTN_STTS'][0]
            road = json_data['CITYDATA']['ROAD_TRAFFIC_STTS']['AVG_ROAD_DATA']
            weather = json_data['CITYDATA']['WEATHER_STTS'][0]
            events = json_data['CITYDATA'].get('EVENT_STTS', [])
        except (KeyError, IndexError, TypeError) as e:
            print(f"JSON 처리 오류: {e}")
            return None

        data = {
            'AREA_NM': ppltn.get('AREA_NM'),
            'AREA_CD': ppltn.get('AREA_CD'),
            'CALL_API_TIME': datetime.now(),
            'AREA_CONGEST_LVL': ppltn.get('AREA_CONGEST_LVL'),
            'MALE_PPLTN_RATE': self._safe_float_convert(ppltn.get('MALE_PPLTN_RATE')),
            'FEMALE_PPLTN_RATE': self._safe_float_convert(ppltn.get('FEMALE_PPLTN_RATE')),
            'PPLTN_RATE_0': self._safe_float_convert(ppltn.get('PPLTN_RATE_0')),
            'PPLTN_RATE_10': self._safe_float_convert(ppltn.get('PPLTN_RATE_10')),
            'PPLTN_RATE_20': self._safe_float_convert(ppltn.get('PPLTN_RATE_20')),
            'PPLTN_RATE_30': self._safe_float_convert(ppltn.get('PPLTN_RATE_30')),
            'PPLTN_RATE_40': self._safe_float_convert(ppltn.get('PPLTN_RATE_40')),
            'PPLTN_RATE_50': self._safe_float_convert(ppltn.get('PPLTN_RATE_50')),
            'PPLTN_RATE_60': self._safe_float_convert(ppltn.get('PPLTN_RATE_60')),
            'PPLTN_RATE_70': self._safe_float_convert(ppltn.get('PPLTN_RATE_70')),
            'RESNT_PPLTN_RATE': self._safe_float_convert(ppltn.get('RESNT_PPLTN_RATE')),
            'NON_RESNT_PPLTN_RATE': self._safe_float_convert(ppltn.get('NON_RESNT_PPLTN_RATE')),
            'ROAD_TRAFFIC_IDX': road.get('ROAD_TRAFFIC_IDX'),
            'ROAD_TRAFFIC_SPD': self._safe_float_convert(road.get('ROAD_TRAFFIC_SPD')),
            'TEMP': self._safe_float_convert(weather.get('TEMP')),
            'SENSIBLE_TEMP': self._safe_float_convert(weather.get('SENSIBLE_TEMP')),
            'HUMIDITY': self._safe_float_convert(weather.get('HUMIDITY')),
            'WIND_SPD': self._safe_float_convert(weather.get('WIND_SPD')),
            'PRECIPITATION': weather.get('PRECIPITATION'),
            'UV_INDEX_LVL': int(self._safe_float_convert(weather.get('UV_INDEX_LVL'))),
            'PM25': self._safe_float_convert(weather.get('PM25')),
            'PM10': self._safe_float_convert(weather.get('PM10')),
            'AIR_IDX_MVL': self._safe_float_convert(weather.get('AIR_IDX_MVL')),
            'EVENT_COUNT': len(events),
        }
        return pd.DataFrame([data])

    def _create_time_features(self, df):
        df['adjusted_datetime'] = pd.to_datetime(df['CALL_API_TIME']).apply(
            lambda dt: dt.replace(minute=0 if dt.minute < 30 else 30, second=0, microsecond=0)
        )
        df['month'] = df['adjusted_datetime'].dt.month
        df['day'] = df['adjusted_datetime'].dt.day
        df['weekday'] = df['adjusted_datetime'].dt.dayofweek
        df['hour_30'] = df['adjusted_datetime'].apply(lambda x: x.hour * 2 + x.minute // 30)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_sin'] = np.sin(2 * np.pi * df['day']/df['adjusted_datetime'].dt.days_in_month)
        df['day_cos'] = np.cos(2 * np.pi * df['day']/df['adjusted_datetime'].dt.days_in_month)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/7)
        df['hour_30_sin'] = np.sin(2 * np.pi * df['hour_30']/48)
        df['hour_30_cos'] = np.cos(2 * np.pi * df['hour_30']/48)
        return df

    def _encode(self, df):
        congest_map = {'여유': 1, '보통': 2, '약간 붐빔': 3, '붐빔': 4}
        traffic_map = {'원활': 1, '서행': 2, '정체': 3}
        precip_map = {'-': 0.0, '~1mm': 0.5, '1.0mm': 1.0, '1.4mm': 1.4, '1.5mm': 1.5, '1.7mm': 1.7, '1.8mm': 1.8, '2.0mm': 2.0, '2.5mm': 2.5}

        for col in [c for c in df.columns if 'CONGEST' in c or 'AFTER' in c]:
            df[col] = df[col].map(congest_map)
        df['ROAD_TRAFFIC_IDX'] = df['ROAD_TRAFFIC_IDX'].map(traffic_map)
        df['PRECIPITATION'] = df['PRECIPITATION'].map(precip_map).fillna(0)
        return pd.get_dummies(df, columns=['AREA_NM'], prefix='AREA')

    def preprocess_for_prediction(self, live_json_data):
        df = self.process_initial_json(live_json_data)
        if df is None or df.empty:
            return None
        df = self._create_time_features(df)
        df = self._encode(df)
        return df

# 환경 변수 로드python -m app.python_schedule.predict_congestion_schedule
env_path = Path(__file__).resolve().parents[2] / ".env" 
load_dotenv(dotenv_path="/home/ubuntu/MLOps/MLOps/.env")
BASE_URL = "http://openapi.seoul.go.kr:8088"
API_KEY = os.getenv("SEOUL_API_KEY")

print(f"[DEBUG] API_KEY: {API_KEY}")

def load_data_from_api_local():
    print(" 1. API 데이터 불러오는 중...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    area_info_path = "/home/ubuntu/MLOps/MLOps/app/python_schedule/data/seoul_rtd_categories.csv"
    df = pd.read_csv(area_info_path)
    df.rename(columns={"area_nm": "AREA_NM", "category": "CATEGORY", "x": "X", "y": "Y"}, inplace=True)

    all_data = []

    for _, row in df.iterrows():
        area_encoded = urllib.parse.quote(row['AREA_NM'])  # 한글 깨짐 방지
        url = f"{BASE_URL}/{API_KEY}/json/citydata/1/5/{area_encoded}"
        response = requests.get(url)

        print(f"[DEBUG] 응답 상태코드: {response.status_code}")
        print(f"[DEBUG] 응답 내용: {response.text[:100]}")  # 응답 앞 100자 출력

        if response.status_code == 200:
            if response.text.strip():  # 응답이 비어있지 않다면
                try:
                    json_data = response.json()
                    converted_data = { "CITYDATA": json_data.get("CITYDATA", {}) }

                    all_data.append({
                        "AREA_NM": row["AREA_NM"],
                        "CATEGORY": row["CATEGORY"],
                        "X": row["X"],
                        "Y": row["Y"],
                        "raw_json": converted_data
                    })
                except Exception as e:
                    print(f" JSON 파싱 실패: {url} / 오류: {e}")
            else:
                print(f" 빈 응답입니다: {url}")
        else:
            print(f" API 호출 실패: {url} / 상태코드: {response.status_code}")

    print(f"[DEBUG] 수집된 전체 데이터 수: {len(all_data)}")
    for i, d in enumerate(all_data[:3]):
        print(f"[DEBUG] 샘플 {i+1}: {d}")

    return all_data  



def predict_local(json_data, area_info):
    import glob
    
    print(" 3. CatBoost 예측 중...")

    preprocessor = CongestionDataPreprocessor()
    predict_df = preprocessor.preprocess_for_prediction(json_data)
    print(" 2. 전처리 완료")
    print(f"[DEBUG] 생성된 DataFrame 길이: {len(predict_df)}")
    print(f"[DEBUG] 컬럼 목록: {predict_df.columns}")
    print(f"[DEBUG] 첫 줄: {predict_df.head(1)}")    

    if predict_df is None or predict_df.empty:
        raise ValueError("예측에 사용할 데이터가 없습니다.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    model_files = sorted(glob.glob(os.path.join(model_dir, "catboost_model_AFTER_*.pkl")))

    CONGEST_LVL_MAP_INV = {1: '여유', 2: '보통', 3: '약간 붐빔', 4: '붐빔'}
    results = []

    for model_path in model_files:
        try:
            with open(model_path, 'rb') as f:
                model = joblib.load(f)

            final_df = predict_df.reindex(columns=model.feature_names_, fill_value=0)
            pred_class = model.predict(final_df)[0].item()
            pred_proba = model.predict_proba(final_df)[0]

            target_name = os.path.basename(model_path).replace("catboost_model_", "").replace(".pkl", "")
            proba_dict = {
                CONGEST_LVL_MAP_INV.get(cls, f"클래스 {cls}"): f"{p*100:.2f}%"
                for cls, p in zip(model.classes_, pred_proba)
            }

            print(f"[DEBUG] 결과 타겟: {target_name}")
            results.append({
                "AREA_NM": area_info["AREA_NM"],
                "CATEGORY": area_info["CATEGORY"],
                "AREA_CONGEST_LVL": predict_df.iloc[0].get("AREA_CONGEST_LVL", ""),
                "X": area_info["X"],
                "Y": area_info["Y"],
                "target": target_name,
                "prediction": CONGEST_LVL_MAP_INV.get(pred_class, "알 수 없음"),
                "probabilities": proba_dict
            })

        except Exception as e:
            print(f"[ 예측 실패] {model_path}: {e}")
            continue

    return results


from collections import defaultdict

def save_result_local(results):
    from datetime import datetime
    import pandas as pd
    import os

    now = datetime.now().strftime("%Y%m%d%H")
    output_dir = "data/live_crowd"
    os.makedirs(output_dir, exist_ok=True)

    hour_map = {
        "AFTER_ONE_HOUR": 1,
        "AFTER_TWO_HOUR": 2,
        "AFTER_THREE_HOUR": 3,
        "AFTER_SIX_HOUR": 6,
        "AFTER_TWELVE_HOUR": 12
    }

    #  결과를 타겟별로 그룹화
    grouped = defaultdict(list)
    for res in results:
        grouped[res["target"]].append(res)

    for target, res_list in grouped.items():
        file_index = hour_map.get(target)
        if file_index is None:
            continue  # 매핑 안 되면 저장 안 함

        filename = f"crowdness_pred_{now}_{file_index}.csv"
        file_path = os.path.join(output_dir, filename)

        df = pd.DataFrame([{
            "AREA_NM": r["AREA_NM"],
            "CATEGORY": r["CATEGORY"],
            "AREA_CONGEST_LVL": r["AREA_CONGEST_LVL"],
            "X": r["X"],
            "Y": r["Y"],
            "PREDICTED_CONGESTION": r["prediction"]
        } for r in res_list])

        df.to_csv(file_path, index=False)
        print(f" 저장 완료: {file_path} / 총 {len(df)}개 장소")


def send_slack_alert(message: str):
    import os
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    
    if not SLACK_WEBHOOK_URL:
        print(" 슬랙 URL이 없어요")
        return

    payload = {"text": message}
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code == 200:
            print(" 슬랙 알림 전송 성공")
        else:
            print(f"❌ 슬랙 실패: {response.status_code} / {response.text}")
    except Exception as e:
        print(f" 슬랙 전송 에러: {e}")


def run_pipeline():
    print("\n=== 혼잡도 예측 파이프라인 시작 ===")

    try:
        all_data = load_data_from_api_local()

        results = []
        for area in all_data:
            area_results = predict_local(area["raw_json"], area)
            results.extend(area_results)

        save_result_local(results)
        print("=== 파이프라인 완료 ===")

    except Exception as e:
        error_message = f" 혼잡도 예측 파이프라인 실패!\n사유: {str(e)}"
        send_slack_alert(error_message)
        print(error_message)

import sys

if len(sys.argv) > 1 and sys.argv[1] == "now":
    # 수동 실행
    run_pipeline()
else:
    # 자동 실행: 정각마다 실행되도록 예약
    schedule.every().hour.at(":00").do(run_pipeline)
    run_pipeline()  # 처음 실행 시 1번 실행
    while True:
        schedule.run_pending()
        time.sleep(1)