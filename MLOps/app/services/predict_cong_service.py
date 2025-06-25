import sys
import os
import requests
import pandas as pd
from dotenv import load_dotenv
import joblib
from datetime import datetime, timedelta
from collections import defaultdict
import schedule
import time

# --- 경로 설정 및 초기화 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.src.congestion_preprocessing import CongestionDataPreprocessor

# --- 환경 변수 로드 ---
load_dotenv()

BASE_URL = "http://openapi.seoul.go.kr:8088"
API_KEY = os.getenv("SEOUL_API_KEY")
PLACE_PATH = os.getenv("PLACE_PATH")

# --- 모델 경로 로드 ---
MODEL_PATHS = {
    "1-hour": os.getenv("ONE_HOUR_MODEL_PATH"),
    "2-hour": os.getenv("TWO_HOUR_MODEL_PATH"),
    "3-hour": os.getenv("THREE_HOUR_MODEL_PATH"),
    "6-hour": os.getenv("SIX_HOUR_MODEL_PATH"),
    "12-hour": os.getenv("TWELVE_HOUR_MODEL_PATH"),
}
# 파일명 접미사 매핑
MODEL_FILE_SUFFIX = {"1-hour": 1, "2-hour": 2, "3-hour": 3, "6-hour": 6, "12-hour": 12}


PRED_PATH = os.getenv("PRED_PATH")

# --- 전역 객체 및 모델 로드 ---
preprocessor = CongestionDataPreprocessor()
models = {}
for name, path in MODEL_PATHS.items():
    if path and os.path.exists(path):
        try:
            models[name] = joblib.load(path)
            print(f"✅ Model '{name}' loaded from {path}")
        except Exception as e:
            print(f"❌ Failed to load model '{name}': {e}")
    else:
        print(f"⚠️ Model path for '{name}' not found or not set.")

CONGEST_LVL_MAP = {1: '여유', 2: '보통', 3: '약간 붐빔', 4: '붐빔'}

def predict_and_save_all_locations():
    """
    모든 장소의 실시간 혼잡도를 예측하고 모델별로 분리된 CSV 파일로 저장합니다.
    """
    if not models:
        print("❌ No models are loaded. Exiting.")
        return

    if not PLACE_PATH or not os.path.exists(PLACE_PATH):
        print(f"❌ PLACE_PATH is not set or file not found: {PLACE_PATH}")
        return

    area_info = pd.read_csv(PLACE_PATH)
    area_info.rename(columns={'area_nm': 'AREA_NM'}, inplace=True)
    
    # 모델별 예측 결과를 저장할 딕셔너리
    all_predictions = defaultdict(list)

    for _, row in area_info.iterrows():
        area_name = row['AREA_NM']
        try:
            print(f"Processing {area_name}...")
            # 1. API 호출 및 전처리
            url = f"{BASE_URL}/{API_KEY}/json/citydata/1/5/{area_name}"
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()
            predict_df = preprocessor.preprocess_for_prediction(json_data)

            if predict_df is None:
                print(f"  -> Preprocessing failed.")
                continue

            current_congestion_num = int(predict_df['AREA_CONGEST_LVL'].iloc[0])
            current_congestion_str = CONGEST_LVL_MAP.get(current_congestion_num, "N/A")

            # 2. 모델별 예측 실행
            for name, model in models.items():
                final_predict_df = predict_df.reindex(columns=model.feature_names_, fill_value=0)
                prediction_num = int(model.predict(final_predict_df)[0][0])
                prediction_str = CONGEST_LVL_MAP.get(prediction_num, "Unknown")
                
                location_result = {
                    'AREA_NM': area_name,
                    'CURRENT_CONGESTION': current_congestion_str,
                    'PREDICTED_CONGESTION_NUM': prediction_num,
                    'PREDICTED_CONGESTION_STR': prediction_str
                }
                all_predictions[name].append(location_result)
            print(f"  -> OK")

        except Exception as e:
            print(f"  -> Failed: {e}")
            continue

    # 3. 결과 취합 및 파일 저장
    if not all_predictions:
        print("No predictions were made.")
        return

    # 타임스탬프 (현재 시간 + 1시간)
    timestamp = (datetime.now() + timedelta(hours=1)).strftime("%Y%m%d%H")
    output_dir = PRED_PATH if PRED_PATH and os.path.isdir(PRED_PATH) else os.path.dirname(__file__)

    # 저장된 파일 경로들을 저장할 리스트
    saved_files = []

    for name, predictions_list in all_predictions.items():
        if not predictions_list:
            continue
        
        predictions_df = pd.DataFrame(predictions_list)
        final_df = pd.merge(predictions_df, area_info, on='AREA_NM', how='left')

        column_order = [
            'AREA_NM', 'category', 'x', 'y', 
            'CURRENT_CONGESTION', 
            'PREDICTED_CONGESTION_NUM', 'PREDICTED_CONGESTION_STR'
        ]
        final_df = final_df[[col for col in column_order if col in final_df.columns]]

        file_suffix = MODEL_FILE_SUFFIX.get(name)
        output_filename = f"congestion_predictions_{timestamp}_{file_suffix}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ Predictions for '{name}' saved to: {output_path}")
        saved_files.append(output_path)
    
    return saved_files

if __name__ == "__main__":
    print("Congestion prediction service started.")
    print("Initial prediction run...")
    predict_and_save_all_locations()

    schedule.every().hour.at(":50").do(predict_and_save_all_locations)

    print("Scheduler is running. Waiting for the next scheduled run at XX:50.")
    while True:
        schedule.run_pending()
        time.sleep(1)