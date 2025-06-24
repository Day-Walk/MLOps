from airflow import DAG  
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
from datetime import datetime, timedelta
from io import StringIO
import pandas as pd
import requests
import logging
import os
from dotenv import load_dotenv
from sklearn.preprocessing import OrdinalEncoder
import joblib
import json

# 환경 변수 로드
load_dotenv()

# API 기본 정보
BASE_URL = "http://openapi.seoul.go.kr:8088"
API_KEY = os.getenv("SEOUL_API_KEY")

# CrowdPreprocessor 클래스 추가
class CrowdPreprocessor:
    def __init__(self):
        self.df = None
    
    def safe_float_convert(self, value):
        """안전한 float 변환 함수"""
        if value is None or value == '' or value == 'null':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def process_json(self, json_data):
        try:
            citydata = json_data.get('CITYDATA', {})

            # 안전한 데이터 추출
            ppltn_data = citydata.get('LIVE_PPLTN_STTS')
            road_data = citydata.get('ROAD_TRAFFIC_STTS')
            weather_data = citydata.get('WEATHER_STTS')

            # 인구 데이터
            if isinstance(ppltn_data, list) and len(ppltn_data) > 0:
                ppltn = ppltn_data[0]
            elif isinstance(ppltn_data, dict):
                ppltn = ppltn_data
            else:
                logging.warning("LIVE_PPLTN_STTS 데이터를 찾을 수 없습니다.")
                return None

            # 도로 데이터
            if isinstance(road_data, dict) and 'AVG_ROAD_DATA' in road_data:
                road = road_data['AVG_ROAD_DATA']
            elif isinstance(road_data, list) and len(road_data) > 0:
                road = road_data[0].get('AVG_ROAD_DATA', {})
            else:
                logging.warning("ROAD_TRAFFIC_STTS 데이터를 찾을 수 없습니다.")
                road = {}

            # 날씨 데이터
            if isinstance(weather_data, list) and len(weather_data) > 0:
                weather = weather_data[0]
            elif isinstance(weather_data, dict):
                weather = weather_data
            else:
                logging.warning("WEATHER_STTS 데이터를 찾을 수 없습니다.")
                weather = {}

        except (KeyError, IndexError, TypeError) as e:
            logging.error(f"데이터 구조 파싱 오류: {e}")
            return None

        # 모델이 기대하는 모든 피처 생성
        data = {
            'AREA_NM': ppltn.get('AREA_NM', ''),
            'AREA_CONGEST_LVL': ppltn.get('AREA_CONGEST_LVL', ''),

            'PPLTN_RATE_0': self.safe_float_convert(ppltn.get('PPLTN_RATE_0')),
            'PPLTN_RATE_10': self.safe_float_convert(ppltn.get('PPLTN_RATE_10')),
            'PPLTN_RATE_20': self.safe_float_convert(ppltn.get('PPLTN_RATE_20')),
            'PPLTN_RATE_30': self.safe_float_convert(ppltn.get('PPLTN_RATE_30')),
            'PPLTN_RATE_40': self.safe_float_convert(ppltn.get('PPLTN_RATE_40')),
            'PPLTN_RATE_50': self.safe_float_convert(ppltn.get('PPLTN_RATE_50')),
            'PPLTN_RATE_60': self.safe_float_convert(ppltn.get('PPLTN_RATE_60')),
            'PPLTN_RATE_70': self.safe_float_convert(ppltn.get('PPLTN_RATE_70')),

            'MALE_PPLTN_RATE': self.safe_float_convert(ppltn.get('MALE_PPLTN_RATE')),
            'FEMALE_PPLTN_RATE': self.safe_float_convert(ppltn.get('FEMALE_PPLTN_RATE')),
            'RESNT_PPLTN_RATE': self.safe_float_convert(ppltn.get('RESNT_PPLTN_RATE')),
            'NON_RESNT_PPLTN_RATE': self.safe_float_convert(ppltn.get('NON_RESNT_PPLTN_RATE')),

            'ROAD_TRAFFIC_IDX': self.safe_float_convert(road.get('ROAD_TRAFFIC_IDX')),
            'ROAD_TRAFFIC_SPD': self.safe_float_convert(road.get('ROAD_TRAFFIC_SPD')),

            'TEMP': self.safe_float_convert(weather.get('TEMP')),
            'HUMIDITY': self.safe_float_convert(weather.get('HUMIDITY')),
            'SENSIBLE_TEMP': self.safe_float_convert(weather.get('SENSIBLE_TEMP')),
            'PRECIPITATION': self.safe_float_convert(weather.get('PRECIPITATION')),
            'UV_INDEX_LVL': self.safe_float_convert(weather.get('UV_INDEX_LVL')),
            'WIND_SPD': self.safe_float_convert(weather.get('WIND_SPD')),

            'PM10': self.safe_float_convert(weather.get('PM10')),
            'PM25': self.safe_float_convert(weather.get('PM25')),
            'AIR_IDX_MVL': self.safe_float_convert(weather.get('AIR_IDX_MVL')),

            'EVENT_COUNT': 0,
            'CALL_API_TIME': datetime.now()
        }

        self.df = pd.DataFrame([data])
        return self.df

# 콜백 함수들
def success_callback(context):
    logging.info(f"작업 성공: {context['task_instance'].task_id}")

def slack_alert(context):
    import os
    import json
    import requests

    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logging.warning("SLACK_WEBHOOK_URL이 설정되지 않았습니다.")
        return
        
    task_instance = context.get('task_instance')
    dag_id = context.get('dag').dag_id
    task_id = task_instance.task_id
    execution_date = context.get('execution_date')
    log_url = task_instance.log_url

    message = {
        "text": f"""
        ❌ *Task Failed!*
        - DAG: `{dag_id}`
        - Task: `{task_id}`
        - Time: `{execution_date}`
        - <{log_url}|View logs>
        """
    }

    try:
        requests.post(
            webhook_url, data=json.dumps(message),
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        logging.error(f"Slack 알림 전송 실패: {e}")

def failure_callback(context):
    logging.error(f"작업 실패: {context['task_instance'].task_id}")

# 1. API 호출하는 코드 
def load_data_from_api(**context):
    # API 키 검증
    if not API_KEY:
        raise ValueError("SEOUL_API_KEY 환경변수가 설정되지 않았습니다.")
    
    # 파일 경로 검증
    base_dir = os.path.dirname(os.path.abspath(__file__))
    area_info_path = os.path.join(base_dir, "../data/seoul_rtd_categories.csv")
    
    if not os.path.exists(area_info_path):
        raise FileNotFoundError(f"area_info.xlsx 파일을 찾을 수 없습니다: {area_info_path}")
    df = pd.read_csv(area_info_path)
    df.rename(columns={
    "area_nm": "AREA_NM"
}, inplace=True)

    locations = []
    for _, row in df.iterrows():
        locations.append({"name": row["AREA_NM"]})

    all_data = []

    for i, loc in enumerate(locations):
        url = f"{BASE_URL}/{API_KEY}/json/citydata/1/5/{loc['name']}"
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()

            # 첫 번째 지역만 상세 분석
            if i == 0:
                logging.info(f"=== {loc['name']} API 응답 구조 분석 (첫 번째 지역) ===")
                if 'CITYDATA' in json_data:
                    citydata = json_data['CITYDATA']
                    logging.info(f"CITYDATA keys: {list(citydata.keys())}")
                    
                    # 각 섹션의 타입 확인
                    if 'LIVE_PPLTN_STTS' in citydata:
                        logging.info(f"LIVE_PPLTN_STTS type: {type(citydata['LIVE_PPLTN_STTS'])}")
                        if isinstance(citydata['LIVE_PPLTN_STTS'], list):
                            logging.info(f"LIVE_PPLTN_STTS length: {len(citydata['LIVE_PPLTN_STTS'])}")
                            if len(citydata['LIVE_PPLTN_STTS']) > 0:
                                logging.info(f"LIVE_PPLTN_STTS[0] keys: {list(citydata['LIVE_PPLTN_STTS'][0].keys())}")
                        elif isinstance(citydata['LIVE_PPLTN_STTS'], dict):
                            logging.info(f"LIVE_PPLTN_STTS keys: {list(citydata['LIVE_PPLTN_STTS'].keys())}")
                    
                    if 'ROAD_TRAFFIC_STTS' in citydata:
                        logging.info(f"ROAD_TRAFFIC_STTS type: {type(citydata['ROAD_TRAFFIC_STTS'])}")
                        if isinstance(citydata['ROAD_TRAFFIC_STTS'], dict):
                            logging.info(f"ROAD_TRAFFIC_STTS keys: {list(citydata['ROAD_TRAFFIC_STTS'].keys())}")
                            if 'AVG_ROAD_DATA' in citydata['ROAD_TRAFFIC_STTS']:
                                road_data = citydata['ROAD_TRAFFIC_STTS']['AVG_ROAD_DATA']
                                logging.info(f"AVG_ROAD_DATA type: {type(road_data)}")
                                if isinstance(road_data, dict):
                                    logging.info(f"AVG_ROAD_DATA keys: {list(road_data.keys())}")
                        
                    if 'WEATHER_STTS' in citydata:
                        logging.info(f"WEATHER_STTS type: {type(citydata['WEATHER_STTS'])}")
                        if isinstance(citydata['WEATHER_STTS'], list):
                            logging.info(f"WEATHER_STTS length: {len(citydata['WEATHER_STTS'])}")
                            if len(citydata['WEATHER_STTS']) > 0:
                                logging.info(f"WEATHER_STTS[0] keys: {list(citydata['WEATHER_STTS'][0].keys())}")
                        elif isinstance(citydata['WEATHER_STTS'], dict):
                            logging.info(f"WEATHER_STTS keys: {list(citydata['WEATHER_STTS'].keys())}")
                else:
                    logging.warning(f"{loc['name']}: CITYDATA 키가 없습니다. 전체 응답 구조: {list(json_data.keys())}")
            else:
                logging.info(f"{loc['name']} API 호출 성공")
            
            # API 응답을 CrowdPreprocessor가 기대하는 구조로 변환
            converted_data = {
                 "CITYDATA": json_data.get("CITYDATA", {})  
            }
            all_data.append({
                "AREA_NM": loc["name"], 
                "raw_json": converted_data
            })
        else:
            logging.warning(f"{loc['name']} 호출 실패: {response.status_code}")

    context["ti"].xcom_push(key="raw_api_data", value=json.dumps(all_data))
    logging.info(f"API 데이터 로드 완료: {len(all_data)}개 지역")

# 2. 전처리
def preprocess(**context):
    # API 데이터 가져오기
    raw_data_json = context["ti"].xcom_pull(key="raw_api_data", task_ids="load_data_from_api")
    raw_data = json.loads(raw_data_json)
    
    logging.info(f"전처리 시작: {len(raw_data)}개 지역 데이터")
    
    preprocessor = CrowdPreprocessor()
    processed_dataframes = []
    
    for item in raw_data:
        try:
            processed_df = preprocessor.process_json(item["raw_json"])
            if processed_df is not None:
                processed_dataframes.append(processed_df)
                logging.info(f"{item['AREA_NM']} 전처리 완료")
        except Exception as e:
            logging.error(f"{item['AREA_NM']} 전처리 중 오류: {str(e)}")
    
    if len(processed_dataframes) == 0:
        raise ValueError("처리된 데이터가 없습니다.")
    
    # 데이터 합치기
    combined_df = pd.concat(processed_dataframes, ignore_index=True)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    area_info_path = os.path.join(base_dir, "../data/seoul_rtd_categories.csv")
    area_info = pd.read_csv(area_info_path)[["area_nm", "category", "x", "y"]]
    area_info.columns = ["AREA_NM", "CATEGORY", "X", "Y"]
    combined_df = combined_df.merge(area_info, on="AREA_NM", how="left")

    # 모델 학습 시와 동일한 구조 확인
    # airflow.py에서 제외한 컬럼들: AREA_NM, AREA_CD, CALL_API_TIME, AREA_CONGEST_LVL
    logging.info(f"전체 컬럼: {list(combined_df.columns)}")
    
    # 결측치 처리
    combined_df = combined_df.fillna(0)
    
    # XCom으로 전체 데이터 전달 (예측 함수에서 피처 선택)
    context["ti"].xcom_push(key="df_sample", value=combined_df.to_json(orient="split"))
    
    # 호환성을 위해 X_data도 전달 (사용하지 않지만)
    exclude_cols = ["AREA_NM", "CALL_API_TIME", "AREA_CONGEST_LVL"]
    feature_cols = [col for col in combined_df.columns if col not in exclude_cols]
    X = combined_df[feature_cols].copy()
    context["ti"].xcom_push(key="X_data", value=X.to_json(orient="split"))
    
    logging.info("전처리 완료")


# 3. 예측 (배치 단위 처리)
def predict(**context):
    # 이전 task에서 데이터 가져오기
    X_json = context["ti"].xcom_pull(key="X_data", task_ids="preprocess")
    df_json = context["ti"].xcom_pull(key="df_sample", task_ids="preprocess")
    
    X = pd.read_json(StringIO(X_json), orient="split")
    df_sample = pd.read_json(StringIO(df_json), orient="split")
    
    # 모델 로드
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "../model/rf_model.pkl")
    
    if not os.path.exists(model_path):
        alternative_paths = [
            os.path.join(base_dir, "../model/rf_model.pkl"),
            "/home/ubuntu/MLOps/MLOps/app/airflow/model/rf_model.pkl"
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            raise FileNotFoundError("모델 파일을 찾을 수 없습니다.")
    
    model = joblib.load(model_path)
    
    # 모델 학습 시와 동일한 피처 구조로 맞추기
    exclude_cols = ["AREA_NM", "CALL_API_TIME", "AREA_CONGEST_LVL"]
    feature_cols = [col for col in df_sample.columns if col not in exclude_cols]
    
    # 피처 데이터 추출
    X_for_prediction = df_sample[feature_cols].copy()

    # 예측에 불필요한 컬럼 제거
    drop_cols = ["CATEGORY", "X", "Y"]
    X_for_prediction = X_for_prediction.drop(columns=drop_cols, errors="ignore")
    
    # 새로운 일관된 인코딩 함수 적용
    def apply_consistent_encoding(df):
        df_encoded = df.copy()
        
        # 범주형 컬럼별로 고정된 인코딩 적용
        categorical_mappings = {
            # 실제 학습 데이터에서 사용된 매핑을 여기에 정의
            # 현재 API 데이터에서 나올 수 있는 범주형 변수들
            'AREA_CONGEST_LVL': {'여유': 0, '보통': 1, '약간 붐빔': 2, '붐빔': 3},
            # 추가 범주형 변수가 있다면 여기에 추가
        }
        
        for col in df_encoded.select_dtypes(include="object").columns:
            if col in categorical_mappings:
                df_encoded[col] = df_encoded[col].map(categorical_mappings[col]).fillna(0)
            else:
                # 알 수 없는 범주형 변수는 숫자로 변환
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)
        
        return df_encoded
    
    # 일관된 인코딩 적용
    X_for_prediction = apply_consistent_encoding(X_for_prediction)
    
    #  추가: numpy 배열로 변환하여 피처 이름 검증 우회
    X_for_prediction = pd.DataFrame(X_for_prediction.values, 
                                   columns=X_for_prediction.columns)

    # 결측치 처리
    X_for_prediction = X_for_prediction.fillna(0)
    
    # 데이터 타입 확인
    for col in X_for_prediction.columns:
        if X_for_prediction[col].dtype == 'object':
            X_for_prediction[col] = pd.to_numeric(X_for_prediction[col], errors='coerce').fillna(0)
    
    logging.info(f"예측용 피처 컬럼: {list(X_for_prediction.columns)}")
    logging.info(f"예측 데이터 shape: {X_for_prediction.shape}")
    
    # 배치 예측
    batch_size = 1000
    predictions = []
    
    for i in range(0, len(X_for_prediction), batch_size):
        batch_X = X_for_prediction.iloc[i:i+batch_size]
        
        # 수정된 부분: 오류 처리 추가
        try:
            batch_pred = model.predict(batch_X)
        except ValueError as e:
            if "feature names should match" in str(e):
                logging.warning("피처 이름 불일치로 인한 오류 발생. numpy 배열로 변환하여 재시도합니다.")
                # 피처 이름 검증을 우회하기 위해 numpy 배열로 변환
                batch_pred = model.predict(batch_X.values)
            else:
                logging.error(f"예측 중 예상치 못한 오류 발생: {str(e)}")
                raise e
        
        predictions.extend(batch_pred)
        
        if i % (batch_size * 5) == 0 or i + batch_size >= len(X_for_prediction):
            logging.info(f"예측 진행률: {min(i + batch_size, len(X_for_prediction))}/{len(X_for_prediction)}")
    
    # 결과 저장
    df_sample["PREDICTED_CONGESTION"] = predictions
    context["ti"].xcom_push(key="result_df", value=df_sample.to_json(orient="split"))
    
    logging.info("예측 완료")

# 4. 저장
def save_result(**context):
    # 이전 task인 predict에서 XCom으로 전달한 JSON 문자열을 가져옴
    df_json = context["ti"].xcom_pull(key="result_df", task_ids="predict")
    result_df = pd.read_json(StringIO(df_json), orient="split")

    # 저장 경로 설정 (절대 경로 사용)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "../data/live_crowd")
    os.makedirs(output_dir, exist_ok=True)

    # 파일에 시간정보 붙여서 중복 방지
    now = datetime.now().strftime("%Y%m%d%H")
    file_name = f"crowdness_pred_{now}.csv"
    file_path = os.path.join(output_dir, file_name)

    required_columns = ["AREA_NM", "CATEGORY", "AREA_CONGEST_LVL", "X", "Y", "PREDICTED_CONGESTION"]
    for col in required_columns:
        if col not in result_df.columns:
            result_df[col] = None  # 또는 기본값 지정

    # 필요한 컬럼만 저장
    save_df = result_df[["AREA_NM", "CATEGORY", "AREA_CONGEST_LVL", "X", "Y", "PREDICTED_CONGESTION"]]
    save_df.to_csv(file_path, index=False)

    logging.info(f"결과 저장 완료: {file_path}")

# DAG 정의
default_args = {
    'owner': 'day_walk',
    'start_date': datetime(2025, 6, 16),
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'on_failure_callback': slack_alert,
    'on_success_callback': success_callback,
}

# DAG 생성
dag = DAG(
    dag_id="predict_congestion_hourly_xcom",
    default_args=default_args,
    description="혼잡도 예측 모델을 1시간마다 실행 (XCom 기반)",
    schedule_interval="0 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "prediction", "xcom"]
)

# Task 정의
t1 = PythonOperator(
    task_id="load_data_from_api",
    python_callable=load_data_from_api,
    dag=dag
)

t2 = PythonOperator(
    task_id="preprocess",
    python_callable=preprocess,
    dag=dag
)

t3 = PythonOperator(
    task_id="predict",
    python_callable=predict,
    dag=dag
)

t4 = PythonOperator(
    task_id="save_result",
    python_callable=save_result,
    dag=dag
)

# Task 의존성 설정
t1 >> t2 >> t3 >> t4

globals()["predict_congestion_hourly_xcom"] = dag
