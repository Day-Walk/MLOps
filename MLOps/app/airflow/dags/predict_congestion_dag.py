from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
from datetime import datetime, timedelta
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

# 콜백 함수들
# 작업 성공 시 어떤 task가 성공했는지 로그로 기록
def success_callback(context):
    logging.info(f"작업 성공: {context['task_instance'].task_id}")

def slack_alert(context):
    import os
    import json
    import requests

    webhook_url = os.getenv("SLACK_WEBHOOK_URL") # .env에 정의함
    task_instance = context.get('task_instance')
    dag_id = context.get('dag').dag_id  # 어떤 dag에서 실패했는지 task 실패인지 언제 실행한건지 확인
    task_id = task_instance.task_id
    execution_date = context.get('execution_date')
    log_url = task_instance.log_url # Airflow 웹에서 해당 Task의 로그를 볼 수 있는 링크

    message = {
        "text": f"""
        ❌ *Task Failed!*
        - DAG: `{dag_id}`
        - Task: `{task_id}`
        - Time: `{execution_date}`
        - <{log_url}|View logs>
        """
    }

    requests.post(
        webhook_url, data=json.dumps(message),
        headers={"Content-Type": "application/json"}
    )


# 작업 실패 시 로그에 실패 메시지를 남김
def failure_callback(context):
    logging.error(f"작업 실패: {context['task_instance'].task_id}")

# 1. 데이터 불러오기
# **context는 Airflow가 전달하는 실행 환경(context) 객체
# XCom을 통해 값을 전달하거나 받아올 수 있음
# 이 함수 역할 경로설정, 예외처리(파일 없으면 DAG 실패 처리), dag 처리, xcom 전송(json 변환), 로그 출력
# 서울시 API에서 데이터 수집
def load_data_from_api(**context):

    area_info_path = "data/area_info.xlsx"
    df = pd.read_excel(area_info_path)

    locations = []
    for _, row in df.iterrows():
        locations.append({"code": row["AREA_CD"], "name": row["AREA_NM"]})

    all_data = []
    now = datetime.now().strftime("%Y%m%d%H%M")

    for loc in locations:
        url = f"{BASE_URL}/{API_KEY}/json/citydata/1/5/{loc['code']}"
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()
            all_data.append({"AREA_CD": loc["code"], "AREA_NM": loc["name"], "data": json_data})
        else:
            logging.warning(f"{loc['name']} 호출 실패: {response.status_code}")

    # JSON → DataFrame 변환 (샘플용 구조)
    records = []
    for item in all_data:
        try:
            congestion = item["data"]["SeoulRtd.citydata"]['LIVE_PPLTN_STTS']['AREA_CONGEST_LVL']
            records.append({
                "AREA_CD": item["AREA_CD"],
                "AREA_NM": item["AREA_NM"],
                "AREA_CONGEST_LVL": congestion,
                "CALL_API_TIME": now
            })
        except Exception as e:
            logging.warning(f"데이터 파싱 오류: {item['AREA_NM']}, {e}")

    df = pd.DataFrame(records)
    df_json = df.to_json(orient="split")
    context["ti"].xcom_push(key="raw_data", value=df_json)
    logging.info(f"API 데이터 로드 완료: {len(df)}행")

# 2. 전처리
def preprocess(**context):
    df_json = context["ti"].xcom_pull(key="raw_data", task_ids="load_data_from_api")
    # json 문자열을 판다스로 복원 ,orient='split 데이터 프레임 복원할때 정확도 높이기 위해
    df = pd.read_json(df_json, orient="split")

    # 예측에 사용하지 않을 불필요한 컬럼 목록을 정의
    required_cols = ["AREA_NM", "AREA_CD", "CALL_API_TIME", "AREA_CONGEST_LVL"]
    # DataFrame에 필요한 컬럼이 실제로 있는지 확인하는 단계
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")

    df_sample = df.copy()
    X = df_sample.drop(columns=required_cols)
    categorical_cols = X.select_dtypes(include="object").columns
    if len(categorical_cols) > 0:
        # fit_transform()은 학습 +D 변환을 동시에 수행
        X[categorical_cols] = OrdinalEncoder().fit_transform(X[categorical_cols])

    # 전처리 완료된 입력 데이터(X)를 JSON으로 변환해서 XCom에 저장
    # 다음 Task에서 꺼내 쓸 수 있도록 "X_data"라는 이름으로 저장
    context["ti"].xcom_push(key="X_data", value=X.to_json(orient="split"))
    # 전처리 전 원본 샘플링 데이터(df_sample)도 저장
    # 나중에 예측값을 붙이기 위해 사용함
    context["ti"].xcom_push(key="df_sample", value=df_sample.to_json(orient="split"))
    logging.info("전처리 완료")

# 3. 예측 (배치 단위 처리)
def predict(**context):
    # 이전 task (preprocess)에서 XCom으로 저장한 데이터를 가져옴
    X_json = context["ti"].xcom_pull(key="X_data", task_ids="preprocess")
    df_json = context["ti"].xcom_pull(key="df_sample", task_ids="preprocess")

    # JSON 문자열을 다시 Pandas DataFrame으로 복원
    X = pd.read_json(X_json, orient="split")
    df_sample = pd.read_json(df_json, orient="split")

    # 학습된 모델이 저장되어 있는 경로
    model_path = "/opt/airflow/models/rf_model.pkl"

    # 해당 경로에 모델 파일이 없으면 DAG을 실패로 종료
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    model = joblib.load(model_path)

    logging.info(f"총 샘플 수: {len(X)}")

    # 배치 단위 예측 수행
    batch_size = 1000
    predictions = []

    for i in range(0, len(X), batch_size):
        batch_X = X.iloc[i:i+batch_size]
        batch_pred = model.predict(batch_X)
        predictions.extend(batch_pred)
        logging.info(f"{i} ~ {i + len(batch_X)} 번째 예측 완료")

    # 전체 결과를 DataFrame에 저장
    df_sample["PREDICTED_CONGESTION"] = predictions
    context["ti"].xcom_push(key="result_df", value=df_sample.to_json(orient="split"))
    logging.info("전체 예측 완료")

# 4. 저장
def save_result(**context):
    # 이전 task인 predict에서 XCom으로 전달한 JSON 문자열을 가져옴
    df_json = context["ti"].xcom_pull(key="result_df", task_ids="predict")
    result_df = pd.read_json(df_json, orient="split")

    # 저장 경로 설정
    output_dir = "./data/live_crowd"
    os.makedirs(output_dir, exist_ok=True)  # 폴더 없으면 생성

    # 파일에 시간정보 붙여서 중복 방지 + 추적 가능하게 함
    now = datetime.now().strftime("%Y%m%d%H%M")  # 현재 날짜와 시간을 YYYYMMDDHHMM 형식으로 생성
    file_name = f"place_crowd_{now}.csv"
    file_path = os.path.join(output_dir, file_name)

    # 필요한 컬럼만 저장
    save_df = result_df[["AREA_NM", "PREDICTED_CONGESTION"]]
    save_df.to_csv(file_path, index=False)

    logging.info(f"결과 저장 완료: {file_path}")

# DAG 기본 설정 (추가로 슬랙 알림이나 이메일 알림 추가 가능) -> 내 생각 슬랙 추천 이메일 비효율
default_args = {
    'owner': 'day_walk',  # DAG 팀 이름 정의
    'start_date': datetime(2025, 6, 16),  # 시작 날짜 정의
    'retries': 2,  # task 실패시 최대 몇 번 재시도 할지 정하는거 우리가 정할 수 있음
    'retry_delay': timedelta(minutes=3),  # 재시도 간격 설정. 실패한 후 3분 뒤 다시 시도함 이것도 정할 수 있음
    'on_failure_callback': slack_alert,  # Task 실패 시 호출할 함수. 로그에 실패 정보 기록
    'on_success_callback': success_callback,  # Task 성공 시 호출할 함수. 로그에 성공 정보 기록
}

with DAG(
    dag_id="predict_congestion_hourly_xcom",  # DAG의 ID. Airflow UI 및 로그 등에서 이 이름으로 나타남
    default_args=default_args,
    description="혼잡도 예측 모델을 1시간마다 실행 (XCom 기반)",
    schedule_interval="0 * * * *",  # 실행 주기 설정(이 코드가 한시간마다 가져오게 함)
    catchup=False,  # 과거 실행 안 함(true 하면 DAG를 늦게 등록해도 과거 실행 이력을 모두 따라잡아서 실행) -> 굳이??
    max_active_runs=1,  # 1로 설정하면 중복 실행 방지
    tags=["ml", "prediction", "xcom"]
) as dag:

    t1 = PythonOperator(
        task_id="load_data_from_api",
        python_callable=load_data_from_api,
        provide_context=True
    )

    t2 = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess,
        provide_context=True
    )

    t3 = PythonOperator(
        task_id="predict",
        python_callable=predict,
        provide_context=True
    )

    t4 = PythonOperator(
        task_id="save_result",
        python_callable=save_result,
        provide_context=True
    )

    t1 >> t2 >> t3 >> t4

    globals()["predict_congestion_hourly_xcom"] = dag