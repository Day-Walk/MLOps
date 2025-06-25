from fastapi import APIRouter, HTTPException
from app.schema.crowd_schema import CrowdResponse, CrowdLevel, CrowdInfo
from datetime import datetime, timedelta
import pandas as pd
import os

router = APIRouter(prefix="/api", tags=["crowdness"])

# .env에서 PRED_PATH를 찾지 못할 경우를 대비한 기본 경로 설정
PRED_PATH = os.getenv("PRED_PATH")
if not PRED_PATH:
    print("PRED_PATH is not set")

@router.get("/crowd", response_model=CrowdResponse)
def get_crowd_prediction(hour: int):
    """
    ## 시간대별 혼잡도 예측 결과 조회 API
    - **hour**: 조회할 예측 시간 (1, 2, 3, 6, 12 중 하나)
    """
    try:
        # 1. 대상 파일명 생성
        target_timestamp = (datetime.now()).strftime("%Y%m%d%H")
        filename = f"congestion_predictions_{target_timestamp}_{hour}.csv"
        file_path = os.path.join(PRED_PATH, filename)
        
        print(f"Attempting to read: {file_path}")

        # 2. 파일 존재 확인 및 데이터 로드
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Prediction file not found for T+{hour}h. It may not have been generated yet.")

        df = pd.read_csv(file_path)

        # 3. 응답 데이터 형식으로 변환
        crowd_info_list = []
        for _, row in df.iterrows():
            crowd_info_list.append(CrowdInfo(
                area_nm=str(row['AREA_NM']),
                x=str(row['x']),
                y=str(row['y']),
                category=str(row['category']),
                area_congest_lvl=str(row['PREDICTED_CONGESTION_STR']),
                area_congest_num=int(row['PREDICTED_CONGESTION_NUM'])
            ))

        crowd_level = CrowdLevel(
            total=len(crowd_info_list),
            row=crowd_info_list
        )

        return CrowdResponse(
            success=True,
            message=f"조회 성공!",
            crowdLevel=crowd_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")