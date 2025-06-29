from fastapi import APIRouter, HTTPException, Query
from typing import List
from app.schema.crowd_schema import CrowdResponse, CrowdLevel, CrowdInfo
from datetime import datetime, timedelta
import pandas as pd
import os

router = APIRouter(prefix="/api", tags=["crowdness"])

@router.get("/crowd", response_model=CrowdResponse)
def get_crowd_prediction(hour: int, area: List[str] = Query(["all"])):
    """
    ## 시간대별 혼잡도 예측 결과 조회 API
    - **hour**: 조회할 예측 시간 (1, 2, 3, 6, 12 중 하나)
    - **area**: 조회할 지역 리스트. 파라미터를 여러 번 사용하여 다수 지역 조회 가능 (예: `?area=홍대&area=강남`). 기본값은 'all'.
    """
    try:
        # 1. 대상 S3 경로 생성
        target_timestamp = (datetime.now()).strftime("%Y%m%d%H")
        filename = f"congestion_predictions_{target_timestamp}_{hour}.csv"

        s3_bucket_name = os.getenv("S3_BUCKET_NAME")
        if not s3_bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is not set.")
        
        s3_directory = os.getenv("S3_DIRECTORY_PATH", "")
        s3_key = os.path.join(s3_directory, filename).replace("\\", "/") # S3는 /를 사용
        s3_path = f"s3://{s3_bucket_name}/{s3_key}"
        
        print(f"Attempting to read from S3: {s3_path}")

        # 2. S3에서 데이터 로드
        df = pd.read_csv(s3_path)

        # 3. 'area' 파라미터에 따른 데이터 필터링
        if "all" not in area:
            df = df[df['AREA_NM'].isin(area)]
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No data found for areas: {', '.join(area)}")

        # 4. 응답 데이터 형식으로 변환
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
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Prediction file not found: {s3_path}. It may not have been generated yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")