from fastapi import APIRouter
from app.schema.crowd_schema import CrowdResponse, CrowdLevel, CrowdInfo

router = APIRouter(prefix="/api", tags=["crowdness"])

@router.get("/crowd", response_model=CrowdResponse)
def get_crowd(hour: int):
    """
    ## 혼잡도 예측 API
    - **hour**: 시간 (정수)
    """
    dummy_row_data = [
        CrowdInfo(
            area_nm="충정로역",
            x="37.55969632013057",
            y="126.96369132536704",
            area_congest_lvl="붐빔",
            area_congest_num=4
        ),
        CrowdInfo(
            area_nm="서울역",
            x="37.55659428234287",
            y="126.97302795181167",
            area_congest_lvl="붐빔",
            area_congest_num=4
        )
    ]

    dummy_crowd_level = CrowdLevel(
        total=120,
        row=dummy_row_data
    )

    return CrowdResponse(
        success=True,
        message="혼잡도 예측 성공!",
        crowdLevel=dummy_crowd_level
    ) 