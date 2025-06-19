from pydantic import BaseModel
from typing import List

class CrowdInfo(BaseModel):
    area_nm: str
    x: str
    y: str
    area_congest_lvl: str
    area_congest_num: int

class CrowdLevel(BaseModel):
    total: int
    row: List[CrowdInfo]

class CrowdResponse(BaseModel):
    success: bool
    message: str
    crowdLevel: CrowdLevel 