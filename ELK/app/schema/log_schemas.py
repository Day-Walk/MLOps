from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

class Location(BaseModel):
    lat: float
    lng: float

class Place(BaseModel):
    placeId: str
    name: str
    address: str
    imgUrl: str
    location: Location

class Answer(BaseModel):
    placeList: List[Place]
    detail: str

class LogRequest(BaseModel):
    userId: str
    question: str
    answer: Answer
    createAt: datetime

class LogResponse(BaseModel):
    isSuccess: bool
    message: str

class ClickLogRequest(BaseModel):
    userId: str  # UUID 형식
    placeId: str  # UUID 형식
    timestamp: datetime

class ClickLogResponse(BaseModel):
    success: bool
    message: str