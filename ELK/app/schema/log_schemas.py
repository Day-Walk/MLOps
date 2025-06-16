from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

class Place(BaseModel):
    placeId: str
    name: str
    address: str
    imgUrl: str

class Answer(BaseModel):
    title: str
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