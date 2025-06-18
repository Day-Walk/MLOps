FROM python:3.10-slim

WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U langchain-chroma chromadb

# app 디렉토리만 복사 (중요!)
COPY ./app ./app

# Python 경로 설정
ENV PYTHONPATH=/app

# 포트 노출
EXPOSE 8000

# uvicorn 실행
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
