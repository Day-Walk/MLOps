# ELK/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY ./app ./app

# 포트 노출
EXPOSE 9201

# uvicorn 실행 (절대 경로로 명시적 실행)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9201"]
