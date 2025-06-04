# start_all.sh
#!/bin/bash

echo "=== MLOps 전체 시스템 시작 ==="

# 1. 네트워크 생성 (이미 있으면 무시됨)
echo "1. 네트워크 생성..."
docker network create mlops-network 2>/dev/null || echo "네트워크가 이미 존재합니다."

# 2. ELK 서버 시작
echo "2. ELK 서버 시작..."
cd ./ELK/
docker-compose up -d

# ELK 서버 준비 대기
echo "ELK 서버 준비 중..."
sleep 30

# Elasticsearch 상태 확인
until curl -s http://localhost:9200 > /dev/null; do
  echo "Elasticsearch 시작 대기 중..."
  sleep 5
done
echo "Elasticsearch 준비 완료!"

# 3. MLOps 서버 시작
echo "3. MLOps 서버 시작..."
cd ../MLOps/
docker-compose up -d

# MLOps 서버 준비 대기
echo "MLOps 서버 준비 중..."
sleep 15

until curl -s http://localhost:8000 > /dev/null; do
  echo "MLOps 서버 시작 대기 중..."
  sleep 5
done
echo "MLOps 서버 준비 완료!"

echo "=== 모든 서비스 시작 완료 ==="
echo "ELK API: http://localhost:9201"
echo "MLOps API: http://localhost:8000"
echo "Kibana: http://localhost:5601"
echo "API 문서: http://localhost:8000/docs"
