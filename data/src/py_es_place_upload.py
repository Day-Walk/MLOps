from utils.es_place_upload import KoreanContentElasticsearch

# 사용 예시
if __name__ == "__main__":
    # 한국어 콘텐츠 Elasticsearch 클라이언트 초기화
    print("한국어 콘텐츠 Elasticsearch 클라이언트 초기화 중...")
    korean_es = KoreanContentElasticsearch(host="localhost", port=9200)
    
    # 인덱스 생성
    print("\n=== 한국어 콘텐츠 인덱스 생성 ===")
    index_name = "place_data_v3"
    success = korean_es.create_korean_content_index(index_name)
    
    if success:
        print(f"인덱스 '{index_name}' 생성 성공")
        
        # json 파일 삽입
        csv_file_path = "data/place.csv"
        korean_es.insert_data_from_csv(index_name, csv_file_path)
        
    else:
        print("인덱스 생성 실패")