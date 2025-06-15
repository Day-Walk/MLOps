import csv
import os
import sys
import time
from xml.etree import ElementTree as ET

import requests
from dotenv import load_dotenv

def load_environment():
    """환경변수를 로드하고 필요한 키가 있는지 확인합니다."""
    load_dotenv()
    
    api_key = os.getenv('DATA_API_KEY')
    if not api_key:
        print("❌ 오류: DATA_API_KEY 환경변수가 설정되지 않았습니다.")
        print("💡 .env 파일에 다음과 같이 설정해주세요:")
        print("DATA_API_KEY=your_api_key_here")
        sys.exit(1)
    
    print("✅ 환경변수가 성공적으로 로드되었습니다.")
    return api_key

def get_category_data(content_type_id, api_key, cat1=None, cat2=None):
    """카테고리 데이터를 API에서 가져옵니다."""
    url = "http://apis.data.go.kr/B551011/KorService2/categoryCode2"
    
    # API 요청 파라미터 설정
    params = {
        'numOfRows': '2000',
        'MobileOS': 'WEB', 
        'MobileApp': 'apptest',
        'serviceKey': api_key,
        'contentTypeId': str(content_type_id)
    }
    
    if cat1:
        params['cat1'] = cat1
    if cat2:
        params['cat2'] = cat2
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # HTTP 에러 체크
        
        root = ET.fromstring(response.content)
        items = []
        
        for item in root.findall('.//item'):
            code = item.find('code')
            name = item.find('name')
            
            if code is not None and name is not None:
                items.append({
                    'contentTypeId': content_type_id,
                    'cat1': cat1 if cat1 else '',
                    'cat2': cat2 if cat2 else '',
                    'cat3': code.text if cat2 else '',
                    'code': code.text,
                    'name': name.text
                })
        
        return items
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API 요청 중 오류 발생: {e}")
        return []
    except ET.ParseError as e:
        print(f"❌ XML 파싱 중 오류 발생: {e}")
        return []

def main():
    """메인 실행 함수입니다."""
    # 환경변수 로드
    api_key = load_environment()
    
    # 관광 콘텐츠 타입 ID 목록
    content_type_ids = [12, 14, 28, 38, 39]
    all_items = []
    
    print("📊 카테고리 데이터 수집을 시작합니다...")
    
    for content_type_id in content_type_ids:
        print(f"🔄 콘텐츠 타입 {content_type_id} 처리 중...")
        
        # 1단계: cat1 카테고리 수집
        cat1_items = get_category_data(content_type_id, api_key)
        
        for cat1_item in cat1_items:
            cat1_code = cat1_item['code']
            cat1_name = cat1_item['name']
            
            all_items.append({
                'contentTypeId': content_type_id,
                'cat1': cat1_code,
                'cat2': '',
                'cat3': '',
                'code': cat1_code,
                'name': cat1_name
            })
            
            # 2단계: cat2 카테고리 수집
            cat2_items = get_category_data(content_type_id, api_key, cat1=cat1_code)
            
            for cat2_item in cat2_items:
                cat2_code = cat2_item['code']
                cat2_name = cat2_item['name']
                
                all_items.append({
                    'contentTypeId': content_type_id,
                    'cat1': cat1_code,
                    'cat2': cat2_code,
                    'cat3': '',
                    'code': cat2_code,
                    'name': cat2_name
                })
                
                # 3단계: cat3 카테고리 수집
                cat3_items = get_category_data(content_type_id, api_key, cat1=cat1_code, cat2=cat2_code)
                
                for cat3_item in cat3_items:
                    cat3_code = cat3_item['code']
                    cat3_name = cat3_item['name']
                    
                    all_items.append({
                        'contentTypeId': content_type_id,
                        'cat1': cat1_code,
                        'cat2': cat2_code,
                        'cat3': cat3_code,
                        'code': cat3_code,
                        'name': cat3_name
                    })
                
                time.sleep(0.1)  # API 과다 호출 방지
            time.sleep(0.1)
        time.sleep(0.1)
    
    # CSV 파일로 저장
    output_file_path = './data/categories.csv'
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['contentTypeId', 'cat1', 'cat2', 'cat3', 'code', 'name'])
            writer.writeheader()
            writer.writerows(all_items)
        
        print(f"✅ 카테고리 데이터가 {output_file_path} 파일로 저장되었습니다.")
        print(f"📈 총 {len(all_items)}개의 카테고리가 수집되었습니다.")
        
    except IOError as e:
        print(f"❌ 파일 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 