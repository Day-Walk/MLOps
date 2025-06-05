import pandas as pd
import requests
import json



# 서울 장소 데이터 가져오기
def get_seoul_place():
    URL = "http://apis.data.go.kr/B551011/KorService2/"
    DECODING_KEY = "AcaGFMVEaYDSsY2lsCwLPE8EJitjqgPTxJfifRwSGkDkgSXaPNnKAsFC0sJhT2hJZDUgIzfAONS5yJXN75N1oQ=="
    
    areabasedlist_url = URL + "areaBasedList2"
    areabasedlist_params = {
        'serviceKey' : DECODING_KEY,
        'MobileOS' : 'WEB',
        'MobileApp' : 'Sample',
        '_type' : 'json',
        'areaCode' : '1',
        'numOfRows' : '7800'
    }

    areabasedlist_response = requests.get(areabasedlist_url, params=areabasedlist_params)
    areabasedlist = areabasedlist_response.json()
    seoul_place_list = areabasedlist['response']['body']['items']['item']
    seoul_place_df = pd.DataFrame(seoul_place_list)

    # 필요한 컬럼만 가져오기
    col_list = ['addr1', 'cat1', 'cat2', 'cat3', 'contentid', 'contenttypeid', 'firstimage', 'firstimage2', 'mapx', 'mapy', 'tel', 'title']
    seoul_place_df = seoul_place_df[col_list]

    # 필요한 카테고리
    # 12:관광지, 14:문화시설, 28:레포츠, 38:쇼핑, 39:음식점
    category_list = ['12', '14', '28', '38', '39']
    filtered_df = seoul_place_df[seoul_place_df['contenttypeid'].isin(category_list)]

    # 불필요한 데이터 삭제
    filtered_df_drop = filtered_df[~filtered_df['cat3'].isin(['A03010300', 'A04011000', 'A03010200'])]

    return filtered_df_drop


# 카테고리 이름 가져오기

# image, content, opentime, closedate 가져오기

# 결측값 처리 - image, tel




