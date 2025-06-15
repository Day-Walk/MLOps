import pandas as pd
import numpy as np
import requests
import json
import ast

URL = "http://apis.data.go.kr/B551011/KorService2/"
DECODING_KEY = ""

# 서울 장소 데이터 가져오기
def get_seoul_place():
   
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


# image 가져오기
def get_image_list(contentid):
  image_list = []

  detail_image_url = URL + "detailImage2"
  detail_image_params = {
      'serviceKey' : DECODING_KEY,
      'MobileOS' : 'WEB',
      'MobileApp' : 'Sample',
      '_type' : 'json',
      'contentId' : contentid
  }

  detail_image_response = requests.get(detail_image_url, params=detail_image_params)
  detail_image = detail_image_response.json()
  image_items = detail_image['response']['body']['items']

  # 이미지가 있다면 리스트로 묶기
  if image_items:
    image_item = image_items['item']
    image_list = [item['originimgurl'] for item in image_item]
  else:
    pass

  return image_list

# content 가져오기
def get_content_overview(contentId):
    detail_common_url = URL + "detailCommon2"
    detail_common_params = {
        'serviceKey': DECODING_KEY,
        'MobileOS': 'WEB',
        'MobileApp': 'Sample',
        '_type': 'json',
        'contentId': contentId
    }

    detail_common_response = requests.get(detail_common_url, params=detail_common_params)
    detail_common = detail_common_response.json()

    # 'overview'에 들어있는 값을 content로 저장
    try:
        content = detail_common['response']['body']['items']['item'][0]['overview']
    except Exception:
        content = None

    return content

# closedate, opentime, phoneNum
# 대분류별로 json 구조가 달라서 다르게 처리

def get_detail_12(contentId):
    detail_12_url = URL + "detailIntro2"
    detail_12_params = {
        'serviceKey': DECODING_KEY,
        'MobileOS': 'WEB',
        'MobileApp': 'Sample',
        '_type': 'json',
        'contentTypeId': '12',
        'contentId': contentId
    }

    detail_12_response = requests.get(detail_12_url, params=detail_12_params)
    detail_12 = detail_12_response.json()

    try:
        item = detail_12['response']['body']['items']['item'][0]
        closeDate = item.get('restdate')
        openTime = item.get('usetime')
        phoneNum = item.get('infocenter')
    except Exception:
        closeDate = None
        openTime = None
        phoneNum = None

    return closeDate, openTime, phoneNum

def get_detail_14(contentId):
    detail_14_url = URL + "detailIntro2"
    detail_14_params = {
        'serviceKey': DECODING_KEY,
        'MobileOS': 'WEB',
        'MobileApp': 'Sample',
        '_type': 'json',
        'contentTypeId': '14',
        'contentId': contentId
    }

    detail_14_response = requests.get(detail_14_url, params=detail_14_params)
    detail_14 = detail_14_response.json()

    try:
        item = detail_14['response']['body']['items']['item'][0]
        closeDate = item.get('restdateculture')
        openTime = item.get('usetimeculture')
        phoneNum = item.get('infocenterculture')
    except Exception:
        closeDate = None
        openTime = None
        phoneNum = None

    return closeDate, openTime, phoneNum

def get_detail_28(contentId):
    detail_28_url = URL + "detailIntro2"
    detail_28_params = {
        'serviceKey': DECODING_KEY,
        'MobileOS': 'WEB',
        'MobileApp': 'Sample',
        '_type': 'json',
        'contentTypeId': '28',
        'contentId': contentId
    }

    detail_28_response = requests.get(detail_28_url, params=detail_28_params)
    detail_28 = detail_28_response.json()

    try:
        item = detail_28['response']['body']['items']['item'][0]
        closeDate = item.get('restdateleports')
        openTime = item.get('usetimeleports')
        phoneNum = item.get('infocenterleports')
    except Exception:
        closeDate = None
        openTime = None
        phoneNum = None

    return closeDate, openTime, phoneNum

def get_detail_38(contentId):
    detail_38_url = URL + "detailIntro2"
    detail_38_params = {
        'serviceKey': DECODING_KEY,
        'MobileOS': 'WEB',
        'MobileApp': 'Sample',
        '_type': 'json',
        'contentTypeId': '38',
        'contentId': contentId
    }

    detail_38_response = requests.get(detail_38_url, params=detail_38_params)
    detail_38 = detail_38_response.json()

    try:
        item = detail_38['response']['body']['items']['item'][0]
        closeDate = item.get('restdateshopping')
        openTime = item.get('opentime')
        phoneNum = item.get('infocentershopping')
    except Exception:
        closeDate = None
        openTime = None
        phoneNum = None

    return closeDate, openTime, phoneNum

def get_detail_39(contentId):
    detail_39_url = URL + "detailIntro2"
    detail_39_params = {
        'serviceKey': DECODING_KEY,
        'MobileOS': 'WEB',
        'MobileApp': 'Sample',
        '_type': 'json',
        'contentTypeId': '39',
        'contentId': contentId
    }

    detail_39_response = requests.get(detail_39_url, params=detail_39_params)
    detail_39 = detail_39_response.json()

    try:
        item = detail_39['response']['body']['items']['item'][0]
        closeDate = item.get('restdatefood')
        openTime = item.get('opentimefood')
        phoneNum = item.get('infocenterfood')
    except Exception:
        closeDate = None
        openTime = None
        phoneNum = None

    return closeDate, openTime, phoneNum

def get_details(row):
    content_type_id = row['contenttypeid']
    content_id = row['contentid']

    if content_type_id == 12:
        return get_detail_12(content_id)
    elif content_type_id == 14:
        return get_detail_14(content_id)
    elif content_type_id == 28:
        return get_detail_28(content_id)
    elif content_type_id == 38:
        return get_detail_38(content_id)
    elif content_type_id == 39:
        return get_detail_39(content_id)
    else:
        return None, None, None


def main():
    # 서울 장소 목록 호출
    df = get_seoul_place()

    # 이미지 호출해 저장
    df['imageList'] = df['contentid'].apply(get_image_list)

    # content 호출해 저장
    df['content'] = df['contentid'].apply(get_content_overview)

    # closedate, opentime, phoneNum 호출해 저장
    df[['closeDate', 'openTime', 'phoneNum']] = df.apply(get_details, axis=1, result_type='expand')

    df.to_csv(".data/place_api.csv", index=False)
