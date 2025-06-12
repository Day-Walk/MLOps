import pandas as pd
import json
import haversine
import geopandas as gpd
from shapely.geometry import Point
import re

# 지하철역 추가
def add_subway_info(place_json, subway_json, radius=500):
    
    for place in place_json:
        # 장소데이터 좌표 가져오기
        place_coord = (place['location']['lat'], place['location']['lng'])

        stations_list = []
        for subway in subway_json['DATA']:
            # 지하철역 좌표 가져오기
            subway_coord = (float(subway['lat']), float(subway['lot']))

            # 장소와 지하철역 거리 계산
            distance = haversine.haversine(place_coord, subway_coord, unit='m')

            # 500미터 이내인 경우만
            if distance <= 500:
                # 리스트에 역명 + "역"으로 담아줌
                stations_list.append(subway['bldn_nm']+"역")

        # set으로 중복 제거
        stations_str = ", ".join(set(stations_list))

        # 문자열로 저장
        place['subway'] = stations_str

    return place_json

# 행정구역 추가
def add_dong_info(place_json, dong_region):
    
    # 서울시내 행정구역만
    seoul = dong_region[dong_region["ADM_CD"].str.startswith('11')]

    # 좌표 계산을 위한 좌표계 통일
    seoul = seoul.to_crs("EPSG:4326")

    for place in place_json:
    # 장소 좌표
        point = Point(place['location']['lng'], place['location']['lat'])

        # 행정구역에 포함이 되는지
        matched = seoul[seoul.contains(point)]

        # 포함되는 행정구역이 없다면
        if matched.empty:
            place['dong'] = ""
        else:
            # 행정구역명 담아주고 전처리
            dong = matched['ADM_NM'].values[0]
            # "행정구역이름" + "동" 으로 통일
            place['dong'] = re.sub(r"(\D+)([\d·\.]*(\d+가)*[\d·]*)동", r"\1동", dong)

    return place_json


def main():
    # 장소, 지하철, 행정구역 데이터 불러오기
    with open("/place_vectordb.json", 'r', encoding='utf-8') as f:
        place_json = json.load(f)

    with open("/seoul_subway_info.json", 'r', encoding='utf-8') as f:
        subway_json = json.load(f)

    dong_region = gpd.read_file("/dong_region.shp")

    place_json = add_subway_info(place_json, subway_json)
    place_json = add_dong_info(place_json, dong_region)
    
    # json 파일로 저장
    with open("place_metadata_enrich.json", "w", encoding="utf-8") as f:
        json.dump(place_json, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
