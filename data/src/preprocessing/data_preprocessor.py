import pandas as pd
import numpy as np
import ast

class DataPreprocessor:
    def __init__(self, click_path, place_path, like_path, user_path):
        # 파일 경로 저장
        self.click_path = click_path
        self.place_path = place_path
        self.like_path = like_path
        self.user_path = user_path

    def load_data(self):
        # CSV 파일 로드
        self.click = pd.read_csv(self.click_path)
        self.place = pd.read_csv(self.place_path)
        self.like = pd.read_csv(self.like_path)
        self.user = pd.read_csv(self.user_path)

    def preprocess_click(self):
        # 클릭 데이터 전처리
        click = self.click.copy()
        click["yn"] = 1
        click = click[["HEX(user_id)", "HEX(place_id)", "yn"]]
        click.rename(columns={"HEX(user_id)": "user_id", "HEX(place_id)": "place_id"}, inplace=True)
        click["user_id"] = click["user_id"].astype(str)
        click["place_id"] = click["place_id"].astype(str)

        # 음성 샘플 생성(장소 선택 안한 장소 중 랜덤 샘플링)
        user_ids = click["user_id"].unique()
        place_ids = click["place_id"].unique()
        neg_samples = []
        for user in user_ids:
            clicked = click[click['user_id'] == user]['place_id'].unique()
            unclicked = np.setdiff1d(place_ids, clicked)
            n_samples = min(len(unclicked), len(clicked) * 4)
            sampled = np.random.choice(unclicked, size=n_samples, replace=False)
            for p in sampled:
                neg_samples.append([user, p, 0])
        click_neg = pd.DataFrame(neg_samples, columns=["user_id", "place_id", "yn"])

        # 양성 샘플과 음성 샘플 합치고 셔플
        self.click_all = pd.concat([click, click_neg], ignore_index=True).sample(frac=1).reset_index(drop=True)

    def preprocess_place(self):
        # 장소 데이터 전처리
        place = self.place.copy()
        place = place[["HEX(id)", "name", "category", "sub_category"]]
        place.rename(columns={"HEX(id)": "place_id", "name": "place_name"}, inplace=True)
        place["category"] = place["category"].replace("음식점", "음식점&카페")
        place["place_id"] = place["place_id"].astype(str)
        self.place_clean = place

    def preprocess_like(self):
        # 유저 선호 태그 전처리
        like = self.like.copy()
        like = like[["HEX(user_id)", "category", "tag_list"]]
        like.rename(columns={"HEX(user_id)": "user_id", "category": "user_category"}, inplace=True)
        like["user_id"] = like["user_id"].astype(str)
        like["tag_list"] = like["tag_list"].apply(ast.literal_eval)

        # 카테고리 태그 조합
        like["category_tag"] = like.apply(
            lambda row: [f"{row['user_category']}_{tag}" for tag in row["tag_list"]],
            axis=1
        )
        like_final = like.groupby("user_id")["category_tag"].sum().reset_index()
        like_final.rename(columns={"category_tag": "like_list"}, inplace=True)

        # 문자열 태그 숫자 인코딩
        key2index = {}
        def encode(x):
            for k in x:
                if k not in key2index:
                    key2index[k] = len(key2index) + 1
            return [key2index[k] for k in x]

        like_final["like_list"] = like_final["like_list"].apply(encode)
        self.like_final = like_final

    def preprocess_user(self):
        # 유저 정보 전처리
        user = self.user.copy()
        user = user[["HEX(id)", "name", "age", "gender"]]
        user.rename(columns={"HEX(id)": "user_id", "name": "user_name"}, inplace=True)
        user["user_id"] = user["user_id"].astype(str)
        self.user_clean = user

    def merge_all(self):
        # 클릭 + 장소 + 선호 태그 + 유저 정보 병합
        df = pd.merge(self.click_all, self.place_clean, on="place_id", how="left")
        df = pd.merge(df, self.like_final, on="user_id", how="left")
        df = pd.merge(df, self.user_clean, on="user_id", how="left")
        df.dropna(inplace=True)
        self.final_df = df

    def run(self, output_path="final_click_log.csv"):
        # 전체 전처리 파이프라인 실행
        self.load_data()
        self.preprocess_click()
        self.preprocess_place()
        self.preprocess_like()
        self.preprocess_user()
        self.merge_all()
        self.final_df.to_csv(output_path, index=False)
        print(f"✅ 전처리 완료! → {output_path}")