import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
import ast
import os
import pickle
import sys
import json

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
mlops_app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if mlops_app_root not in sys.path:
    sys.path.insert(0, mlops_app_root)

from ELK.app.services.elasticsearch_service import ElasticsearchService
from app.services.db_connection import DatabaseService
from tqdm import tqdm

def pad_sequences(sequences, maxlen, padding='post', value=0):
    """
    NumPy를 이용한 pad_sequences의 간단한 구현
    """
    padded = np.full((len(sequences), maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if padding == 'post':
            padded[i, :len(seq)] = seq[:maxlen]
        else:  # 'pre' padding
            padded[i, -len(seq):] = seq[-maxlen:]
    return padded

class DeepFMModdelTrain:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = None
        self.sparse_features = ["userid", "name", "age", "gender", "place_id", "place_name","category", "subcategory"]
        self.sequence_feature = "like_list"
        self.linear_feature_columns = None
        self.dnn_feature_columns = None
        self.feature_names = None
        self.model_input = None
        self.target = "yn"        
        self.model_path = os.environ.get("DEEPFM_TRAIN_MODEL_PATH", "")
        self.encoders_path = os.environ.get("DEEPFM_TRAIN_ENCODERS_PATH", "")
        self.key2index_path = os.environ.get("DEEPFM_TRAIN_KEY2INDEX_PATH", "")
        self.model = None
        self.max_len = None
        self.label_encoders = {}
        self.key2index = {}
    
    def load_data(self):
        """
        Elasticsearch에서 상호작용 데이터를, DB에서 메타데이터를 조회하여 학습 데이터를 생성합니다.
        """
        # 1. Elasticsearch에서 상호작용 데이터 가져오기
        print("1. Elasticsearch에서 상호작용 데이터를 불러오는 중...")
        es_service = ElasticsearchService()
        interaction_df = es_service.get_training_data_for_all_users()

        if interaction_df is None or interaction_df.empty:
            print("[오류] Elasticsearch에서 데이터를 불러오지 못했습니다. 파이프라인을 중단합니다.")
            self.data = pd.DataFrame()
            return

        # 2. 고유 ID 추출 및 DB 서비스 초기화
        unique_users = list(interaction_df['userid'].unique())
        unique_places = list(interaction_df['place_id'].unique())
        
        # 3. DB에서 메타데이터 일괄 조회
        db_service = DatabaseService()
        print("\n2. DB에서 사용자 및 장소 정보를 한번에 가져오는 중...")
        user_df = db_service.get_users_info_by_user_ids(unique_users)
        place_df = db_service.get_places_info_by_place_ids(unique_places)
        db_service.close_connection()

        # 4. 데이터프레임 병합
        print("\n3. 데이터프레임 병합 중...")
        data = interaction_df.merge(user_df, on='userid', how='left')
        self.data = data.merge(place_df, on='place_id', how='left')

        # 5. 결측값 처리
        if not self.data.empty:
            self.data['like_list'] = self.data['like_list'].fillna('[]')
        
        print(f"\n총 {len(self.data)}개의 학습 데이터 준비 완료.")

    def preprocess(self):
        if self.data is None or self.data.empty:
            print("학습 데이터가 없습니다. 먼저 load_data()를 호출하세요.")
            return False

        # sparse feature에 대해 결측값 처리 및 label encoding 수행
        for feature in self.sparse_features:
            self.data[feature] = self.data[feature].fillna("unknown")
            encoder = LabelEncoder()
            self.data[feature] = encoder.fit_transform(self.data[feature])
            self.label_encoders[feature] = encoder
        
        # 'category_tag' 포맷의 'like_list'를 생성하고, JSON 문자열을 리스트로 변환
        print("전처리: 'category'와 'like_list'를 조합하여 최종 피처 생성 중...")
        def create_category_prefixed_likes(row):
            category = row['category']
            like_list_str = row['like_list']
            
            # category나 like_list가 비어있으면 빈 리스트 반환
            if pd.isna(category) or pd.isna(like_list_str):
                return []
            
            try:
                # DB에서 온 JSON 문자열을 파이썬 리스트로 변환
                tags = json.loads(like_list_str)
                if not isinstance(tags, list):
                    return []
                # '카테고리_태그' 형태로 조합
                return [f"{category}_{tag}" for tag in tags]
            except (json.JSONDecodeError, TypeError):
                return []

        self.data['like_list'] = self.data.apply(create_category_prefixed_likes, axis=1)

        # sequence feature 인코딩
        def encode(x):
            for k in x:
                if k not in self.key2index:
                    self.key2index[k] = len(self.key2index) + 1
            return [self.key2index[k] for k in x]
        self.data[self.sequence_feature] = self.data[self.sequence_feature].apply(encode)
        
        # sequence feature의 최대 길이만 계산하고, 데이터프레임은 수정하지 않음
        self.max_len = max(len(x) for x in self.data[self.sequence_feature])
        
        # 최종 feature 생성
        sparse_feature_columns = [SparseFeat(feature, 
                                           vocabulary_size=self.data[feature].nunique(), 
                                           embedding_dim=4) for feature in self.sparse_features]
        
        sequence_feature_columns = [VarLenSparseFeat(SparseFeat(self.sequence_feature, 
                                                             vocabulary_size=len(self.key2index) + 1, 
                                                             embedding_dim=4), maxlen=self.max_len, combiner="mean")]
        
        self.linear_feature_columns = sparse_feature_columns + sequence_feature_columns
        self.dnn_feature_columns = sparse_feature_columns + sequence_feature_columns
        
        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)
        
        # 학습 데이터 생성 (모델 입력은 Numpy 배열 형태여야 함)
        self.model_input = {}
        for name in self.feature_names:
            if name == self.sequence_feature:
                # 시퀀스 피처는 이 단계에서 패딩을 적용하여 Numpy 배열로 변환
                sequences = pad_sequences(self.data[name].values, maxlen=self.max_len, padding='post', value=0)
                self.model_input[name] = sequences
            else:
                # 다른 피처들은 .values를 사용하여 Numpy 배열로 변환
                self.model_input[name] = self.data[name].values

        self.model_input[self.target] = self.data[self.target].values
        
        # 인코더들 저장
        with open(self.encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        with open(self.key2index_path, 'wb') as f:
            pickle.dump({'key2index': self.key2index, 'max_len': self.max_len}, f)
        
        return True # 전처리 성공

    def train(self):
        if self.linear_feature_columns is None or self.dnn_feature_columns is None:
            print("피처 컬럼이 초기화되지 않았습니다. preprocess()를 먼저 실행하거나 데이터 로딩을 확인하세요.")
            return None
        
        if self.data is None:
            print("학습 데이터(self.data)가 없어 학습을 진행할 수 없습니다.")
            return None

        model = DeepFM(self.linear_feature_columns, 
                       self.dnn_feature_columns, 
                       task="regression",
                       device=str(self.device))
        
        model.compile("adam", "mse", metrics=["mse"])
        
        model.fit(self.model_input, 
                  self.data[self.target].values, 
                  batch_size=256, 
                  epochs=10, 
                  verbose=2,
                  validation_split=0.2)
        
        # 모델 저장 
        torch.save(model.state_dict(), self.model_path)
        print(f"모델이 {self.model_path}에 저장되었습니다.")
        
        return model
    
    def predict(self, input_data):
        # 저장된 인코더들 로드
        with open(self.encoders_path, 'rb') as f:
            self.label_encoders = pickle.load(f)
        with open(self.key2index_path, 'rb') as f:
            key2index_data = pickle.load(f)
            if isinstance(key2index_data, dict) and 'key2index' in key2index_data:
                self.key2index = key2index_data['key2index']
                self.max_len = key2index_data['max_len']
            else:
                # 이전 버전 호환성
                self.key2index = key2index_data
                self.max_len = 50  # 기본값

        sparse_feature_names = ["userid", "name", "age", "gender", "place_id", "place_name", "category", "subcategory"]
        
        reconstructed_sparse_features = [SparseFeat(feat, vocabulary_size=len(self.label_encoders[feat].classes_), embedding_dim=4)
                                         for feat in sparse_feature_names]
        
        reconstructed_sequence_feature = [VarLenSparseFeat(SparseFeat(self.sequence_feature,
                                                                      vocabulary_size=len(self.key2index) + 1,
                                                                      embedding_dim=4),
                                                           maxlen=self.max_len, combiner='mean')]

        self.linear_feature_columns = reconstructed_sparse_features + reconstructed_sequence_feature
        self.dnn_feature_columns = reconstructed_sparse_features + reconstructed_sequence_feature
        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)
        
        # 입력 데이터를 DataFrame으로 변환
        if isinstance(input_data, dict):
            input_df = pd.DataFrame(input_data)
        else:
            input_df = input_data.copy()
        
        # sparse feature 전처리
        for feature in sparse_feature_names:
            encoder = self.label_encoders[feature]
            known_classes = set(encoder.classes_)
            
            # 'unknown'이 학습되었는지 확인
            unknown_in_classes = 'unknown' in known_classes
            
            def transform_element(x):
                if pd.isna(x) or x not in known_classes:
                    return 'unknown' if unknown_in_classes else encoder.classes_[0]
                return x

            input_df[feature] = input_df[feature].apply(transform_element)
            input_df[feature] = self.label_encoders[feature].transform(input_df[feature])
        
        # sequence feature 전처리
        sequence_feature_name = "like_list"
        input_df[sequence_feature_name] = input_df[sequence_feature_name].apply(ast.literal_eval)
        
        def encode_sequence(x):
            return [self.key2index.get(k, 0) for k in x]  # 없는 키는 0으로 처리
        
        input_df[sequence_feature_name] = input_df[sequence_feature_name].apply(encode_sequence)
        
        # 모델 입력 형태로 변환
        model_input = {}
        for name in self.feature_names:
            if name == sequence_feature_name:
                # 시퀀스 피처는 이 단계에서 패딩을 적용하여 Numpy 배열로 변환
                sequences = pad_sequences(input_df[name].values, maxlen=self.max_len, padding='post', value=0)
                model_input[name] = sequences
            else:
                # 다른 피처들은 .values를 사용하여 Numpy 배열로 변환
                model_input[name] = input_df[name].values
        
        # 모델 로드 및 예측
        model = DeepFM(self.linear_feature_columns, 
                       self.dnn_feature_columns, 
                       task="regression",
                       device=str(self.device))
        model.load_state_dict(torch.load(self.model_path))
        model.compile("adam", "mse", metrics=["mse"])
        
        return model.predict(model_input)
    
if __name__ == "__main__":
    deepfm_train = DeepFMModdelTrain()
    deepfm_train.load_data()
    
    # 전처리가 성공한 경우에만 학습 진행
    if deepfm_train.preprocess():
        model = deepfm_train.train()
        
        if model:
            # 예시 데이터
            input_data = {
                "userid": ["0x06fa1ba7a7e44621a2338e6093e53341", "0x6d132cda535848e295b8e489486ea841", "0x0fa0a9c4a283451181b77d91e3229c91"],
                "name": ["딩딩이", "댕댕이 언니", "에구궁"],
                "age": [30, 60, 50],
                "gender": [1, 1, 0],
                "place_id": ["0xeb37b72b1fa54dc6a3867517ac2df6ef", "0x0528fbb073104d51974112a71d72b4e4", "0x1226fc5501194d2eba00383748045c20"],
                "place_name": ["롯데월드 쇼핑몰", "청아라 생선구이", "시골보쌈"],
                "category": ["쇼핑", "음식점&카페", "음식점&카페"],
                "subcategory": ["전문매장/상가", "한식", "한식"],
                "like_list": ['["태그1", "태그2"]', '["태그3"]', '[]'] 
            }
            # predict 메서드를 사용하기 위한 별도의 인스턴스 생성
            # 학습 시 사용된 인코더 등을 그대로 사용하기 위함
            predictor = DeepFMModdelTrain() 
            prediction = predictor.predict(input_data)
            print("예측 결과:", prediction)