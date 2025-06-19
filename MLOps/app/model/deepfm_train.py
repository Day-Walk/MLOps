import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
import ast
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle

class DeepFMModdelTrain:
    def __init__(self, data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = pd.read_csv(data_path)
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
    
    def preprocess(self): 
        # sparse feature 이름 저장
        sparse_feature_names = self.sparse_features.copy()
        
        # sparse feature에 대해 결측값 처리 및 label encoding 수행
        for feature in sparse_feature_names:
            self.data[feature] = self.data[feature].fillna("unknown")
            encoder = LabelEncoder()
            self.data[feature] = encoder.fit_transform(self.data[feature])
            self.label_encoders[feature] = encoder
        
        # sequence feature 리스트 변환
        self.data[self.sequence_feature] = self.data[self.sequence_feature].apply(ast.literal_eval)
        
        # sequence feature 인코딩
        def encode(x):
            for k in x:
                if k not in self.key2index:
                    self.key2index[k] = len(self.key2index) + 1
            return [self.key2index[k] for k in x]
        self.data[self.sequence_feature] = self.data[self.sequence_feature].apply(encode)
        
        # sequence feature 패딩
        self.max_len = max(len(x) for x in self.data[self.sequence_feature])
        self.data[self.sequence_feature] = pad_sequences(self.data[self.sequence_feature], maxlen=self.max_len, padding='post', value=0)
        
        # 최종 feature 생성
        self.sparse_features = [SparseFeat(feature, 
                                           vocabulary_size=self.data[feature].nunique(), 
                                           embedding_dim=4) for feature in sparse_feature_names]
        
        self.sequence_feature = [VarLenSparseFeat(SparseFeat(self.sequence_feature, 
                                                             vocabulary_size=len(self.key2index) + 1, 
                                                             embedding_dim=4), maxlen=self.max_len, combiner="mean")]
        
        self.linear_feature_columns = self.sparse_features + self.sequence_feature
        self.dnn_feature_columns = self.sparse_features + self.sequence_feature
        
        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)
        
        # 학습 데이터 생성
        self.model_input = {name: self.data[name] for name in self.feature_names}
        self.model_input[self.target] = self.data[self.target]
        
        # 인코더들 저장
        with open(self.encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        with open(self.key2index_path, 'wb') as f:
            pickle.dump(self.key2index, f)
        
    def train(self):
        model = DeepFM(self.linear_feature_columns, 
                       self.dnn_feature_columns, 
                       task="regression",
                       device=self.device)
        
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
            self.key2index = pickle.load(f)

        # 예측에 필요한 메타데이터 재구성
        temp_like_list = self.data[self.sequence_feature].apply(ast.literal_eval)
        self.max_len = max(len(x) for x in temp_like_list)

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
        input_df[sequence_feature_name] = pad_sequences(input_df[sequence_feature_name], maxlen=self.max_len, padding='post', value=0)
        
        # 모델 입력 형태로 변환
        model_input = {}
        for name in self.feature_names:
            if name == sequence_feature_name:
                model_input[name] = np.array(list(input_df[name]))
            else:
                model_input[name] = input_df[name].values
        
        # 모델 로드 및 예측
        model = DeepFM(self.linear_feature_columns, 
                       self.dnn_feature_columns, 
                       task="regression",
                       device=self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.compile("adam", "mse", metrics=["mse"])
        
        return model.predict(model_input)
    
if __name__ == "__main__":
    deepfm_train = DeepFMModdelTrain(os.environ.get("CLICK_LOG", ""))
    deepfm_train.preprocess()
    model = deepfm_train.train()
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
        "like_list": ["[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]", "[26, 22, 29, 44]", "[11, 28, 14, 29, 10, 22, 8, 25, 30]"]
    }
    prediction = deepfm_train.predict(input_data)
    print("예측 결과:", prediction)