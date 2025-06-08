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

class DeepFMModdelTrain:
    def __init__(self, data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = pd.read_csv(data_path)
        self.sparse_features = ["user_id", "user_name", "age", "gender", "place_id", "place_name","category", "sub_category"]
        self.sequence_feature = "like_list"
        self.linear_feature_columns = None
        self.dnn_feature_columns = None
        self.feature_names = None
        self.model_input = None
        self.target = "yn"        
        self.model_path = "/home/ubuntu/working/MLOps/MLOps/app/model/deepfm_model.pt"
        self.model = None
    
    def preprocess(self): 
        # sparse feature 이름 저장
        sparse_feature_names = self.sparse_features.copy()
        
        # sparse feature에 대해 결측값 처리 및 label encoding 수행
        for feature in sparse_feature_names:
            self.data[feature] = self.data[feature].fillna("unknown")
            self.data[feature] = LabelEncoder().fit_transform(self.data[feature])
        
        # sequence feature 리스트 변환
        self.data[self.sequence_feature] = self.data[self.sequence_feature].apply(ast.literal_eval)
        
        # sequence feature 인코딩
        key2index = {}
        def encode(x):
            for k in x:
                if k not in key2index:
                    key2index[k] = len(key2index) + 1
            return [key2index[k] for k in x]
        self.data[self.sequence_feature] = self.data[self.sequence_feature].apply(encode)
        
        # sequence feature 패딩
        max_len = max(len(x) for x in self.data[self.sequence_feature])
        self.data[self.sequence_feature] = pad_sequences(self.data[self.sequence_feature], maxlen=max_len, padding='post', value=0)
        
        # 최종 feature 생성
        self.sparse_features = [SparseFeat(feature, 
                                           vocabulary_size=self.data[feature].nunique(), 
                                           embedding_dim=4) for feature in sparse_feature_names]
        
        self.sequence_feature = [VarLenSparseFeat(SparseFeat(self.sequence_feature, 
                                                             vocabulary_size=len(key2index) + 1, 
                                                             embedding_dim=4), maxlen=max_len, combiner="mean")]
        
        self.linear_feature_columns = self.sparse_features + self.sequence_feature
        self.dnn_feature_columns = self.sparse_features + self.sequence_feature
        
        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)
        
        # 학습 데이터 생성
        self.model_input = {name: self.data[name] for name in self.feature_names}
        self.model_input[self.target] = self.data[self.target]
        
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
    
    def predict(self):
        model = DeepFM(self.linear_feature_columns, 
                       self.dnn_feature_columns, 
                       task="regression",
                       device=self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.compile("adam", "mse", metrics=["mse"])
        return model.predict(self.model_input)
    
if __name__ == "__main__":
    deepfm_train = DeepFMModdelTrain("/home/ubuntu/working/MLOps/data/final_click_log.csv")
    deepfm_train.preprocess()
    model = deepfm_train.train()
    print(model.predict(deepfm_train.model_input))