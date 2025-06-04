# app/models/deepfm_model.py
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
import ast

class DeepFMClickPredictor:
    """DeepFM 기반 클릭 예측 모델"""
    
    def __init__(self, dataframe, sparse_features, target, sequence_feature, max_len, vocab_size):
        """
        DeepFM 클릭 예측 모델 초기화
        
        Args:
            dataframe: 학습 데이터
            sparse_features: 범주형 피처 리스트
            target: 예측 대상 컬럼명
            sequence_feature: 시퀀스 피처 컬럼명
            max_len: 시퀀스 최대 길이
            vocab_size: 어휘 사전 크기
        """
        self.df = dataframe.copy()
        self.sparse_features = sparse_features
        self.sequence_feature = sequence_feature
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.target = target
        self.model = None
        self.feature_columns = None
        self.feature_names = None
        self.train_model_input = None
        self.test_model_input = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # LabelEncoder 저장 (예측 시 사용)
        self.label_encoders = {}

    def preprocess(self):
        """데이터 전처리 수행"""
        # 범주형 피처 인코딩
        for feat in self.sparse_features:
            self.df[feat] = self.df[feat].fillna('-1')
            lbe = LabelEncoder()
            self.df[feat] = lbe.fit_transform(self.df[feat].astype(str))
            self.label_encoders[feat] = lbe  # 인코더 저장
        
        # 타겟 변수 처리
        self.df[self.target] = self.df[self.target].astype(float)

        # 피처 컬럼 정의
        self.feature_columns = [
            SparseFeat(feat, vocabulary_size=self.df[feat].nunique(), embedding_dim=8)
            for feat in self.sparse_features
        ] + [
            VarLenSparseFeat(
                SparseFeat(self.sequence_feature, vocabulary_size=self.vocab_size, embedding_dim=8),
                maxlen=self.max_len,
                combiner='mean'
            )
        ]

        self.feature_names = get_feature_names(self.feature_columns)

        # 데이터 분할
        train, test = train_test_split(self.df, test_size=0.2, random_state=42)
        
        # 모델 입력 데이터 준비
        self.train_model_input = {
            name: np.array(list(train[name])) if name == self.sequence_feature else train[name]
            for name in self.feature_names
        }
        self.test_model_input = {
            name: np.array(list(test[name])) if name == self.sequence_feature else test[name]
            for name in self.feature_names
        }
        self.train_target = train[self.target].values
        self.test_target = test[self.target].values

    def train(self, epochs=10, batch_size=32):
        """모델 학습"""
        self.model = DeepFM(
            linear_feature_columns=self.feature_columns,
            dnn_feature_columns=self.feature_columns,
            task='binary',  # 클릭 예측이므로 binary로 변경
            device=self.device
        )
        
        self.model.compile("adam", "binary_crossentropy", metrics=['auc'])
        
        self.model.fit(
            self.train_model_input,
            self.train_target,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2
        )

    def evaluate(self):
        """모델 평가"""
        pred_ans = self.model.predict(self.test_model_input, batch_size=100)
        
        try:
            auc_score = round(roc_auc_score(self.test_target, pred_ans), 4)
        except:
            auc_score = 'N/A'

        return {
            "LogLoss": round(log_loss(self.test_target, pred_ans), 4),
            "AUC": auc_score
        }
    
    def predict_single(self, user_features, place_features):
        """
        단일 사용자-장소 조합에 대한 CTR 예측
        
        Args:
            user_features: 사용자 특징 딕셔너리
            place_features: 장소 특징 딕셔너리
        
        Returns:
            float: CTR 예측값 (0~1)
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 입력 데이터 준비
        input_data = {}
        
        # 범주형 피처 처리
        for feat in self.sparse_features:
            if feat.startswith('user_'):
                value = user_features.get(feat, '-1')
            else:
                value = place_features.get(feat, '-1')
            
            # LabelEncoder로 인코딩
            if feat in self.label_encoders:
                try:
                    encoded_value = self.label_encoders[feat].transform([str(value)])[0]
                except ValueError:
                    # 학습 시 보지 못한 값은 -1로 처리
                    encoded_value = self.label_encoders[feat].transform(['-1'])[0]
            else:
                encoded_value = 0
            
            input_data[feat] = np.array([encoded_value])
        
        # 시퀀스 피처 처리
        like_list = user_features.get(self.sequence_feature, [])
        if isinstance(like_list, str):
            like_list = ast.literal_eval(like_list)
        
        # 패딩 처리
        if len(like_list) > self.max_len:
            like_list = like_list[:self.max_len]
        else:
            like_list = like_list + [0] * (self.max_len - len(like_list))
        
        input_data[self.sequence_feature] = np.array([like_list])
        
        # 예측 수행
        ctr_score = self.model.predict(input_data, batch_size=1)[0][0]
        return float(ctr_score)

    def save_model(self, path):
        """모델 저장"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'feature_columns': self.feature_columns,
                'feature_names': self.feature_names,
                'label_encoders': self.label_encoders,
                'vocab_size': self.vocab_size,
                'max_len': self.max_len
            }, path)
    
    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.feature_columns = checkpoint['feature_columns']
        self.feature_names = checkpoint['feature_names']
        self.label_encoders = checkpoint['label_encoders']
        self.vocab_size = checkpoint['vocab_size']
        self.max_len = checkpoint['max_len']
        
        self.model = DeepFM(
            linear_feature_columns=self.feature_columns,
            dnn_feature_columns=self.feature_columns,
            task='binary',
            device=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
