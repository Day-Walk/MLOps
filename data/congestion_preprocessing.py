import pandas as pd
import numpy as np
from datetime import datetime
import json


# 클래스 CongestionDataPreprocessor (JSON -> 정제된 DataFrame)
class CongestionDataPreprocessor:
    """
    실시간 도시 데이터(JSON)를 받아 머신러닝 모델에 사용 가능한
    정제된 데이터프레임으로 변환하는 클래스 (1차 전처리)
    """
    def __init__(self):
        self.numeric_cols = [
            'MALE_PPLTN_RATE', 'FEMALE_PPLTN_RATE', 'PPLTN_RATE_0', 'PPLTN_RATE_10', 
            'PPLTN_RATE_20', 'PPLTN_RATE_30', 'PPLTN_RATE_40', 'PPLTN_RATE_50', 
            'PPLTN_RATE_60', 'PPLTN_RATE_70', 'RESNT_PPLTN_RATE', 'NON_RESNT_PPLTN_RATE',
            'ROAD_TRAFFIC_SPD', 'TEMP', 'SENSIBLE_TEMP', 'HUMIDITY', 'WIND_SPD', 
            'UV_INDEX_LVL', 'PM25', 'PM10', 'AIR_IDX_MVL', 'EVENT_COUNT'
        ]
        self.non_numeric_cols = ['AREA_CONGEST_LVL', 'ROAD_TRAFFIC_IDX', 'PRECIPITATION']

    def _safe_float_convert(self, value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def process_initial_json(self, json_data):
        try:
            ppltn = json_data['CITYDATA']['LIVE_PPLTN_STTS'][0]
            road = json_data['CITYDATA']['ROAD_TRAFFIC_STTS']['AVG_ROAD_DATA']
            weather = json_data['CITYDATA']['WEATHER_STTS'][0]
            events = json_data['CITYDATA'].get('EVENT_STTS', [])
        except (KeyError, IndexError, TypeError) as e:
            print(f"JSON 데이터 처리 중 오류 발생: {e}")
            return None

        data = {
            'AREA_NM': ppltn.get('AREA_NM'),
            'AREA_CD': ppltn.get('AREA_CD'),
            'CALL_API_TIME': datetime.now(),
            'AREA_CONGEST_LVL': ppltn.get('AREA_CONGEST_LVL'),
            'MALE_PPLTN_RATE': self._safe_float_convert(ppltn.get('MALE_PPLTN_RATE')),
            'FEMALE_PPLTN_RATE': self._safe_float_convert(ppltn.get('FEMALE_PPLTN_RATE')),
            'PPLTN_RATE_0': self._safe_float_convert(ppltn.get('PPLTN_RATE_0')),
            'PPLTN_RATE_10': self._safe_float_convert(ppltn.get('PPLTN_RATE_10')),
            'PPLTN_RATE_20': self._safe_float_convert(ppltn.get('PPLTN_RATE_20')),
            'PPLTN_RATE_30': self._safe_float_convert(ppltn.get('PPLTN_RATE_30')),
            'PPLTN_RATE_40': self._safe_float_convert(ppltn.get('PPLTN_RATE_40')),
            'PPLTN_RATE_50': self._safe_float_convert(ppltn.get('PPLTN_RATE_50')),
            'PPLTN_RATE_60': self._safe_float_convert(ppltn.get('PPLTN_RATE_60')),
            'PPLTN_RATE_70': self._safe_float_convert(ppltn.get('PPLTN_RATE_70')),
            'RESNT_PPLTN_RATE': self._safe_float_convert(ppltn.get('RESNT_PPLTN_RATE')),
            'NON_RESNT_PPLTN_RATE': self._safe_float_convert(ppltn.get('NON_RESNT_PPLTN_RATE')),
            'ROAD_TRAFFIC_IDX': road.get('ROAD_TRAFFIC_IDX'),
            'ROAD_TRAFFIC_SPD': self._safe_float_convert(road.get('ROAD_TRAFFIC_SPD')),
            'TEMP': self._safe_float_convert(weather.get('TEMP')),
            'SENSIBLE_TEMP': self._safe_float_convert(weather.get('SENSIBLE_TEMP')),
            'HUMIDITY': self._safe_float_convert(weather.get('HUMIDITY')),
            'WIND_SPD': self._safe_float_convert(weather.get('WIND_SPD')),
            'PRECIPITATION': weather.get('PRECIPITATION'),
            'UV_INDEX_LVL': int(self._safe_float_convert(weather.get('UV_INDEX_LVL'))),
            'PM25': self._safe_float_convert(weather.get('PM25')),
            'PM10': self._safe_float_convert(weather.get('PM10')),
            'AIR_IDX_MVL': self._safe_float_convert(weather.get('AIR_IDX_MVL')),
            'EVENT_COUNT': len(events),
        }
        return pd.DataFrame([data])

    def _adjust_and_deduplicate(self, df):
        df['CALL_API_TIME'] = pd.to_datetime(df['CALL_API_TIME'])
        def adjust_minute(dt):
            return dt.replace(minute=0 if dt.minute < 30 else 30, second=0, microsecond=0)
        df['adjusted_datetime'] = df['CALL_API_TIME'].apply(adjust_minute)
        
        df = df.sort_values(by=['AREA_NM', 'CALL_API_TIME'], ascending=False)
        df = df.drop_duplicates(subset=['AREA_NM', 'adjusted_datetime'], keep='last')
        return df.sort_values(by=['AREA_NM', 'adjusted_datetime'])

    def _interpolate(self, df):
        filled_df = pd.DataFrame()
        for area_nm, group in df.groupby('AREA_NM'):
            group = group.set_index('adjusted_datetime')
            full_range = pd.date_range(start=group.index.min(), end=group.index.max(), freq='30min')
            group = group.reindex(full_range)
            group['AREA_NM'] = area_nm
            group[self.numeric_cols] = group[self.numeric_cols].interpolate(method='time', limit_direction='both')
            group[self.non_numeric_cols] = group[self.non_numeric_cols].ffill().bfill()
            filled_df = pd.concat([filled_df, group])
        return filled_df.reset_index().rename(columns={'index': 'adjusted_datetime'})

    def _create_time_features(self, df):
        """데이터프레임에 시간 관련 피처(sin/cos 변환 포함)를 추가합니다."""
        df['month'] = df['adjusted_datetime'].dt.month
        df['day'] = df['adjusted_datetime'].dt.day
        df['weekday'] = df['adjusted_datetime'].dt.dayofweek
        df['hour_30'] = df['adjusted_datetime'].apply(lambda x: x.hour * 2 + x.minute // 30)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_sin'] = np.sin(2 * np.pi * df['day']/df['adjusted_datetime'].dt.days_in_month)
        df['day_cos'] = np.cos(2 * np.pi * df['day']/df['adjusted_datetime'].dt.days_in_month)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/7)
        df['hour_30_sin'] = np.sin(2 * np.pi * df['hour_30']/48)
        df['hour_30_cos'] = np.cos(2 * np.pi * df['hour_30']/48)
        return df

    def _create_features(self, df):
        """[학습용] 타겟 변수와 시간 피처를 생성합니다."""
        # Target(Future Congestion) Columns
        time_shifts = {'ONE_HOUR': 2, 'TWO_HOUR': 4, 'THREE_HOUR': 6, 'SIX_HOUR': 12, 'TWELVE_HOUR': 24}
        for name, shift in time_shifts.items():
            df[f'AFTER_{name}'] = df.groupby('AREA_NM')['AREA_CONGEST_LVL'].shift(-shift)
        
        # Time-based Features (공통 메서드 호출)
        df = self._create_time_features(df)
        
        return df.dropna()

    def _encode(self, df):
        # Categorical Feature Encoding
        congest_map = {'여유': 1, '보통': 2, '약간 붐빔': 3, '붐빔': 4}
        traffic_map = {'원활': 1, '서행': 2, '정체': 3}
        precip_map = {'-': 0.0, '~1mm': 0.5, '1.0mm': 1.0, '1.4mm': 1.4, '1.5mm': 1.5, '1.7mm': 1.7, '1.8mm': 1.8, '2.0mm': 2.0, '2.5mm': 2.5}

        for col in [c for c in df.columns if 'CONGEST' in c or 'AFTER' in c]:
            df[col] = df[col].map(congest_map)
        df['ROAD_TRAFFIC_IDX'] = df['ROAD_TRAFFIC_IDX'].map(traffic_map)
        df['PRECIPITATION'] = df['PRECIPITATION'].map(precip_map).fillna(0)
        
        return pd.get_dummies(df, columns=['AREA_NM'], prefix='AREA')

    def run(self, df):
        """전체 1차 전처리 파이프라인을 실행"""
        df_processed = self._adjust_and_deduplicate(df)
        df_processed = df_processed.drop(columns=['AREA_CD', 'CALL_API_TIME'])
        df_processed = self._interpolate(df_processed)
        df_processed = self._create_features(df_processed)
        df_processed = self._encode(df_processed)
        return df_processed

    def preprocess_for_prediction(self, live_json_data):
        """
        [예측용] 실시간 JSON 데이터 1건을 받아 예측에 사용할 DataFrame으로 전처리합니다.
        """
        # 1. JSON -> DataFrame
        df = self.process_initial_json(live_json_data)
        if df is None or df.empty:
            return None

        # 2. 예측을 위한 adjusted_datetime 생성
        df['adjusted_datetime'] = pd.to_datetime(df['CALL_API_TIME']).apply(
            lambda dt: dt.replace(minute=0 if dt.minute < 30 else 30, second=0, microsecond=0)
        )
        
        # 3. 시간 관련 피처 생성 (공통 메서드 호출)
        df = self._create_time_features(df)
        
        # 4. 범주형 데이터 인코딩
        df = self._encode(df)
        
        return df


# 클래스 ModelDataPreprocessor (정제된 DataFrame -> 학습/테스트 데이터셋)
class ModelDataPreprocessor:
    """
    1차 전처리된 데이터프레임을 받아 모델 학습에 필요한
    피처(X)와 타겟(y) 데이터셋으로 분리하는 클래스 (2차 전처리)
    """
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("입력값은 반드시 pandas DataFrame이어야 합니다.")
        self.df = df.copy()
        self.target_cols = ['AFTER_ONE_HOUR', 'AFTER_TWO_HOUR', 'AFTER_THREE_HOUR', 'AFTER_SIX_HOUR', 'AFTER_TWELVE_HOUR']
        self.features = self._prepare_features()

    def _prepare_features(self):
        """학습에 사용할 피처 목록을 정의합니다."""
        exclude_cols = self.target_cols.copy()
        # 원본 스크립트의 로직을 반영하여 object, datetime 컬럼도 제외 리스트에 추가
        object_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        
        exclude_cols.extend(object_cols)
        exclude_cols.extend(datetime_cols)
        
        features = [col for col in self.df.columns if col not in set(exclude_cols)]
        print(f"학습에 사용될 피처 수: {len(features)}")
        print(f"제외될 컬럼 목록: {list(set(exclude_cols))}")
        return features

    def get_target_columns(self):
        """학습할 모든 타겟 변수 목록을 반환합니다."""
        return self.target_cols

    def split_data_for_target(self, target_variable, test_size=0.3):
        """
        주어진 타겟 변수에 대해 시계열을 고려하여 학습/테스트 데이터셋을 분리합니다.
        
        Args:
            target_variable (str): 분리할 타겟 변수의 이름.
            test_size (float): 테스트 데이터셋의 비율.
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if target_variable not in self.target_cols:
            raise ValueError(f"'{target_variable}'은 유효한 타겟 변수가 아닙니다. 다음 중 하나를 선택하세요: {self.target_cols}")

        # 시간순 정렬 (안정성을 위해 재정렬)
        time_column = 'adjusted_datetime'
        if time_column in self.df.columns:
            self.df = self.df.sort_values(by=time_column).reset_index(drop=True)
        
        X = self.df[self.features]
        y = self.df[target_variable].round().astype(int) # 타겟은 정수형으로 변환

        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        print(f"\n[{target_variable}] 데이터 분리 결과:")
        print(f"  - X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  - X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test

# ==============================================================================
# 사용 예시 (단일 JSON 파일로 실시간 예측)
# ==============================================================================
if __name__ == '__main__':
    import glob
    import pickle

    # --- 예측 설정 ---
    JSON_PATH = '서울대공원_20250520_000003.json'
    MODEL_DIR = './'
    CONGEST_LVL_MAP_INV = {1: '여유', 2: '보통', 3: '약간 붐빔', 4: '붐빔'}

    # --- 1. JSON 데이터 로드 및 전처리 ---
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            live_json_data = json.load(f)
        
        preprocessor = CongestionDataPreprocessor()
        predict_df = preprocessor.preprocess_for_prediction(live_json_data)

    except Exception as e:
        print(f"데이터 준비 중 오류 발생: {e}")
        predict_df = None

    # --- 2. 모델 로드 및 예측 수행 ---
    if predict_df is not None:
        print(f"\n기준 시각: {live_json_data['CITYDATA']['LIVE_PPLTN_STTS'][0]['PPLTN_TIME']}")
        print(f"기준 장소: {live_json_data['CITYDATA']['AREA_NM']}")
        print("\n--- 예측 결과 ---")
        
        model_files = sorted(glob.glob(f"{MODEL_DIR.rstrip('/')}/catboost_model_AFTER_*.pkl"))

        if not model_files:
            print(f"경고: '{MODEL_DIR}'에서 모델 파일(*.pkl)을 찾을 수 없습니다.")

        for model_path in model_files:
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # 모델이 기억하는 피처 순서에 맞춰 데이터 준비
                final_predict_df = predict_df.reindex(columns=model.feature_names_, fill_value=0)

                # 예측
                prediction = model.predict(final_predict_df)[0][0]
                prediction_proba = model.predict_proba(final_predict_df)[0]
                
                # 결과 출력
                target_name = model_path.split('catboost_model_')[1].replace('.pkl', '')
                predicted_label = CONGEST_LVL_MAP_INV.get(prediction, "알 수 없음")
                proba_dict = {
                    CONGEST_LVL_MAP_INV.get(c, f"클래스 {c}"): f"{p*100:.2f}%"
                    for c, p in zip(model.classes_, prediction_proba)
                }

                print(f"[{target_name}]")
                print(f"  -> 예측 혼잡도: {predicted_label} {proba_dict}")

            except Exception as e:
                print(f"'{model_path}' 모델 예측 중 오류 발생: {e}")