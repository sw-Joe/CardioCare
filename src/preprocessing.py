from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder



# 데이터 상세 기록에 명시된 컬럼명(스키마)
HEART_DISEASE_SCHEMA: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    데이터 누수(Data Leakage)를 방지하는 프로덕션 레벨 커스텀 변환기
    - 훈련 데이터셋(Train Set)에서 계산된 IQR 경계값을 고정 저장
    - 새로운 데이터(Test/Inference) 유입 시 동일한 기준선으로 이상치를 Clipping
    """
    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor
        self.lower_bounds_: List[float] = []
        self.upper_bounds_: List[float] = []

    def fit(self, X: np.ndarray, y: getattr = None) -> "OutlierClipper":
        """훈련 데이터의 특성별 사분위수, IQR 상하한 임계치 학습"""
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = []
        self.upper_bounds_ = []
        
        for col in X_df.columns:
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_.append(q1 - self.factor * iqr)
            self.upper_bounds_.append(q3 + self.factor * iqr)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """학습된 상하한 경계값을 기준으로 극단값 변동성을 제어(Winsorization)"""
        X_df = pd.DataFrame(X).copy()
        for i, col in enumerate(X_df.columns):
            X_df[col] = X_df[col].clip(lower=self.lower_bounds_[i], upper=self.upper_bounds_[i])
        return X_df.to_numpy()


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    파이프라인 외부에서 데이터 스키마 자체의 결함 정제 함수
    중복 행 제거, 값 전체가 결측치인 빈 컬럼 제거, 타깃 변수 이진화를 수행.
    """
    cleaned_df = df.copy()
    
    # 1. 타깃 이진 분류 스키마 강제 (0: 정상 / 1 이상: 심장병 위험군)
    if "target" in cleaned_df.columns:
        cleaned_df["target"] = cleaned_df["target"].apply(lambda x: 1 if x > 0 else 0)
        
    # 2. 훈련 데이터 누수 및 과적합 유발인자 중복 제거
    duplicate_count = cleaned_df.duplicated().sum()
    if duplicate_count > 0:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        
    # 3. 측정 에러로 유입된 빈 컬럼 탈락 방어선
    empty_cols = [col for col in cleaned_df.columns if cleaned_df[col].isnull().all()]
    if empty_cols:
        cleaned_df = cleaned_df.drop(columns=empty_cols)
        
    return cleaned_df


def load_n_clean_data(data_path: Path) -> pd.DataFrame:
    """
    원본 CSV 소스로부터 데이터를 안전하게 수신하여 스키마를 강제 주입,
    결측치 심볼(?) 파싱 및 1차 정제가 완료된 완결형 데이터프레임을 반환.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"정밀 진단 에러: 마스터 데이터 소스가 지정된 경로에 없습니다 -> {data_path}")
        
    raw_df = pd.read_csv(
        data_path, 
        header=None, 
        names=HEART_DISEASE_SCHEMA, 
        na_values="?"
    )
    return clean_raw_data(raw_df)


def build_production_pipeline(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    새로운 데이터 및 서빙 환경에 무수정 재적용 가능한
    sk-learn Pipeline 기반 컬럼 통합 변환기 빌더.
    """
    # 수치형 변수: 중앙값 대치 -> 변동성 제어 클리핑 -> 표준 스케일링
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", OutlierClipper(factor=1.5)),
        ("scaler", StandardScaler())
    ])

    # 범주형 변수: 최빈값 대치 -> 알 수 없는 토큰 에러 우회형 원핫 인코딩
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 명시되지 않은 알 수 없는 노이즈 컬럼 유입 시 제거
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    
    return preprocessor