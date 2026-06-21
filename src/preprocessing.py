import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==========================================
# [1] 프로덕션 대응형 커스텀 변환기 설계
# ==========================================

HEART_DISEASE_SCHEMA = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Train 데이터셋에서 계산된 IQR 기준 경계값을 바탕으로 
    이상치를 안전하게 클리핑(Winsorization)하여 수치 변동성을 제어하는 변환기입니다.
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = []
        self.upper_bounds_ = []

    def fit(self, X, y=None):
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

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for i, col in enumerate(X_df.columns):
            X_df[col] = X_df[col].clip(lower=self.lower_bounds_[i], upper=self.upper_bounds_[i])
        return X_df.to_numpy()


# ==========================================
# [2] 통합 데이터 처리 및 파이프라인 빌더
# ==========================================

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    파이프라인 유입 전, 데이터 레이아웃 자체를 정제하는 메인 함수입니다.
    중복 제거 및 전체가 비어있는 빈 컬럼(Empty Column) 유입 에러를 방어합니다.
    """
    cleaned_df = df.copy()
    
    # 1. 중복 데이터 탐지 및 제거
    duplicate_count = cleaned_df.duplicated().sum()
    if duplicate_count > 0:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        
    # 2. 모든 값이 결측치인 빈 컬럼(Empty Column) 제거 방어선
    empty_cols = [col for col in cleaned_df.columns if cleaned_df[col].isnull().all()]
    if empty_cols:
        cleaned_df = cleaned_df.drop(columns=empty_cols)

    # target 클래스 값을 이진화(target=0 정상 / =1 심장병)
    cleaned_df.loc[cleaned_df['target'] > 1, 'target'] = 1
        
    return cleaned_df


def build_production_pipeline(numeric_features, categorical_features):
    """
    새로운 데이터셋에 그대로 100% 재적용 가능한 
    sk-learn Pipeline 및 ColumnTransformer 통합 객체를 반환합니다.
    """
    
    # 수치형 변수 파이프라인: 중앙값 대치 -> 이상치 클리핑 -> 표준 스케일링
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", OutlierClipper(factor=1.5)),
        ("scaler", StandardScaler())
    ])

    # 범주형 변수 파이프라인: 최빈값 대치 -> 원핫 인코딩 (미지 변수 에러 무시 설정)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 컬럼별 독립적 변환 결합 및 스키마 강제
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"  # 명시되지 않은 이상 컬럼 유입 시 자동 탈락 (보안)
    )
    
    return preprocessor


# ==========================================
# [3] 재현성 검증 엔트리포인트
# ==========================================
if __name__ == "__main__":
    # 데이터 정의 및 가상 로드 시뮬레이션
    NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    # 테스트용 모크 데이터 생성 (결측치, 중복, 빈 컬럼 강제 포함)
    mock_data = pd.DataFrame({
        "age": [63, 37, 41, 56, 56, np.nan],  # 결측치
        "trestbps": [145, 130, 130, 120, 120, 140],  # 중복 행 유발용
        "chol": [233, 250, 204, 409, 409, 600],  # 600은 이상치
        "thalach": [150, 187, 172, 178, 178, 120],
        "oldpeak": [2.3, 3.5, 1.4, 0.8, 0.8, 1.0],
        "sex": [1, 1, 0, 1, 1, np.nan],
        "cp": [3, 2, 1, 3, 3, 4],
        "fbs": [1, 0, 0, 0, 0, 0],
        "restecg": [0, 1, 0, 0, 0, 1],
        "exang": [0, 0, 0, 0, 0, 1],
        "slope": [0, 0, 2, 1, 1, 1],
        "ca": [0, 0, 0, 0, 0, np.nan],
        "thal": [1, 2, 2, 3, 3, 2],
        "completely_empty": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] # 빈 컬럼 에지케이스
    })
    
    print(f"변환 전 원본 Shape: {mock_data.shape}")
    
    # 1. 데이터 클리닝 적용 (중복, 빈 컬럼 제거)
    cleaned_df = clean_raw_data(mock_data)
    print(f"클리닝 후 (중복/빈컬럼 제거) Shape: {cleaned_df.shape}")
    
    # 2. 파이프라인 초기화 및 학습 데이터 적합 (fit_transform)
    X = cleaned_df.drop(columns=["completely_empty"], errors="ignore")
    pipeline = build_production_pipeline(NUM_COLS, CAT_COLS)
    
    transformed_X = pipeline.fit_transform(X)
    print(f"최종 파이프라인 변환 완료 numpy 배열 Shape: {transformed_X.shape}")
    print("성공: 데이터 누수 및 이상치가 제어된 일관된 피처 매트릭스가 반환되었습니다.")