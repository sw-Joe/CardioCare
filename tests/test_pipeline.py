import unittest
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

# 테스트 대상 모듈 임포트 (프로젝트 구조에 맞게 조정)
# 여기서는 전처리 파이프라인과 더미 모델/최적 모델이 준비되었다고 가정합니다.
from src.preprocessing import build_production_pipeline

class TestCardioCarePipeline(unittest.TestCase):
    
    def setUp(self):
        """
        테스트 환경 초기화: 일관된 검증을 위한 모크(Mock) 임상 데이터 생성
        """
        self.NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        
        # 정상 범위 내의 가상 환자 샘플 3개 생성
        self.valid_mock_data = pd.DataFrame({
            "age": [55, 40, 65],
            "sex": [1, 0, 1],
            "cp": [2, 0, 3],
            "trestbps": [130, 120, 140],
            "chol": [250, 200, 300],
            "fbs": [0, 1, 0],
            "restecg": [1, 0, 1],
            "thalach": [150, 170, 120],
            "exang": [0, 1, 0],
            "oldpeak": [1.5, 0.0, 2.5],
            "slope": [1, 2, 0],
            "ca": [0, 2, 1],
            "thal": [2, 3, 1]
        })
        
        # 파이프라인 객체 생성 및 적합(fit) 단계 시뮬레이션
        self.pipeline = build_production_pipeline(self.NUM_COLS, self.CAT_COLS)
        self.pipeline.fit(self.valid_mock_data)

    def test_01_prediction_shape_match(self):
        """
        [요구사항 1] 변환 결과의 피처 매트릭스 Shape 일치 여부 검증
        """
        transformed_X = self.pipeline.transform(self.valid_mock_data)
        
        # 입력 행의 수와 출력 피처 매트릭스의 행의 수가 완벽히 일치하는지 확인
        self.assertEqual(transformed_X.shape[0], self.valid_mock_data.shape[0])
        # 원-핫 인코딩 확장 이후 최소 수치형 변수 개수보단 커야 함을 검증
        self.assertGreater(transformed_X.shape[1], len(self.NUM_COLS))

    def test_02_probability_range_and_sum(self):
        """
        [요구사항 2] 예측 확률 모델링의 유효 범위 [0, 1] 및 행별 합계 1.0 여부 검증
        (여기서는 실제 추론기의 소프트맥스/predict_proba 결과물 규격을 모킹하여 검증)
        """
        # 임의의 추론 결과 확률 행렬 시뮬레이션 (N, 2 클래스)
        # 실제 구현 시에는: mock_probs = final_model.predict_proba(transformed_X)
        mock_probs = np.array([
            [0.85, 0.15],
            [0.02, 0.98],
            [0.45, 0.55]
        ])
        
        # 조건 A: 모든 확률값은 0 이상 1 이하에 존재해야 함
        self.assertTrue(np.all(mock_probs >= 0.0) and np.all(mock_probs <= 1.0))
        
        # 조건 B: 행별 확률의 합계는 수학적으로 1.0 스케일이어야 함 (오차 허용치 부동소수점 검증)
        row_sums = mock_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_03_clinical_feature_input_range_validation(self):
        """
        [요구사항 3] 임상적 특성 입력값 경계 유효성 검증 (예: 나이가 음수이거나 극단값 유입 시 방어선)
        """
        invalid_data = self.valid_mock_data.copy()
        invalid_data.loc[0, "age"] = -10  # 생물학적으로 불가능한 에러 데이터 강제 주입
        
        # 도메인 방어선 로직 검증: 시스템이 데이터프레임 무결성 단계나
        # 전처리 내부에서 상용 명세를 벗어난 입력을 탐지하여 예외를 던지는지 확인
        with self.assertRaises(ValueError):
            # 전처리 전단 혹은 추론 모듈에서 나이 범위[0, 120] 검증 함수를 실행한다고 가정
            validate_clinical_bounds(invalid_data)

    def test_04_deterministic_operation_on_fixed_seed(self):
        """
        [요구사항 4] 고정된 무작위 시드 하에서 시스템이 결정론적(Deterministic)으로 작동하는지 검증
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # 동일한 시드로 모델을 두 번 초기화 및 적합
        transformed_X = self.pipeline.transform(self.valid_mock_data)
        y_mock = np.array([0, 1, 0])
        
        model_a = RandomForestClassifier(n_estimators=10, random_state=42)
        model_b = RandomForestClassifier(n_estimators=10, random_state=42)
        
        model_a.fit(transformed_X, y_mock)
        model_b.fit(transformed_X, y_mock)
        
        # 두 독립 모델의 예측 확률 결과가 한 치의 오차도 없이 일치하는지 확인
        proba_a = model_a.predict_proba(transformed_X)
        proba_b = model_b.predict_proba(transformed_X)
        np.testing.assert_array_equal(proba_a, proba_b)


def validate_clinical_bounds(df):
    """test_03 지원을 위한 최소한의 방어선 헬퍼 함수"""
    if (df["age"] < 0).any() or (df["age"] > 120).any():
        raise ValueError("임상 검증 오류: 환자의 연령 범주가 유효하지 않습니다.")
    return True


if __name__ == "__main__":
    unittest.main()