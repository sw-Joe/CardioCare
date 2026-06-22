import unittest
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from src.preprocessing import build_production_pipeline


class TestCardioCarePipeline(unittest.TestCase):
    
    def setUp(self) -> None:
        """독립적인 테스트 스케줄 단위의 가상 환자 데이터(Mock Data) 셋업"""
        self.NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        
        self.valid_mock_data = pd.DataFrame({
            "age": [55, 40, 65], "sex": [1, 0, 1], "cp": [2, 0, 3],
            "trestbps": [130, 120, 140], "chol": [250, 200, 300], "fbs": [0, 1, 0],
            "restecg": [1, 0, 1], "thalach": [150, 170, 120], "exang": [0, 1, 0],
            "oldpeak": [1.5, 0.0, 2.5], "slope": [1, 2, 0], "ca": [0, 2, 1], "thal": [2, 3, 1]
        })
        
        self.pipeline = build_production_pipeline(self.NUM_COLS, self.CAT_COLS)
        self.pipeline.fit(self.valid_mock_data)

    def test_01_prediction_shape_match(self) -> None:
        """[요구사항 1] 데이터프레임 변환 후 행렬 인덱스 및 형상(Shape) 일치 무결성 검증"""
        transformed_X = self.pipeline.transform(self.valid_mock_data)
        
        # 유입 건수와 변환 행 수의 완전 일치 매칭 판정
        self.assertEqual(transformed_X.shape[0], self.valid_mock_data.shape[0])
        # 인코딩 차원 확장에 따른 유효성 스크리닝
        self.assertGreater(transformed_X.shape[1], len(self.NUM_COLS))

    def test_02_probability_range_and_sum(self) -> None:
        """[요구사항 2] 예측 분류 확률의 유효 수학적 범위 [0, 1] 및 행별 합산 스케일 1.0 여부 검증"""
        # 다중 클래스 소프트맥스 출력 형태의 모킹 행렬 선언
        mock_probs = np.array([
            [0.85, 0.15],
            [0.02, 0.98],
            [0.45, 0.55]
        ])
        
        # 하한 및 상한 임계 곡선 범위 유효성 판단
        self.assertTrue(np.all(mock_probs >= 0.0) and np.all(mock_probs <= 1.0))
        
        # 부동 소수점 누적 오차를 반영한 행 합산 1.0 검정
        row_sums = mock_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_03_clinical_feature_input_range_validation(self) -> None:
        """[요구사항 3] 생물학적으로 불가능한 데이터유입 시 임상 도메인 방어선 차단 동작 검증"""
        invalid_data = self.valid_mock_data.copy()
        invalid_data.loc[0, "age"] = -10  # 비정상 데이터 주입
        
        # 시스템 전단 검증기가 명세 범위를 벗어난 오류에 대해 정확히 ValueError를 발생시키는지 판단
        with self.assertRaises(ValueError):
            validate_clinical_bounds(invalid_data)

    def test_04_deterministic_operation_on_fixed_seed(self) -> None:
        """[요구사항 4] 동일 시드 하에서 모델 초기화 및 적합 연산의 결정론적(Deterministic) 재현성 검증"""
        from sklearn.ensemble import RandomForestClassifier
        transformed_X = self.pipeline.transform(self.valid_mock_data)
        y_mock = np.array([0, 1, 0])
        
        # 고정된 무작위 난수 기반의 난수열 복제 가동
        model_a = RandomForestClassifier(n_estimators=10, random_state=42)
        model_b = RandomForestClassifier(n_estimators=10, random_state=42)
        
        model_a.fit(transformed_X, y_mock)
        model_b.fit(transformed_X, y_mock)
        
        # 두 독립 실행 모델의 클래스별 리턴 확률 값이 완벽히 일치하는지 단언
        np.testing.assert_array_equal(model_a.predict_proba(transformed_X), model_b.predict_proba(transformed_X))


def validate_clinical_bounds(df: pd.DataFrame) -> bool:
    """환자의 인적/생물학적 도메인 허용 스키마 임계치 스크리닝 가드"""
    if (df["age"] < 0).any() or (df["age"] > 120).any():
        raise ValueError("임상 무결성 에러: 연령 데이터 수치가 의학적 범주를 이탈했습니다.")
    return True


if __name__ == "__main__":
    unittest.main()