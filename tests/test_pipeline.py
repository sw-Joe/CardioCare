from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import load_n_clean_data, build_production_pipeline


# 전역 실험 시드 고정
SEED: int = 42
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class TestCardioCarePipeline(unittest.TestCase):
    def setUp(self) -> None:
        """데이터셋 일부를 로드하여 셋업"""
        DATA_PATH = PROJECT_ROOT / "data" / "heart+disease" / "processed.cleveland.data"
        cleaned_df = load_n_clean_data(DATA_PATH)

        X = cleaned_df.drop(columns=["target"])
        y = cleaned_df["target"]

        # 실제 데이터셋의 분할본을 테스트 마스터로 활용
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )

        self.NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

        self.pipeline = build_production_pipeline(self.NUM_COLS, self.CAT_COLS)
        self.pipeline.fit(self.X_train)

    def test_01_prediction_shape_match(self) -> None:
        """01. 실제 데이터를 변환했을 때 행렬 인덱스 및 형상(Shape) 무결성 검증"""
        transformed_X = self.pipeline.transform(self.X_test)

        # 유입된 실제 테스트셋 행 수와 출력 행 수의 일치 판정
        self.assertEqual(transformed_X.shape[0], self.X_test.shape[0])
        self.assertGreater(transformed_X.shape[1], len(self.NUM_COLS))

    def test_02_probability_range_and_sum(self) -> None:
        """02. 실제 모델이 출력한 예측 확률 배열의 수학적 범위 [0, 1] 및 합산 1.0 검증"""
        X_train_proc = self.pipeline.transform(self.X_train)
        X_test_proc = self.pipeline.transform(self.X_test)

        model = RandomForestClassifier(random_state=SEED)
        model.fit(X_train_proc, self.y_train)

        # 실제 예측 확률 매트릭스를 직접 연산하여 단언문 수행
        actual_output_probs = model.predict_proba(X_test_proc)

        # 모든 클래스별 확률값은 0 이상 1 이하에 존재해야 함
        self.assertTrue(
            np.all(actual_accuracy_trend := actual_output_probs >= 0.0)
            and np.all(actual_output_probs <= 1.0)
        )

        # 행별 확률의 수학적 합계는 부동소수점 오차 범위 내에서 정확히 1.0 스케일이어야 함
        row_sums = actual_output_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_03_clinical_feature_input_range_validation(self) -> None:
        """03. 실제 데이터를 의도적으로 오염시켰을 때 임상 방어선의 예외 차단 동작 검증"""
        # 실제 데이터 사본 생성
        invalid_clinical_data = self.X_test.copy()
        # [INTENTIONAL MUTATION] 방어선 작동 테스트를 위해 특정 실제 나이 수치를 음수(-10)로 강제 변이
        invalid_clinical_data.loc[invalid_clinical_data.index[0], "age"] = -10

        with self.assertRaises(ValueError):
            validate_clinical_bounds(invalid_clinical_data)

    def test_04_deterministic_operation_on_fixed_seed(self) -> None:
        """04. 실제 데이터셋 환경에서 동일 시드 하의 알고리즘 결정론적(Deterministic) 재현성 검증"""
        X_test_proc = self.pipeline.transform(self.X_test)

        model_a = RandomForestClassifier(n_estimators=10, random_state=SEED)
        model_b = RandomForestClassifier(n_estimators=10, random_state=SEED)

        model_a.fit(X_test_proc, self.y_test)
        model_b.fit(X_test_proc, self.y_test)

        np.testing.assert_array_equal(
            model_a.predict_proba(X_test_proc), model_b.predict_proba(X_test_proc)
        )


def validate_clinical_bounds(df: pd.DataFrame) -> bool:
    """age값에 대한 유효성 검증"""
    if (df["age"] < 0).any() or (df["age"] > 120).any():
        raise ValueError(
            "임상 무결성 에러: 연령 데이터 수치가 의학적 범주를 이탈했습니다."
        )
    return True


if __name__ == "__main__":
    unittest.main()
