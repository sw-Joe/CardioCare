import logging
import os
from pathlib import Path
import sys
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import load_n_clean_data, build_production_pipeline


# 호스트 볼륨 보존 레이어 감지
LOG_DIR = Path("/app/logs") if os.path.exists("/.dockerenv") else Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / "inference.log"

# logger 선언
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("CardioCare_Inference")

# 전역 실험 시드 고정
SEED: int = 42
# 패키지 경로 탐색 최적화
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def main() -> None:
    logger.info("=========================================")
    logger.info("CardioCare 실제 데이터 기반 실시간 추론 엔진 가동")
    logger.info("=========================================")

    DATA_PATH = PROJECT_ROOT / "data" / "heart+disease" / "processed.cleveland.data"
    cleaned_df = load_n_clean_data(DATA_PATH)

    X = cleaned_df.drop(columns=["target"])
    y = cleaned_df["target"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    real_patient_df = X_test.iloc[0:1]
    logger.info(
        f"▶ [EVENT] 실제 환자 임상 데이터 유입 -> 인덱스 ID: {real_patient_df.index[0]} | 구조: {real_patient_df.shape}"
    )

    NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    preprocessor = build_production_pipeline(NUM_COLS, CAT_COLS)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train_proc = preprocessor.fit_transform(X_train)

    model = RandomForestClassifier(random_state=SEED)
    model.fit(X_train_proc, y_train)

    real_patient_proc = preprocessor.transform(real_patient_df)

    time.sleep(0.5)
    real_prediction_proba = model.predict_proba(real_patient_proc)
    disease_risk_score = real_prediction_proba[0][1]

    logger.info("--- [배포 시스템 계측 메타데이터] ---")
    logger.info("DEPLOY_MODEL_VERSION : 1.0.0")
    logger.info(
        f"PATIENT_DISEASE_RISK : {disease_risk_score * 100:.2f}% (실제 모델 연산치)"
    )
    logger.info(f"ACTUAL_GROUND_TRUTH  : {y_test.iloc[0]} (실제 임상 진단 확정값)")

    if disease_risk_score >= 0.5:
        logger.warning(
            "⚠️ [위험 경보] 심장질환 발병 임계점 초과. 담당의 의사결정 보조 신호를 발송합니다."
        )
    else:
        logger.info(
            "✅ [정상 소견] 임상 수치가 안정권입니다. 정기 모니터링 군으로 분류합니다."
        )


if __name__ == "__main__":
    main()
