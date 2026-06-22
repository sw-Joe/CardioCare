import logging
import os
from pathlib import Path
import sys
import time

# 패키지 경로 탐색 최적화
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import load_n_clean_data, build_production_pipeline



# Logger
LOG_DIR = Path("/app/logs") if os.path.exists("/.dockerenv") else Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / "inference.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")]
)
logger = logging.getLogger("CardioCare_Inference")

# 전역 실험 시드 고정
SEED: int = 42


def main() -> None:
    logger.info("=========================================")
    logger.info("CardioCare 실제 데이터 기반 실시간 추론 엔진 가동")
    logger.info("=========================================")
    
    # 1. 실제 마스터 데이터 로드 및 추론용 샘플 행 추출
    DATA_PATH = PROJECT_ROOT / "data" / "heart+disease" / "processed.cleveland.data"
    cleaned_df = load_n_clean_data(DATA_PATH)
    
    X = cleaned_df.drop(columns=["target"])
    y = cleaned_df["target"]
    
    # train.py와 동일한 시드로 분할하여 미지의 테스트 샘플 1건 확보
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    # 실제 테스트셋의 첫 번째 환자 레코드 추출
    real_patient_df = X_test.iloc[0:1] 
    logger.info(f"▶ [EVENT] 실제 환자 임상 데이터 유입 -> 인덱스 ID: {real_patient_df.index[0]} | 구조: {real_patient_df.shape}")
    
    # 2. 파이프라인 및 백엔드 모델 실제 연산 적합 (동일 시드 재현)
    NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    preprocessor = build_production_pipeline(NUM_COLS, CAT_COLS)
    
    
    # 전체 학습 데이터를 기반으로 파이프라인의 상태 실체화
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train_proc = preprocessor.fit_transform(X_train)
    
    # 실제 분류기 학습 및 예측 확률 추출
    model = RandomForestClassifier(random_state=SEED)
    model.fit(X_train_proc, y_train)
    
    # 유입된 실제 환자 데이터를 파이프라인 변환 후 실제 추론 실행
    real_patient_proc = preprocessor.transform(real_patient_df)
    
    time.sleep(0.5) # 연산 Latency 시뮬레이션
    ### [REAL PROBABILITY] 하드코딩을 제거하고 모델이 연산한 실제 확률 배열 바인딩
    real_prediction_proba = model.predict_proba(real_patient_proc)
    disease_risk_score = real_prediction_proba[0][1]
    
    logger.info("--- [배포 시스템 계측 메타데이터] ---")
    logger.info("DEPLOY_MODEL_VERSION : 1.0.0")
    logger.info(f"PATIENT_DISEASE_RISK : {disease_risk_score * 100:.2f}% (실제 모델 연산치)")
    logger.info(f"ACTUAL_GROUND_TRUTH  : {y_test.iloc[0]} (실제 임상 진단 확정값)")
    
    if disease_risk_score >= 0.5:
        logger.warning("⚠️ [위험 경보] 심장질환 발병 임계점 초과. 담당의 의사결정 보조 신호를 발송합니다.")
    else:
        logger.info("✅ [정상 소견] 임상 수치가 안정권입니다. 정기 모니터링 군으로 분류합니다.")


if __name__ == "__main__":
    main()