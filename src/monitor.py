import os
import sys
import time
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import load_n_clean_data, build_production_pipeline

# 인프라 레이어 경로 확립 & 로깅 보존 디렉토리 생성
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH: Path = LOG_DIR / "monitor.log"

# logger 선언 & 파일 출력
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    ]
)
logger = logging.getLogger("CardioCare_Monitor")


def main() -> None:
    DATA_PATH = PROJECT_ROOT / "data" / "heart+disease" / "processed.cleveland.data"
    cleaned_df = load_n_clean_data(DATA_PATH)

    X = cleaned_df.drop(columns=["target"])
    y = cleaned_df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    preprocessor = build_production_pipeline(NUM_COLS, CAT_COLS)
    X_train_proc = preprocessor.fit_transform(X_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_proc, y_train)

    days = [f"Day {i}" for i in range(1, 8)]
    actual_accuracy_trend = []
    TARGET_FEATURE = "thalach"
    
    # 합성 타임스탬프 생성을 위한 기준 시간축 고정 (현재 런타임시점)
    base_unix_time = time.time()
    MODEL_VERSION = "1.0.0"

    logger.info("=== [운영 시나리오] 일자별 실제 데이터 드리프트 및 성능 하락 연산 ===")
    for day_idx in range(7):
        # 연속형 특성 최소 하나(thalach)의 분포를 인위적으로 shift
        current_shift = day_idx * -5.0
        X_test_operational = X_test.copy()
        X_test_operational[TARGET_FEATURE] = X_test_operational[TARGET_FEATURE] + current_shift

        X_test_op_proc = preprocessor.transform(X_test_operational)
        preds = model.predict(X_test_op_proc)
        current_acc = balanced_accuracy_score(y_test, preds)

        actual_accuracy_trend.append(current_acc)
        
        # 시계열 추적용 가상 타임스탬프 계산 (24시간 단위 mocking 가산)
        synthetic_timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(base_unix_time + day_idx * 86400)
        )

        # 추론 경로 커스텀 logging 계측 파일 출력 실행
        logger.info(
            f"[INFERENCE_TRACK] TS: {synthetic_timestamp} | VER: {MODEL_VERSION} | "
            f"INPUT_SHAPE: {X_test_operational.shape} | "
            f"PREDS_RISK_COUNT: {int(preds.sum())}/{len(preds)} | "
            f"ACTUAL_TRUE_COUNT: {int(y_test.sum())}"
        )
        
        logger.info(
            f"■ {days[day_idx]} | 적용 Shift 편향: {current_shift:>5.1f} | "
            f"계산된 실제 Balanced Accuracy: {current_acc:.4f}"
        )

    # 각 연속형 특성에 대하여 이표본 KS 검정 전수 전개 및 플래그 주입
    logger.info("=== [최종] 전 연속형 특성 계열 대상 이표본 KS 검정 진단 ===")
    for col in NUM_COLS:
        ks_stat, p_value = ks_2samp(X_train[col], X_test_operational[col])
        logger.info(f"피처 [{col:<8}] -> KS 통계량 거리: {ks_stat:.4f} | 유의확률 p-value: {p_value:.5e}")
        
        if p_value < 0.05:
            logger.warning(
                f"⚠️ [ALERT DRIFT FLAG] 특성 '{col}'에서 통계적 분포 오염 감지. "
                f"임계치(p < 0.05)를 이탈했습니다."
            )

    # 시각화 및 지표 연관성 차트 출력
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(
        X_train[TARGET_FEATURE], bins=15, alpha=0.5, label="Train Baseline", density=True, color="blue"
    )
    axes[0].hist(
        X_test_operational[TARGET_FEATURE], bins=15, alpha=0.5, label="Production (Day 7)", density=True, color="red"
    )
    axes[0].set_title(f"Feature Distribution Shift: {TARGET_FEATURE}")
    axes[0].set_xlabel("Values")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].plot(
        days, actual_accuracy_trend, marker="o", color="purple", linewidth=2, label="Calculated Balanced Accuracy"
    )
    axes[1].axhline(y=0.75, color="r", linestyle="--", label="Alert Threshold (0.75)")
    axes[1].set_title("Real Operational Performance Degradation Over Time")
    axes[1].set_xlabel("Timeline")
    axes[1].set_ylabel("Metric Score")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    report_path = PROJECT_ROOT / "notebooks" / "drift_monitoring_report.png"
    plt.tight_layout()
    plt.savefig(report_path)
    plt.close()
    
    logger.info(f"종합 모니터링 시각화 플롯 출력 완료 -> 경로: {report_path}")


if __name__ == "__main__":
    main()