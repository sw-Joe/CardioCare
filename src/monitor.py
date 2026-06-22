import logging
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import load_n_clean_data, build_production_pipeline


# logger 선언
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s"
)
logger = logging.getLogger("CardioCare_Monitor")

# 패키지 경로 탐색 최적화
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


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

    logger.info("=== [운영 시나리오] 일자별 실제 데이터 드리프트 및 성능 하락 연산 ===")
    for day_idx in range(7):
        current_shift = day_idx * -5.0
        X_test_operational = X_test.copy()
        X_test_operational[TARGET_FEATURE] = (
            X_test_operational[TARGET_FEATURE] + current_shift
        )

        X_test_op_proc = preprocessor.transform(X_test_operational)
        preds = model.predict(X_test_op_proc)
        current_acc = balanced_accuracy_score(y_test, preds)

        actual_accuracy_trend.append(current_acc)
        logger.info(
            f"■ {days[day_idx]} | 적용 Shift 편향: {current_shift:>5.1f} | "
            f"계산된 실제 Balanced Accuracy: {current_acc:.4f}"
        )

    logger.info("=== [최종 단계] Day 7 기준 비모수형 이표본 KS 검정 진단 ===")
    ks_stat, p_value = ks_2samp(
        X_train[TARGET_FEATURE], X_test_operational[TARGET_FEATURE]
    )

    logger.info(f"KS 통계량 (최대 수렴 거리)  : {ks_stat:.4f}")
    logger.info(f"유의 확률 (p-value)          : {p_value:.5e}")

    # 통계적 유의성 임계치 기반 조건부 다이내믹 얼럿 로깅
    if p_value < 0.05:
        logger.warning(
            f"⚠️ [ALERT] 데이터 드리프트 유의성 확보. 특성 '{TARGET_FEATURE}'의 분포 오염을 감지했습니다."
        )
    else:
        logger.info(
            "✅ 운영 유입 데이터의 통계적 분포가 기준선과 유의미하게 일치합니다."
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(
        X_train[TARGET_FEATURE],
        bins=15,
        alpha=0.5,
        label="Train Baseline",
        density=True,
        color="blue",
    )
    axes[0].hist(
        X_test_operational[TARGET_FEATURE],
        bins=15,
        alpha=0.5,
        label="Production (Day 7)",
        density=True,
        color="red",
    )
    axes[0].set_title(f"Feature Distribution Shift: {TARGET_FEATURE}")
    axes[0].set_xlabel("Values")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].text(
        0.05,
        0.95,
        f"KS p-val: {p_value:.2e}",
        transform=axes[0].transAxes,
        verticalalignment="top",
    )

    axes[1].plot(
        days,
        actual_accuracy_trend,
        marker="o",
        color="purple",
        linewidth=2,
        label="Calculated Balanced Accuracy",
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

    logger.info(
        f"종합 모니터링 시각화 플롯 출력 및 아티팩트 보존 완료 -> 경로: {report_path}"
    )


if __name__ == "__main__":
    main()
