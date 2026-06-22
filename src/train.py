import os
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import mlflow
import mlflow.sklearn

# 패키지 경로 탐색 최적화
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import clean_raw_data, build_production_pipeline, HEART_DISEASE_SCHEMA

# 전역 실험 무작위 시드 및 RDBMS 백엔드 경로 설정
SEED: int = 42
DB_PATH: Path = PROJECT_ROOT / "mlflow.db"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def evaluate_and_log_metrics(y_true: pd.Series, y_pred: np.ndarray, run_name: str) -> Dict[str, float]:
    """임상적 오진(False Negative) 제어를 고려한 4대 평가지표 산출 및 오차행렬 로깅"""
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }
    
    for metric_name, val in metrics.items():
        mlflow.log_metric(f"test_{metric_name}", val)
        
    print(f"[{run_name:>23}] Balanced Acc: {metrics['balanced_accuracy']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f}")
    
    # 혼동 행렬 시각화 아티팩트 보존 프로세스
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Disease"], yticklabels=["Normal", "Disease"])
    plt.title(f"Confusion Matrix - {run_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    
    cm_path = f"confusion_matrix_{run_name}.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    
    mlflow.log_artifact(cm_path)
    if os.path.exists(cm_path):
        os.remove(cm_path)
        
    return metrics


def main() -> None:
    # 1. 원본 데이터 로드 및 인프라 파라미터 초기화
    DATA_PATH = PROJECT_ROOT / "data" / "heart+disease" / "processed.cleveland.data"
    
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    mlflow.set_experiment("CardioCare_Heart_Disease_Prediction")
    
    raw_df = pd.read_csv(DATA_PATH, header=None, names=HEART_DISEASE_SCHEMA, na_values="?")
    cleaned_df = clean_raw_data(raw_df)
    
    X = cleaned_df.drop(columns=["target"])
    y = cleaned_df["target"]
    
    # 2. 계층화 데이터 분할 수행 (Stratified Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # 3. 데이터 누수가 격리 차단된 피처 엔지니어링 적합
    NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    preprocessor = build_production_pipeline(NUM_COLS, CAT_COLS)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # 4. 중요도 기반 피처 선택 기법 적용 (Feature Selection)
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=SEED), threshold="median")
    X_train_sel = selector.fit_transform(X_train_proc, y_train)
    X_test_sel = selector.transform(X_test_proc)
    
    # 5. 다중 모델 계열 후보군 정의 및 실험 실행
    models: Dict[str, Any] = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Support_Vector_Machine": SVC(probability=True, random_state=SEED),
        "Random_Forest": RandomForestClassifier(random_state=SEED)
    }
    
    best_baseline_family: str = None
    best_f1: float = -1.0
    
    print("\n=== [1단계] 후보군 베이스라인 실험 및 로깅 개시 ===")
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"Baseline_{model_name}"):
            mlflow.set_tag("model_family", model_name)
            mlflow.set_tag("stage", "baseline")
            
            model.fit(X_train_sel, y_train)
            preds = model.predict(X_test_sel)
            
            mlflow.log_params(model.get_params())
            res = evaluate_and_log_metrics(y_test, preds, f"Baseline_{model_name}")
            mlflow.sklearn.log_model(model, f"model_{model_name}")
            
            if res["f1_score"] > best_f1:
                best_f1 = res["f1_score"]
                best_baseline_family = model_name

    print(f"\n>> 그리드 서치 하이퍼파라미터 튜닝 대상 선정 계열: {best_baseline_family}")

    # 6. 최적 모델 대상 5-Fold 교차 검증 및 하이퍼파라미터 최적화
    print("\n=== [2단계] 하이퍼파라미터 튜닝 및 최적화 GridSearch 개시 ===")
    with mlflow.start_run(run_name=f"Tuned_{best_baseline_family}_GridSearch"):
        mlflow.set_tag("model_family", best_baseline_family)
        mlflow.set_tag("stage", "hyperparameter_tuning")
        
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        if best_baseline_family == "Random_Forest":
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]}
            estimator = RandomForestClassifier(random_state=SEED)
        elif best_baseline_family == "Logistic_Regression":
            param_grid = {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"]}
            estimator = LogisticRegression(max_iter=1000, random_state=SEED)
        else:
            param_grid = {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
            estimator = SVC(probability=True, random_state=SEED)
            
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv_strategy, scoring="f1", n_jobs=-1)
        grid_search.fit(X_train_sel, y_train)
        
        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        final_preds = best_model.predict(X_test_sel)
        evaluate_and_log_metrics(y_test, final_preds, f"Tuned_{best_baseline_family}")
        mlflow.sklearn.log_model(best_model, "final_optimized_model")
        print("\n종단간 모델 훈련 및 최적화 아티팩트 적재 프로세스 완료.")


if __name__ == "__main__":
    main()