import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import mlflow
import mlflow.sklearn


# 공통 전처리 모듈 경로 인식
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import clean_raw_data, build_production_pipeline, HEART_DISEASE_SCHEMA


SEED = 42


def evaluate_and_log_metrics(y_true, y_pred, run_name):
    """
    임상적 맥락을 고려한 5대 핵심 평가지표를 산출하고 MLflow에 기록합니다.
    """
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }
    
    # MLflow 지표 적재
    for metric_name, val in metrics.items():
        mlflow.log_metric(f"test_{metric_name}", val)
        
    print(f"[{run_name}] Balanced Acc: {metrics['balanced_accuracy']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f}")
    
    # Confusion Matrix 이미지 생성 및 아티팩트 저장
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


def main():
    # 1. 환경 설정 및 데이터 로드
    DATA_PATH = PROJECT_ROOT / "data" / "heart+disease" / "processed.cleveland.data"
    
    # 로컬 MLflow 실행 기록 경로 명시
    DB_PATH = PROJECT_ROOT / "mlflow.db"
    # ARTIFACT_PATH = PROJECT_ROOT / "mlruns"

    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    mlflow.set_experiment("CardioCare_Heart_Disease_Prediction")
    
    raw_df = pd.read_csv(DATA_PATH, header=None, names=HEART_DISEASE_SCHEMA, na_values="?")
    cleaned_df = clean_raw_data(raw_df)
    
    X = cleaned_df.drop(columns=["target"])
    y = cleaned_df["target"]
    
    # 2. [요구사항 만족] 결정론적 데이터 분할 (80:20, 계층화, 시드 고정)
    # 근거: 소규모 데이터셋의 파티션 편향을 방지하고 균등 분할 보장
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # 3. 공통 전처리 파이프라인 적용 (격리 수행으로 누수 차단)
    NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    preprocessor = build_production_pipeline(NUM_COLS, CAT_COLS)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # 4. [요구사항 만족] 특성 선택 (Feature Selection)
    # 근거: 트리 기반 중요도를 산출하여 다중공선성 결함 유발 특성을 배제
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=SEED), threshold="median")
    X_train_sel = selector.fit_transform(X_train_proc, y_train)
    X_test_sel = selector.transform(X_test_proc)
    
    # 5. [요구사항 만족] 3개 계열 후보 모델 정의
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Support_Vector_Machine": SVC(probability=True, random_state=SEED),
        "Random_Forest": RandomForestClassifier(random_state=SEED)
    }
    
    best_baseline_family = None
    best_f1 = -1
    
    # 각 계열별 Baseline 모델 실험 루프 및 MLflow 자동 추적
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"Baseline_{model_name}") as run:
            # 태그 적재 (요구사항 명시)
            mlflow.set_tag("model_family", model_name)
            mlflow.set_tag("stage", "baseline")
            
            # 모델 학습 및 예측
            model.fit(X_train_sel, y_train)
            preds = model.predict(X_test_sel)
            
            # 파라미터 및 아티팩트 로깅
            mlflow.log_params(model.get_params())
            res = evaluate_and_log_metrics(y_test, preds, f"Baseline_{model_name}")
            mlflow.shadow_models = mlflow.sklearn.log_model(model, f"model_{model_name}")
            
            # 상위 후보 모델 선정 조건 스크리닝 (F1-score 기준)
            if res["f1_score"] > best_f1:
                best_f1 = res["f1_score"]
                best_baseline_family = model_name

    print(f"\n★ 교차 검증 및 하이퍼파라미터 튜닝 대상 선정 계열: {best_baseline_family}")

    # 6. [요구사항 만족] 5-Fold Stratified CV 및 하이퍼파라미터 튜닝 수행
    # 선정 근거: 클래스 분포 불균형 환경에서 가장 안전한 폴드 격리 구조 검증 방식
    with mlflow.start_run(run_name=f"Tuned_{best_baseline_family}_GridSearch") as parent_run:
        mlflow.set_tag("model_family", best_baseline_family)
        mlflow.set_tag("stage", "hyperparameter_tuning")
        
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        if best_baseline_family == "Random_Forest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]
            }
            estimator = RandomForestClassifier(random_state=SEED)
        elif best_baseline_family == "Logistic_Regression":
            param_grid = {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"]
            }
            estimator = LogisticRegression(max_iter=1000, random_state=SEED)
        else: # Support Vector Machine
            param_grid = {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            }
            estimator = SVC(probability=True, random_state=SEED)
            
        # 임상적 가치를 고려하여 평가지표를 f1 또는 recall 변형 탐색 설정
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring="f1",
            n_jobs=-1
        )
        
        grid_search.fit(X_train_sel, y_train)
        best_model = grid_search.best_estimator_
        
        # 최적 하이퍼파라미터 및 교차 검증 결과 저장
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # 최종 최적 모델 테스트셋 평가
        final_preds = best_model.predict(X_test_sel)
        print("\n=== 최종 최적화 모델 테스트 결과 ===")
        evaluate_and_log_metrics(y_test, final_preds, f"Tuned_{best_baseline_family}")
        
        # 최종 모델 아티팩트 등록
        mlflow.sklearn.log_model(best_model, "final_optimized_model")
        print("최종 최적화 모델 및 실험 아티팩트가 MLflow 레지스트리에 저장되었습니다.")


if __name__ == "__main__":
    main()