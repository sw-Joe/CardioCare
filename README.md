# CardioCare: 심장병 발병 가능성 예측을 위한 종단간 ML 시스템

본 저장소는 임상 데이터로부터 심장병 발병 가능성을 예측하여 심장 전문의의 의사결정을 보조하는 종단간 머신러닝 시스템입니다. 데이터 누수 방지, 결정론적 재현성, SQLite RDBMS 백엔드 기반 실험 추적 및 프로덕션 레벨의 모니터링 아키텍처를 준수하여 설계되었습니다.

> **윤리적 관점 선언**: 본 시스템은 오직 정보를 알리는 용도일 뿐이며, 단독으로 의료적 결정을 내리는 시스템이 아닙니다. 최종 진단 책임은 의료 전문가에게 있습니다.

---

## 1. 프로젝트 디렉터리 구조

```text
├── data/
│   └── heart+disease/
│       └── processed.cleveland.data  # 데이터셋 마스터 파일
├── notebooks/
│   └── 01_eda_preprocessing.ipynb    # §5.1 EDA 및 전처리 분석 노트북
├── src/
│   ├── preprocessing.py              # 공통 전처리 파이프라인 및 클리닝 모듈
│   ├── train.py                      # §5.2 다중 모델 학습 및 MLflow 실험 추적 스크립트
│   ├── inference.py                  # §5.3 프로덕션 추론 엔트리포인트
│   └── monitor.py                    # §5.4 통계적 데이터 드리프트 탐지 스크립트
├── tests/
│   └── test_pipeline.py              # §5.3 4대 필수 요건 유닛 테스트
├── Dockerfile                        # 경량 보안 컨테이너 빌드 명세
├── requirements.txt                  # 의존성 고정 파일
└── .github/workflows/ci.yml          # GitHub Actions 지속적 통합 설정
```

## 2. 개발 환경 구축 및 로컬 파이프라인 재현 절차

본 프로젝트는 Python 3.10 환경에서 동작하며 무작위 시드는 `42`로 결정론적 고정되었습니다.
모든 명령어는 프로젝트 루트 디렉토리(`cardio_care/`)에서 실행해야 파이썬 패키지 내부 경로 탐색이 정상 작동합니다.

### 2.1 가상환경 초기화 및 패키지 설치
```Bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 의존성 고정 설치
pip install --upgrade pip
pip install -r requirements.txt
```

2.2 스타일 검증 및 자동화 단위 테스트 (§5.3)

Ruff 정적 분석 도구를 통한 PEP 8 스타일 검증 및 4대 필수 단위 테스트를 수행합니다.
```Bash
# Ruff 린터 및 포매터 검증
ruff check .
ruff format --check .

# 유닛 테스트 모듈 실행 (반드시 -m 플래그 사용)
python -m unittest discover -s tests -p "test_*.py"
```

2.3 머신러닝 모델 학습 및 실험 추적 실행 (§5.2)

3개 계열 베이스라인 학습 후 최적 후보군을 선별하여 5-Fold 교차 검증을 가동합니다. 파이썬 모듈 시스템 패키징 구조로 실행합니다.
```Bash
# 모델 훈련 파이프라인 가동 (반드시 -m 플래그 사용)
python -m src.train
```
- MLflow 실험 대시보드 확인:
```Bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

2.4 운영 모니터링 및 데이터 드리프트 탐지 (§5.4)

실제 데이터셋 기반의 일자별 점진적 센서 열화 수치를 직접 통계 연산하여 드리프트 경고 트리거와 성능 저하 시계열 리포트를 추출합니다.
```Bash
# 모니터링 파이프라인 가동 (반드시 -m 플래그 사용)
python -m src.monitor
```

## 3. Docker 컨테이너를 통한 배포 및 재현 절차 (§5.3)

본 가이드는 프로덕션 서빙 격리 환경에서의 무결한 작동을 보장합니다. 멀티 스테이지 빌드 및 비루트 보안 권한이 적용되어 있습니다.
WSL2 환경에서 실행 시, 파일 권한 이슈를 방지하기 위해 저장소가 Linux 파일 시스템 영역(/home/)에 위치해 있는지 확인하십시오.

### 3.1 도커 이미지 빌드

```Bash
docker build -t cardiocare-system:1.0.0 .
```

### 3.2 컨테이너 추론 엔트리포인트 구동 테스트

컨테이너 내부의 비루트 유저 권한(appuser) 및 가상 환경 내 환경 변수 매핑 상태에서 src/inference.py가 정상 구동되는지 테스트합니다. 추론 프로세스가 개시되며 런타임 로그가 적재됩니다.

```Bash
docker run --rm cardiocare-system:1.0.0
```

### 3.3 아티팩트 보존을 위한 볼륨 마운트 실행 (선택 사항)

추론 결과 로그 및 모니터링 시계열 지표를 로컬 호스트와 동기화하여 검증하고자 할 경우 아래 명령을 수행합니다.

```Bash
docker run --rm \
  -v $(pwd)/logs:/app/logs \
  cardiocare-system:1.0.0
```