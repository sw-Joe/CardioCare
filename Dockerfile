# 1. 베이스 이미지 선정 (Python 표준 경량화 버전 채택)
FROM python:3.13-slim AS builder

# 2. 이미지 레이어 최적화 및 빌드 필수 도구 설치
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. 가상 환경 설정 및 의존성 캐싱 레이어 분리
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 프로덕션 실행 레이어 구성 (경량화 멀티 스테이지 포맷)
FROM python:3.13-slim AS runner
WORKDIR /app

# 빌더 레이어에서 유저 라이브러리 및 패키지 아티팩트 복사
COPY --from=builder /opt/venv /opt/venv
COPY src/ /app/src/
COPY data/ /app/data/
# 학습 단계에서 추출된 MLflow 최적 아티팩트 모델 복사 파트 (생략 가능 혹은 저장 위치 동기화)
# COPY mlruns/ /app/mlruns/ 

# 런타임 환경변수 동기화
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# 상용 보안 요건: 비루트 실행 권한 강제 및 소유권 맵핑
RUN useradd -u 8888 appuser && chown -R appuser:appuser /app /opt/venv
USER appuser

# 6. 추론 엔트리포인트 실행 구문 지정
ENTRYPOINT ["python", "src/inference.py"]