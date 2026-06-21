# 1. 베이스 이미지 선정 (Python 표준 경량화 버전 채택)
FROM python:3.10-slim as builder

# 2. 이미지 레이어 최적화 및 빌드 필수 도구 설치
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. 가상 환경 설정 및 의존성 캐싱 레이어 분리
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 4. 프로덕션 실행 레이어 구성 (경량화 멀티 스테이지 포맷)
FROM python:3.10-slim as runner
WORKDIR /app

# 빌더 레이어에서 유저 라이브러리 및 패키지 아티팩트 복사
COPY --from=builder /root/.local /root/.local
COPY src/ /app/src/
COPY data/ /app/data/
# 학습 단계에서 추출된 MLflow 최적 아티팩트 모델 복사 파트 (생략 가능 혹은 저장 위치 동기화)
# COPY mlruns/ /app/mlruns/ 

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# 5. 보안 요건 충족: 루트 권한 악용 방지를 위한 비루트 실행 유저 강제 설정
RUN useradd -u 8888 appuser && chown -R appuser:appuser /app
USER appuser

# 6. 추론 엔트리포인트 실행 구문 지정
ENTRYPOINT ["python", "src/inference.py"]