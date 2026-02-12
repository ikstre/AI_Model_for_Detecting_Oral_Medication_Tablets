# Seungman 작업 폴더 README

이 문서는 `seungman/` 폴더 내부 파일을 기준으로, 역할/구조/실행 흐름을 정리합니다.

## 1) 프로젝트 목표
- 이미지 시각 효과에 따른 학습 성능 측정
- 시각 효과 기반 훈련 시행 및 결과 비교

## 2) 폴더 개요
`seungman/`은 팀 공용 파이프라인에서 **승만님 실험 분기(EDA, 개선 파이프라인, 시각화 테스트, 가중치 비교)**를 모아둔 작업 공간입니다.

---

## 3) 파일 구조 요약

### 핵심 파이프라인/코드
- `pill_detection_pipeline_v23_fixed.ipynb` : 개선된 메인 노트북 파이프라인
- `pill_detection_pipeline_v23_fixed_pv1.ipynb` : 시각효과/보정 실험 버전
- `pill_model_setup_v4_max4_fixed.py` : 모델/학습 설정 유틸
- `pill_preprocess_v3_robust.py` : 전처리 유틸
- `global_category_index.json` : 클래스 매핑 기준 파일

### 분석/검증 노트북
- `EDA_Seungman_CLEANED_Q1Q3.ipynb`
- `eda_seungman.ipynb`
- `eda_seungman_add1.ipynb`
- `eda_seungman_graph.ipynb`
- `seungman_visualtest.ipynb`
- `1.5_seungman.ipynb`
- `test.ipynb`
- `seungman_make_dataset_local.ipynb`

### 결과물/모델
- `best.pt`, `best(0.985).pt`, `best(pv1).pt`
- `submission_final3_sahi.csv`

---

## 4) 실행 방법

### A. 노트북 실행(권장)
프로젝트 루트에서:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install jupyter ultralytics pandas numpy matplotlib seaborn pycocotools optuna
jupyter lab
```

실행 우선순위:
1. `pill_detection_pipeline_v23_fixed.ipynb`
2. 필요 시 `pill_detection_pipeline_v23_fixed_pv1.ipynb` 비교 실행
3. 결과 확인(`submission_final3_sahi.csv` 포맷 기준)

### B. 보조 Python 파일 사용
- 전처리: `pill_preprocess_v3_robust.py`
- 설정/학습 유틸: `pill_model_setup_v4_max4_fixed.py`

> 주의: 일부 경로가 로컬 절대경로 기반일 수 있으니, 실행 전 데이터/모델 경로를 현재 환경에 맞게 수정하세요.

---

## 5) 기존 작업 요약(유지)
- 알약 데이터셋 구축: 추가 데이터 정제 및 대량 이미지 YOLO 포맷 변환
- 데이터 분석: BBOX 오류 데이터 확인 및 클래스 분포 분석
- 인덱스 동기화: 글로벌 JSON 기준 클래스 ID 정렬
- 버그 수정: 제출 CSV 생성 시 카테고리 ID 오프셋 이슈 개선
- 성능 개선: 시각효과(전처리/증강) 기반 실험
