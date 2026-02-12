# AI_07기 1팀 초급 프로젝트 미션
### 팀장 : 이태호 / 팀원 : 조찬영, 손승만, 양수빈

## 목표
사진 속 최대 4개의 알약의 이름(클래스)과 위치(바운딩 박스)를 검출합니다.

## 협업일지 링크
https://www.notion.so/4231e9d21f0e4eef92ed2adb231ca085?v=75ec033e7b49459e984e101424d40fd1

---

## 문서 안내 (메인/서브 + EDA)
프로젝트 문서를 실행 우선순위 기준으로 구분하면 아래와 같습니다.

### A. 메인 파이프라인
1. **function 기반 메인 코드** → `README_py.md`  
   - 엔트리: `main.py`
   - 단계: `gci → offset → dataset → train → inference`
   - 결과물 기본 루트: `<base-dir>/outputs/`
2. **Notebook 기반 메인 파이프라인** → `README_IPYNB.md`  
   - 대상: `pill_detection_pipeline_v23_fixed.ipynb`
   - 결과물 예시: `yolo_runs/`, `exports_pill/`, `submission*.csv`

### B. 서브 실험 파이프라인
3. **Seungman 실험 폴더** → `seungman/README.md`  
   - 시각 효과/실험 분기, 가중치 비교, 실험 제출물 관리
4. **YOLOv8l 실험 코드** → `function_v8l/`  
   - `function/`과 유사한 구조로 v8l 학습/데이터셋 실험

### C. EDA / 데이터 분석 문서
5. **EDA 노트북/요약** → `eda/`  
   - 예: `eda/EDA_SUMMARY(chan).md`, `eda/*.ipynb`
   - 분석 산출물: `eda/eda_outputs_taeho/`

---

## 목차 (프로젝트 본문)
#### 1. DATA 구성 및 EDA, 전처리
#### 2. 모델 구성
#### 3. 학습 설정
#### 4. 모델 학습 결과 분석
#### 5. 종합평가

---

## 빠른 실행 방법

### 1) 환경 준비
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2) 메인 코드 실행(function 중심)
```bash
python main.py --base-dir <데이터_루트>
```
예시:
```bash
python main.py --base-dir E:\download --steps gci offset dataset
python main.py --base-dir E:\download --steps train
python main.py --base-dir E:\download --steps inference
```

### 3) Notebook 실행(메인/서브 실험)
```bash
jupyter lab
```
- 메인 노트북: `pill_detection_pipeline_v23_fixed.ipynb`
- 서브 실험: `seungman/*.ipynb`, `eda/*.ipynb`

---

## 디렉토리 핵심 구조 (상세)
```text
AI_07_basic/
├── README.md
├── README_py.md                      # function 메인 파이프라인 문서
├── README_IPYNB.md                   # 메인 notebook 파이프라인 문서
├── main.py                           # function 기반 메인 엔트리
│
├── function/                         # 메인 코드 (GCI/ID보정/데이터셋/학습/추론)
├── function_v8l/                     # v8l 기반 서브 실험 코드
├── pill_detection_pipeline_v23_fixed.ipynb
├── seungman/                         # 서브 실험(workspace)
├── eda/                              # EDA 노트북 + 요약 문서
│
├── global_category_index/            # 버전 관리된 GCI 결과
├── model/                            # 사전학습/실험 weight
├── results/                          # 실험 이미지/리포트/결과 CSV
├── exports_pill/                     # notebook 추론 산출물(제출/GT/GCI)
└── data/                             # 학습/테스트 원천 데이터
```

---

## 실행별 주요 결과물 경로 (코드 기준 확인)

### 1) `main.py` 실행 시 (`function` 메인)
- `<base-dir>/outputs/global_category_index/`
- `<base-dir>/outputs/datasets/`
- `<base-dir>/outputs/yolo_runs/`
- `<base-dir>/outputs/submissions/`

### 2) `function/dataset_generator.py` 실행 시
- `<output_dir>/<dataset_name>/...`
- `<output_dir>/<dataset_name>/yolo_format/data.yaml`

### 3) `function/csv_generator.py` 실행 시
- `out_csv_path`로 지정한 경로에 제출 CSV 생성

### 4) `function_v8l` 실험 코드 실행 시
- 데이터셋: `output_dir/dataset_name` 하위에 CSV/리포트/`yolo_format/`
- 학습(`train_latest_v8l.py` 기본 예시): `/content/drive/MyDrive/yolo_runs`

### 5) Notebook/실험 폴더 관례 경로
- 메인/실험 노트북 추론 산출물: `exports_pill/`, `results/`, `seungman/*.csv`
