# AI_07기 1팀 초급 프로젝트 미션
### 팀장 : 이태호 / 팀원 : 조찬영, 손승만, 양수빈

## 목표
사진 속 최대 4개의 알약의 이름(클래스)과 위치(바운딩 박스)를 검출합니다.

## 협업일지 링크
https://www.notion.so/4231e9d21f0e4eef92ed2adb231ca085?v=75ec033e7b49459e984e101424d40fd1

---

## 문서 안내 (우선순위 반영)
프로젝트의 메인 코드/운영 기준 우선순위에 맞춰 문서를 아래 순서로 확인하세요.

1. **function 기반 메인 코드 문서** → `README_py.md`
2. **Notebook 파이프라인 문서** → `README_IPYNB.md`
3. **Seungman 실험 폴더 문서** → `seungman/README.md`

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

### 3) Notebook 실행(보조/실험)
```bash
jupyter lab
```
- 메인 노트북: `pill_detection_pipeline_v23_fixed.ipynb`

---

## 디렉토리 핵심 구조
```text
AI_07_basic/
├── README.md
├── README_py.md
├── README_IPYNB.md
├── main.py
├── function/
├── pill_detection_pipeline_v23_fixed.ipynb
├── seungman/
├── evaluator/
├── eda/
├── global_category_index/
├── model/
└── results/
```
