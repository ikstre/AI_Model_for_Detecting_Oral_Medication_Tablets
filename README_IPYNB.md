# 약품 식별 AI 파이프라인 (Notebook 버전)

YOLOv8 아키텍처를 활용한 약품 알약 검출 및 분류를 위한 종합 머신러닝 파이프라인입니다. Jupyter Notebook 기반으로 셀 단위 실행과 중간 결과 확인이 가능하며, Optuna 하이퍼파라미터 탐색을 통한 성능 최적화를 지원합니다.

## 목차

- [시스템 개요](#시스템-개요)
- [시스템 요구사항](#시스템-요구사항)
- [설치 가이드](#설치-가이드)
- [디렉토리 구조](#디렉토리-구조)
- [노트북 셀 구조](#노트북-셀-구조)
- [사용 방법](#사용-방법)
- [파이프라인 구조](#파이프라인-구조)
- [설정 관리](#설정-관리)
- [실험 분석 가이드](#실험-분석-가이드)
- [성능 최적화](#성능-최적화)
- [문제 해결 가이드](#문제-해결-가이드)

---

## 시스템 개요

본 파이프라인은 약품 식별을 위한 8단계 프로세스를 Jupyter Notebook 환경에서 구현합니다:

**1단계: 어노테이션 로드 및 COCO 구성** (Cell 6~7)  
JSON 어노테이션 파일을 수집하고, 이미지 인덱스를 구축하여 COCO 포맷의 통합 데이터셋을 생성합니다. GCI 기반 57개 카테고리 필터링을 적용합니다.

**1.5단계: Global Category Index (GCI) 로드** (Cell 9)  
사전 구축된 `gci_57_MODEL_SORTED_BY_CATEGORY_ID.json`을 로드하여 57개 카테고리의 `category_id ↔ global_index(0..56)` 매핑을 설정합니다.

**1.6단계: Class Weight 생성** (Cell 11)  
클래스 불균형 보정을 위한 가중치를 계산합니다. 빈도 기반 가중치로 오버샘플링에 활용됩니다.

**3단계: category_id ↔ YOLO cls 매핑** (Cell 17)  
GCI의 `global_index`를 YOLO 클래스 인덱스로 직접 사용하는 매핑 테이블을 구성합니다.

**4단계: YOLO 데이터셋 export** (Cell 18)  
이미지 복사 + YOLO 형식 라벨 생성 + `dataset.yaml` 작성으로 학습 데이터를 구축합니다.

**5단계: 베이스라인 학습** (Cell 22)  
Optuna 전 파이프라인 검증용 빠른 학습을 수행합니다.

**7단계: Optuna 하이퍼파라미터 탐색 → 최종 재학습** (Cell 31~36)  
Optuna로 최적 파라미터를 탐색한 뒤, best params로 전체 데이터 최종 학습을 수행합니다. 또는 **7-ALT** 셀로 Optuna를 건너뛰고 기존 가중치를 직접 사용할 수 있습니다.

**8단계: 테스트 추론 및 제출 CSV 생성** (Cell 39~41)  
최종 가중치로 테스트 이미지를 추론하고, GCI 역매핑을 적용하여 대회 형식의 CSV를 생성합니다.

---

## 시스템 요구사항

### 하드웨어 사양

**최소 요구사항:**
- CPU: Intel i5 또는 AMD Ryzen 5 (4코어 이상)
- RAM: 16GB 시스템 메모리
- 저장공간: 100GB 사용 가능한 디스크 공간
- GPU: 선택사항이나 학습 시 권장

**권장 사양:**
- CPU: Intel i7/i9 또는 AMD Ryzen 7/9 (8코어 이상)
- RAM: 32GB 시스템 메모리
- 저장공간: 200GB SSD 저장공간
- GPU: NVIDIA RTX 3080/4080 이상 (8GB 이상 VRAM)

### 소프트웨어 의존성

**운영체제:**
- Windows 10/11 (64비트)
- Ubuntu 20.04 LTS 이상
- macOS 11.0 이상

**Python 환경:**
- Python 3.8 이상
- CUDA Toolkit 11.0 이상 (GPU 가속용)
- cuDNN 8.0 이상 (GPU 가속용)
- Jupyter Notebook 또는 JupyterLab

**필수 패키지:**
```bash
ultralytics>=8.0.0
torch>=1.12.0
pandas>=1.3.0
numpy>=1.21.0
pillow>=8.0.0
tqdm>=4.62.0
matplotlib>=3.4.0
seaborn>=0.11.0
pycocotools>=2.0.0
optuna>=3.0.0
pyyaml>=5.4.0
```

---

## 설치 가이드

### 환경 설정

**1단계: 저장소 복제**

```bash
git clone https://github.com/ikstre/AI_07_basic.git
cd AI_07_basic
```

**2단계: 가상환경 생성**

```bash
python -m venv pipeline_env
source pipeline_env/bin/activate  # Linux/macOS
pipeline_env\Scripts\activate     # Windows
```

**3단계: 의존성 설치**

```bash
pip install -r requirements.txt
```

또는 노트북 Cell 1에서 직접 설치:

```python
!pip -q install ultralytics==8.* pycocotools optuna pyyaml
```

### 저장소 구조

```
AI_07_basic/
├── pill_detection_pipeline_v23_fixed.ipynb   # 메인 노트북
├── pill_model_setup_v4_max4_fixed.py         # CFG, set_seed, train_yolo 등
├── pill_preprocess_v3_robust.py              # 전처리 유틸리티
├── requirements.txt                          # 의존성 목록
├── README_IPYNB.md                           # 본 문서
├── README_py.md                              # py 버전 문서
└── global_category_index/
    └── gci_57_MODEL_SORTED_BY_CATEGORY_ID.json
```

### 데이터 준비

다음 구조에 따라 데이터를 구성하세요:

```
project_root/
├── data/
│   ├── label_aug/                        # 어노테이션 파일
│   │   ├── train_annotations/            # 주 학습 데이터 (category_id 원본)
│   │   ├── TL_4_단일/                    # 단일 알약 이미지 라벨
│   │   ├── VL_1_조합/                    # 검증 조합 데이터
│   │   └── ...
│   ├── train_aug/                        # 학습 이미지
│   └── test_images/                      # 테스트 이미지
├── global_category_index/                # GCI 매핑 파일
│   └── gci_57_MODEL_SORTED_BY_CATEGORY_ID.json
└── pill_detection_pipeline_v23_fixed.ipynb
```

**중요:** `gci_57_MODEL_SORTED_BY_CATEGORY_ID.json` 파일은 반드시 `global_category_index/` 폴더에 위치해야 합니다. 이 파일이 없으면 GCI 로드 셀(Cell 9)에서 오류가 발생합니다.

---

## 디렉토리 구조

파이프라인 실행 시 생성되는 출력 구조:

```
project_root/
├── yolo_pill_ds/                         # YOLO 학습 데이터셋
│   ├── images/                           # 복사된 학습 이미지
│   ├── labels/                           # YOLO 형식 라벨 (.txt)
│   ├── train.txt                         # 학습 이미지 목록
│   ├── val.txt                           # 검증 이미지 목록
│   ├── dataset.yaml                      # YOLO 데이터 설정
│   └── local_categories.csv              # 카테고리 매핑 표
├── yolo_pill_ds_optuna/                  # Optuna 서브셋 데이터셋 (선택)
├── yolo_runs/                            # 학습 결과
│   ├── pill_baseline_augdata/            # 베이스라인 학습
│   │   ├── weights/
│   │   │   ├── best.pt                   # 최고 성능 가중치
│   │   │   └── last.pt                   # 마지막 epoch 가중치
│   │   └── results.csv                   # epoch별 메트릭
│   ├── optuna/                           # Optuna trial 결과
│   └── final/                            # Optuna best params 최종 학습
│       └── *_final_from_optuna_best/
│           └── weights/best.pt
├── exports_pill/                         # 제출용 파일
│   ├── global_category_index.json        # GCI 복사본
│   └── submission_*.csv                  # 제출 CSV
└── global_category_index/                # GCI 원본
    ├── gci_57_MODEL_SORTED_BY_CATEGORY_ID.json
    ├── global_category_index.csv
    └── local_categories.csv
```

---

## 노트북 셀 구조

| 셀 번호 | 구분 | 내용 | 실행 필수 |
|---------|------|------|-----------|
| 0 | Markdown | 파이프라인 개요 | — |
| 1 | Code | 패키지 설치 | 최초 1회 |
| 2 | Code | 한글 폰트 설정 (시각화용) | ✅ |
| 3 | Code | GracefulStopper 정의 (학습 중단 유틸) | ✅ |
| **4** | **Code** | **⚙️ 전체 설정 (cfg)** — 경로, 학습 파라미터, Optuna 설정 | **✅** |
| 5 | Markdown | 어노테이션 로드 설명 | — |
| **6** | **Code** | **1) JSON + 이미지 → COCO 구성** | **✅** |
| 7 | Code | 1-1) COCO 사후 검증 통계 | ✅ |
| 8 | Markdown | GCI 설명 | — |
| **9** | **Code** | **1.5) GCI 로드** (57개 카테고리) | **✅** |
| 10 | Markdown | Class Weight 설명 | — |
| 11 | Code | 1.6) Class Weight 생성/로드 | ✅ |
| 12 | Markdown | 시각화 설명 | — |
| 13 | Code | 이미지 경로 resolve 유틸 | ✅ |
| 14 | Markdown | Train/Val 분리 설명 | — |
| 15 | Code | 2) Train/Val 분리 | ✅ |
| 16 | Markdown | YOLO export 설명 | — |
| 17 | Code | 3) category_id ↔ YOLO cls 매핑 | ✅ |
| **18** | **Code** | **4) YOLO 데이터셋 export** | **✅** |
| 19 | Markdown | GCI 라벨 치환 설명 | — |
| 20 | Code | 4.5) GCI 라벨 치환 (호환, 보통 스킵) | 선택 |
| 21 | Markdown | 베이스라인 학습 설명 | — |
| **22** | **Code** | **5) 베이스라인 학습** | **✅** |
| 23~24 | Code | (옵션) 여러 YOLO 모델 비교 | 선택 |
| 25 | Markdown | Validation 평가 설명 | — |
| 26~28 | Code | 6) Val 평가 + best.pt 탐색 | ✅ |
| 29 | Markdown | Optuna 설명 | — |
| 30 | Code | YOLO→COCO 변환 + mAP 평가 유틸 | ✅ |
| **7-ALT** | **Code** | **⚡ Optuna 건너뛰기 — 수동 파라미터 설정 후 7-1로 이동** | **택 1** |
| 31 | Code | 7-0) Optuna 서브셋 구축 | 택 1 |
| 32~33 | Code | 7) Optuna 탐색 실행 | 택 1 |
| 34~36 | Code | 7-1) Optuna best params 최종 재학습 | 택 1 |
| 37 | Code | (디버그) Trainer args 확인 | 선택 |
| 38 | Markdown | 테스트 추론 설명 | — |
| **39~41** | **Code** | **8) 테스트 추론 + 제출 CSV 생성** | **✅** |
| 42 | Code | (빈 셀) | — |

---

## 사용 방법

### 실행 경로 A: 전체 파이프라인 (Optuna 포함)

전체 데이터 탐색과 최적 학습을 수행하는 표준 경로입니다.

```
Cell 1~4 (설정) → Cell 6~18 (데이터 구축) → Cell 22 (베이스라인)
  → Cell 31~33 (Optuna 탐색) → Cell 34~36 (최종 재학습) → Cell 39~41 (추론/제출)
```

### 실행 경로 B: Optuna 건너뛰기 (수동 파라미터로 최종 학습)

이미 학습된 best.pt가 있거나, 수동으로 파라미터를 지정하여 Optuna 탐색 없이 최종 학습을 수행하는 경로입니다.

```
Cell 1~4 (설정) → Cell 6~18 (데이터 구축) → Cell 22 (베이스라인)
  → 7-ALT (파라미터 설정) → 7-1 (최종 재학습) → Cell 37~38 (가중치 확정) → Cell 39~41 (추론/제출)
```

**7-ALT 셀 사용법:**
```python
# 방법 1: cfg 기본값 그대로 사용 (수정 없이 실행)
best_params = {
    "model":   getattr(cfg, "base_model", "yolov8n.pt"),
    "batch":   cfg.batch,
    "lr0":     cfg.lr0,
    ...
}

# 방법 2: 이전 실험에서 확인한 값으로 수정
best_params = {
    "model":   "yolov8s.pt",
    "batch":   16,
    "lr0":     0.001,
    ...
}

# 방법 3: 기존 best.pt를 pretrained로 fine-tuning
PRETRAINED_WEIGHTS = "yolo_runs/pill_baseline_augdata/weights/best.pt"
```

### 실행 경로 C: 추론만 실행

데이터 구축 후, 학습 없이 기존 가중치로 추론합니다. 7-1 최종 재학습 셀에서 가중치 경로만 확정한 후 추론 셀로 이동합니다.

```
Cell 1~4 (설정) → Cell 6~18 (데이터 구축)
  → 7-ALT (기존 best.pt 지정) → Cell 39~41 (추론/제출)
```

### 학습 중단 및 재개

노트북에는 `GracefulStopper`가 내장되어 있어 안전한 학습 중단이 가능합니다.

**중단 방법:**
- Jupyter 중단 버튼 (■) 또는 `Ctrl+C` 1회 → 현재 epoch 완료 후 안전 종료
- 2회 연속 → 즉시 강제 종료
- 또는 새 셀에서 `stopper.stopped = True` 실행

**재개:** 가중치는 `cfg.save_period`(기본 10 epoch)마다 자동 저장되므로, 중단 후 마지막 체크포인트에서 이어서 학습 가능합니다.

---

## 파이프라인 구조

### 1단계: 어노테이션 로드 및 COCO 구성 (Cell 6)

**목적:**
여러 어노테이션 폴더의 JSON 파일을 스캔하고, 이미지 인덱스와 매칭하여 통합 COCO 데이터셋을 생성합니다.

**핵심 처리:**

**이미지 인덱싱:** `cfg.image_roots` 하위를 재귀 탐색하여 O(1) 파일명→경로 해시맵을 구축합니다.

**JSON 수집:** `cfg.train_ann_dirs` 하위의 모든 `*.json`을 재귀 수집합니다.

**카테고리 ID 정규화:** `normalize_category_id` 함수가 폴더별로 차등 처리합니다:
- `train_annotations/` 하위 → ID를 그대로 유지 (`keep_subdir`)
- 나머지 폴더 → +1 오프셋 적용 (`shift_other`)

**GCI 필터링:** `cfg.gci_filter_allowed_ids = True`일 때, 사전 구축된 GCI의 57개 `category_id`에 해당하지 않는 어노테이션을 제외합니다.

**품질 검증:**
- 바운딩 박스 존재 및 형식 검증
- 좌표 경계 확인 (이미지 크기 초과 여부)
- 퇴화 bbox 필터링 (width ≤ 0 또는 height ≤ 0)
- 인덱스 파일 제거 (`_index` 접미사)
- 중복 bbox 제거 (파일명 + cat_id + 좌표 기준)

**정상 출력 예시:**
```
[BUILD] images: 232 | annotations: 771 | categories: 56
[BUILD] cat_norm_stats: {'cat_norm_keep_subdir': 771, 'cat_norm_shift_other': 193968}
[BUILD] invalid_counts: {'not_in_train57': 193968}
```

---

### 1.5단계: Global Category Index 로드 (Cell 9)

**목적:**
사전 구축된 GCI JSON을 로드하여 전체 파이프라인에서 사용할 `category_id ↔ global_index` 매핑을 설정합니다.

**입력 파일:**
```
global_category_index/gci_57_MODEL_SORTED_BY_CATEGORY_ID.json
```

**GCI JSON 구조:**
```json
{
  "metadata": {
    "total_categories": 57,
    "total_annotations": 763,
    "index_range": "0-56"
  },
  "category_map": {
    "1900": {"category_id": 1900, "category_name": "보령부스파정 5mg", ...},
    ...
  },
  "index_to_id": {"0": 1900, "1": 2483, ...},
  "id_to_index": {"1900": 0, "2483": 1, ...}
}
```

**역할:**
- YOLO 라벨 기록: `global_index`를 class로 사용
- 제출 CSV 역매핑: `global_index → category_id`로 원본 ID 복원
- 카테고리 필터링: 57개 ID 기준으로 학습 데이터 선별

---

### 1.6단계: Class Weight 생성/로드 (Cell 11)

**목적:**
클래스 불균형 보정을 위한 가중치를 계산합니다.

**처리 방식:**
- `category_csv` + `submission_csv`가 모두 있으면 빈도 + confidence 기반 가중치 생성
- 없으면 COCO 어노테이션 빈도 기반 가중치로 fallback

**활용:** `cfg.balance_enable = True`일 때, `train.txt`에 오버샘플링 방식으로 적용됩니다.

---

### 4단계: YOLO 데이터셋 export (Cell 18)

**목적:**
COCO 데이터를 YOLO 학습 형식으로 변환합니다.

**출력:**
- `yolo_pill_ds/images/`: 원본 이미지 복사
- `yolo_pill_ds/labels/`: YOLO 형식 라벨 (class, cx, cy, w, h — 정규화 좌표)
- `yolo_pill_ds/dataset.yaml`: 학습 설정 파일
- `yolo_pill_ds/train.txt`, `val.txt`: 분할 목록

**좌표 변환:**
```
x_center_norm = (bbox_x + bbox_width/2) / image_width
y_center_norm = (bbox_y + bbox_height/2) / image_height
width_norm = bbox_width / image_width
height_norm = bbox_height / image_height
```

**클래스 불균형 보정:** `cfg.balance_enable = True`이면 소수 클래스를 `balance_extra_ratio`(기본 30%)만큼 추가 복사하여 `train.txt`에 append합니다.

---

### 5단계: 베이스라인 학습 (Cell 22)

**목적:**
Optuna 전에 데이터와 파이프라인이 정상인지 빠르게 확인합니다.

**모델 설정:**

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| 아키텍처 | YOLOv8n~s | `cfg.base_model`로 지정 |
| 입력 해상도 | 640×640 | `cfg.imgsz` |
| 학습 에폭 | 300 | `cfg.epochs` (조기 종료: patience=30) |
| 배치 크기 | 8 | `cfg.batch` |
| 옵티마이저 | AdamW | `cfg.optimizer` |
| 학습률 | 0.0005 | `cfg.lr0` |

**데이터 증강:**
```python
# 기하학적 증강
degrees=10.0        # 회전 (±10°)
translate=0.1       # 이동 (10%)
scale=0.2           # 스케일 (±20%)
fliplr=0.5          # 수평 뒤집기 (50%)
flipud=0.5          # 수직 뒤집기 (50%)
mosaic=0.25         # 모자이크 (25%)
erasing=0.2         # 랜덤 지우기 (20%)

# 비활성화 (알약 색상 보존)
hsv_h=0.0           # 색조 변화 없음
hsv_s=0.0           # 채도 변화 없음
hsv_v=0.0           # 명도 변화 없음
```

**학습 기능:**
- 혼합 정밀도 학습 (AMP) — 메모리 효율 향상
- 자동 데이터 분할 (90% 학습, 10% 검증)
- 주기적 체크포인트 (10 epoch마다)
- GracefulStopper를 통한 안전한 중단/재개

---

### 7단계: Optuna 하이퍼파라미터 탐색 (Cell 31~36)

**목적:**
Optuna TPE Sampler + Hyperband Pruner를 사용하여 최적 학습 파라미터를 자동 탐색하고, best params로 최종 재학습합니다.

**탐색 대상:**
```
model, batch, lr0, weight_decay, optimizer,
degrees, translate, scale, mosaic, mixup, erasing
```

**서브셋 모드:** `cfg.optuna_use_subset = True`일 때 전체 데이터의 일부로 빠르게 탐색합니다.
- 방법 1: `cfg.optuna_ann_dirs`에 특정 폴더 지정
- 방법 2: 폴더 미지정 시 전체 데이터에서 `cfg.optuna_sample_ratio`(기본 30%) 랜덤 샘플링

**7-ALT: Optuna 건너뛰기**  
Optuna 탐색 없이 수동으로 하이퍼파라미터를 설정하여 7-1 최종 재학습으로 바로 넘어갑니다:
```python
# cfg 기본값 사용 (수정 없이 실행)
best_params = { "model": "yolov8n.pt", "batch": 8, "lr0": 0.0005, ... }

# 기존 best.pt로 fine-tuning
PRETRAINED_WEIGHTS = "yolo_runs/pill_baseline_augdata/weights/best.pt"
```

---

### 8단계: 테스트 추론 및 제출 CSV 생성 (Cell 39~41)

**목적:**
최종 가중치로 테스트 이미지를 추론하고 대회 형식의 CSV를 생성합니다.

**추론 설정:**
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| 신뢰도 임계값 | 0.001 | 낮게 설정하여 포괄적 검출 |
| NMS IoU 임계값 | 0.7 | 중복 제거 |
| 이미지당 최대 검출 | 4 | `cfg.max_det_per_image` |
| 반정밀도 (FP16) | True | 추론 속도 2배 향상 |

**ID 역매핑:**
GCI의 `index_to_id`를 사용하여 YOLO 클래스 인덱스를 원본 `category_id`로 복원합니다. `cfg.submit_catid_offset = 0`으로 설정되어 있어 추가 오프셋 없이 GCI ID가 그대로 제출됩니다.

**가중치 우선순위:**
1. `cfg.final_weights` (Optuna 최종 재학습 또는 7-ALT 지정)
2. `FINAL_WEIGHTS` 전역 변수
3. `best_pt` (Validation 평가 셀에서 탐색)
4. 자동 탐색 fallback

**CSV 스키마:**
```
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score
```

**평가 지표:** mAP@[0.75:0.95] (IoU=0.75~0.95, step 0.05)

---

## 설정 관리

### 주요 설정 (Cell 4)

모든 설정은 Cell 4의 `cfg` 객체에서 관리됩니다. 이 셀만 수정하면 전체 파이프라인에 반영됩니다.

**데이터 경로:**
```python
cfg.label_root = "data/label_aug"          # 어노테이션 루트
cfg.image_root = "data/train_aug"          # 학습 이미지 루트
cfg.test_img_dir = "data/test_images"      # 테스트 이미지
cfg.yolo_dataset_dir = "yolo_pill_ds"      # YOLO export 경로
```

**카테고리 ID 처리:**
```python
cfg.train_ann_cid_shift = 1                # train_annotations 제외 나머지에 +1
cfg.apply_shift_only_to_subdir = False     # False: subdir 제외, 나머지에 shift
cfg.gci_filter_allowed_ids = True          # 57개 카테고리만 필터링
cfg.submit_catid_offset = 0               # 제출 시 추가 오프셋 없음
```

**학습 하이퍼파라미터:**
```python
cfg.epochs = 300          # 최대 에폭
cfg.patience = 30         # 조기 종료 인내도
cfg.batch = 8             # 배치 크기
cfg.imgsz = 640           # 입력 해상도
cfg.lr0 = 0.0005          # 초기 학습률
cfg.optimizer = "AdamW"   # 옵티마이저
```

**Optuna 설정:**
```python
cfg.optuna_use_subset = True       # 서브셋 모드
cfg.optuna_sample_ratio = 0.3      # 전체의 30% 샘플링
cfg.optuna_batch_sizes = [8, 16]   # 배치 크기 후보
```

---

## 실험 분석 가이드

### 학습 결과 확인

**학습 로그 위치:**
```
yolo_runs/<실험명>/results.csv
```

**주요 메트릭 컬럼:**
- `train/box_loss`, `train/cls_loss`: 학습 손실
- `val/box_loss`, `val/cls_loss`: 검증 손실
- `metrics/mAP50`: mAP@0.5
- `metrics/mAP50-95`: mAP@0.5:0.95
- `metrics/precision`, `metrics/recall`: 정밀도/재현율

**TensorBoard 모니터링:**
```bash
tensorboard --logdir yolo_runs/
```

### 실험 비교 체크리스트

새 실험을 시작하기 전, 이전 실험 결과를 다음 항목으로 분석하세요:

**1) 빌드 통계 확인 (Cell 7 출력):**
```
[VERIFY] images: 232 | annotations: 771 | categories: 56
```
- 이미지/어노테이션 수가 예상과 일치하는지 확인
- `not_in_train57` 필터링 수가 비정상적으로 많지 않은지 확인

**2) 학습 곡선 분석:**
- 학습 손실이 지속적으로 감소하는지
- 검증 손실이 특정 시점부터 증가하지 않는지 (과적합 징후)
- mAP가 수렴했는지 또는 아직 상승 중인지

**3) 클래스별 성능 확인:**
- 특정 카테고리의 AP가 극단적으로 낮은지
- 소수 클래스의 recall이 충분한지

**4) Optuna 결과 분석 (사용 시):**
```python
# Optuna study 결과 확인
print("Best params:", study.best_params)
print("Best value:", study.best_value)

# Trial별 결과
for t in study.trials:
    print(f"Trial {t.number}: {t.value:.4f} | {t.params}")
```

### 성능 개선 실험 방향

**데이터 측면:**
- 오버샘플링 비율 조정: `cfg.balance_extra_ratio` (기본 0.3)
- 증강 강도 조정: `mosaic`, `erasing`, `degrees` 값 변경
- 이미지 해상도 변경: `cfg.imgsz` (416, 512, 640, 800)

**모델 측면:**
- 모델 크기 변경: `yolov8n.pt` → `yolov8s.pt` → `yolov8m.pt`
- 학습률 스케줄 조정: `cfg.lr0` 값 변경
- 옵티마이저 변경: `AdamW`, `SGD`, `Adam`

**후처리 측면:**
- NMS IoU 임계값: 추론 셀의 `iou` 파라미터
- 신뢰도 임계값: `cfg.submit_conf`
- 이미지당 최대 검출 수: `cfg.max_det_per_image`

---

## 성능 최적화

### 학습 가속화

**GPU 메모리 최적화:**
```python
cfg.batch = 4          # 메모리 부족 시 감소
cfg.imgsz = 416        # 더 작은 입력 크기
cfg.amp = True         # 혼합 정밀도 (기본 활성화)
```

**CPU 활용:**
```python
cfg.workers = 4        # CPU 코어 수에 맞게 조정 (기본 0)
```

### 추론 가속화

**배치 크기 및 형식 최적화:**
```python
# Cell 41 추론 실행 시
pred_batch=8           # GPU 메모리에 따라 증가
chunk_size=500         # 큰 청크로 처리
half=True              # FP16 추론 (기본 활성화)
```

**모델 최적화:**
```python
model.export(format='onnx')     # ONNX 형식
model.export(format='engine')   # TensorRT (NVIDIA GPU)
```

---

## 문제 해결 가이드

### 빌드 시 이미지/어노테이션이 0건

**증상:**
```
[BUILD] images: 0 | annotations: 0 | categories: 0
[BUILD] invalid_counts: {'not_in_train57': 23411}
```

**원인 및 해결:**
- GCI JSON의 ID와 어노테이션의 ID가 불일치
- `cfg.train_ann_cid_shift` 값 확인 (현재 +1)
- `cfg.apply_shift_only_to_subdir` 설정 확인 (현재 False)
- 진단 셀을 추가하여 실제 raw ID와 GCI ID를 비교

### GCI 파일을 찾을 수 없음

**증상:**
```
FileNotFoundError: 사전 구축된 GCI JSON을 찾을 수 없습니다
```

**해결:**
- `global_category_index/gci_57_MODEL_SORTED_BY_CATEGORY_ID.json` 경로에 파일 배치
- `cfg.gci_output_dir` 경로가 올바른지 확인

### CUDA 메모리 부족

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결:**
- `cfg.batch = 4` 또는 `cfg.batch = 2`로 감소
- `cfg.imgsz = 416` 또는 `cfg.imgsz = 512`로 감소
- `cfg.amp = True` 확인 (혼합 정밀도)
- GPU 캐시 비우기: 새 셀에서 `torch.cuda.empty_cache()` 실행

### 학습 중단 후 가중치를 찾을 수 없음

**해결:**
- `yolo_runs/` 하위에서 `best.pt` 또는 `last.pt` 수동 확인
- Cell 27 (가중치 탐색 셀) 실행하여 자동 탐색
- 7-ALT 셀에서 `MANUAL_WEIGHTS`에 직접 경로 입력

### Optuna study가 없습니다 오류

**증상:**
```
RuntimeError: Optuna study가 없습니다
```

**해결:**
- Optuna 셀(Cell 32~33)을 먼저 실행
- 또는 7-ALT 셀을 사용하여 Optuna를 건너뛰기

### 제출 CSV의 category_id가 맞지 않음

**확인 사항:**
- `cfg.submit_catid_offset` 값 (현재 0이어야 함)
- GCI `index_to_id` 매핑이 제출 기준 ID와 일치하는지
- 진단: 생성된 CSV의 `category_id` 컬럼에서 상위 값이 GCI의 ID와 동일한지 확인

---

## 로깅 및 모니터링

### 학습 진행 확인

**실시간 메트릭:** Ultralytics가 매 epoch 후 콘솔에 출력하며, `results.csv`에 기록합니다.

**TensorBoard:**
```bash
tensorboard --logdir yolo_runs/
```

### 주요 모니터링 항목

| 항목 | 위치 | 확인 포인트 |
|------|------|-------------|
| 빌드 통계 | Cell 7 출력 | images, annotations, categories 수 |
| GCI 로드 | Cell 9 출력 | total_categories: 57 |
| YOLO export | Cell 18 출력 | train/val 분할 수 |
| 학습 메트릭 | `results.csv` | loss 감소, mAP 수렴 |
| 가중치 경로 | Cell 36/7-ALT | `FINAL_WEIGHTS` 출력 |
| 제출 CSV | Cell 41 출력 | 행 수, category_id 범위 |

---

**문서 버전:** 1.0  
**최종 업데이트:** 2025년 2월 11일  
**호환성:** Python 3.8+, YOLOv8, CUDA 11.0+, Jupyter Notebook/Lab  
**대상 파일:** `pill_detection_pipeline_v23_fixed.ipynb`
