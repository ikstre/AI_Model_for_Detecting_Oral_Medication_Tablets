# AI 알약 식별 프로젝트

### AI_07기 1팀 초급 프로젝트 미션
**팀장** : 이태호 / **팀원** : 조찬영, 손승만, 양수빈

## 목표
사진 속 최대 4개의 알약의 이름(클래스)과 위치(바운딩 박스)를 검출합니다.

## 협업일지 링크
https://www.notion.so/4231e9d21f0e4eef92ed2adb231ca085?v=75ec033e7b49459e984e101424d40fd1

## 보고서 링크
https://drive.google.com/file/d/1Y2ZYo8t5ajR9vZBcurE8Oa7G4vpb3XSJ/view?usp=sharing

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

1. [DATA 구성 및 EDA, 전처리](#1-data-구성-및-eda-전처리)
2. [모델 구성](#2-모델-구성)
3. [학습 설정](#3-학습-설정)
4. [모델 학습 결과 분석](#4-모델-학습-결과-분석)
5. [종합평가](#5-종합평가)

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

---

## 1. DATA 구성 및 EDA, 전처리

### 1.1 데이터 소스 및 규모

| 항목 | 세부 정보 |
|------|-----------|
| 데이터 출처 | AI Hub 경구약제 이미지 (260만+ 라벨링) |
| 이미지 해상도 | 976 × 1,280 (단일 규격) |
| Train Set | 232장 / 어노테이션 763 → 771개 (누락 8건 복구) |
| 외부 데이터 | +16,012장 추가 (희소 클래스 보강) |
| 최종 데이터셋 | 12,219건 품질 검증 완료 / 56종 약물 카테고리 |
| 평가 지표 | mAP@[0.75:0.95] (IoU=0.75~0.95, step 0.05) |

AI Hub 원천 데이터는 두 가지 구조를 가집니다. 각 이미지가 하나의 약품에 대응되는 **단일 약품 데이터(1:1 매칭)** 와, 하나의 이미지에 여러 약품이 포함되어 복합적 매칭 로직이 필요한 **조합 약품 데이터(N:1 매칭)** 입니다.

### 1.2 파편화 문제 및 해결

원본 데이터는 이미지 1장당 객체별로 개별 JSON 파일이 생성되어 총 763개의 JSON이 분산 저장되어 있었습니다. 통합 없이는 다른 객체를 배경으로 오인할 위험이 있었으며, 초기 매칭률은 0%였습니다.

`json_finder.py`의 `validate_pill_dataset()` 함수가 이미지-JSON 매칭을 전수 조사하고, `validate_json_content_quality()`로 JSON 품질 검사(중복 BBox, 경계 이탈 등)를 수행합니다. glob 기반 파일명-경로 매핑으로 매칭률을 **0% → 100%** 로 개선하였습니다.

### 1.3 카테고리 ID 충돌 해결 및 GCI 시스템

Train set의 `category_id=100`과 추가 데이터의 `category_id=99`가 상이하여 전면 오분류 위험이 존재했습니다. 스크립트와 노트북에서 각각 다른 방향의 오프셋을 적용하여 동일한 정합 결과를 달성합니다.

| 환경 | 오프셋 대상 | 오프셋 값 | 결과 |
|---|---|---|---|
| 스크립트 (`main.py`) | 보조 데이터셋(TL/VL) | +1 | 보조 99 → 100으로 맞춤 |
| 노트북 (`.ipynb`) | train_annotations | −1 | train 100 → 99로 맞춤 |

**(노트북)** `cfg.train_ann_cid_shift = -1`, `cfg.apply_shift_only_to_subdir = True`로 설정하여 `train_annotations` 폴더 내 JSON에만 −1 오프셋을 적용합니다. 또한 `cfg.submit_catid_offset = 1`로 제출 시 +1 보정하여 원본 ID를 복원합니다.

**Global Category Index(GCI)** 는 전체 데이터셋의 카테고리 ID를 연속적인 인덱스(0..K-1)로 통일하는 핵심 시스템입니다.

**(노트북)** `build_gci_light()` 함수가 단일 프로세스 fallback으로 구현되어, `gci_generator.py`의 `GCIBuilder`(병렬처리)가 실패하는 환경(Windows/Jupyter 등)에서도 안정적으로 GCI를 생성합니다. `cfg.gci_force_light_scan = True`이면 오프셋 일관성을 위해 항상 light scan을 강제합니다.

| GCI 통계 항목 | 값 |
|---|---|
| 전체 카테고리 수 | 4,565개 |
| 전체 annotation 수 | 2,658,679개 |
| 전체 데이터셋 수 | 100개 |
| 카테고리당 평균 샘플 | 582.4개 |

### 1.4 데이터 품질 검증 (QA)

총 9건의 핵심 오류를 자동 탐지 및 처리하여 학습 노이즈를 차단하였습니다.

| 검증 항목 | 발견 | 처리 방법 | 영향 |
|---|---|---|---|
| 중복 BBox | 6개 | 자동 제거 | 학습 노이즈 방지 |
| 라벨링 상호 모순 | 2개 | 라벨링 수정 | 치명적 오분류 방지 |
| 경계 이탈 | 1개 | 좌표 보정 (6567→567) | 학습 안정성 확보 |
| BBox 결측 | 0개 | 필터링 | 데이터 무결성 보장 |

**(노트북)** COCO 빌드 시 `seen_bbox` set으로 `(file_name, cat_id, x, y, w, h)` 기준 중복 bbox를 인라인 제거합니다. 또한 `_index` 접미사 파일 필터링(`cfg.exclude_index_images`), 퇴화 bbox(w≤0 or h≤0) 필터링, 좌표 경계(OOB) 검증을 COCO 구성 단계에서 일괄 수행합니다.

**Dataset Generator 처리 통계** (`dataset_generator_v2_patched_final.py`):

| 구분 | 처리 건수 | 비고 |
|---|---|---|
| BBox 중복 | 122개 | 자동 제거 |
| BBox 누락 | 163개 | 필터링 |
| Area mismatch | 378개 (52.1%) | w × h ≠ bbox_area → 제거 |
| OOB (Out of Bound) | 34개 | 좌표 보정 |
| Missing Image | 28개 | 제외 |
| Parsing Error | 3개 | 오류 파일 제외 |

### 1.5 탐색적 데이터 분석 (EDA)

> 상세 분석은 `eda/` 폴더의 노트북 및 `eda/EDA_SUMMARY(chan).md`를 참고하세요.

**색상 분포**: 상위 5개 색상이 전체의 77.2%, 단색 알약 89.6%.

| 색상 | 비율 | 색상 | 비율 | 색상 | 비율 |
|---|---|---|---|---|---|
| 하얀색 | 39.7% | 분홍색 | 14.7% | 노란색 | 9.3% |
| 주황색 | 7.6% | 갈색 | 5.9% | 기타 | 22.8% |

**BBox 통계**: 평균 크기 235×237px, 이미지 대비 점유율 5.6%, AR ≈ 1.02, 이미지당 평균 3.96개.

**식별 특징**: 분할선 보유 36.3%, 마크 각인 13.5% — 희소하지만 강력한 식별 특징.

**제조사 편향 & Shortcut Learning 위험**: 특정 제조사-색상 상관관계(일양약품=주황, SK케미칼=초록 등)가 발견되어 색상 보존형 증강 전략의 근거가 됨.

**메타데이터 표준화**: 40+ 속성 구조화, `color_final` 통합(공백 제거, '하양'→'하얀색'), `color_class2` 결측 711건 'None' 통일. 식별자 무결성(`drug_N ↔ item_seq` 1:1), 법적 정합성(Product Code ↔ EDI Code 오류 0건) 검증 완료.

### 1.6 전처리 및 증강 전략

**보존형 증강 철학**: HSV 파라미터를 모두 0.0으로 설정하여 색상 변조 원천 차단.

**CLAHE Transform** (`clahe_transform.py`):

| 기법 | 파라미터 | 설명 |
|---|---|---|
| CLAHETransform | clipLimit=2.5, p=0.5 | LAB→L채널 CLAHE→색상 보존형 명암 강화 |
| AdaptiveSharpen | amount=0.3, p=0.3 | 언샤프 마스킹 기반 엣지 강조 |
| MorphologyEdgeEnhance | kernel=3, p=0.2 | 형태학적 경계선 추출 후 합성 |
| Rotation/Translate/Scale | 10°/0.1/0.2 | 위치/크기 변이 시뮬레이션 |
| Mosaic/Flip/Erasing | 0.25/0.5/0.2 | 배치/반전/은폐 강건성 |
| GaussNoise | var=5~15, p=0.2 | 현실적 노이즈 |

**과도한 증강 부작용**: HSV(h=0.015, s=0.4, v=0.4) + scale=0.3 → Kaggle **0.945 → 0.888 (−6.03%)**

**클래스 불균형 대응**: 희소 클래스 +16,012장 추가, `class_weight.py` 빈도+신뢰도 기반 가중치 → 클래스당 평균 214개 / 최소 45 ~ 최대 1,188개.

**(노트북)** `get_balance_class_weights()` 함수가 class_weight_json → COCO 빈도 기반 balanced weight 순으로 fallback하며, `cfg.balance_enable = True` + `balance_extra_ratio = 0.30`일 때 `make_weighted_train_list()`로 소수 클래스를 30% 추가 오버샘플링합니다.

---

## 2. 모델 구성

### 2.1 YOLO 파이프라인 전체 흐름

| 단계 | 명칭 | 핵심 모듈 |
|---|---|---|
| 1 | 데이터 스캔/검증 | `json_finder.py` |
| 2 | GCI 생성 | `gci_generator.py` → `apply_gci_to_yolo.py` |
| 3 | COCO 병합/정제 | `pill_preprocess_v3_robust.py` |
| 4 | YOLO 포맷 내보내기 | images/labels + data.yaml 생성 |
| 5 | 학습 | `train_latest.py` / `fine_tuning.py` |
| 6 | 평가/시각화 | F1/Recall curve, confusion matrix |
| 7 | 추론 | `csv_generator.py` (GCI 역매핑) |
| 8 | 앙상블 (선택) | Expert/Generalist confidence 규칙 |

> 각 단계의 상세 실행 방법은 **메인 스크립트**(`README_py.md`)와 **노트북**(`README_IPYNB.md`)에서 별도로 안내합니다.

### 2.2 핵심 전처리 함수

| 함수명 | 역할 |
|---|---|
| `collect_json_files()` | rglob으로 JSON 재귀 수집 |
| `merge_coco_from_many_json()` | 분산 JSON → 단일 COCO 통합 (중복 ID 제거) |
| `sanitize_coco_inplace()` | BBox 유효성, 이미지 경계 clip, area 재계산 |
| `split_coco_by_image()` | 이미지 단위 Train/Val 분리 (seed 재현) |
| `build_cat_maps()` | category_id ↔ YOLO class 양방향 매핑 |
| `resolve_image_path()` | 4단계 우선순위 이미지 경로 해석 |
| `make_weighted_train_list()` | 클래스 불균형 보정 가중 샘플링 |

**(노트북)** `build_image_index()` 함수가 이미지 루트를 재귀 스캔하여 O(1) 파일명→경로 해시맵(`IMAGE_INDEX`)을 구축하며, `resolve_train_image_path()`가 이를 활용합니다. `normalize_category_id()` 함수가 폴더별 차등 오프셋을 적용하고, `_path_contains_subdir()`로 경로 내 특정 하위 폴더 포함 여부를 판별합니다.

### 2.3 Expert-Generalist 앙상블

56종 학습 모델의 한계로 232개 미지 데이터가 발견되어 앙상블 전략 도입.

| 역할 | 모델 | 특징 |
|---|---|---|
| Expert | YOLOv8n (56종) | 목표 카테고리 특화, 경량 |
| Generalist | YOLOv8s (118종) | 넓은 범위, 미지 대응 |

**앙상블 로직**: 고신뢰(≥0.85) → Expert 채택 / 저신뢰(<0.15)+높은 IoU → Generalist 교체 / 중간 → Expert 유지, 미탐지 시 Generalist 백업.

### 2.4 코드 버그 수정

| ID | 문제 | 해결 | 환경 |
|---|---|---|---|
| FIX-M1 | `train_yolo()` shear/perspective 하드코딩 | cfg에서 읽도록 수정 | 공통 |
| FIX-M2 | `CFG.val_ratio` 클래스 변수 참조 불가 | 명시적 0.1 고정 | 공통 |
| FIX-P1~P3 | 중복 import, 동일 분기, image_index 미지원 | 유지보수성 개선 | 공통 |
| ID 오프셋 | train_annotations vs 외부 +1 차이 | `apply_gci_to_yolo.py` 정규화 | 스크립트 |
| (노트북) shear/perspective | hasattr 체크가 항상 True → 무의미 | `cfg.shear=0.0`, `cfg.perspective=0.0` 직접 할당 | 노트북 |
| (노트북) `_extract_raw_category_id` | 정의만 되고 한 번도 호출되지 않음 | 함수 제거, 인라인 로직만 유지 | 노트북 |
| (노트북) Optuna val metric | `_on_val_end` 콜백으로 mAP 추출 | `_extract_map5095_from_validator()` 유틸 구현 | 노트북 |

---

## 3. 학습 설정

### 3.1 기본 하이퍼파라미터

**스크립트(`main.py`) 기본값:**

| 파라미터 | 값 | 파라미터 | 값 |
|---|---|---|---|
| base_model | yolov8n.pt | epochs | 300 |
| batch | 8 | lr0 | 5e-4 |
| optimizer | AdamW | imgsz | 640 |
| patience | 15 | max_det | 4 |
| seed | 42 | weight_decay | 1e-4 |

**(노트북) 기본값 차이점:**

| 파라미터 | 노트북 기본값 | 스크립트 기본값 | 비고 |
|---|---|---|---|
| epochs | 100 | 300 | 빠른 실험 사이클 |
| batch | 16 | 8 | GPU 메모리에 따라 |
| patience | 10 | 15 | |
| submit_catid_offset | 1 | 0 | 오프셋 방향 차이 보정 |
| train_ann_cid_shift | −1 | +1 | train_annotations에 적용 |

**데이터 증강 (색상 보존, 양쪽 동일):**
```python
hsv_h=0.0, hsv_s=0.0, hsv_v=0.0  # 비활성화
degrees=10.0, translate=0.1, scale=0.2
fliplr=0.5, flipud=0.5, mosaic=0.25, erasing=0.2
```

### 3.2 클래스 가중치

`class_weight.py`의 `calculate_class_weights()`: 빈도 기반(제곱근 스케일링) + Confidence 기반(선형 스케일링) 결합. 희소 클래스 재현율 **0.523 → 0.612 (+0.089)**.

**(노트북)** `get_balance_class_weights()` 함수가 가중치 결정 우선순위를 통합 관리: (1) class_weight_json → (2) COCO 빈도 기반 balanced weight.

### 3.3 Fine-tuning 전략

`fine_tuning.py`에서 Stage 1 베스트 → 2단계 미세 조정: lr0=1e-4(×1/3), mosaic=0.3, mixup=0.01, Cosine LR, warmup 1 epoch.

### 3.4 Optuna (노트북 전용)

> 상세 설정은 `README_IPYNB.md`의 7단계를 참고하세요.

TPESampler + HyperbandPruner 기반 Bayesian Optimization. 탐색: model, batch, lr0, weight_decay, degrees, translate, scale, mosaic.

**(노트북)** Optuna 서브셋 모드 지원: `cfg.optuna_use_subset = True`일 때 전체 데이터의 30%(`optuna_sample_ratio`)로 빠르게 탐색 후, 최종 재학습은 전체 데이터로 수행. `_extract_map5095_from_validator()` 유틸이 Ultralytics 버전별 mAP 추출 호환성을 보장합니다.

### 3.5 7-ALT: 3가지 실행 모드 (노트북 전용)

**(노트북)** Optuna를 건너뛰거나 기존 학습을 이어갈 때 사용하는 3가지 모드:

| MODE | 설명 | 다음 단계 |
|------|------|-----------|
| `"train"` | 수동 파라미터로 처음부터 학습 | → 7-1 → 8 |
| `"resume"` | 기존 가중치(last.pt)에서 이어서 추가 학습 | → 7-1 → 8 |
| `"infer_only"` | 이미 학습된 best.pt로 바로 추론 | → 8 |

**(노트북)** Optuna 탐색 결과 최적 파라미터(예시):
```python
MANUAL_BEST_PARAMS = {
    'model': 'yolov9c.pt',
    'batch': 16,
    'lr0': 0.0006842,
    'weight_decay': 6.95e-06,
    'degrees': 0.095,
    'translate': 0.026,
    'scale': 0.080,
    'mosaic': 0.328
}
```

### 3.6 추론 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| conf | 0.001 | 최소 confidence |
| iou | 0.7 | NMS IoU |
| max_det_per_image | 4 | 이미지당 박스 상한 |
| imgsz | 640 | 추론 해상도 |
| half | True | FP16 (기본 권장) |

**(노트북)** `submit_conf`를 별도로 설정(기본 0.3)하여, 추론 conf(0.001, 포괄적)과 제출 score 임계값을 분리합니다. `run_infer_to_csv()`가 stream+chunking 방식으로 대규모 추론을 처리하며, 최종 CSV에서 `annotation_id`를 1..N으로 재부여합니다.

### 3.7 학습 중단 및 재개

**(노트북)** `GracefulStopper` 클래스가 `signal.SIGINT` 핸들러 + `on_train_epoch_end` 콜백을 조합하여, Ultralytics가 SIGINT 핸들러를 덮어씌우는 문제를 우회합니다.

- **1회 중단**: 현재 epoch 완료 후 안전 종료 (`trainer.stop = True`)
- **2회 연속**: 즉시 강제 종료
- **재개**: `cfg.save_period`(기본 10 epoch)마다 자동 저장, `MODE = "resume"`으로 이어서 학습

---

## 4. 모델 학습 결과 분석

### 4.1 단계별 성능 개선

| 모델 구성 | 해상도 | mAP@0.5 | 추론시간 | 비고 |
|---|---|---|---|---|
| YOLOv8n (베이스라인) | 640 | 0.687 | 12ms | Baseline |
| + CLAHE 전처리 | 640 | 0.694 (+0.007) | 12ms | 대비 강화 |
| + Class Weight | 640 | Recall +0.089 | — | 불균형 보정 |
| YOLOv8s (스케일업) | 640 | 0.743 (+0.049) | 18ms | 표준 운영 |
| YOLOv8l (고해상도) | 1024 | 0.768 (+0.025) | 67ms | 고정밀도 |

### 4.2 Kaggle Leaderboard

| 단계 | 점수 | 변화량 | 비고 |
|---|---|---|---|
| 기본 모델 (스크립트) | 0.946 | — | Baseline |
| CLAHE 전처리 | 0.94943 | +0.00343 | 보존형 증강 |
| 데이터 추가 (16,012장) | **0.988** | **+4.6%p** | 스크립트 최고 |
| **(노트북) Optuna+yolov9c** | **0.984** | — | 노트북 최종 |
| 과도 증강 (부작용) | 0.888 | −6.03% | HSV 색상 변조 |

**(노트북)** 노트북 파이프라인에서 Optuna 탐색(yolov9c.pt, lr0=0.0007, mosaic=0.33 등) 후 최종 재학습 결과 **Kaggle 0.984**를 달성하여, 스크립트 기반 파이프라인(0.988)과 유사한 수준의 성능을 확인하였습니다. 스크립트와 노트북의 점수 차이(0.004)는 주로 데이터 추가량/모델 스케일 차이에 기인하며, 파이프라인 자체의 정합성은 양쪽 모두 검증되었습니다.

### 4.3 Error Analysis / Per-Class Evaluation

| 구분 | kaggle score | evaluator score | evaluator score(56) |
|---|---|---|---|
| v8n | 0.945 | 0.461 | 0.872 |
| v8s(56) | 0.984 | 0.476 | 0.901 |
| v8s(118) | 0.988 | 0.902 | 0.923 |

Kaggle 테스트(842장)만으로는 일반화 평가에 표본 부족 → 5,000장+ 자체 Evaluator 구축, 카테고리별 mAP 50 / mAP 75:95 산출.

### 4.4 학습 결과 상세

- **F1-Confidence**: 최대 F1 ≈ 0.98 @ confidence ≈ 0.84
- **Recall-Confidence**: 낮은 conf에서 Recall ≈ 0.99, 0.9 이상에서 급감
- **Confusion Matrix**: 유사 색상/각인의 원형·타원형에서 국소 혼동
- **학습 수렴**: train/val loss 안정적 감소, 과적합 징후 제한적

---

## 5. 종합평가

### 5.1 주요 성과

**데이터 엔지니어링:**
- 260만 건+ 파편화 데이터 통합 (GCI 시스템)
- 725건 노이즈 자동 탐지/제거
- 클래스 불균형 개선: min 3개 → 45개

**모델 성능:**
- mAP@0.5 베이스라인 대비 **+11.8%** (0.687 → 0.768)
- 스크립트 Kaggle 최종 **0.988**, 노트북 Kaggle 최종 **0.984**
- 두 파이프라인 모두 동일 품질의 데이터 정합성 확보


### 5.2 기술적 기여도

- **GCI 기반 데이터 통합**: 대규모 파편화 데이터셋을 효율적으로 관리하는 범용적 접근법
- **자동화된 품질 관리**: 데이터 무결성 검증과 노이즈 제거의 체계적 자동화
- **지능적 클래스 병합**: 서로 다른 데이터 소스 간 카테고리 충돌 해결 알고리즘
- **보존형 증강 전략 설계**: 약품 식별의 핵심인 색상 정보를 보존하는 CLAHE 기반 증강 파이프라인으로 과도 증강 부작용 방지
- **이중 파이프라인 검증 체계**: 스크립트(0.988)와 노트북(0.984) 파이프라인의 독립적 실행으로 데이터 정합성과 학습 로직의 신뢰성 이중 검증
- **오답 분석 기반 피드백 루프**: 오탐/미탐 샘플 자동 수집 및 시각적 검증 워크플로우를 통한 데이터 정제 우선순위 결정 체계 구축

### 5.3 종합 성과

| 영역 | 성과 |
|---|---|
| Kaggle 성능 | 스크립트 0.988 / 노트북 0.984 / mAP@0.5 +11.8% |
| 데이터 품질 | 오류 725건 탐지/제거, 핵심 9건 정정, 누락 8건 복구 |
| 관리 체계 | 데이터셋 771개, GCI 정합성 100%, 확장 가능 아키텍처 |
| 파이프라인 | 스크립트(자동화) + 노트북(실험) 이중 검증 체계 |

### 5.4 완료 체크리스트

| | | |
|---|---|---|
| ✅ 보존형 전처리/증강 | ✅ JSON 통합 (771개) | ✅ GCI 시스템 구축 |
| ✅ QA (725건 처리) | ✅ EDA 분석 완료 | ✅ YOLO 파이프라인 |
| ✅ Kaggle 자동화 | ✅ Dataset Generator | ✅ Data Acquisition |
| ✅ Class Weight 적용 | ✅ Fine-tuning | ✅ 앙상블 추론 |
| ✅ (노트북) GracefulStopper | ✅ (노트북) Optuna 탐색 | ✅ (노트북) 3-mode 7-ALT |

### 5.5 향후 발전 방향

**기술적 확장**: 모델 경량화(실시간), 다중 뷰(앞/뒷면 동시 인식), 유사 약품 구분 강화, OCR 기반 각인 문자 인식과 객체 검출의 멀티태스크 학습 통합, Transformer 기반 검출 모델(RT-DETR 등)과의 성능 비교 및 앙상블 확장

**시스템 고도화**: 클라우드 API, 모바일 최적화, 약품 정보 DB 연동, 신규 약품 등록 시 자동 재학습(Continuous Learning) 파이프라인 구축, 복약 관리 앱과의 통합을 통한 환자 안전성 향상 서비스 개발

---

## 프로젝트 파일 구성

### 실행 진입점

| 파일 | 역할 | 상세 문서 |
|---|---|---|
| `main.py` | 스크립트 메인 파이프라인 CLI | `README_py.md` |
| `pill_detection_pipeline_v23_fixed.ipynb` | 노트북 심화 파이프라인 (45셀) | `README_IPYNB.md` |
| `pill_preprocess_v3_robust.py` | 데이터 통합/정합화 (공통) | — |
| `pill_model_setup_v4_max4_fixed.py` | YOLO 학습/추론 헬퍼 (공통) | — |

### 핵심 모듈 (`function/`)

| 파일 | 역할 |
|---|---|
| `json_finder.py` | 파편 JSON 탐색, 이미지-JSON 매칭 검증 |
| `gci_generator.py` | GCI 생성/통계 산출 |
| `apply_gci_to_yolo.py` | 로컬→글로벌 라벨 치환, data.yaml 생성 |
| `dataset_generator_v2_patched_final.py` | 데이터셋 생성/검증/내보내기 (병렬) |
| `clahe_transform.py` | CLAHE/AdaptiveSharpen/MorphologyEdge |
| `class_weight.py` | 빈도+신뢰도 클래스 가중치 |
| `csv_generator.py` | stream+chunking 추론, 제출 CSV (GCI 역매핑) |
| `train_latest.py` | YOLO 학습/재개 |
| `fine_tuning.py` | 2단계 Fine-tuning |

### 출력 디렉토리 (`main.py` 기준)

```
<base-dir>/outputs/
├── global_category_index/    # 전역 매핑 + 통계
├── datasets/pill_detection/  # dataset.csv, category_map.csv, yolo_format/
├── yolo_runs/<run_name>/     # weights(best.pt/last.pt), results.csv
└── submissions/              # submission_YYYYMMDD_HHMMSS.csv
```

---

## 시스템 요구사항

### 하드웨어

| 구분 | 최소 | 권장 |
|---|---|---|
| CPU | i5 / Ryzen 5 (4코어) | i7+ / Ryzen 7+ (8코어) |
| RAM | 16GB | 32GB |
| 저장공간 | 100GB | 200GB SSD |
| GPU | 선택사항 | RTX 3080/4080+ (VRAM 8GB+) |

### 소프트웨어

- Python 3.8+, CUDA 11.0+, cuDNN 8.0+
- Jupyter Notebook/Lab (노트북 모드)

### 필수 패키지

```
ultralytics>=8.0.0    torch>=1.12.0     pandas>=1.3.0
numpy>=1.21.0         pillow>=8.0.0     tqdm>=4.62.0
matplotlib>=3.4.0     seaborn>=0.11.0   pycocotools>=2.0.0
optuna>=3.0.0         pyyaml>=5.4.0
```

---

## 문제 해결 가이드 (Quick Reference)

| 증상 | 원인 | 해결 |
|---|---|---|
| `CUDA out of memory` | GPU 메모리 부족 | batch=4/2, imgsz=416, `torch.cuda.empty_cache()` |
| 빌드 시 annotations 0건 | GCI ID ↔ 어노테이션 ID 불일치 | `train_ann_cid_shift` 확인, raw ID vs GCI ID 비교 |
| GCI 파일 없음 | 경로 오류 | `global_category_index/` 하위 파일 배치 확인 |
| (노트북) GCI 병렬 생성 실패 | Windows/Jupyter 멀티프로세스 제한 | `cfg.gci_force_light_scan = True` (자동 fallback) |
| 학습 중단 후 가중치 분실 | 체크포인트 미저장 | `yolo_runs/` 하위 `best.pt`/`last.pt` 수동 확인 |
| 제출 CSV category_id 불일치 | 오프셋 설정 오류 | 스크립트: offset=0 / 노트북: offset=1 확인 |
| (노트북) `Optuna study가 없습니다` | 7-0~7 미실행 | 7-ALT 셀로 Optuna 건너뛰기 |

> 상세한 문제 해결은 `README_py.md` 또는 `README_IPYNB.md`의 해당 섹션을 참고하세요.

---

**문서 버전:** 3.0  
**최종 업데이트:** 2026. 02. 12  
**호환성:** Python 3.8+, YOLOv8, CUDA 11.0+
