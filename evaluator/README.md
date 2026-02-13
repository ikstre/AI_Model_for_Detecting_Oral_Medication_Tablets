# 알약 검출 모델 평가 파이프라인

이 저장소는 YOLO 기반 알약 검출 모델의 성능을 평가하기 위한 도구들을 제공합니다.

## 📋 목차
1. [개요](#개요)
2. [사전 준비](#사전-준비)
3. [단계별 실행 가이드](#단계별-실행-가이드)
4. [파일 설명](#파일-설명)
5. [출력 결과 이해하기](#출력-결과-이해하기)
6. [문제 해결](#문제-해결)

---

## 개요

### 평가 파이프라인 구조
```
1. GCI 생성 → 2. CSV 예측 생성 → 3. GT CSV 생성 → 4. 평가 (mAP 계산)
```

### 필요한 파일
- **모델 가중치** (`.pt` 파일)
- **평가용 이미지** (`train_eval/` 폴더)
- **라벨 JSON 파일** (`label_eval/` 폴더)

---

## 사전 준비

### 1. 디렉토리 구조 확인

```
project/
├── train_eval/                    # 평가용 이미지
│   ├── TS_2_조합/
│   │   ├── K-002483-022362-023223-025438/
│   │   │   ├── K-002483-022362-023223-025438_0_2_0_2_70_000_200.png
│   │   │   ├── K-002483-022362-023223-025438_0_2_0_2_75_000_200.png
│   │   │   └── K-002483-022362-023223-025438_index.png  ⚠️ 이 파일은 자동 스킵됨
│   │   └── ...
│   ├── TS_6_조합/
│   └── ...
│
├── label_eval/                    # 라벨 JSON 파일
│   ├── TL_2_조합/
│   │   ├── K-003544-010221-016551-029451_json/
│   │   │   ├── K-029451/
│   │   │   │   ├── K-003544-010221-016551-029451_0_2_0_2_70_000_200.json
│   │   │   │   └── ...
│   │   │   └── K-003544/
│   │   └── ...
│   └── ...
│
└── weights/
    └── best.pt                    # 학습된 모델 가중치
```

### 2. 필수 라이브러리 설치

```bash
pip install ultralytics numpy tqdm
```

---

## 단계별 실행 가이드

### ⚠️ 중요: GCI 생성 및 전처리 (필수!)

평가를 시작하기 전에 **반드시** 다음 작업을 수행해야 합니다:

#### Step 0-1: 학습 데이터셋의 GCI 생성
```python
# gci_generator.py를 실행하여 학습 데이터셋의 GCI 생성
# (이 스크립트는 별도로 제공되어야 함)

# 예시 경로:
TRAIN_DATA_DIR = r"E:\download\train_data"
OUTPUT_GCI = r"E:\download\train_gci.json"
```

#### Step 0-2: 평가 데이터셋의 GCI 생성
```python
# 평가 데이터셋에 대해서도 GCI 생성
EVAL_LABEL_DIR = r"E:\download\label_eval"
OUTPUT_GCI = r"E:\download\eval_gci.json"
```

#### Step 0-3: **🚨 increment_id 실행 (필수!)**
```python
# 두 개의 GCI JSON에 대해 increment_id 함수를 "꼭" 실행해야 합니다!
# 안 그러면 category_id가 0부터 시작하여 평가 점수가 0점이 나옵니다.

def increment_id(gci_json_path: str):
    """
    GCI JSON의 index_to_id 값을 모두 +1 증가시킵니다.
    
    이유: JSON의 dl_idx는 0부터 시작하지만, 실제 category_id는 1부터 시작해야 합니다.
    """
    import json
    from pathlib import Path
    
    with open(gci_json_path, "r", encoding="utf-8") as f:
        gci = json.load(f)
    
    # index_to_id의 모든 값에 +1
    if "index_to_id" in gci:
        gci["index_to_id"] = {k: int(v) + 1 for k, v in gci["index_to_id"].items()}
    
    # 원본 백업
    backup_path = Path(gci_json_path).with_suffix('.json.backup')
    import shutil
    shutil.copy(gci_json_path, backup_path)
    print(f"✓ 백업 생성: {backup_path}")
    
    # 업데이트된 파일 저장
    with open(gci_json_path, "w", encoding="utf-8") as f:
        json.dump(gci, f, indent=2, ensure_ascii=False)
    
    print(f"✓ GCI 업데이트 완료: {gci_json_path}")

# 실행 예시
increment_id(r"E:\download\train_gci.json")
increment_id(r"E:\download\eval_gci.json")
```

---

### Step 1: 모델 예측 CSV 생성

`csv_generator_eval.py`를 사용하여 평가 이미지에 대한 모델 예측 결과를 CSV로 생성합니다.

```python
# csv_generator_eval.py의 main 부분 수정

if __name__ == "__main__":
    WEIGHTS = r"E:\yolo_runs\pill_baseline_augdata_refined\weights\best.pt"
    TRAIN_EVAL_DIR = r"E:\download\train_eval"  # 평가용 이미지 폴더
    OUT_CSV = r"E:\download\submission_eval\submission_eval.csv"
    GCI_JSON = r"E:\download\train_gci.json"  # ⚠️ 학습 데이터셋의 GCI 사용!
    
    run_infer_to_csv(
        weights_path=WEIGHTS,
        test_img_dir=TRAIN_EVAL_DIR,
        out_csv_path=OUT_CSV,
        gci_json_path=GCI_JSON,  # 모델이 학습한 카테고리 정보
        imgsz=640,
        pred_batch=4,
        chunk_size=200,
        half=True,
        device=0,
    )
```

**실행:**
```bash
python csv_generator_eval.py
```

**출력 예시:**
```
총 5999개의 이미지 발견
추론 진행: 100%|██████████| 5999/5999 [10:23<00:00, 9.61it/s]
✅ 완료
- 저장: E:\download\submission_eval\submission_eval.csv
- 이미지: 5999
- 탐지없음: 12
- 이 탐지: 23456
```

**⚠️ 주의사항:**
- `_index.png`로 끝나는 파일은 자동으로 스킵됩니다 (라벨 이미지)
- `GCI_JSON`은 반드시 **모델이 학습한 데이터셋의 GCI**를 사용해야 합니다!
- `image_id`는 파일명 전체 (확장자 제외)를 사용합니다

---

### Step 2: Ground Truth CSV 생성

`gt_generator.py`를 사용하여 라벨 JSON 파일로부터 Ground Truth CSV를 생성합니다.

```python
# gt_generator.py의 main 부분 수정

if __name__ == "__main__":
    LABEL_EVAL_DIR = r"E:\download\label_eval"
    OUT_GT_CSV = r"E:\download\submission_eval\ground_truth.csv"
    
    generate_gt_csv(
        label_eval_dir=LABEL_EVAL_DIR,
        out_csv_path=OUT_GT_CSV,
        image_width=976,   # 이미지 너비 (기본값)
        image_height=1280  # 이미지 높이 (기본값)
    )
```

**실행:**
```bash
python gt_generator.py
```

**출력 예시:**
```
총 5993개의 JSON 파일 발견
GT CSV 생성: 100%|██████████| 5993/5993 [00:15<00:00, 380.45it/s]
✅ Ground Truth CSV 생성 완료
- 저장: E:\download\submission_eval\ground_truth.csv
- JSON 파일: 5993
- 총 annotations: 23450

검증 통계:
  ✓ 유효: 23450
  ✗ bbox 누락: 0
  ✗ bbox 형식 오류: 0
  ✗ bbox 크기 오류 (w≤0 or h≤0): 15
  ✗ 면적 불일치: 23
  ✗ 면적 값 오류: 0
  ✗ 이미지 경계 초과: 8
  총 제외된 annotations: 46
```

**bbox 검증 항목:**
1. bbox 누락 여부
2. bbox 형식 (4개 값이 모두 숫자인지)
3. bbox 크기 (w > 0, h > 0)
4. 면적 불일치 (JSON의 area 값과 w*h 비교, 0.5% 오차 허용)
5. 이미지 경계 초과 (bbox가 이미지 범위 내에 있는지)

---

### Step 3: 모델 평가 (mAP 계산)

`evaluator_optimized.py`를 사용하여 예측 CSV와 GT CSV를 비교하고 mAP를 계산합니다.

```python
# evaluator_optimized.py의 main 부분 수정

if __name__ == "__main__":
    PRED_CSV = r"E:\download\submission_eval\submission_eval.csv"
    GT_CSV = r"E:\download\submission_eval\ground_truth.csv"
    OUTPUT_TXT = r"E:\download\submission_eval\mAP_75_95_detailed.txt"
    GCI_JSON = r"E:\download\eval_gci.json"  # 선택사항 (중점 카테고리 분석용)
    
    results = evaluate(
        pred_csv_path=PRED_CSV,
        gt_csv_path=GT_CSV,
        output_txt_path=OUTPUT_TXT,
        gci_json_path=GCI_JSON  # None으로 설정하면 중점 카테고리 분석 스킵
    )
```

**실행:**
```bash
python evaluator_optimized.py
```

**출력 예시:**
```
============================================================
평가 시작
============================================================
✓ GCI 로드 완료: 56개 중점 카테고리 식별
예측 CSV 로딩 중...
예측: 5999개 이미지
GT CSV 로딩 중...
GT: 5993개 이미지
⚠️  예측에만 있는 이미지: 6개 (평가에서 제외됨)

총 카테고리 수: 1024
중점 카테고리 중 데이터에 존재: 56개

mAP 계산 중 (IoU threshold: 0.5~0.95, 총 10개)...
mAP@75:95: 100%|██████████| 1024/1024 [01:23<00:00, 12.31it/s]

============================================================
평가 결과
============================================================
전체 mAP@50:    0.9080
전체 mAP@75:95: 0.8987
------------------------------------------------------------
중점 카테고리 mAP@50:    0.9250
중점 카테고리 mAP@75:95: 0.9150
(총 56개 중점 카테고리)
============================================================

카테고리별 AP@50 (상위 10개):
  Category 38954: 1.0000 🎯
  Category 13395: 1.0000 🎯
  Category 33878: 1.0000
  ...

중점 카테고리 AP@50 (하위 10개):
  Category 12081: 0.7532 🎯
  Category 22362: 0.7821 🎯
  ...

✅ mAP@75:95 상세 결과 저장: E:\download\submission_eval\mAP_75_95_detailed.txt
```

---

## 파일 설명

### 입력 파일

| 파일명 | 설명 | 필수 여부 |
|--------|------|-----------|
| `csv_generator_eval.py` | 모델 예측 CSV 생성 | 필수 |
| `gt_generator.py` | Ground Truth CSV 생성 | 필수 |
| `evaluator_optimized.py` | mAP 평가 및 상세 분석 | 필수 |
| `train_gci.json` | 학습 데이터셋의 카테고리 매핑 | 필수 |
| `eval_gci.json` | 평가 데이터셋의 카테고리 매핑 (중점 분석용) | 선택 |

### 출력 파일

| 파일명 | 설명 |
|--------|------|
| `submission_eval.csv` | 모델 예측 결과 (annotation_id, image_id, category_id, bbox, score) |
| `ground_truth.csv` | Ground Truth (annotation_id, image_id, category_id, bbox) |
| `mAP_75_95_detailed.txt` | 카테고리별 상세 mAP 분석 결과 |

### CSV 파일 형식

**예측 CSV (submission_eval.csv):**
```csv
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score
1,K-003544-010221-016551-029451_0_2_0_2_70_000_200,29451,105,252,397,246,0.952341
2,K-003544-010221-016551-029451_0_2_0_2_70_000_200,3544,210,150,180,200,0.891234
...
```

**Ground Truth CSV (ground_truth.csv):**
```csv
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h
1,K-003544-010221-016551-029451_0_2_0_2_70_000_200,29451,105,252,397,246
2,K-003544-010221-016551-029451_0_2_0_2_70_000_200,3544,210,150,180,200
...
```

---

## 출력 결과 이해하기

### mAP 지표 설명

- **mAP@50**: IoU threshold 0.5에서의 평균 정밀도
  - 일반적으로 높게 나옴 (0.8~0.95)
  - 객체를 "대략" 맞췄는지 평가

- **mAP@75:95**: IoU threshold 0.5~0.95 (0.05 간격)의 평균 mAP
  - 더 엄격한 평가 지표
  - bbox 위치가 얼마나 정확한지 평가
  - COCO 데이터셋 평가 방식과 동일

### 상세 결과 파일 (mAP_75_95_detailed.txt)

```
================================================================================
mAP@75:95 카테고리별 상세 계산 결과
================================================================================

계산 방식:
1. IoU threshold를 0.5부터 0.95까지 0.05 간격으로 설정 (총 10개)
2. 각 IoU threshold에서 모든 카테고리의 AP를 계산
3. 각 카테고리의 mAP@75:95 = 해당 카테고리의 10개 AP 평균
4. 전체 mAP@75:95 = 모든 카테고리의 mAP@75:95 평균

전체 mAP@50:    0.908012
전체 mAP@75:95: 0.898734
총 카테고리 수: 1024

--------------------------------------------------------------------------------
중점 카테고리 (GCI 기준) 성능
--------------------------------------------------------------------------------
중점 카테고리 mAP@50:    0.925012
중점 카테고리 mAP@75:95: 0.915023
중점 카테고리 수: 56

================================================================================
카테고리별 mAP@75:95 (낮은 점수 순)
================================================================================

   1. Category  12081: mAP@75:95 = 0.653421 🎯  ← 개선 필요!
   2. Category  22362: mAP@75:95 = 0.682134 🎯
   ...

================================================================================
중점 카테고리 mAP@75:95 (낮은 점수 순)
================================================================================

   1. Category  12081: mAP@75:95 = 0.653421 🎯  ← 중점 카테고리 중 가장 낮음
   2. Category  22362: mAP@75:95 = 0.682134 🎯
   ...

================================================================================
통계 정보
================================================================================

전체 카테고리:
  평균 (mAP@75:95):     0.898734
  중간값:                0.912345
  표준편차:              0.123456
  최소값:                0.653421
  최대값:                1.000000

중점 카테고리:
  평균 (mAP@75:95):     0.915023
  중간값:                0.925123
  표준편차:              0.098765
  최소값:                0.653421
  최대값:                1.000000

점수 구간별 카테고리 분포 (전체):
  0.0 ~ 0.5:    5 카테고리
  0.5 ~ 0.6:   12 카테고리
  0.6 ~ 0.7:   45 카테고리
  0.7 ~ 0.8:  123 카테고리
  0.8 ~ 0.9:  456 카테고리
  0.9 ~ 1.0:  383 카테고리

점수 구간별 카테고리 분포 (중점 카테고리):
  0.0 ~ 0.5:    0 카테고리
  0.5 ~ 0.6:    1 카테고리
  0.6 ~ 0.7:    3 카테고리
  0.7 ~ 0.8:    8 카테고리
  0.8 ~ 0.9:   20 카테고리
  0.9 ~ 1.0:   24 카테고리
```

### 🎯 표시의 의미
- `GCI_JSON`을 제공한 경우, 해당 파일의 `index_to_id`에 포함된 카테고리는 🎯로 표시됩니다
- 이는 모델이 학습한 주요 카테고리들을 의미합니다

---

## 문제 해결

### Q1: 평가 점수가 0점 또는 매우 낮게 나옵니다
**A:** `increment_id()` 함수를 실행하지 않았을 가능성이 높습니다.
```python
increment_id(r"E:\download\train_gci.json")
increment_id(r"E:\download\eval_gci.json")
```
위 코드를 실행한 후 다시 Step 1부터 진행하세요.

### Q2: 예측 이미지와 GT 이미지 개수가 다릅니다
**A:** 정상입니다. GT에 없는 예측은 무시되고, 예측에 없는 GT는 False Negative로 처리됩니다.

### Q3: bbox 검증에서 많은 annotations이 제외됩니다
**A:** 다음을 확인하세요:
- bbox가 이미지 경계를 벗어나는지
- w 또는 h가 0 이하인지
- JSON의 area 값과 w*h가 일치하는지 (0.5% 오차 허용)

### Q4: 평가가 너무 오래 걸립니다
**A:** `evaluator_optimized.py`를 사용하고 있는지 확인하세요. 최적화된 버전은 1~2분 내에 완료됩니다.

### Q5: CUDA out of memory 에러가 발생합니다
**A:** `csv_generator_eval.py`에서 다음 파라미터를 조정하세요:
```python
imgsz=512,        # 640 → 512로 축소
pred_batch=2,     # 4 → 2로 축소
chunk_size=100,   # 200 → 100으로 축소
half=True,        # 유지 (FP16 사용)
```

---

## 전체 실행 순서 요약

```bash
# 0. GCI 생성 및 전처리 (필수!)
python gci_generator.py  # 학습 데이터셋용
python gci_generator.py  # 평가 데이터셋용
python increment_id.py   # 두 GCI 모두에 대해 실행 (필수!)

# 1. 모델 예측 CSV 생성
python csv_generator_eval.py

# 2. Ground Truth CSV 생성
python gt_generator.py

# 3. 평가 (mAP 계산)
python evaluator_optimized.py
```

---

## 라이선스 및 문의

이 도구는 알약 검출 프로젝트를 위해 개발되었습니다.

문의사항이 있으시면 이슈를 등록해주세요.
