# 약품 식별 AI 파이프라인

YOLOv8 아키텍처를 활용한 약품 알약 검출 및 분류를 위한 종합 머신러닝 파이프라인입니다. 본 시스템은 어노테이션 데이터를 처리하고, 객체 검출 모델을 학습하며, 대회 제출용 결과물을 생성합니다.

## 목차

- [시스템 개요](#시스템-개요)
- [시스템 요구사항](#시스템-요구사항)
- [설치 가이드](#설치-가이드)
- [디렉토리 구조](#디렉토리-구조)
- [사용 방법](#사용-방법)
- [파이프라인 구조](#파이프라인-구조)
- [설정 관리](#설정-관리)
- [성능 최적화](#성능-최적화)
- [문제 해결 가이드](#문제-해결-가이드)

---

## 시스템 개요

본 파이프라인은 약품 식별을 위한 5단계 프로세스를 구현합니다:

**1단계: 전역 카테고리 인덱스 생성**  
여러 어노테이션 데이터셋의 카테고리 정보를 통합 인덱싱 시스템으로 집계합니다.

**2단계: 카테고리 ID 오프셋 조정**  
학습 데이터셋과 보조 데이터셋 간의 카테고리 ID 충돌을 체계적인 오프셋 적용과 병합을 통해 해결합니다.

**3단계: YOLO 데이터셋 구축**  
처리된 어노테이션을 포괄적인 검증 및 정규화를 거쳐 YOLO 호환 형식으로 변환합니다.

**4단계: 모델 학습**  
최적화된 하이퍼파라미터와 증강 전략을 사용하여 YOLOv8 모델을 학습합니다.

**5단계: 추론 및 제출**  
테스트 데이터셋에 대한 예측을 생성하고 대회 형식의 CSV 제출 파일을 생성합니다.

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
```

---

## 설치 가이드

### 환경 설정

**1단계: 저장소 복제**

```bash
git clone <repository-url>
cd pharmaceutical-identification-pipeline
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

### 데이터 준비

다음 구조에 따라 데이터를 구성하세요:

```
project_root/
├── label_aug/                   # 어노테이션 파일
│   ├── train_annotations/
│   ├── TL_1_조합/
│   └── ...
├── train_aug/                   # 학습 이미지
└── sprint_ai_project1_data/     # 테스트 데이터
    └── test_images/
```

---

## 디렉토리 구조

파이프라인은 다음과 같은 출력 구조를 생성합니다:

```
outputs/
├── global_category_index/       # 카테고리 매핑 파일
│   ├── global_category_index.json
│   ├── global_category_index_updated.json
│   └── category_statistics.txt
├── datasets/                    # YOLO 형식 데이터셋
│   └── pill_detection/
│       ├── dataset.csv
│       ├── category_map.csv
│       └── yolo_format/
├── yolo_runs/                   # 학습 결과
│   └── pill_baseline/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       └── results.csv
└── submissions/                 # 생성된 CSV 파일
    └── submission_YYYYMMDD_HHMMSS.csv
```

---

## 사용 방법

### 명령줄 인터페이스

파이프라인은 실행 제어를 위한 명령줄 인터페이스를 사용합니다:

**기본 구문:**

```bash
python main.py --base-dir <경로> [--steps <단계1> <단계2> ...]
```

### 실행 예시

**전체 파이프라인 실행:**

```bash
python main.py --base-dir "E:\download"
```

**데이터 전처리만 실행:**

```bash
python main.py --base-dir "E:\download" --steps gci offset dataset
```

**학습만 실행:**

```bash
python main.py --base-dir "E:\download" --steps train
```

**추론만 실행:**

```bash
python main.py --base-dir "E:\download" --steps inference
```

**사용자 정의 단계 조합:**

```bash
python main.py --base-dir "E:\download" --steps gci dataset train
```

### 사용 가능한 단계

- `gci`: 전역 카테고리 인덱스 생성
- `offset`: 카테고리 ID 오프셋 조정
- `dataset`: YOLO 데이터셋 구축
- `train`: 모델 학습
- `inference`: 추론 및 제출 생성
- `all`: 모든 단계를 순차적으로 실행 (기본값)

---

## 파이프라인 구조

### 1단계: 전역 카테고리 인덱스 생성

**목적:**
모든 어노테이션 데이터셋에 대한 통합 카테고리 매핑 시스템을 생성합니다.

**입력 소스:**

- `label_aug/train_annotations/`: 주 학습 어노테이션
- `label_aug/TL_*_조합/`: 증강 학습 조합 (TL_2 제외)
- `label_aug/VL_1_조합/`: 검증 조합 데이터셋

**처리 워크플로우:**

시스템은 지정된 디렉토리 내의 모든 JSON 어노테이션 파일을 재귀적으로 스캔합니다. 각 어노테이션 파일에서 카테고리 정보를 추출하고 카테고리별 샘플 수를 누적합니다. 그런 다음 0부터 시작하는 연속적인 인덱스 매핑을 생성하여 모든 데이터셋에 걸쳐 일관된 카테고리 표현을 보장합니다.

**출력 결과물:**

- `global_category_index.json`: 메타데이터를 포함한 완전한 카테고리 매핑
- `global_category_index.csv`: 분석용 테이블 형식
- `category_statistics.txt`: 데이터셋별 통계 요약

**예상 소요 시간:** 데이터셋 크기에 따라 5-10분

---

### 2단계: 카테고리 ID 오프셋 조정

**목적:**
주 학습 데이터와 보조 데이터셋 간의 카테고리 ID를 조화롭게 통합합니다.

**처리 로직:**

**규칙 1:** train_annotations에만 존재하는 카테고리는 원래 ID를 유지합니다.

**규칙 2:** 보조 데이터셋(TL_, VL_)의 카테고리는 +1 ID 오프셋을 받습니다.

**규칙 3:** 오프셋 적용으로 train_annotations와 ID 충돌이 발생하면, 시스템은 지능적 병합을 수행합니다:

- 샘플 수를 합산
- 데이터셋 출처 정보를 train_annotations 항목 아래 통합
- 연속성을 유지하도록 매핑 인덱스를 재계산

**변환 예시:**

```
변환 전: 카테고리 1899 (보조) + 카테고리 1900 (train_annotations)
변환 후: 카테고리 1899 → 1900 (오프셋) → 기존 1900과 병합
결과:   통합 통계를 가진 단일 카테고리 1900
```

**예상 소요 시간:** 1-2초

---

### 3단계: YOLO 데이터셋 구축

**목적:**
포괄적인 검증을 거쳐 어노테이션 데이터를 YOLO 호환 학습 형식으로 변환합니다.

**핵심 처리 구성요소:**

**이미지 인덱싱 시스템:**
O(1) 파일명-경로 해상도를 위한 해시 기반 조회 테이블을 구축하여 데이터셋 구축 중 효율적인 이미지 매칭을 가능하게 합니다.

**병렬 어노테이션 처리:**
멀티프로세싱을 활용하여 JSON 파일을 동시에 파싱하고, 대규모 어노테이션 세트의 처리 시간을 크게 단축합니다.

**품질 보증 파이프라인:**

- 바운딩 박스 존재 및 형식 검증
- 좌표 경계 확인 (이미지 크기 내)
- 면적 일관성 검사 (너비 × 높이 검증)
- 인덱스 파일 필터링 (_index 접미사 파일 제거)

**카테고리 ID 조화:**
2단계의 업데이트된 카테고리 매핑을 적용하며, 가능한 경우 dl_idx 값을 우선시하고 보조 데이터에 대한 체계적인 오프셋 보정을 수행합니다.

**중복 해결:**
동일한 이미지에 대한 중복 어노테이션을 식별하고 제거하며, 가장 완전한 어노테이션 세트를 보존합니다.

**좌표 정규화:**
COCO 형식의 절대 좌표를 YOLO 형식의 정규화된 좌표로 변환합니다:

```
x_center_norm = (bbox_x + bbox_width/2) / image_width
y_center_norm = (bbox_y + bbox_height/2) / image_height
width_norm = bbox_width / image_width
height_norm = bbox_height / image_height
```

**YOLO 레이블 형식:**

```
<class_index> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```

**예상 소요 시간:** 데이터셋 크기에 따라 10-30분

---

### 4단계: 모델 학습

**목적:**
최적화된 하이퍼파라미터를 사용하여 YOLOv8 객체 검출 모델을 학습합니다.

**모델 설정:**

- **아키텍처:** YOLOv8s (속도와 정확도의 균형을 위한 Small 변형)
- **입력 해상도:** 640×640 픽셀
- **학습 에폭:** 500 (조기 종료: patience=30)
- **배치 크기:** 8 (GPU 메모리에 따라 조정 가능)
- **옵티마이저:** AdamW (초기 학습률 0.001)

**데이터 증강 전략:**

```python
# 기하학적 증강
fliplr=0.5          # 수평 뒤집기 (50% 확률)
flipud=0.5          # 수직 뒤집기 (50% 확률)
mosaic=0.1          # 모자이크 증강 (10% 확률)
erasing=0.1         # 랜덤 지우기 (10% 확률)

# 비활성화된 증강 (알약 검출에 최적화)
hsv_h=0.0           # 색조 변화 비활성화
hsv_s=0.0           # 채도 변화 비활성화
hsv_v=0.0           # 명도 변화 비활성화
degrees=0.0         # 회전 비활성화
translate=0.0       # 이동 비활성화
scale=0.0           # 스케일링 비활성화
```

**학습 기능:**

- 메모리 효율성을 위한 혼합 정밀도 학습 (AMP)
- 자동 데이터 분할 (90% 학습, 10% 검증)
- 주기적 모델 체크포인트 (10 에폭마다)
- 포괄적인 메트릭 로깅

**예상 소요 시간:** 하드웨어 구성에 따라 4-24시간

---

### 5단계: 추론 및 제출

**목적:**
테스트 데이터셋에 대한 예측을 생성하고 대회 형식의 제출 파일을 생성합니다.

**추론 파이프라인:**

**배치 처리 시스템:**
테스트 이미지는 메모리 사용량과 처리 속도를 최적화하기 위해 구성 가능한 배치로 처리됩니다. 시스템은 사용 가능한 GPU 메모리에 따라 배치 크기를 자동으로 조정합니다.

**비최대 억제 (NMS):**
IoU 기반 필터링을 적용하여 중복 검출을 제거합니다:

- IoU 임계값: 0.7
- 신뢰도 임계값: 0.001 (포괄적 접근)
- 이미지당 최대 검출 수: 4

**ID 매핑 해결:**
업데이트된 GCI 매핑을 사용하여 YOLO 클래스 인덱스를 원래 카테고리 ID로 다시 변환하여 대회 요구사항과의 제출 호환성을 보장합니다.

**이미지 ID 추출:**
정규 표현식 패턴 매칭을 사용하여 파일명에서 숫자 이미지 식별자를 자동으로 추출하며, 다양한 명명 규칙을 처리합니다.

**CSV 생성:**
다음 스키마로 대회 표준 CSV를 생성합니다:

```
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score
```

**성능 최적화:**

- 2배 속도 향상을 위한 반정밀도 추론 (FP16)
- 메모리 사용량 관리를 위한 청크 처리
- GPU 성능에 기반한 배치 크기 최적화

**예상 소요 시간:** 일반적인 테스트 세트에 대해 5-20분

---

## 설정 관리

### 학습 파라미터 조정

`_execute_model_training` 메서드를 편집하여 학습 하이퍼파라미터를 수정하세요:

```python
training_results = model.train(
    epochs=300,              # 더 빠른 학습을 위해 감소
    batch=16,                # GPU 메모리가 허용하면 증가
    imgsz=512,               # 메모리 제약 시 감소
    lr0=0.002,               # 학습률 조정
    patience=50,             # 조기 종료 인내도 수정
)
```

### 추론 파라미터 튜닝

`_execute_inference` 메서드에서 추론 설정을 조정하세요:

```python
run_infer_to_csv(
    conf=0.01,               # 신뢰도 임계값
    iou=0.6,                 # NMS IoU 임계값
    max_det_per_image=6,     # 최대 검출 수
    pred_batch=8,            # 추론 배치 크기
    chunk_size=500,          # 처리 청크 크기
)
```

---

## 성능 최적화

### 학습 가속화

**GPU 메모리 최적화:**

```python
# 메모리 제약 시스템을 위한 배치 크기 감소
batch=4  # 매우 제한적인 메모리의 경우 2

# 이미지 해상도 조정
imgsz=416  # 더 빠른 학습을 위한 작은 입력 크기

# 혼합 정밀도 활성화/비활성화
amp=True  # 대부분의 경우 활성화 유지
```

**CPU 활용:**

```python
# CPU 코어에 따라 워커 수 조정
workers=8  # CPU 코어 수로 설정

# 반복 접근을 위한 이미지 캐싱 활성화
cache='ram'  # 메모리가 허용하면 RAM 캐싱 사용
```

### 추론 가속화

**배치 크기 최적화:**

```python
# 더 빠른 추론을 위해 배치 크기 증가
pred_batch=16  # GPU 메모리에 따라 더 높게

# 청크 처리 최적화
chunk_size=1000  # 더 나은 처리량을 위한 큰 청크
```

**모델 최적화:**

```python
# 최적화된 형식으로 내보내기
model.export(format='onnx')     # ONNX 형식
model.export(format='engine')   # TensorRT (NVIDIA GPU)
```

### 하드웨어별 최적화

**NVIDIA GPU 시스템:**

- 최대 추론 속도를 위해 TensorRT 활성화
- 최적의 호환성을 위해 CUDA 11.8 이상 사용
- 대규모 데이터셋의 경우 멀티 GPU 학습 고려

**AMD GPU 시스템:**

- ROCm 호환 PyTorch 빌드 사용
- 최적의 메모리 활용을 위해 배치 크기 조정

**CPU 전용 시스템:**

- 배치 크기를 크게 감소 (batch=1-2)
- CPU 코어에 맞춰 워커 수 증가
- 속도 향상을 위한 모델 양자화 고려

---

## 문제 해결 가이드

### 메모리 관련 문제

**CUDA 메모리 부족 오류:**

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**해결 방법:**

- 배치 크기 감소: `batch=4` 또는 `batch=2`
- 이미지 크기 감소: `imgsz=416` 또는 `imgsz=512`
- 혼합 정밀도 비활성화: `amp=False`
- GPU 캐시 비우기: `torch.cuda.empty_cache()` 호출 추가

**시스템 메모리 고갈:**

```
MemoryError: Unable to allocate array
```

**해결 방법:**

- 워커 수 감소: `max_workers=2`
- 더 작은 청크로 데이터 처리: `chunk_size=100`
- 모든 데이터 로드 대신 데이터 스트리밍 활성화

---

### 파일 시스템 문제

**권한 오류:**

```
PermissionError: [Errno 13] Permission denied
```

**해결 방법:**

- 관리자/sudo 권한으로 실행
- 디렉토리 쓰기 권한 확인
- 안티바이러스 소프트웨어가 파일 접근을 차단하지 않는지 확인

**경로 해석 오류:**

```
FileNotFoundError: No such file or directory
```

**해결 방법:**

- 상대 경로 대신 절대 경로 사용
- 디렉토리 구조가 문서와 일치하는지 확인
- 경로명에 특수 문자가 있는지 확인
- 적절한 경로 구분자 사용 확인 (Windows vs. Unix)

---

### 모델 학습 문제

**학습 정지 또는 충돌:**

```
RuntimeError: DataLoader worker is killed by signal
```

**해결 방법:**

- 멀티프로세싱 비활성화: `workers=0` 설정
- 메모리 문제 방지를 위해 배치 크기 감소
- 손상된 파일에 대한 데이터셋 무결성 확인
- CUDA 드라이버 호환성 확인

**낮은 학습 성능:**

```
낮은 mAP 점수 또는 높은 손실 값
```

**해결 방법:**

- 데이터셋 품질 및 어노테이션 정확도 확인
- 학습률 조정: `lr0=0.0001` 또는 `lr0=0.01` 시도
- 데이터에 맞게 증강 설정 수정
- 학습 에폭 증가 또는 인내도 조정
- 데이터셋의 클래스 균형 확인

---

### 추론 문제

**느린 추론 속도:**

**해결 방법:**

- 반정밀도 활성화: `half=True`
- 배치 크기 증가: `pred_batch=8` 또는 더 높게
- 가능한 경우 GPU 사용: `device=0`
- 모델을 최적화된 형식으로 내보내기 (TensorRT/ONNX)

**잘못된 예측:**

**해결 방법:**

- 모델 가중치 경로가 올바른지 확인
- GCI 매핑 파일 일관성 확인
- 신뢰도 및 IoU 임계값 검증
- 테스트 이미지 전처리가 학습과 일치하는지 확인

---

### 데이터셋 문제

**카테고리 매핑 오류:**

```
KeyError: Category ID not found in mapping
```

**해결 방법:**

- GCI 생성이 성공적으로 완료되었는지 확인
- ID 오프셋 조정이 적용되었는지 확인
- 데이터셋 전체에서 일관된 카테고리 ID 보장
- JSON 어노테이션 파일 형식 검증

**이미지 로딩 실패:**

```
PIL.UnidentifiedImageError: cannot identify image file
```

**해결 방법:**

- 이미지 파일 무결성 확인
- 지원되는 이미지 형식 확인 (jpg, png, jpeg)
- 손상된 이미지 파일 제거
- 이미지-어노테이션 대응 검증

---

## 로깅 및 모니터링

### 로그 파일 위치

**파이프라인 실행 로그:**
```
outputs/pipeline_YYYYMMDD_HHMMSS.log
```

**설정 기록:**
```
outputs/pipeline_config.json
```

**학습 로그:**
```
outputs/yolo_runs/pill_baseline/results.csv
```

### 로그 레벨 설정

로깅 시스템은 여러 상세도 수준을 제공합니다:

- **DEBUG:** 상세한 진단 정보
- **INFO:** 일반 진행 정보
- **WARNING:** 실행을 중지하지 않는 잠재적 문제
- **ERROR:** 실패를 야기할 수 있는 심각한 문제
- **CRITICAL:** 실행을 종료하는 심각한 오류

### 학습 진행 모니터링

**TensorBoard 통합:**

```bash
tensorboard --logdir outputs/yolo_runs/
```

**실시간 메트릭:**

`results.csv` 파일을 모니터링하여 에폭별 메트릭을 확인하세요:

- 학습 및 검증 손실
- mAP@0.5 및 mAP@0.5:0.95
- 정밀도 및 재현율 값
- 학습률 스케줄

---

**문서 버전:** 2.0  
**최종 업데이트:** 2025년 2월 11일  
**호환성:** Python 3.8+, YOLOv8, CUDA 11.0+
