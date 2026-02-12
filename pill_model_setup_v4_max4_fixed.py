import os, json, random
from dataclasses import dataclass

import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

@dataclass
class CFG:
    seed = 42
    train_img_dir = "data/train_images"
    train_ann_dir = "data/train_annotations"
    test_img_dir  = "data/test_images"

    # yolo dataset export
    yolo_dataset_dir = "yolo_pill_ds"
    export_dir = "exports_pill"
    split = 0.9  # train fraction (train=split, val=1-split)
    val_ratio = 1 - split  # val fraction

    # runs/paths
    work_dir = "runs_pill"
    baseline_name = "baseline"

    # Ultralytics run output
    project = r"yolo_runs"
    name = "pill_baseline_augdata"
    optuna_subdir = "optuna"

    # -------------------------
    # 모델 후보(다른 YOLO 버전 포함)
    # - 설치된 ultralytics가 지원하는 모델만 선택해서 사용하시면 됩니다.
    # -------------------------
    model_candidates = [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov9c.pt",
        "yolov10n.pt",
        "yolo11n.pt",
    ]
    base_model = "yolov8n.pt"

    # -------------------------
    # 학습 기본 하이퍼파라미터
    # -------------------------
    imgsz = 640
    epochs = 300
    batch = 8
    lr0 = 5e-4
    weight_decay = 1e-4
    optimizer = "AdamW"
    device = 0  # 0 or "cpu"
    patience = 30
    workers = 0
    amp = True
    save_period = 10

    # -------------------------
    # 증강(색상은 꺼두고, 기하/혼합만 사용하도록 기본값 구성)
    # - Ultralytics train args 그대로 전달됩니다.
    # -------------------------
    hsv_h = 0.0
    hsv_s = 0.0
    hsv_v = 0.0

    degrees = 10.0
    translate = 0.10
    scale = 0.20
    shear = 0.0
    perspective = 0.0

    fliplr = 0.5
    flipud = 0.5

    mosaic = 0.25
    mixup = 0.0
    close_mosaic = 0

    erasing = 0.2  # Random erasing probability

    # -------------------------
    # 불균형 보정(옵션): train.txt에 추가로 반복 샘플링할 비율
    # - 예: 0.5면 train 이미지 수의 50%만큼을 추가로 더 뽑아서 train.txt에 append
    # -------------------------
    balance_enable = True
    # train list oversample 강도 (balance_enable=True일 때만 의미)
    # - 0.3이면 train 이미지 수의 30%만큼을 "추가로" 뽑아 train.txt에 append
    balance_extra_ratio = 0.30
    balance_power = 1.0

    # -------------------------
    # class_weight.py 기반 클래스 가중치(옵션)
    # - class_weights.json이 존재하면(또는 생성하면) 그 값을 우선 사용
    # - Ultralytics는 data.yaml의 class_weights를 기본적으로 사용하지 않으므로,
    #   본 파이프라인에서는 "train.txt oversample"에 가중치를 반영하는 방식으로 적용합니다.
    # -------------------------
    class_weight_enable = True
    class_weight_json = ""  # 비워두면 (work_dir/class_weights.json)을 자동 사용
    class_weight_category_csv = ""   # (선택) global_category_index.csv 등
    class_weight_submission_csv = "" # (선택) score_mean_base 컬럼이 있는 csv
    class_weight_count_threshold = 15  # class_weight.py의 기본값

    # -------------------------
    # 데이터 품질 검사( dataset_generator_v2_patched_final 참고 )
    # -------------------------
    exclude_index_images = True   # file_name에 '_index'가 포함된 샘플 제외
    image_width = 976
    image_height = 1280
    strict_sanitize = True        # bbox/area/bounds 검사 강화
    dedup_images = True           # 동일 file_name 중복 제거

    # -------------------------
    # Optuna
    # -------------------------
    use_optuna = True
    n_trials = 100
    optuna_epochs = 100

    # -------------------------
    # Inference / submission
    # -------------------------
    test_image_id_mode = "digits"  # ✅ 제출 규칙: 파일명 숫자 사용
    conf = 0.001
    iou = 0.7
    max_det = 10 # ✅ 이미지당 탐지 개수 제한(요청대로 넉넉히)
    max_det_per_image = 4  # ✅ 도메인 제약: 이미지당 최대 알약 수(제출/평가에 적용)

def build_yolo_model(base_model):
    from ultralytics import YOLO
    return YOLO(base_model)

def train_yolo(model, data_yaml, cfg, epochs=None, **overrides):
    epochs = int(epochs if epochs is not None else cfg.epochs)

    # 기본 args
    args = dict(
        data=data_yaml,
        project=str(getattr(cfg, 'project', getattr(cfg, 'work_dir', 'runs_pill'))),
        name=str(getattr(cfg, 'name', getattr(cfg, 'baseline_name', 'baseline'))),
        exist_ok=True,
        imgsz=int(cfg.imgsz),
        epochs=epochs,
        batch=int(cfg.batch),
        lr0=float(cfg.lr0),
        weight_decay=float(cfg.weight_decay),
        optimizer=str(cfg.optimizer),
        device=cfg.device,
        patience=int(getattr(cfg, 'patience', 50)),
        workers=int(getattr(cfg, 'workers', 8)),
        amp=bool(getattr(cfg, 'amp', True)),
        save_period=int(getattr(cfg, 'save_period', -1)),
        close_mosaic=int(getattr(cfg, "close_mosaic", 0)),
        verbose=False,

        # augmentation (affine 중심, 색상은 기본 0)
        hsv_h=float(getattr(cfg, "hsv_h", 0.0)),
        hsv_s=float(getattr(cfg, "hsv_s", 0.0)),
        hsv_v=float(getattr(cfg, "hsv_v", 0.0)),
        degrees=float(getattr(cfg, "degrees", 0.0)),
        translate=float(getattr(cfg, "translate", 0.0)),
        scale=float(getattr(cfg, "scale", 0.0)),
        shear = 0.0,
        perspective = 0.0,
        fliplr=float(getattr(cfg, "fliplr", 0.0)),
        flipud=float(getattr(cfg, "flipud", 0.0)),
        mosaic=float(getattr(cfg, "mosaic", 0.0)),
        mixup=float(getattr(cfg, "mixup", 0.0)),
        erasing=float(getattr(cfg, "erasing", 0.0)),
    )

    # trial별 overrides
    args.update(overrides)
    return model.train(**args)
