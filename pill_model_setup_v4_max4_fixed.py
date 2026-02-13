"""
pill_model_setup_v4_max4_fixed.py  (수정 버전)

변경 사항:
  [FIX-M1] train_yolo()에서 shear/perspective가 0.0으로 하드코딩 → cfg에서 읽도록 수정
           (CFG에 이미 shear=0.0, perspective=0.0이 정의되어 있으므로 결과는 동일하나,
            Optuna 등에서 shear/perspective를 탐색할 때 override가 무시되는 버그 방지)
  [FIX-M2] CFG.val_ratio = 1 - split 은 클래스 변수 레벨에서는 참조 불가 → property 또는
           __post_init__으로 처리해야 하나, dataclass가 아닌 일반 필드이므로 명시적 값으로 수정
"""

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
    # [FIX-M2] 클래스 레벨에서 `1 - split`은 CFG.split이 아닌 builtins를 참조할 수 있음
    #          → 명시적 값으로 고정 (노트북 Cell 3에서 cfg.val_ratio를 다시 설정하므로 실질 영향 없음)
    val_ratio = 0.1

    # runs/paths
    work_dir = "runs_pill"
    baseline_name = "baseline"

    # Ultralytics run output
    project = r"yolo_runs"
    name = "pill_baseline_augdata"
    optuna_subdir = "optuna"

    # -------------------------
    # 모델 후보(다른 YOLO 버전 포함)
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
    batch = 16
    lr0 = 5e-4
    weight_decay = 1e-4
    optimizer = "AdamW"
    device = 0  # 0 or "cpu"
    patience = 10
    workers = 0
    amp = True
    save_period = 10

    # -------------------------
    # 증강(색상은 꺼두고, 기하/혼합만 사용하도록 기본값 구성)
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
    # 불균형 보정(옵션)
    # -------------------------
    balance_enable = True
    balance_extra_ratio = 0.30
    balance_power = 1.0

    # -------------------------
    # class_weight.py 기반 클래스 가중치(옵션)
    # -------------------------
    class_weight_enable = True
    class_weight_json = ""
    class_weight_category_csv = ""
    class_weight_submission_csv = ""
    class_weight_count_threshold = 15

    # -------------------------
    # 데이터 품질 검사
    # -------------------------
    exclude_index_images = True
    image_width = 976
    image_height = 1280
    strict_sanitize = True
    dedup_images = True

    # -------------------------
    # Optuna
    # -------------------------
    use_optuna = True
    n_trials = 300
    optuna_epochs = 10

    # -------------------------
    # Inference / submission
    # -------------------------
    test_image_id_mode = "digits"
    conf = 0.001
    iou = 0.7
    max_det = 10
    max_det_per_image = 4

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
        patience=int(getattr(cfg, 'patience', 10)),
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
        # [FIX-M1] 하드코딩(0.0) → cfg에서 읽기 (Optuna override도 정상 반영됨)
        shear=float(getattr(cfg, "shear", 0.0)),
        perspective=float(getattr(cfg, "perspective", 0.0)),
        fliplr=float(getattr(cfg, "fliplr", 0.0)),
        flipud=float(getattr(cfg, "flipud", 0.0)),
        mosaic=float(getattr(cfg, "mosaic", 0.0)),
        mixup=float(getattr(cfg, "mixup", 0.0)),
        erasing=float(getattr(cfg, "erasing", 0.0)),
    )

    # trial별 overrides
    args.update(overrides)
    return model.train(**args)
