from ultralytics import YOLO
import torch
import os

if __name__ == "__main__":
    # GPU 상태 확인
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.get_device_name(0)}")

    # 1. 모델 로드 (yolov8s.pt로 초기화 학습)
    model = YOLO("yolov8l.pt")

    # 2. 학습 실행
    results = model.train(
        # --- 핵심 경로 설정 (코랩 환경에 맞춰 수정 필요) ---
        data="/content/datasets/train_plus_extra_refined/yolo_format/data.yaml", # unzip한 폴더 내 yaml 경로
        project="/content/drive/MyDrive/yolo_runs",     # 결과물은 드라이브에 저장
        name="pill_1024_precision_v8l",
        
        # --- 아키텍트님 요청 및 추천 파라미터 ---
        imgsz=1024,          # 0.998을 위한 고해상도 필수 조건
        batch=8,             # T4 VRAM 16GB 한계점 고려
        epochs=500,          
        patience=50,         # 고해상도이므로 수렴을 위해 기존보다 조금 더 기다림
        
        # --- 박스 정밀도 최적화 세팅 ---
        box=12.0,            # BBox 정밀도에 가중치 대폭 상향 (핵심!)
        lr0=0.01,            # 초기화 학습이므로 0.01로 시작 (수렴 가속)
        overlap_mask=False,  # 객체 중첩이 적으므로 마스킹 정밀도 향상
        optimizer="AdamW",
        amp=True,
        
        # --- 기존 0.988 모델의 증강 세팅 유지 ---
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.1,
        mixup=0.0,
        erasing=0.1,

        device=0,
        save_period=10,      # 드라이브 끊김 대비 10 에폭마다 저장
        workers=4            # 코랩 CPU 사양에 맞춤
    )