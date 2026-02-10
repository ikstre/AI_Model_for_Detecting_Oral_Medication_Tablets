from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")



    

    model = YOLO("yolov8n.pt")

    results = model.train(
    data=r"E:\download\datasets\train_plus_extra\yolo_format\data.yaml",
    split=0.9,

    epochs=300,
    patience=30,
    imgsz=640,

    batch=8,
    workers=0,
    amp=True,

    optimizer="AdamW",
    lr0=0.0005,

    # 증강(현실적 노이즈 위주)
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=10.0,
    translate=0.1,
    scale=0.2,
    fliplr=0.5,
    flipud=0.5,      # 상하반전은 약하게
    mosaic=0.25,     # 데이터 충분하니 더 약하게
    mixup=0.0,
    erasing=0.2,     # 가림/반사/부분손상 대응

    device=0,

    project=r"E:\yolo_runs",
    name="pill_baseline_augdata",
    save_period=10
)
