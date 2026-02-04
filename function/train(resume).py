from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    model = YOLO(r"C:\Users\chocy\AI_07_basic\runs\detect\pill_resume_stable\weights\best.pt")

    results = model.train(
        data=r"E:\download\datasets\original_trainset\yolo_format\data.yaml",
        epochs=50,
        batch=8,
        imgsz=640,
        optimizer="AdamW",
        lr0=3e-4,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=2.0,

        mosaic=0.5,
        mixup=0.02,
        copy_paste=0.0,

        workers=0,
        amp=True,
        cache=False,
        device=0,
        name="pill_finetune_safe"
    )

