from ultralytics import YOLO
import torch

# if __name__ == "__main__":
#     print(f"CUDA: {torch.cuda.is_available()}")
#     print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")



    

#     model = YOLO("yolov8n.pt")

#     results = model.train(
#         data=r"E:\download\datasets\original_trainset\yolo_format\data.yaml",
#         split=0.9,

#         epochs=100,
#         patience=20,
#         imgsz=640,

#         batch=16,          # <- 32에서 낮추기
#         workers=0,         # <- 안정성 우선 (안 죽으면 2~4로)
#         amp=True,

#         optimizer="AdamW",
#         lr0=0.001,

#         # 증강: 일단 안정적으로
#         hsv_h=0.0,
#         hsv_s=0.0,
#         hsv_v=0.0,
#         degrees=10.0,
#         translate=0.1,
#         scale=0.2,
#         fliplr=0.5,
#         mosaic=0.5,        # <- 1.0에서 낮추기
#         mixup=0.0,         

#         device=0,

#         project=r"E:\yolo_runs",
#         name="pill_baseline",
#         save_period=5
#     )

# clahe fine tuning
if __name__ == "__main__":
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(r"C:\Users\chocy\AI_07_basic\runs\detect\runs\detect\base_235ep\weights\best.pt")

    model.train(
        data=r"E:\download\datasets\original_trainset\yolo_clahe\clahe_data.yaml",
        epochs=100,          # 파인튜닝이면 20~50 정도부터
        batch=8,
        imgsz=640,
        lr0=1e-4,           
        lrf=0.05,
        warmup_epochs=1.0,

        mosaic=0.3,         # 파인튜닝 때는 mosaic 줄이는 편이 안정적
        mixup=0.0,
        copy_paste=0.0,

        optimizer="AdamW",
        device=0,
        workers=0,          
        amp=True,
        cache=False,
        save_period=10,
        name="pill_clahe_finetune",
    )