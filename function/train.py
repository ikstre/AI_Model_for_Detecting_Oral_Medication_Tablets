# from ultralytics import YOLO
# import torch

# if __name__ == "__main__":
#     print(f"CUDA: {torch.cuda.is_available()}")
#     print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")



    

#     model = YOLO("yolov8n.pt")

#     results = model.train(
#     data=r"E:\download\datasets\train_plus_extra\yolo_format\data.yaml",
#     split=0.9,

#     epochs=300,
#     patience=30,
#     imgsz=640,

#     batch=8,
#     workers=0,
#     amp=True,

#     optimizer="AdamW",
#     lr0=0.0005,

#     # 증강(현실적 노이즈 위주)
#     hsv_h=0.0,
#     hsv_s=0.0,
#     hsv_v=0.0,
#     degrees=10.0,
#     translate=0.1,
#     scale=0.2,
#     fliplr=0.5,
#     flipud=0.5,      # 상하반전은 약하게
#     mosaic=0.25,     # 데이터 충분하니 더 약하게
#     mixup=0.0,
#     erasing=0.2,     # 가림/반사/부분손상 대응

#     device=0,

#     project=r"E:\yolo_runs",
#     name="pill_baseline_augdata",
#     save_period=10
# )


# if __name__ == "__main__":
#     print(f"CUDA: {torch.cuda.is_available()}")
#     print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

#     model = YOLO(r"E:\yolo_runs\pill_baseline_compdata2\weights\best.pt")

#     results = model.train(
#         data=r"E:\download\datasets\original_data\yolo_format\data.yaml",
#         epochs=40,           
#         lr0=3e-4,
#         optimizer="AdamW",
#         imgsz=640,
#         batch=8,
#         workers=0,
#         amp=True,
#         device=0,

#         hsv_h=0.015,
#         hsv_s=0.4,
#         hsv_v=0.4,
#         degrees=10.0,
#         translate=0.1,
#         scale=0.3,
#         fliplr=0.5,

#         mosaic=0.1,   # 또는 0.0
#         mixup=0.0,

#         project=r"E:\yolo_runs",
#         name="pill_baseline_compdata_240",
#         save_period=10
#     )


# # clahe fine tuning
# if __name__ == "__main__":
#     print("CUDA:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         print("GPU:", torch.cuda.get_device_name(0))

#     model = YOLO(r"C:\Users\chocy\AI_07_basic\runs\detect\pill_clahe_finetune2\weights\best.pt")

#     model.train(
#         data=r"E:\download\datasets\original_data\yolo_clahe\data.yaml",
#         epochs=50,          # 파인튜닝이면 20~50 정도부터
#         batch=8,
#         imgsz=640,
#         lr0=1e-4,           
#         lrf=0.05,
#         warmup_epochs=1.0,

#         mosaic=0.3,         # 파인튜닝 때는 mosaic 줄이는 편이 안정적
#         mixup=0.0,
#         copy_paste=0.0,

#         optimizer="AdamW",
#         device=0,
#         workers=0,          
#         amp=True,
#         cache=False,
#         name="pill_clahe_finetune_focal",
#     )




# import torch
# from ultralytics import YOLO
# from sklearn.model_selection import KFold
# import yaml
# import os
# from pathlib import Path

# # 1. 경로 설정
# base_data_path = Path(r"E:\download\datasets\original_data\yolo_clahe")
# images_path = base_data_path / "images"  # 이미지들이 모여있는 곳
# original_yaml = base_data_path / "data.yaml"

# def run_kfold_training(k=5):
#     # 이미지 파일 리스트 확보
#     img_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
#     kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
#     # 원본 yaml 읽기
#     with open(original_yaml, 'r', encoding='utf-8') as f:
#         data_config = yaml.safe_load(f)

#     for fold, (train_idx, val_idx) in enumerate(kf.split(img_files)):
#         print(f"\n🚀 --- Starting Fold {fold+1}/{k} ---")
        
#         # 각 fold를 위한 임시 txt 파일 생성 (YOLO가 읽을 리스트)
#         train_list = [str(images_path / img_files[i]) for i in train_idx]
#         val_list = [str(images_path / img_files[i]) for i in val_idx]
        
#         fold_train_txt = base_data_path / f"train_fold_{fold}.txt"
#         fold_val_txt = base_data_path / f"val_fold_{fold}.txt"
        
#         with open(fold_train_txt, 'w',encoding='utf-8') as f: f.write('\n'.join(train_list))
#         with open(fold_val_txt, 'w', encoding='utf-8') as f: f.write('\n'.join(val_list))
        
#         # 이 fold 전용 yaml 생성
#         data_config['train'] = str(fold_train_txt)
#         data_config['val'] = str(fold_val_txt)
        
#         fold_yaml = base_data_path / f"data_fold_{fold}.yaml"
#         with open(fold_yaml, 'w', encoding='utf-8') as f:
#             yaml.dump(data_config, f)

#         # 2. 모델 학습 시작
#         # 각 Fold마다 깨끗한 best.pt에서 다시 시작하거나, 이전 fold의 가중치를 이어받을 수 있음
#         model = YOLO(r"C:\Users\chocy\AI_07_basic\runs\detect\pill_clahe_finetune_comp\weights\best.pt")
        
#         model.train(
#             data=str(fold_yaml),
#             epochs=50,             # K-Fold는 여러 번 반복하므로 에폭을 조금 줄여도 됨
#             batch=8,
#             imgsz=640,
#             lr0=1e-4,
#             lrf=0.05,
#             warmup_epochs=1.0,
#             mosaic=0.3,
#             optimizer="AdamW",
#             device=0,
#             name=f"pill_clahe_fold_{fold}",
#         )
#         print(f"✅ Fold {fold+1} Completed.\n")

# if __name__ == "__main__":
#     print("CUDA:", torch.cuda.is_available())
#     run_kfold_training(k=5)  # 5-Fold 추천




from ultralytics import YOLO

LAST = r"E:\yolo_runs\pill_baseline_augdata2\weights\last.pt"
# 또는 (가끔 이렇게 생성됨)
# LAST = r"E:\yolo_runs\detect\pill_baseline_augdata\weights\last.pt"

model = YOLO(LAST)

results = model.train(
    resume=True,
    epochs=500,      # ✅ 총 500까지 (이미 300이면 301~500 수행)
    patience=80,     # ✅ plateau 오래 보는 용도
    save_period=10
)