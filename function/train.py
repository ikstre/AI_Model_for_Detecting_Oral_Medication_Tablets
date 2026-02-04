from ultralytics import YOLO
import torch

if __name__ == '__main__':  # 이 줄 추가!
    # GPU 체크
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    # 모델 초기화
    model = YOLO('yolov8n.pt')
    
    # 학습 시작
    results = model.train(
        data=r'E:\download\datasets\original_trainset\yolo_format\data.yaml',
        split=0.9,
        epochs=100,
        patience=20,
        batch=32,
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        
        # 색상 보존 증강
        hsv_h=0.0, 
        hsv_s=0.0, 
        hsv_v=0.0,
        degrees=15.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        
        device=0,
        workers=4,  # 또는 workers=0 (싱글 프로세스)
        amp=True,
        
        project='runs/detect',
        name='pill_baseline',
        save_period=10
    )
    
    # 평가
    print("\n" + "="*80)
    print("학습 완료!")
    print("="*80)