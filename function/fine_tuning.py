from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # 현재 best 모델 로드
    model = YOLO(r"C:\Users\chocy\AI_07_basic\runs\detect\runs\detect\pill_stage1_finetune\weights\best.pt")
    
    # Stage 1: 안정적 추가 학습
    results = model.train(
        data=r"E:\download\datasets\original_trainset\yolo_format\data.yaml",
        
        # 기본 설정
        epochs=40,  # 20 에폭 추가
        batch=8,
        imgsz=640,
        
        # 더 작은 Learning Rate (미세 조정)
        optimizer="AdamW",
        lr0=1e-4,  # 3e-4 → 1e-4 (1/3로 감소)
        lrf=0.01,  # 최종 lr = 1e-6
        cos_lr=True,
        warmup_epochs=1.0,  # 짧은 warmup
        
        # 증강 약화 (안정화)
        mosaic=0.3,  # 0.5 → 0.3
        mixup=0.01,  # 0.02 → 0.01
        copy_paste=0.0,
        
        # 기타
        workers=0,
        amp=True,
        cache=False,
        device=0,
        
        # 중요: resume 대신 새 폴더
        project="runs/detect",
        name="pill_stage2_finetune",
        exist_ok=False  # 새 폴더 생성
    )
    
    print("\n" + "="*80)
    print("Stage 2 완료!")
    print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.5f}")
    print(f"Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.5f}")
    print("="*80)