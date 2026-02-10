import pandas as pd
import numpy as np
import json
from pathlib import Path

def calculate_class_weights(
    category_csv: str,
    submission_csv: str,
    output_json: str = "class_weights.json"
):
    """
    클래스 가중치 계산
    
    Args:
        category_csv: 카테고리 통계 CSV
        submission_csv: 제출 파일 (confidence 통계)
        output_json: 가중치 JSON 저장 경로
    """
    
    # 1. 카테고리 통계 로드
    df_cat = pd.read_csv(category_csv)
    
    # 2. Submission 통계 로드
    df_sub = pd.read_csv(submission_csv)
    
    # 3. 병합
    df = df_cat.merge(
        df_sub[['category_id', 'score_mean_base']],
        on='category_id',
        how='left'
    )
    
    # 4. 빈도 기반 가중치 (15회 이하만)
    def freq_weight(count):
        if count >= 15:
            return 1.0
        # 제곱근 스케일링: 3회→2.24, 6회→1.58, 9회→1.29, 15회→1.0
        return np.sqrt(15.0 / count)
    
    df['freq_weight'] = df['total_count'].apply(freq_weight)
    
    # 5. Confidence 기반 가중치
    def conf_weight(score_mean):
        if pd.isna(score_mean):
            return 1.0  # 데이터 없으면 기본값
        
        if score_mean >= 0.6:
            return 1.0
        
        # 선형 스케일링: 0.35→1.22, 0.40→1.12, 0.45→1.0
        return 1.0 + (0.6 - score_mean) * 0.5
    
    df['conf_weight'] = df['score_mean_base'].apply(conf_weight)
    
    # 6. 최종 가중치
    df['final_weight'] = df['freq_weight'] * df['conf_weight']
    
    # 7. 정규화 (평균 = 1.0)
    mean_weight = df['final_weight'].mean()
    df['final_weight'] = df['final_weight'] / mean_weight
    
    # 8. JSON 생성 (global_index → weight)
    weight_dict = {}
    for _, row in df.iterrows():
        global_idx = int(row['global_index'])
        weight = float(row['final_weight'])
        weight_dict[global_idx] = weight
    
    # 저장
    with open(output_json, 'w') as f:
        json.dump(weight_dict, f, indent=2)
    
    # 통계 출력
    print("=" * 80)
    print("클래스 가중치 계산 완료")
    print("=" * 80)
    
    print(f"\n📊 통계:")
    print(f"   전체 클래스: {len(df)}개")
    print(f"   15회 이하: {len(df[df['total_count'] <= 15])}개")
    print(f"   낮은 conf (<0.45): {len(df[df['score_mean_base'] < 0.45])}개")
    
    print(f"\n⚖️  가중치 분포:")
    print(f"   최소: {df['final_weight'].min():.3f}")
    print(f"   최대: {df['final_weight'].max():.3f}")
    print(f"   평균: {df['final_weight'].mean():.3f}")
    print(f"   중앙값: {df['final_weight'].median():.3f}")
    
    print(f"\n🔝 상위 10개 (가중치 높음):")
    top_10 = df.nlargest(10, 'final_weight')[
        ['global_index', 'category_name', 'total_count', 'score_mean_base', 'final_weight']
    ]
    print(top_10.to_string(index=False))
    
    print(f"\n💾 저장: {output_json}")
    
    return df


# 실행
# if __name__ == "__main__":
#     df_weights = calculate_class_weights(
#         category_csv=r"E:\download\global_category_index(train_set_56)\global_category_index.csv",
#         submission_csv=r"E:\download\model_comparison_clahe.csv",  # 네가 만든 CSV
#         output_json=r"E:\download\class_weights.json"
#     )






from ultralytics import YOLO
import torch
import json
import yaml
from pathlib import Path

def train_with_class_weights(
    data_yaml: str,
    weights_json: str,
    model_weights: str = "yolov8n.pt",
    epochs: int = 100,
    batch: int = 8,
    device: int = 0
):
    """
    클래스 가중치 적용 학습
    
    Args:
        data_yaml: 데이터셋 YAML
        weights_json: 클래스 가중치 JSON
        model_weights: 초기 가중치 (또는 이전 best.pt)
        epochs: 에폭 수
        batch: 배치 크기
        device: GPU 번호
    """
    
    print("=" * 80)
    print("클래스 가중치 학습 시작")
    print("=" * 80)
    
    # 1. 클래스 가중치 로드
    print(f"\n[1단계] 클래스 가중치 로드...")
    with open(weights_json, 'r', encoding='utf-8') as f:
        class_weights = json.load(f)
    
    # global_index 순서대로 정렬 (0, 1, 2, ...)
    weight_list = [class_weights[str(i)] for i in range(len(class_weights))]
    weight_tensor = torch.tensor(weight_list, dtype=torch.float32)
    
    print(f"   ✅ 가중치 로드: {len(weight_list)}개 클래스")
    print(f"   범위: {weight_tensor.min():.3f} ~ {weight_tensor.max():.3f}")
    
    # 2. 모델 로드
    print(f"\n[2단계] 모델 로드...")
    model = YOLO(model_weights)
    
    # 3. 클래스 가중치 적용
    print(f"\n[3단계] 클래스 가중치 적용...")
    
    # YOLOv8에서 클래스 가중치는 loss 함수에 직접 적용
    # 방법: data.yaml에 클래스 가중치 추가 또는 커스텀 loss 사용
    
    # 🔧 방법 1: data.yaml 수정 (추천)
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # class_weights 추가
    data_config['class_weights'] = weight_list
    
    # 임시 YAML 저장
    temp_yaml = Path(data_yaml).parent / "data_weighted.yaml"
    with open(temp_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f)
    
    print(f"   ✅ 가중치 적용 완료")
    print(f"   임시 YAML: {temp_yaml}")
    
    # 4. 학습
    print(f"\n[4단계] 학습 시작...")
    
    results = model.train(
        data=str(temp_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=640,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # 증강 (색상 보존)
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=15.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
        
        # 기타
        device=device,
        workers=0,
        amp=True,
        patience=20,
        
        project='runs/detect',
        name='weighted_training',
        exist_ok=False
    )
    
    print("\n" + "=" * 80)
    print("학습 완료!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # GPU 확인
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    # 학습 실행
    results = train_with_class_weights(
        data_yaml=r"E:\download\datasets\original_data\yolo_clahe\data.yaml",
        weights_json=r"E:\download\class_weights.json",
        model_weights=r"C:\Users\chocy\AI_07_basic\runs\detect\pill_clahe_finetune2\weights\best.pt",  # 이전 best
        epochs=50,
        batch=8,
        device=0
    )
    
    print(f"\n🎯 Best mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")