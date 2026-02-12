import csv
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

def load_csv(csv_path: str):
    """CSV 파일을 로드하여 image_id별로 그룹화"""
    data = defaultdict(list)
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            
            bbox = [
                float(row["bbox_x"]),
                float(row["bbox_y"]),
                float(row["bbox_w"]),
                float(row["bbox_h"])
            ]
            category_id = int(row["category_id"])
            
            detection = {
                "bbox": bbox,
                "category_id": category_id,
            }
            
            # 예측 CSV의 경우 score 포함
            if "score" in row:
                detection["score"] = float(row["score"])
            
            data[image_id].append(detection)
    
    return data

def calculate_iou(box1, box2):
    """
    두 bbox의 IoU 계산
    box format: [x, y, w, h]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 좌표를 (x1, y1, x2, y2) 형식으로 변환
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2
    
    # 교집합 영역 계산
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # 합집합 영역 계산
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou

def calculate_ap(precisions, recalls):
    """
    precision-recall 커브에서 AP 계산 (11-point interpolation 방식)
    """
    # recall 값을 0에서 1까지 0.1 간격으로
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        # recall >= t인 precision 값 중 최대값
        prec_values = [p for r, p in zip(recalls, precisions) if r >= t]
        if len(prec_values) > 0:
            ap += max(prec_values)
    ap /= 11.0
    return ap

def evaluate_detections_all_thresholds(pred_data, gt_data, iou_thresholds):
    """
    여러 IoU threshold에 대해 한번에 평가 (최적화 버전)
    
    Args:
        pred_data: 예측 데이터 (image_id -> list of detections)
        gt_data: Ground Truth 데이터 (image_id -> list of detections)
        iou_thresholds: IoU 임계값 리스트 (예: [0.5, 0.55, 0.6, ...])
    
    Returns:
        results: {iou_threshold: {category_id: ap}} 형태의 딕셔너리
    """
    # 모든 카테고리 수집
    all_categories = set()
    for detections in gt_data.values():
        for det in detections:
            all_categories.add(det["category_id"])
    
    # 각 카테고리별로 모든 예측과 GT 수집 (한번만)
    category_data = {}
    
    for category_id in all_categories:
        all_preds = []
        all_gts = []
        
        for image_id in gt_data.keys():
            gt_dets = [d for d in gt_data[image_id] if d["category_id"] == category_id]
            pred_dets = [d for d in pred_data.get(image_id, []) if d["category_id"] == category_id]
            
            # GT와 예측을 image_id와 함께 저장
            for gt in gt_dets:
                all_gts.append({
                    "image_id": image_id,
                    "bbox": gt["bbox"]
                })
            
            for pred in pred_dets:
                all_preds.append({
                    "image_id": image_id,
                    "bbox": pred["bbox"],
                    "score": pred.get("score", 1.0)
                })
        
        if len(all_gts) == 0:
            continue
        
        # 예측을 score 기준으로 내림차순 정렬
        all_preds.sort(key=lambda x: x["score"], reverse=True)
        
        # 모든 예측-GT 쌍에 대해 IoU 미리 계산
        iou_matrix = []
        for pred in all_preds:
            pred_image_id = pred["image_id"]
            pred_bbox = pred["bbox"]
            
            # 같은 이미지의 GT들과 IoU 계산
            image_gts = [(i, gt) for i, gt in enumerate(all_gts) if gt["image_id"] == pred_image_id]
            
            if len(image_gts) == 0:
                iou_matrix.append([])
            else:
                ious = [(gt_idx, calculate_iou(pred_bbox, gt["bbox"])) for gt_idx, gt in image_gts]
                iou_matrix.append(ious)
        
        category_data[category_id] = {
            'preds': all_preds,
            'gts': all_gts,
            'iou_matrix': iou_matrix
        }
    
    # 각 IoU threshold에 대해 AP 계산
    results = {}
    
    for iou_threshold in iou_thresholds:
        ap_per_category = {}
        
        for category_id, data in category_data.items():
            all_preds = data['preds']
            all_gts = data['gts']
            iou_matrix = data['iou_matrix']
            
            # TP, FP 계산
            tp = np.zeros(len(all_preds))
            fp = np.zeros(len(all_preds))
            
            # 각 이미지별 GT가 이미 매칭되었는지 추적
            gt_matched = defaultdict(set)
            
            for pred_idx, pred in enumerate(all_preds):
                image_id = pred["image_id"]
                
                # 미리 계산된 IoU 사용
                ious = iou_matrix[pred_idx]
                
                if len(ious) == 0:
                    fp[pred_idx] = 1
                    continue
                
                # 최대 IoU 찾기
                max_iou = max(iou for _, iou in ious)
                max_gt_idx = max(ious, key=lambda x: x[1])[0]
                
                # IoU 임계값 이상이고 아직 매칭되지 않은 GT라면 TP
                if max_iou >= iou_threshold and max_gt_idx not in gt_matched[image_id]:
                    tp[pred_idx] = 1
                    gt_matched[image_id].add(max_gt_idx)
                else:
                    fp[pred_idx] = 1
            
            # Cumulative sum
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            # Recall과 Precision 계산
            recalls = tp_cumsum / len(all_gts)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # AP 계산
            ap = calculate_ap(precisions.tolist(), recalls.tolist())
            ap_per_category[category_id] = ap
        
        results[iou_threshold] = ap_per_category
    
    return results

def evaluate(pred_csv_path: str, gt_csv_path: str, output_txt_path: str = None, gci_json_path: str = None):
    """
    예측 CSV와 GT CSV를 비교하여 mAP@50과 mAP@75:95 계산
    
    Args:
        pred_csv_path: 예측 결과 CSV 파일 경로
        gt_csv_path: Ground Truth CSV 파일 경로
        output_txt_path: mAP@75:95 상세 결과를 저장할 txt 파일 경로
        gci_json_path: Global Category Index JSON 파일 경로 (선택사항, 제공시 해당 카테고리 중점 분석)
    """
    print("=" * 60)
    print("평가 시작")
    print("=" * 60)
    
    # GCI JSON 로드 (선택적)
    focused_categories = None
    if gci_json_path:
        import json
        try:
            with open(gci_json_path, "r", encoding="utf-8") as f:
                gci_data = json.load(f)
                index_to_id = gci_data.get("index_to_id", {})
                focused_categories = set(int(cat_id) for cat_id in index_to_id.values())
                print(f"✓ GCI 로드 완료: {len(focused_categories)}개 중점 카테고리 식별")
        except Exception as e:
            print(f"⚠️  GCI 로드 실패: {e}")
            focused_categories = None
    
    # CSV 파일 로드
    print("예측 CSV 로딩 중...")
    pred_data = load_csv(pred_csv_path)
    print(f"예측: {len(pred_data)}개 이미지")
    
    print("GT CSV 로딩 중...")
    gt_data = load_csv(gt_csv_path)
    print(f"GT: {len(gt_data)}개 이미지")
    
    # 이미지 개수 차이 확인
    pred_only = set(pred_data.keys()) - set(gt_data.keys())
    gt_only = set(gt_data.keys()) - set(pred_data.keys())
    
    if len(pred_only) > 0:
        print(f"⚠️  예측에만 있는 이미지: {len(pred_only)}개 (평가에서 제외됨)")
    if len(gt_only) > 0:
        print(f"⚠️  GT에만 있는 이미지: {len(gt_only)}개 (False Negative로 처리됨)")
    
    # 모든 카테고리 수집
    all_categories = set()
    for detections in gt_data.values():
        for det in detections:
            all_categories.add(det["category_id"])
    all_categories = sorted(list(all_categories))
    
    print(f"\n총 카테고리 수: {len(all_categories)}")
    
    # 중점 카테고리 확인
    if focused_categories:
        focused_in_data = [c for c in all_categories if c in focused_categories]
        print(f"중점 카테고리 중 데이터에 존재: {len(focused_in_data)}개")
    
    # IoU threshold 설정
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    # 한번에 모든 threshold에 대해 평가 (최적화!)
    print(f"\nmAP 계산 중 (IoU threshold: 0.5~0.95, 총 {len(iou_thresholds)}개)...")
    all_results = evaluate_detections_all_thresholds(pred_data, gt_data, iou_thresholds)
    
    # mAP@50 추출
    mAP50 = np.mean(list(all_results[0.5].values()))
    
    # 카테고리별 mAP@75:95 계산
    print("카테고리별 mAP@75:95 계산 중...")
    category_map_75_95 = {}
    
    for category_id in tqdm(all_categories, desc="mAP@75:95"):
        category_aps = []
        
        for iou_thresh in iou_thresholds:
            if category_id in all_results[iou_thresh]:
                category_aps.append(all_results[iou_thresh][category_id])
            else:
                category_aps.append(0.0)
        
        # 해당 카테고리의 mAP@75:95 = 모든 IoU threshold에서의 AP 평균
        category_map_75_95[category_id] = np.mean(category_aps)
    
    # 전체 mAP@75:95 = 모든 카테고리의 mAP@75:95 평균
    mAP_75_95 = np.mean(list(category_map_75_95.values()))
    
    # 중점 카테고리 mAP 계산
    focused_mAP_50 = None
    focused_mAP_75_95 = None
    if focused_categories:
        focused_ap50_values = [all_results[0.5][c] for c in all_categories if c in focused_categories and c in all_results[0.5]]
        focused_map_values = [category_map_75_95[c] for c in all_categories if c in focused_categories and c in category_map_75_95]
        
        if focused_ap50_values:
            focused_mAP_50 = np.mean(focused_ap50_values)
        if focused_map_values:
            focused_mAP_75_95 = np.mean(focused_map_values)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"전체 mAP@50:    {mAP50:.4f}")
    print(f"전체 mAP@75:95: {mAP_75_95:.4f}")
    
    if focused_categories and focused_mAP_50 is not None:
        print("-" * 60)
        print(f"중점 카테고리 mAP@50:    {focused_mAP_50:.4f}")
        print(f"중점 카테고리 mAP@75:95: {focused_mAP_75_95:.4f}")
        print(f"(총 {len(focused_in_data)}개 중점 카테고리)")
    
    print("=" * 60)
    
    # 카테고리별 AP@50 출력 (상위 10개)
    print("\n카테고리별 AP@50 (상위 10개):")
    sorted_cats = sorted(all_results[0.5].items(), key=lambda x: x[1], reverse=True)
    for cat_id, ap in sorted_cats[:10]:
        marker = " 🎯" if focused_categories and cat_id in focused_categories else ""
        print(f"  Category {cat_id}: {ap:.4f}{marker}")
    
    # 중점 카테고리 하위 10개 출력
    if focused_categories:
        print("\n중점 카테고리 AP@50 (하위 10개):")
        focused_sorted = sorted(
            [(c, all_results[0.5].get(c, 0.0)) for c in all_categories if c in focused_categories],
            key=lambda x: x[1]
        )
        for cat_id, ap in focused_sorted[:10]:
            print(f"  Category {cat_id}: {ap:.4f} 🎯")
    
    # mAP@75:95 상세 결과를 txt 파일로 저장
    if output_txt_path:
        output_txt_path = Path(output_txt_path)
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("mAP@75:95 카테고리별 상세 계산 결과\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("계산 방식:\n")
            f.write("1. IoU threshold를 0.5부터 0.95까지 0.05 간격으로 설정 (총 10개)\n")
            f.write("2. 각 IoU threshold에서 모든 카테고리의 AP를 계산\n")
            f.write("3. 각 카테고리의 mAP@75:95 = 해당 카테고리의 10개 AP 평균\n")
            f.write("4. 전체 mAP@75:95 = 모든 카테고리의 mAP@75:95 평균\n\n")
            
            f.write(f"전체 mAP@50:    {mAP50:.6f}\n")
            f.write(f"전체 mAP@75:95: {mAP_75_95:.6f}\n")
            f.write(f"총 카테고리 수: {len(all_categories)}\n")
            
            if focused_categories and focused_mAP_50 is not None:
                f.write("\n" + "-" * 80 + "\n")
                f.write("중점 카테고리 (GCI 기준) 성능\n")
                f.write("-" * 80 + "\n")
                f.write(f"중점 카테고리 mAP@50:    {focused_mAP_50:.6f}\n")
                f.write(f"중점 카테고리 mAP@75:95: {focused_mAP_75_95:.6f}\n")
                f.write(f"중점 카테고리 수: {len(focused_in_data)}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("카테고리별 mAP@75:95 (낮은 점수 순)\n")
            f.write("=" * 80 + "\n\n")
            
            # mAP@75:95 기준으로 오름차순 정렬 (낮은 점수부터)
            sorted_categories = sorted(category_map_75_95.items(), key=lambda x: x[1])
            
            for rank, (cat_id, map_score) in enumerate(sorted_categories, 1):
                marker = " 🎯" if focused_categories and cat_id in focused_categories else ""
                f.write(f"{rank:4d}. Category {cat_id:6d}: mAP@75:95 = {map_score:.6f}{marker}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("카테고리별 mAP@75:95 (높은 점수 순)\n")
            f.write("=" * 80 + "\n\n")
            
            # mAP@75:95 기준으로 내림차순 정렬 (높은 점수부터)
            sorted_categories_desc = sorted(category_map_75_95.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (cat_id, map_score) in enumerate(sorted_categories_desc, 1):
                marker = " 🎯" if focused_categories and cat_id in focused_categories else ""
                f.write(f"{rank:4d}. Category {cat_id:6d}: mAP@75:95 = {map_score:.6f}{marker}\n")
            
            # 중점 카테고리만 따로 정리
            if focused_categories:
                f.write("\n" + "=" * 80 + "\n")
                f.write("중점 카테고리 mAP@75:95 (낮은 점수 순)\n")
                f.write("=" * 80 + "\n\n")
                
                focused_sorted = sorted(
                    [(c, category_map_75_95[c]) for c in all_categories if c in focused_categories and c in category_map_75_95],
                    key=lambda x: x[1]
                )
                
                for rank, (cat_id, map_score) in enumerate(focused_sorted, 1):
                    f.write(f"{rank:4d}. Category {cat_id:6d}: mAP@75:95 = {map_score:.6f} 🎯\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("중점 카테고리 mAP@75:95 (높은 점수 순)\n")
                f.write("=" * 80 + "\n\n")
                
                focused_sorted_desc = sorted(
                    [(c, category_map_75_95[c]) for c in all_categories if c in focused_categories and c in category_map_75_95],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for rank, (cat_id, map_score) in enumerate(focused_sorted_desc, 1):
                    f.write(f"{rank:4d}. Category {cat_id:6d}: mAP@75:95 = {map_score:.6f} 🎯\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("통계 정보\n")
            f.write("=" * 80 + "\n\n")
            
            map_values = list(category_map_75_95.values())
            f.write(f"전체 카테고리:\n")
            f.write(f"  평균 (mAP@75:95):     {np.mean(map_values):.6f}\n")
            f.write(f"  중간값:                {np.median(map_values):.6f}\n")
            f.write(f"  표준편차:              {np.std(map_values):.6f}\n")
            f.write(f"  최소값:                {np.min(map_values):.6f}\n")
            f.write(f"  최대값:                {np.max(map_values):.6f}\n")
            
            if focused_categories and focused_map_values:
                f.write(f"\n중점 카테고리:\n")
                f.write(f"  평균 (mAP@75:95):     {np.mean(focused_map_values):.6f}\n")
                f.write(f"  중간값:                {np.median(focused_map_values):.6f}\n")
                f.write(f"  표준편차:              {np.std(focused_map_values):.6f}\n")
                f.write(f"  최소값:                {np.min(focused_map_values):.6f}\n")
                f.write(f"  최대값:                {np.max(focused_map_values):.6f}\n")
            
            # 점수 구간별 분포
            f.write("\n점수 구간별 카테고리 분포 (전체):\n")
            bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for i in range(len(bins) - 1):
                count = sum(1 for v in map_values if bins[i] <= v < bins[i+1])
                f.write(f"  {bins[i]:.1f} ~ {bins[i+1]:.1f}: {count:4d} 카테고리\n")
            
            if focused_categories and focused_map_values:
                f.write("\n점수 구간별 카테고리 분포 (중점 카테고리):\n")
                for i in range(len(bins) - 1):
                    count = sum(1 for v in focused_map_values if bins[i] <= v < bins[i+1])
                    f.write(f"  {bins[i]:.1f} ~ {bins[i+1]:.1f}: {count:4d} 카테고리\n")
            
        print(f"\n✅ mAP@75:95 상세 결과 저장: {output_txt_path}")
    
    return {
        "mAP@50": mAP50,
        "mAP@75:95": mAP_75_95,
        "focused_mAP@50": focused_mAP_50,
        "focused_mAP@75:95": focused_mAP_75_95,
        "ap_per_category@50": all_results[0.5],
        "map_per_category@75:95": category_map_75_95
    }

if __name__ == "__main__":
    PRED_CSV = r"E:\download\submission_eval\submission_eval(0.888).csv"
    GT_CSV = r"E:\download\submission_eval\ground_truth.csv"
    OUTPUT_TXT = r"E:\download\submission_eval\mAP_75_95_detailed(0.888).txt"
    GCI_JSON = r"E:\download\global_category_index(train_set_56)\global_category_index.json"  # 선택사항
    
    results = evaluate(
        pred_csv_path=PRED_CSV,
        gt_csv_path=GT_CSV,
        output_txt_path=OUTPUT_TXT,
        gci_json_path=GCI_JSON  # None으로 설정하면 중점 카테고리 분석 스킵
    )