import json
import csv
from pathlib import Path
from tqdm import tqdm

def validate_annotation(annotation, json_path, file_name, image_width=976, image_height=1280):
    """
    Annotation 데이터 검증
    
    Returns:
        dict: {'status': 'valid'/'invalid', 'error_type': str, 'data': dict}
    """
    # bbox 검증
    bbox = annotation.get('bbox')
    if not bbox or len(bbox) != 4:
        return {
            'status': 'invalid',
            'error_type': 'missing_bbox',
            'data': {'file_name': file_name, 'json_path': str(json_path)},
        }

    try:
        x, y, w, h = map(float, bbox)
    except Exception:
        return {
            'status': 'invalid',
            'error_type': 'invalid_bbox_format',
            'data': {'file_name': file_name, 'bbox': bbox, 'json_path': str(json_path)},
        }

    # 음수/0 크기 방지
    if w <= 0 or h <= 0:
        return {
            'status': 'invalid',
            'error_type': 'invalid_bbox_size',
            'data': {'file_name': file_name, 'bbox': [x, y, w, h], 'json_path': str(json_path)},
        }

    # bbox 면적 불일치 제거
    area = annotation.get('area')
    if area is not None:
        try:
            area_val = float(area)
            wh = w * h
            # 정수 반올림/부동소수 오차 감안: 1px^2 또는 0.5% 중 큰 값 허용
            tol = max(1.0, wh * 0.005)
            if abs(area_val - wh) > tol:
                return {
                    'status': 'invalid',
                    'error_type': 'area_mismatch',
                    'data': {
                        'file_name': file_name,
                        'bbox': [x, y, w, h],
                        'area': area_val,
                        'w*h': wh,
                        'tol': tol,
                        'json_path': str(json_path),
                    },
                }
        except Exception:
            return {
                'status': 'invalid',
                'error_type': 'invalid_area',
                'data': {'file_name': file_name, 'area': area, 'json_path': str(json_path)},
            }

    # bbox 경계 검증
    if (
        x < 0
        or y < 0
        or x + w > image_width
        or y + h > image_height
    ):
        return {
            'status': 'invalid',
            'error_type': 'out_of_bounds',
            'data': {
                'file_name': file_name,
                'bbox': [x, y, w, h],
                'overflow_x': max(0.0, x + w - image_width),
                'overflow_y': max(0.0, y + h - image_height),
                'json_path': str(json_path),
            },
        }
    
    return {
        'status': 'valid',
        'bbox': [x, y, w, h]
    }

def generate_gt_csv(label_eval_dir: str, out_csv_path: str, image_width: int = 976, image_height: int = 1280):
    """
    label_eval 폴더를 순회하며 Ground Truth CSV 파일 생성
    
    Args:
        label_eval_dir: label_eval 폴더 경로
        out_csv_path: 출력 CSV 파일 경로
        image_width: 이미지 너비 (기본값: 976)
        image_height: 이미지 높이 (기본값: 1280)
    """
    label_eval_dir = Path(label_eval_dir)
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 모든 JSON 파일 찾기
    json_files = list(label_eval_dir.rglob("*.json"))
    json_files.sort()
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {label_eval_dir}")
    
    print(f"총 {len(json_files)}개의 JSON 파일 발견")
    
    # 검증 통계
    validation_stats = {
        'missing_bbox': 0,
        'invalid_bbox_format': 0,
        'invalid_bbox_size': 0,
        'area_mismatch': 0,
        'invalid_area': 0,
        'out_of_bounds': 0,
        'valid': 0,
    }
    
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "annotation_id", "image_id", "category_id",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h"
        ])
        
        ann_id = 1
        total_annotations = 0
        
        for json_path in tqdm(json_files, desc="GT CSV 생성"):
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                
                # dl_idx에서 category_id 추출 (+1)
                if "images" in data and len(data["images"]) > 0:
                    dl_idx = data["images"][0].get("dl_idx")
                    file_name = data["images"][0].get("file_name", "unknown")
                    
                    if dl_idx:
                        category_id = int(dl_idx) + 1
                    else:
                        print(f"Warning: dl_idx not found in {json_path.name}")
                        continue
                else:
                    print(f"Warning: images field not found in {json_path.name}")
                    continue
                
                # image_id: JSON 파일명에서 .json 제거
                image_id = json_path.stem
                
                # annotations에서 bbox 추출 및 검증
                if "annotations" in data:
                    for ann in data["annotations"]:
                        # bbox 검증
                        validation_result = validate_annotation(
                            ann, json_path, file_name, image_width, image_height
                        )
                        
                        if validation_result['status'] == 'invalid':
                            validation_stats[validation_result['error_type']] += 1
                            continue
                        
                        # 검증 통과한 bbox 저장
                        validation_stats['valid'] += 1
                        x, y, w, h = validation_result['bbox']
                        
                        writer.writerow([
                            ann_id,
                            image_id,
                            category_id,
                            int(x), int(y), int(w), int(h)
                        ])
                        ann_id += 1
                        total_annotations += 1
                else:
                    print(f"Warning: annotations field not found in {json_path.name}")
                    
            except Exception as e:
                print(f"Error processing {json_path.name}: {e}")
                continue
    
    print("✅ Ground Truth CSV 생성 완료")
    print(f"- 저장: {out_csv_path}")
    print(f"- JSON 파일: {len(json_files)}")
    print(f"- 총 annotations: {total_annotations}")
    print("\n검증 통계:")
    print(f"  ✓ 유효: {validation_stats['valid']}")
    print(f"  ✗ bbox 누락: {validation_stats['missing_bbox']}")
    print(f"  ✗ bbox 형식 오류: {validation_stats['invalid_bbox_format']}")
    print(f"  ✗ bbox 크기 오류 (w≤0 or h≤0): {validation_stats['invalid_bbox_size']}")
    print(f"  ✗ 면적 불일치: {validation_stats['area_mismatch']}")
    print(f"  ✗ 면적 값 오류: {validation_stats['invalid_area']}")
    print(f"  ✗ 이미지 경계 초과: {validation_stats['out_of_bounds']}")
    total_invalid = sum(v for k, v in validation_stats.items() if k != 'valid')
    print(f"  총 제외된 annotations: {total_invalid}")


if __name__ == "__main__":
    LABEL_EVAL_DIR = r"E:\download\label_eval"
    OUT_GT_CSV = r"E:\download\submission_eval\ground_truth.csv"
    
    generate_gt_csv(
        label_eval_dir=LABEL_EVAL_DIR,
        out_csv_path=OUT_GT_CSV
    )