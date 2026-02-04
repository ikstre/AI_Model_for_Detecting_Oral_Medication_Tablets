import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
import os
import json
import re
from glob import glob


def create_training_dataset(
    label_root: str,
    image_root: str,
    output_dir: str,
    dataset_name: str,
    dataset_type: str = 'combination',  # 'combination' or 'single'
    exclude_index_images: bool = True
) -> Dict: # Dict를 반환하는 함수임.
    """
    검증된 데이터를 기반으로 학습용 데이터셋 생성
    
    Args:
        label_root: JSON annotation 루트 디렉토리
        image_root: 이미지 루트 디렉토리
        output_dir: 데이터셋을 저장할 디렉토리
        dataset_name: 데이터셋 이름 (예: 'TS_8_combination', 'TS_10_single')
        dataset_type: 'combination' 또는 'single'
        exclude_index_images: _index.png 파일 제외 여부
    
    Returns:
        dict: 데이터셋 생성 결과
    """
    
    print("=" * 80)
    print(f"데이터셋 생성: {dataset_name}")
    print("=" * 80)
    
    # 출력 디렉토리 생성
    dataset_output = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output, exist_ok=True)
    
    # 1. JSON 파일 수집
    print("\n[1단계] JSON 파일 수집 및 파싱...")
    json_files = glob(os.path.join(label_root, "**", "*.json"), recursive=True)
    print(f"   └─ 발견된 JSON 파일: {len(json_files)}개")
    
    # 2. 데이터 파싱 및 검증
    print("\n[2단계] 데이터 파싱 및 검증...")
    
    valid_records = []
    invalid_records = {
        'missing_image': [],
        'missing_bbox': [],
        'invalid_category': [],
        'out_of_bounds': [],
        'is_index_file': []
    }
    
    IMG_W, IMG_H = 976, 1280
    
    for idx, json_path in enumerate(json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_meta = data['images'][0]
            annotation = data['annotations'][0]
            
            file_name = img_meta['file_name']
            
            # _index 파일 필터링
            if exclude_index_images and '_index' in file_name:
                invalid_records['is_index_file'].append({
                    'file_name': file_name,
                    'json_path': json_path
                })
                continue
            
            # bbox 결측값 확인
            bbox = annotation.get('bbox')
            if bbox is None or len(bbox) != 4:
                invalid_records['missing_bbox'].append({
                    'file_name': file_name,
                    'json_path': json_path,
                    'bbox': bbox
                })
                continue
            
            # bbox 경계 이탈 확인
            x, y, w, h = bbox
            if x + w > IMG_W or y + h > IMG_H or x < 0 or y < 0:
                invalid_records['out_of_bounds'].append({
                    'file_name': file_name,
                    'bbox': bbox,
                    'overflow_x': max(0, x + w - IMG_W),
                    'overflow_y': max(0, y + h - IMG_H)
                })
                continue
            
            # Category ID 추출 (dl_idx 우선 사용)
            dl_idx = img_meta.get('dl_idx')
            category_id = annotation.get('category_id')
            
            # dl_idx가 있으면 우선 사용 (추가 데이터)
            if dl_idx and str(dl_idx).strip():
                final_category_id = int(dl_idx)
            elif category_id:
                final_category_id = int(category_id)
            else:
                invalid_records['invalid_category'].append({
                    'file_name': file_name,
                    'dl_idx': dl_idx,
                    'category_id': category_id
                })
                continue
            
            # 이미지 파일 경로 찾기
            image_path = find_image_file(image_root, file_name)
            
            if not image_path or not os.path.exists(image_path):
                invalid_records['missing_image'].append({
                    'file_name': file_name,
                    'expected_path': image_path
                })
                continue
            
            # 유효한 레코드 저장
            valid_records.append({
                'file_name': file_name,
                'image_path': image_path,
                'width': img_meta['width'],
                'height': img_meta['height'],
                'category_id': final_category_id,
                'category_name': img_meta.get('dl_name', 'Unknown'),
                'bbox_x': bbox[0],
                'bbox_y': bbox[1],
                'bbox_w': bbox[2],
                'bbox_h': bbox[3],
                'area': annotation.get('area', bbox[2] * bbox[3]),
                'drug_N': img_meta.get('drug_N', ''),
                'json_path': json_path
            })
            
        except Exception as e:
            invalid_records['missing_bbox'].append({
                'file_name': 'unknown',
                'json_path': json_path,
                'error': str(e)
            })
        
        if (idx + 1) % 500 == 0:
            print(f"   진행중... {idx + 1}/{len(json_files)} 파일 처리 완료")
    
    print(f"\n   ✅ 유효 레코드: {len(valid_records)}개")
    print(f"   ❌ 무효 레코드:")
    print(f"      • _index 파일: {len(invalid_records['is_index_file'])}개")
    print(f"      • 이미지 없음: {len(invalid_records['missing_image'])}개")
    print(f"      • bbox 결측: {len(invalid_records['missing_bbox'])}개")
    print(f"      • bbox 이탈: {len(invalid_records['out_of_bounds'])}개")
    print(f"      • category 오류: {len(invalid_records['invalid_category'])}개")
    
    # 3. DataFrame 생성
    print("\n[3단계] DataFrame 생성...")
    df = pd.DataFrame(valid_records)

    # 3-1. 중복 bbox 검증 및 제거
    print("\n[3-1단계] 중복 bbox 검증 중...")

    # bbox tuple 컬럼 생성
    df['bbox_tuple'] = list(zip(
        df['bbox_x'], df['bbox_y'], df['bbox_w'], df['bbox_h']
    ))

    # file_name + bbox 기준 중복 탐색 (모든 중복 포함)
    duplicate_mask = df.duplicated(
        subset=['file_name', 'bbox_tuple'],
        keep=False   # ★ 핵심: 중복된 모든 행 True
    )

    duplicates = df[duplicate_mask]

    duplicate_details = []

    if not duplicates.empty:
        duplicate_groups = duplicates.groupby(['file_name', 'bbox_tuple'])

        for (filename, bbox_tuple), group in duplicate_groups:
            drugs = group[['drug_N', 'category_name', 'json_path']].to_dict('records')

            duplicate_details.append({
                'file_name': filename,
                'bbox': list(bbox_tuple),
                'count': len(group),
                'records': drugs
            })

    print(f"   └─ 중복 bbox 발견: {len(duplicate_details)}개 케이스")

    # 중복 bbox 전체 제거 (하나도 남기지 않음)
    before_cnt = len(df)
    df = df[~duplicate_mask]
    after_cnt = len(df)

    print(f"   └─ 중복 bbox 전체 제거 완료: {before_cnt - after_cnt}개 삭제됨")

    # 임시 컬럼 제거
    df = df.drop(columns=['bbox_tuple'])
 
    # 4. 카테고리 정보 생성
    print("\n[4단계] 카테고리 정보 생성...")
    category_map = df[['category_id', 'category_name']].drop_duplicates()
    category_map = category_map.sort_values('category_id').reset_index(drop=True)
    
    print(f"   └─ 전체 약 종류: {len(category_map)}개")
    
    # 5. 데이터셋 파일 저장
    print("\n[5단계] 데이터셋 파일 저장...")
    
    # 5-1. 전체 데이터 CSV
    dataset_csv_path = os.path.join(dataset_output, f"{dataset_name}_dataset.csv")
    df.to_csv(dataset_csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 데이터셋 CSV: {dataset_csv_path}")
    
    # 5-2. 카테고리 맵 CSV
    category_csv_path = os.path.join(dataset_output, f"{dataset_name}_categories.csv")
    category_map.to_csv(category_csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 카테고리 CSV: {category_csv_path}")
    
    # 5-3. YOLO 형식 (선택적)
    yolo_output = os.path.join(dataset_output, "yolo_format")
    create_yolo_format(df, category_map, yolo_output)
    
    # 5-4. 무효 레코드 보고서
    invalid_report_path = os.path.join(dataset_output, f"{dataset_name}_invalid_records.json")
    with open(invalid_report_path, 'w', encoding='utf-8') as f:
        json.dump(invalid_records, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 무효 레코드 보고서: {invalid_report_path}")

    # 5-5. 중복 bbox 보고서
    duplicate_report_path = os.path.join(dataset_output, f"{dataset_name}_duplicate_bboxes.json")

    with open(duplicate_report_path, 'w', encoding='utf-8') as f:
        json.dump(duplicate_details, f, ensure_ascii=False, indent=2)

    print(f"   ✅ 중복 bbox 보고서: {duplicate_report_path}")
    
    # 6. 통계 정보
    stats = {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'total_valid_records': len(df),
        'total_images': df['file_name'].nunique(),
        'total_categories': len(category_map),
        'invalid_counts': {
            **{k: len(v) for k, v in invalid_records.items()},
            'duplicate_bbox_removed': before_cnt - after_cnt
        },
        'output_dir': dataset_output,
        'files': {
            'dataset_csv': dataset_csv_path,
            'category_csv': category_csv_path,
            'invalid_report': invalid_report_path
        }
    }
    print_dataset_stats(stats)
    
    return stats


def find_image_file(image_root: str, file_name: str) -> str:
    """
    이미지 파일 경로 찾기 (폴더 구조에 따라 재귀 탐색)
    
    Args:
        image_root: 이미지 루트 디렉토리
        file_name: 찾을 파일명
    
    Returns:
        str: 이미지 전체 경로 (없으면 None)
    """
    
    # 1. 직접 경로 (기존 train set 형식)
    direct_path = os.path.join(image_root, file_name)
    if os.path.exists(direct_path):
        return direct_path
    
    # 2. 조합 폴더 내부 (추가 데이터 형식)
    # 파일명에서 조합 ID 추출: K-001900-010224-016551-029345_...
    base_name = os.path.splitext(file_name)[0]
    pattern = r'(K-[\d-]+)'
    match = re.match(pattern, base_name)
    
    if match:
        combination_id = match.group(1)
        # 조합 폴더 안에 이미지가 있는 경우
        folder_path = os.path.join(image_root, combination_id, file_name)
        if os.path.exists(folder_path):
            return folder_path
    
    # 3. 단일 약 폴더 내부
    # K-022713_0_2_1_1_90_200_200.png → K-022713 폴더
    drug_id_match = re.match(r'(K-\d+)', base_name)
    if drug_id_match:
        drug_id = drug_id_match.group(1)
        single_path = os.path.join(image_root, drug_id, file_name)
        if os.path.exists(single_path):
            return single_path
    
    # 4. 재귀 탐색 (마지막 수단)
    for root, dirs, files in os.walk(image_root):
        if file_name in files:
            return os.path.join(root, file_name)
    
    return None


def create_yolo_format(df: pd.DataFrame, category_map: pd.DataFrame, output_dir: str):
    """
    YOLO 형식으로 데이터셋 생성
    
    Args:
        df: 데이터셋 DataFrame
        category_map: 카테고리 맵
        output_dir: YOLO 출력 디렉토리
    """
    
    print(f"\n   [YOLO 형식 생성]")
    
    # YOLO 디렉토리 구조 생성
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Category ID를 0부터 시작하는 인덱스로 매핑
    category_to_idx = {row['category_id']: idx for idx, row in category_map.iterrows()}
    
    # 이미지별로 그룹화
    grouped = df.groupby('file_name')
    
    for file_name, group in grouped:
        # 이미지 복사
        src_image = group.iloc[0]['image_path']
        dst_image = os.path.join(images_dir, file_name)
        
        if not os.path.exists(dst_image):
            shutil.copy2(src_image, dst_image)
        
        # YOLO 라벨 파일 생성
        label_file = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                # YOLO 형식: class_id center_x center_y width height (정규화)
                img_w, img_h = row['width'], row['height']
                x, y, w, h = row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']
                
                center_x = (x + w / 2) / img_w
                center_y = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                class_idx = category_to_idx[row['category_id']]
                
                f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    
    # data.yaml 생성
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n\n")
        f.write(f"nc: {len(category_map)}\n")
        f.write(f"names:\n")
        for idx, row in category_map.iterrows():
            f.write(f"  {idx}: '{row['category_name']}'\n")
    
    print(f"      ✅ YOLO 이미지: {len(grouped)}개")
    print(f"      ✅ YOLO 라벨: {len(grouped)}개")
    print(f"      ✅ data.yaml: {yaml_path}")


def print_dataset_stats(stats: Dict):
    """데이터셋 통계 출력"""
    
    print("\n" + "=" * 80)
    print("데이터셋 생성 완료")
    print("=" * 80)
    
    print(f"\n📊 {stats['dataset_name']} 통계:")
    print(f"   • 데이터셋 타입: {stats['dataset_type']}")
    print(f"   • 유효 레코드: {stats['total_valid_records']:,}개")
    print(f"   • 고유 이미지: {stats['total_images']:,}개")
    print(f"   • 약 종류: {stats['total_categories']}개")
    
    print(f"\n❌ 제외된 레코드:")
    for reason, count in stats['invalid_counts'].items():
        if count > 0:
            print(f"   • {reason}: {count}개")
    
    print(f"\n💾 출력 파일:")
    print(f"   • 데이터셋: {stats['files']['dataset_csv']}")
    print(f"   • 카테고리: {stats['files']['category_csv']}")
    print(f"   • 무효 보고서: {stats['files']['invalid_report']}")
    
    print("\n" + "=" * 80)


# 사용 예시

if __name__ == "__main__":
    
    # # ===== 기존 Train Set =====
    # print("\n### 기존 Train Set 데이터셋 생성 ###\n")
    # original_stats = create_training_dataset(
    #     label_root=r"E:\download\sprint_ai_project1_data\train_annotations",
    #     image_root=r"E:\download\sprint_ai_project1_data\train_images",
    #     output_dir=r"E:\download\datasets",
    #     dataset_name="original_trainset",
    #     dataset_type="combination",
    #     exclude_index_images=False
    # )
    
    # # ===== 추가 데이터 - 조합 =====
    # print("\n### TS_1 조합 데이터셋 생성 ###\n")
    # ts8_stats = create_training_dataset(
    #     label_root=r"E:\download\label\경구약제조합_5000종\TL_1_조합",
    #     image_root=r"E:\download\image\경구약제조합_5000종\TS_1_조합",
    #     output_dir=r"E:\download\datasets",
    #     dataset_name="TS_8_combination",
    #     dataset_type="combination",
    #     exclude_index_images=True
    # )
    
    # ===== 추가 데이터 - 단일 =====
    print("\n### TS_81 단일 데이터셋 생성 ###\n")
    ts10_stats = create_training_dataset(
        label_root=r"E:\download\label\단일경구약제_5000종\TL_81_단일",
        image_root=r"E:\download\image\단일경구약제_5000종\TS_81_단일",
        output_dir=r"E:\download\datasets",
        dataset_name="TS_10_single",
        dataset_type="single",
        exclude_index_images=False
    )