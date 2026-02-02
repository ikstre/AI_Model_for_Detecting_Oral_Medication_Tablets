import json
import os
import re
from glob import glob
from collections import defaultdict

def build_final_dataset_v2(label_root, image_root):
    # 1. 실제 이미지 파일들의 경로를 먼저 다 훑어서 맵(Dict)으로 만듭니다.
    # 파일명(확장자 제외)을 키로, 전체 경로를 값으로 저장합니다.
    print("실제 이미지 위치 스캔 중...")
    image_path_map = {}
    for img_path in glob(os.path.join(image_root, "**", "*.*"), recursive=True):
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 파일명만 추출 (예: K-001...200)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            image_path_map[base_name] = img_path

    print(f"스캔된 실제 이미지 수: {len(image_path_map)}개")

    # 2. JSON 파편 모으기
    image_groups = defaultdict(list)
    json_files = glob(os.path.join(label_root, "**", "*.json"), recursive=True)
    
    for json_path in json_files:
        if "_index" in json_path: continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        img_meta = data['images'][0]
        # JSON 내 파일명에서 확장자 제거 후 키로 사용
        raw_file_name = os.path.splitext(img_meta['file_name'])[0]
        
        image_groups[raw_file_name].append({
            "category_id": int(img_meta['dl_idx']),
            "pill_name": img_meta['dl_name'],
            "bbox": data['annotations'][0]['bbox']
        })

    # 3. 매칭 검증
    verified_dataset = []
    missing_count = 0

    for file_key, pills in image_groups.items():
        if file_key in image_path_map:
            actual_path = image_path_map[file_key]
            # 고유 ID 추출
            unique_ids = re.findall(r'\d+', file_key)
            unique_img_id = int(unique_ids[-3]) if len(unique_ids) >= 3 else 0
            
            verified_dataset.append({
                "image_path": actual_path,
                "file_name": os.path.basename(actual_path),
                "image_id": unique_img_id,
                "objects": pills
            })
        else:
            missing_count += 1

    print(f"\n--- 검증 결과 ---")
    print(f"유효한 이미지-라벨 세트: {len(verified_dataset)}개")
    print(f"이미지 파일 누락: {missing_count}개")
    
    return verified_dataset

# --- 경로 설정 (압축 해제된 폴더 경로여야 합니다) ---
LABEL_DIR = r"E:\download\label\경구약제조합_5000종\TL_1_조합"
IMAGE_DIR = r"E:\download\image\경구약제조합_5000종\TS_1_조합" 

final_data = build_final_dataset_v2(LABEL_DIR, IMAGE_DIR)


# import torch
# from torch.utils.data import Dataset
# import cv2

# class PillDetectionDataset(Dataset):
#     def __init__(self, verified_data, transforms=None):
#         self.data = verified_data
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
        
#         # 1. 이미지 읽기
#         image = cv2.imread(item['image_path'])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # 2. 정답(Label) 정보 정리
#         boxes = []
#         labels = []
#         for obj in item['objects']:
#             # bbox: [x, y, w, h] -> [xmin, ymin, xmax, ymax] (Pascal VOC 포맷 변환)
#             x, y, w, h = obj['bbox']
#             boxes.append([x, y, x + w, y + h])
#             labels.append(obj['category_id'])
            
#         target = {
#             "boxes": torch.as_tensor(boxes, dtype=torch.float32),
#             "labels": torch.as_tensor(labels, dtype=torch.int64),
#             "image_id": torch.tensor([item['image_id']])
#         }
        
#         if self.transforms:
#             # (선택) Augmentation 적용 시
#             image = self.transforms(image)
            
#         return image, target