#!/usr/bin/env python3
"""
카테고리 ID를 +1 증가시키고 train_annotations와 병합하는 모듈
"""

import json
from typing import Dict, Any
from collections import defaultdict


def increment_category_ids(input_file: str, output_file: str) -> None:
    """
    JSON 파일의 카테고리 ID들을 +1 증가시키고 train_annotations와 병합
    
    Args:
        input_file: 입력 JSON 파일 경로
        output_file: 출력 JSON 파일 경로
    """
    # JSON 파일 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. train_annotations만 있는 카테고리 ID 수집
    train_only_ids = set()
    for cat_id, cat_info in data['category_map'].items():
        datasets = cat_info.get('datasets', {})
        if list(datasets.keys()) == ['train_annotations']:
            train_only_ids.add(int(cat_id))
    
    print(f"Train annotations only 카테고리 수: {len(train_only_ids)}")
    print(f"Train annotations only IDs: {sorted(train_only_ids)[:10]}...")
    
    # 2. 새로운 category_map 생성
    new_category_map = {}
    merge_info = defaultdict(lambda: {'count': 0, 'datasets': {}})
    
    for cat_id_str, cat_info in data['category_map'].items():
        cat_id = int(cat_id_str)
        datasets = cat_info.get('datasets', {})
        
        # train_annotations만 있는 경우 -> 그대로 유지
        if cat_id in train_only_ids:
            new_category_map[cat_id_str] = cat_info.copy()
        else:
            # 다른 데이터셋이 있는 경우 -> ID를 +1
            new_cat_id = cat_id + 1
            new_cat_id_str = str(new_cat_id)
            
            # +1한 ID가 train_only_ids에 있으면 병합 대상
            if new_cat_id in train_only_ids:
                merge_info[new_cat_id]['count'] += cat_info['count']
                for ds_name, ds_count in datasets.items():
                    merge_info[new_cat_id]['datasets'][ds_name] = ds_count
            else:
                # 병합 대상이 아니면 새로운 카테고리로 추가
                new_category_map[new_cat_id_str] = {
                    'category_id': new_cat_id,
                    'category_name': cat_info['category_name'],
                    'dl_idx': new_cat_id_str,
                    'count': cat_info['count'],
                    'datasets': datasets.copy()
                }
    
    # 3. 병합 처리
    print(f"\n병합 대상 카테고리 수: {len(merge_info)}")
    for merge_cat_id, merge_data in merge_info.items():
        merge_cat_id_str = str(merge_cat_id)
        if merge_cat_id_str in new_category_map:
            # count 더하기
            new_category_map[merge_cat_id_str]['count'] += merge_data['count']
            
            # datasets 병합 (train_annotations 뒤에 추가)
            existing_datasets = new_category_map[merge_cat_id_str]['datasets']
            train_count = existing_datasets.get('train_annotations', 0)
            
            # train_annotations를 제외한 나머지 추가
            merged_datasets = {'train_annotations': train_count}
            merged_datasets.update(merge_data['datasets'])
            
            new_category_map[merge_cat_id_str]['datasets'] = merged_datasets
            
            print(f"  병합: ID {merge_cat_id} - {new_category_map[merge_cat_id_str]['category_name']}")
    
    # 4. index_to_id 업데이트
    # 먼저 모든 변환을 계산
    id_mapping = {}  # old_id -> new_id
    for cat_id_str in data['category_map'].keys():
        cat_id = int(cat_id_str)
        if cat_id in train_only_ids:
            id_mapping[cat_id] = cat_id  # train_only는 그대로
        else:
            id_mapping[cat_id] = cat_id + 1  # 나머지는 +1
    
    # index_to_id를 새로 구성 (중복 제거)
    new_index_to_id = {}
    seen_new_ids = set()
    current_new_idx = 0
    
    for idx_str in sorted(data['index_to_id'].keys(), key=int):
        old_cat_id = data['index_to_id'][idx_str]
        new_cat_id = id_mapping.get(old_cat_id, old_cat_id)
        
        # 이미 사용된 new_cat_id는 스킵 (연속 번호의 경우)
        if new_cat_id not in seen_new_ids:
            new_index_to_id[str(current_new_idx)] = new_cat_id
            seen_new_ids.add(new_cat_id)
            current_new_idx += 1
    
    # 5. id_to_index 업데이트 (index_to_id의 역매핑)
    new_id_to_index = {str(cat_id): int(idx) for idx, cat_id in new_index_to_id.items()}
    
    # 6. dataset_stats 업데이트
    new_dataset_stats = {}
    
    for ds_name, ds_data in data['dataset_stats'].items():
        if ds_name == 'train_annotations':
            # train_annotations는 그대로 유지하되, 병합된 카테고리 반영
            new_ds_data = ds_data.copy()
            new_category_counts = new_ds_data['category_counts'].copy()
            
            # 병합된 카테고리의 count 업데이트
            for merge_cat_id, merge_data in merge_info.items():
                merge_cat_id_str = str(merge_cat_id)
                if merge_cat_id_str in new_category_counts:
                    # 병합된 데이터의 총 count를 더함
                    for ds_name_inner, count_inner in merge_data['datasets'].items():
                        if ds_name_inner != 'train_annotations':
                            new_ds_data['total_annotations'] += count_inner
            
            new_ds_data['category_counts'] = new_category_counts
            new_dataset_stats['train_annotations'] = new_ds_data
        else:
            # 다른 데이터셋은 카테고리 ID를 +1
            new_category_counts = {}
            for cat_id_str, count in ds_data['category_counts'].items():
                cat_id = int(cat_id_str)
                new_cat_id = cat_id + 1
                new_category_counts[str(new_cat_id)] = count
            
            new_ds_data = ds_data.copy()
            new_ds_data['category_counts'] = new_category_counts
            new_dataset_stats[ds_name] = new_ds_data
    
    # 7. 최종 데이터 구성
    result = {
        'metadata': data['metadata'].copy(),
        'category_map': new_category_map,
        'index_to_id': new_index_to_id,
        'id_to_index': new_id_to_index,
        'dataset_stats': new_dataset_stats
    }
    
    # 8. 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n처리 완료!")
    print(f"입력 파일: {input_file}")
    print(f"출력 파일: {output_file}")
    print(f"전체 카테고리 수: {len(new_category_map)}")


def main():
    """메인 함수"""
    input_file = 'E:\download\global_category_index_TL2_Only\global_category_index.json'
    output_file = 'E:\download\global_category_index_TL2_Only\global_category_index_updated.json'
    
    increment_category_ids(input_file, output_file)


if __name__ == '__main__':
    main()
