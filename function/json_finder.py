import os
import re
from glob import glob
from collections import defaultdict
from pathlib import Path

import json
import pandas as pd


def validate_pill_dataset(label_root, image_root):
    """
    알약 데이터셋의 JSON 파일 누락을 검증하는 함수
    
    Args:
        label_root: JSON 파일들이 있는 루트 디렉토리
        image_root: 이미지 파일들이 있는 루트 디렉토리
    
    Returns:
        dict: 검증 결과 딕셔너리
    """
    
    print("=" * 80)
    print("알약 데이터셋 JSON 파일 검증 시작")
    print("=" * 80)
    
    # 실제 JSON 폴더 구조 파악 (디버깅용)
    print("\n[0단계] 실제 JSON 폴더 구조 분석 중...")
    sample_combinations = []
    for item in os.listdir(label_root):
        item_path = os.path.join(label_root, item)
        if os.path.isdir(item_path) and item.startswith('K-'):
            sample_combinations.append(item)
            if len(sample_combinations) >= 3:
                break
    
    if sample_combinations:
        print(f"   샘플 조합 폴더: {sample_combinations[0]}")
        sample_path = os.path.join(label_root, sample_combinations[0])
        subdirs = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]
        if subdirs:
            print(f"   └─ 하위 약 폴더 예시: {subdirs[:3]}")
    
    # 이미지 파일 스캔
    print("\n[1단계] 이미지 파일 스캔 중...")
    image_files = []
    for img_path in glob(os.path.join(image_root, "**", "*.*"), recursive=True):
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(img_path)
    
    print(f"   └─ 발견된 이미지 파일: {len(image_files)}개")
    
    # 이미지 파일명 파싱
    print("\n[2단계] 이미지 파일명 분석 중...")
    image_info = []
    
    # 파일명 패턴: K-001900-016548-019607-029451_0_2_0_2_70_000_200.jpg
    pattern = r'(K-[\d-]+)_.*?_(\d+)_\d+_\d+$'
    
    parse_errors = 0
    for img_path in image_files:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        match = re.match(pattern, filename)
        
        if match:
            combination_id = match.group(1)  # K-003544-004543-016548-027993
            angle = match.group(2)  # 70, 75, 90 등
            
            # 조합 ID에서 개별 약 ID 추출 (K- 접두사 포함)
            parts = combination_id.split('-')
            pill_ids = ['K-' + p for p in parts[1:]]  # 모든 부분에 K- 붙이기
            # 결과: ['K-003544', 'K-004543', 'K-016548', 'K-027993']

            image_info.append({
                'image_path': img_path,
                'filename': filename,
                'combination_id': combination_id,
                'angle': angle,
                'pill_ids': pill_ids
            })
        else:
            parse_errors += 1
    
    print(f"   └─ 파싱 완료: {len(image_info)}개 이미지")
    if parse_errors > 0:
        print(f"   └─ 파싱 실패: {parse_errors}개 (패턴 불일치)")
    
    # 실제 JSON 폴더 구조 확인
    print("\n[3단계] 실제 JSON 폴더 패턴 자동 감지 중...")
    
    # 첫 번째 이미지로 폴더 패턴 확인
    folder_suffix = ""
    pill_folder_prefix = ""
    
    if image_info:
        test_img = image_info[0]
        combination_id = test_img['combination_id']
        
        # 가능한 조합 폴더명 패턴 확인
        possible_patterns = [
            combination_id + "_json",
            combination_id,
            combination_id + "_labels"
        ]
        
        for pattern in possible_patterns:
            test_path = os.path.join(label_root, pattern)
            if os.path.exists(test_path):
                folder_suffix = pattern.replace(combination_id, "")
                print(f"   └─ 조합 폴더 패턴 감지: '{combination_id}{folder_suffix}'")
                
                # 하위 약 폴더 패턴 확인
                subdirs = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
                if subdirs:
                    # K-003544 형태인지 003544 형태인지 확인
                    if subdirs[0].startswith('K-'):
                        pill_folder_prefix = ""
                        print(f"   └─ 약 폴더 패턴 감지: 'K-' 접두사 포함")
                    else:
                        pill_folder_prefix = "K-"
                        print(f"   └─ 약 폴더 패턴 감지: 숫자만 사용")
                break
    
    # JSON 파일 존재 여부 확인
    print("\n[4단계] JSON 파일 존재 여부 확인 중...")
    
    results = {
        'total_images': len(image_info),
        'total_expected_jsons': 0,
        'total_existing_jsons': 0,
        'total_missing_jsons': 0,
        'missing_details': [],
        'image_summary': [],
        'folder_pattern': {
            'combination_suffix': folder_suffix,
            'pill_prefix': pill_folder_prefix
        }
    }
    
    for idx, img in enumerate(image_info):
        combination_id = img['combination_id']
        angle = img['angle']
        pill_ids = img['pill_ids']
        filename = img['filename']
        
        expected_json_count = len(pill_ids)
        existing_json_count = 0
        missing_jsons = []
        
        # 조합 폴더 경로
        combination_folder = os.path.join(label_root, combination_id + folder_suffix)
        
        # 각 약 ID별로 JSON 파일 존재 여부 확인
        for pill_id in pill_ids:
            # pill_id가 이미 K-로 시작하면 그대로, 아니면 추가
            if pill_id.startswith('K-'):
                folder_name = pill_id if not pill_folder_prefix else pill_id.replace('K-', '')
            else:
                folder_name = pill_folder_prefix + pill_id if pill_folder_prefix else pill_id
            
            json_folder = os.path.join(combination_folder, folder_name)
            json_filename = f"{filename}.json"
            json_path = os.path.join(json_folder, json_filename)
            
            if os.path.exists(json_path):
                existing_json_count += 1
            else:
                missing_jsons.append({
                    'pill_id': pill_id,
                    'expected_path': json_path,
                    'folder_exists': os.path.exists(json_folder),
                    'combination_folder_exists': os.path.exists(combination_folder)
                })
        
        # 통계 업데이트
        results['total_expected_jsons'] += expected_json_count
        results['total_existing_jsons'] += existing_json_count
        results['total_missing_jsons'] += len(missing_jsons)
        
        # 이미지별 요약
        image_summary = {
            'image_file': os.path.basename(img['image_path']),
            'combination_id': combination_id,
            'angle': angle,
            'expected_jsons': expected_json_count,
            'existing_jsons': existing_json_count,
            'missing_jsons': len(missing_jsons)
        }
        results['image_summary'].append(image_summary)
        
        # 누락된 JSON이 있으면 상세 정보 저장
        if missing_jsons:
            results['missing_details'].append({
                'image_file': os.path.basename(img['image_path']),
                'image_path': img['image_path'],
                'combination_id': combination_id,
                'angle': angle,
                'missing_jsons': missing_jsons
            })
        
        # 진행상황 표시 (100개마다)
        if (idx + 1) % 100 == 0:
            print(f"   진행중... {idx + 1}/{len(image_info)} 이미지 처리 완료")
    
    # 결과 출력
    print_validation_results(results)
    
    return results


def print_validation_results(results):
    """검증 결과를 보기 좋게 출력"""
    
    print("\n" + "=" * 80)
    print("검증 결과 요약")
    print("=" * 80)
    
    print(f"\n 전체 통계:")
    print(f"   • 총 이미지 파일 수: {results['total_images']:,}개")
    print(f"   • 예상 JSON 파일 수: {results['total_expected_jsons']:,}개")
    print(f"   • 실제 존재 JSON 수: {results['total_existing_jsons']:,}개")
    print(f"   • 누락된 JSON 수: {results['total_missing_jsons']:,}개")
    
    if results['total_expected_jsons'] > 0:
        if results['total_missing_jsons'] > 0:
            coverage = (results['total_existing_jsons'] / results['total_expected_jsons']) * 100
            print(f"   • JSON 완성도: {coverage:.2f}%")
        else:
            print(f"   • JSON 완성도: 100% ")
    
    # 감지된 폴더 패턴 정보
    if 'folder_pattern' in results:
        print(f"\n🔍 감지된 폴더 구조:")
        print(f"   • 조합 폴더 접미사: '{results['folder_pattern']['combination_suffix']}'")
        print(f"   • 약 폴더 접두사: '{results['folder_pattern']['pill_prefix']}'")
    
    # 누락된 파일이 있는 이미지 찾기
    incomplete_images = [img for img in results['image_summary'] if img['missing_jsons'] > 0]
    
    if incomplete_images:
        print(f"\n  JSON이 불완전한 이미지: {len(incomplete_images)}개")
        print("\n상위 10개 예시:")
        print(f"{'이미지 파일명':<60} {'각도':<6} {'존재/예상':<10}")
        print("-" * 80)
        
        for img in incomplete_images[:10]:
            status = f"{img['existing_jsons']}/{img['expected_jsons']}"
            print(f"{img['image_file']:<60} {img['angle']:<6} {status:<10}")
    else:
        print(f"\n 모든 이미지의 JSON 파일이 완전합니다!")
    
    # 누락된 JSON 파일 상세 정보
    if results['missing_details']:
        print(f"\n 누락된 JSON 파일 상세 내역 (상위 20개):")
        print("=" * 80)
        
        for i, detail in enumerate(results['missing_details'][:20], 1):
            print(f"\n[{i}] 이미지: {detail['image_file']}")
            print(f"    조합 ID: {detail['combination_id']}")
            print(f"    각도: {detail['angle']}")
            print(f"    누락된 JSON {len(detail['missing_jsons'])}개:")
            
            for missing in detail['missing_jsons']:
                folder_status = " 폴더 있음" if missing['folder_exists'] else "! 폴더 없음"
                combo_status = " (조합 폴더도 없음)" if not missing['combination_folder_exists'] else ""
                print(f"      • 약 ID: {missing['pill_id']} {folder_status}{combo_status}")
                print(f"        경로: {missing['expected_path']}")
        
        if len(results['missing_details']) > 20:
            print(f"\n... 외 {len(results['missing_details']) - 20}개 더 있음")
    
    print("\n" + "=" * 80)


def save_validation_report(results, output_path="validation_report.txt"):
    """검증 결과를 파일로 저장"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("알약 데이터셋 JSON 파일 검증 보고서\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"총 이미지 파일 수: {results['total_images']:,}개\n")
        f.write(f"예상 JSON 파일 수: {results['total_expected_jsons']:,}개\n")
        f.write(f"실제 존재 JSON 수: {results['total_existing_jsons']:,}개\n")
        f.write(f"누락된 JSON 수: {results['total_missing_jsons']:,}개\n\n")
        
        if 'folder_pattern' in results:
            f.write(f"감지된 조합 폴더 접미사: '{results['folder_pattern']['combination_suffix']}'\n")
            f.write(f"감지된 약 폴더 접두사: '{results['folder_pattern']['pill_prefix']}'\n\n")
        
        if results['missing_details']:
            f.write("=" * 80 + "\n")
            f.write("누락된 JSON 파일 전체 목록\n")
            f.write("=" * 80 + "\n\n")
            
            for i, detail in enumerate(results['missing_details'], 1):
                f.write(f"[{i}] 이미지: {detail['image_file']}\n")
                f.write(f"    이미지 경로: {detail['image_path']}\n")
                f.write(f"    조합 ID: {detail['combination_id']}\n")
                f.write(f"    각도: {detail['angle']}\n")
                f.write(f"    누락된 JSON:\n")
                
                for missing in detail['missing_jsons']:
                    f.write(f"      • 약 ID: {missing['pill_id']}\n")
                    f.write(f"        경로: {missing['expected_path']}\n")
                    f.write(f"        폴더 존재: {'예' if missing['folder_exists'] else '아니오'}\n")
                    f.write(f"        조합 폴더 존재: {'예' if missing['combination_folder_exists'] else '아니오'}\n")
                f.write("\n")
        else:
            f.write("모든 이미지에 대한 JSON 파일이 완전합니다!\n")
    
    print(f"\n 상세 보고서 저장 완료: {output_path}")


def validate_json_content_quality(label_root, image_root):
    """
    JSON 파일 내용의 품질을 검증하는 함수
    1. 같은 이미지, 같은 bbox에 여러 약이 있는 중복 케이스
    2. bbox가 이미지 경계를 벗어나는 케이스
    
    Args:
        label_root: JSON 파일들이 있는 루트 디렉토리
        image_root: 이미지 파일들이 있는 루트 디렉토리
    
    Returns:
        dict: 품질 검증 결과
    """
    
    print("\n" + "=" * 80)
    print("JSON 파일 내용 품질 검증 시작")
    print("=" * 80)
    
    # 이미지 크기 정의
    IMG_W, IMG_H = 976, 1280
    
    # 모든 JSON 파일 수집
    print("\n[1단계] JSON 파일 수집 중...")
    json_files = glob(os.path.join(label_root, "**", "*.json"), recursive=True)
    print(f"   └─ 발견된 JSON 파일: {len(json_files)}개")
    
    # 데이터 수집
    print("\n[2단계] JSON 데이터 파싱 중...")
    all_records = []
    parse_errors = []
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_meta = data['images'][0]
            annotation = data['annotations'][0]
            
            record = {
                'file_name': img_meta['file_name'],
                'width': img_meta['width'],
                'height': img_meta['height'],
                'drug_N': img_meta['drug_N'],
                'dl_idx': img_meta['dl_idx'],
                'dl_name': img_meta['dl_name'],
                'bbox': annotation['bbox'],
                'category_id': annotation['category_id'],
                'json_path': json_path
            }
            all_records.append(record)
            
        except Exception as e:
            parse_errors.append({'path': json_path, 'error': str(e)})
    
    print(f"   └─ 파싱 완료: {len(all_records)}개")
    if parse_errors:
        print(f"   └─ 파싱 오류: {len(parse_errors)}개")
    
    # DataFrame으로 변환
    import pandas as pd
    df = pd.DataFrame(all_records)
    
    # 3. 중복 bbox 검증
    print("\n[3단계] 중복 bbox 검증 중...")
    
    # bbox를 tuple로 변환하여 그룹핑
    df['bbox_tuple'] = df['bbox'].apply(tuple)
    
    # 같은 파일명 + 같은 bbox인 케이스 찾기
    duplicates = df.groupby(['file_name', 'bbox_tuple']).filter(lambda x: len(x) > 1)
    duplicate_groups = duplicates.groupby(['file_name', 'bbox_tuple'])
    
    duplicate_details = []
    for (filename, bbox_tuple), group in duplicate_groups:
        drugs = group[['drug_N', 'dl_name', 'json_path']].to_dict('records')
        duplicate_details.append({
            'file_name': filename,
            'bbox': list(bbox_tuple),
            'count': len(group),
            'drugs': drugs
        })
    
    print(f"   └─ 중복 bbox 발견: {len(duplicate_details)}개 케이스")
    
    # 4. 경계 이탈 검증
    print("\n[4단계] 이미지 경계 이탈 검증 중...")
    
    # bbox 경계 계산
    df['bbox_right'] = df['bbox'].apply(lambda x: x[0] + x[2])
    df['bbox_bottom'] = df['bbox'].apply(lambda x: x[1] + x[3])
    
    # 경계 이탈 케이스
    oob_mask = (df['bbox_right'] > IMG_W) | (df['bbox_bottom'] > IMG_H)
    oob_data = df[oob_mask]
    
    oob_details = []
    for idx, row in oob_data.iterrows():
        bbox = row['bbox']
        oob_details.append({
            'file_name': row['file_name'],
            'drug_N': row['drug_N'],
            'dl_name': row['dl_name'],
            'bbox': bbox,
            'bbox_right': row['bbox_right'],
            'bbox_bottom': row['bbox_bottom'],
            'overflow_x': max(0, row['bbox_right'] - IMG_W),
            'overflow_y': max(0, row['bbox_bottom'] - IMG_H),
            'json_path': row['json_path']
        })
    
    print(f"   └─ 경계 이탈 발견: {len(oob_details)}개")
    
    # 5. 결과 정리
    results = {
        'total_json_files': len(all_records),
        'parse_errors': parse_errors,
        'duplicate_bbox': {
            'count': len(duplicate_details),
            'details': duplicate_details
        },
        'out_of_bounds': {
            'count': len(oob_details),
            'details': oob_details
        }
    }
    
    # 6. 결과 출력
    print_quality_results(results)
    
    return results


def print_quality_results(results):
    """품질 검증 결과를 출력"""
    
    print("\n" + "=" * 80)
    print("품질 검증 결과 요약")
    print("=" * 80)
    
    print(f"\n📊 전체 통계:")
    print(f"   • 총 JSON 파일 수: {results['total_json_files']:,}개")
    
    if results['parse_errors']:
        print(f"   • 파싱 오류: {len(results['parse_errors'])}개")
    
    # 중복 bbox
    dup_count = results['duplicate_bbox']['count']
    print(f"\n🔄 중복 bbox 케이스: {dup_count}개")
    
    if dup_count > 0:
        print("\n상위 10개 예시:")
        print(f"{'파일명':<60} {'bbox':<25} {'약 개수':<8}")
        print("-" * 100)
        
        for detail in results['duplicate_bbox']['details'][:10]:
            bbox_str = str(detail['bbox'])
            print(f"{detail['file_name']:<60} {bbox_str:<25} {detail['count']}개")
            for drug in detail['drugs']:
                print(f"    ├─ {drug['drug_N']}: {drug['dl_name']}")
        
        if dup_count > 10:
            print(f"\n... 외 {dup_count - 10}개 더 있음")
    
    # 경계 이탈
    oob_count = results['out_of_bounds']['count']
    print(f"\n⚠️  이미지 경계 이탈: {oob_count}개")
    
    if oob_count > 0:
        print("\n상위 10개 예시:")
        print(f"{'파일명':<50} {'약':<20} {'넘침(X, Y)':<15}")
        print("-" * 90)
        
        for detail in results['out_of_bounds']['details'][:10]:
            overflow = f"({detail['overflow_x']}, {detail['overflow_y']})"
            print(f"{detail['file_name']:<50} {detail['drug_N']:<20} {overflow:<15}")
            print(f"    └─ bbox: {detail['bbox']}")
        
        if oob_count > 10:
            print(f"\n... 외 {oob_count - 10}개 더 있음")
    
    print("\n" + "=" * 80)


def save_quality_report(results, output_path="json_quality_report.txt"):
    """품질 검증 결과를 파일로 저장"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("JSON 파일 내용 품질 검증 보고서\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"총 JSON 파일 수: {results['total_json_files']:,}개\n\n")
        
        # 중복 bbox
        f.write("=" * 80 + "\n")
        f.write(f"중복 bbox 케이스: {results['duplicate_bbox']['count']}개\n")
        f.write("=" * 80 + "\n\n")
        
        for i, detail in enumerate(results['duplicate_bbox']['details'], 1):
            f.write(f"[{i}] 파일: {detail['file_name']}\n")
            f.write(f"    bbox: {detail['bbox']}\n")
            f.write(f"    중복 약 {detail['count']}개:\n")
            for drug in detail['drugs']:
                f.write(f"      • {drug['drug_N']}: {drug['dl_name']}\n")
                f.write(f"        경로: {drug['json_path']}\n")
            f.write("\n")
        
        # 경계 이탈
        f.write("=" * 80 + "\n")
        f.write(f"이미지 경계 이탈: {results['out_of_bounds']['count']}개\n")
        f.write("=" * 80 + "\n\n")
        
        for i, detail in enumerate(results['out_of_bounds']['details'], 1):
            f.write(f"[{i}] 파일: {detail['file_name']}\n")
            f.write(f"    약: {detail['drug_N']} - {detail['dl_name']}\n")
            f.write(f"    bbox: {detail['bbox']}\n")
            f.write(f"    경계 초과: X축 {detail['overflow_x']}px, Y축 {detail['overflow_y']}px\n")
            f.write(f"    경로: {detail['json_path']}\n")
            f.write("\n")
    
    print(f"\n💾 품질 보고서 저장 완료: {output_path}")



if __name__ == "__main__":
    # 경로 설정
    LABEL_DIR = r"C:\Users\TAEHO\Desktop\AI_07_basic\data\train_annotations"
    IMAGE_DIR = r"C:\Users\TAEHO\Desktop\AI_07_basic\data\train_images"
    
    # 1. JSON 파일 누락 검증
    print("\n🔍 1단계: JSON 파일 누락 검증")
    validation_results = validate_pill_dataset(LABEL_DIR, IMAGE_DIR)
    save_validation_report(validation_results, "pill_dataset_validation_report.txt")
    
    # 2. JSON 내용 품질 검증
    print("\n🔍 2단계: JSON 내용 품질 검증")
    quality_results = validate_json_content_quality(LABEL_DIR, IMAGE_DIR)
    save_quality_report(quality_results, "json_quality_report.txt")
    
    print("\n✅ 모든 검증 완료!")