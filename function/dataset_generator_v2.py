import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import pandas as pd


@dataclass
class DatasetConfig:
    """데이터셋 생성 설정"""
    label_root: str
    image_root: str
    output_dir: str
    dataset_name: str
    dataset_type: str = 'combination'
    exclude_index_images: bool = True
    image_width: int = 976
    image_height: int = 1280
    max_workers: int = 4


@dataclass
class ValidationResult:
    """검증 결과를 담는 데이터 클래스"""
    valid_records: List[Dict] = field(default_factory=list)
    invalid_records: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    duplicate_details: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0


class ImageIndexer:
    """이미지 경로 인덱싱 및 캐싱"""
    
    def __init__(self, image_root: str):
        self.image_root = Path(image_root)
        self._index: Dict[str, str] = {}
    
    def build_index(self) -> Dict[str, str]:
        """이미지 디렉터리를 스캔하여 파일명-경로 맵 생성"""
        logging.info("이미지 경로 인덱싱 시작...")
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        count = 0
        
        for image_path in self.image_root.rglob('*'):
            if image_path.suffix.lower() in image_extensions:
                filename = image_path.name
                if filename not in self._index:  # 중복 방지
                    self._index[filename] = str(image_path)
                    count += 1
        
        logging.info(f"인덱싱 완료: {count:,}개 이미지 등록")
        return self._index
    
    def get_path(self, filename: str) -> Optional[str]:
        """파일명으로 경로 조회 (O(1))"""
        return self._index.get(filename)


class AnnotationProcessor:
    """JSON 어노테이션 처리"""
    
    @staticmethod
    def parse_single_json(json_path: Path, config: DatasetConfig, 
                         image_index: Dict[str, str]) -> Dict[str, Any]:
        """단일 JSON 파일 파싱 (병렬 처리용 정적 메서드)"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_meta = data['images'][0]
            annotation = data['annotations'][0]
            file_name = img_meta['file_name']
            
            # 1. _index 파일 필터링
            if config.exclude_index_images and '_index' in file_name:
                return {
                    'status': 'invalid',
                    'error_type': 'is_index_file',
                    'data': {'file_name': file_name, 'json_path': str(json_path)}
                }
            
            # 2. bbox 검증
            bbox = annotation.get('bbox')
            if not bbox or len(bbox) != 4:
                return {
                    'status': 'invalid',
                    'error_type': 'missing_bbox',
                    'data': {'file_name': file_name, 'json_path': str(json_path)}
                }
            
            x, y, w, h = bbox
            
            # 3. bbox 경계 검증
            if (x < 0 or y < 0 or 
                x + w > config.image_width or 
                y + h > config.image_height):
                return {
                    'status': 'invalid',
                    'error_type': 'out_of_bounds',
                    'data': {
                        'file_name': file_name,
                        'bbox': bbox,
                        'overflow_x': max(0, x + w - config.image_width),
                        'overflow_y': max(0, y + h - config.image_height)
                    }
                }
            
            # 4. 카테고리 ID 추출
            dl_idx = img_meta.get('dl_idx')
            category_id = annotation.get('category_id')
            
            if dl_idx and str(dl_idx).strip():
                final_category_id = int(dl_idx)
            elif category_id is not None:
                final_category_id = int(category_id)
            else:
                return {
                    'status': 'invalid',
                    'error_type': 'invalid_category',
                    'data': {
                        'file_name': file_name,
                        'dl_idx': dl_idx,
                        'category_id': category_id
                    }
                }
            
            # 5. 이미지 경로 확인
            image_path = image_index.get(file_name)
            if not image_path or not Path(image_path).exists():
                return {
                    'status': 'invalid',
                    'error_type': 'missing_image',
                    'data': {'file_name': file_name, 'expected_path': image_path}
                }
            
            # 유효한 레코드 반환
            return {
                'status': 'valid',
                'data': {
                    'file_name': file_name,
                    'image_path': image_path,
                    'width': img_meta['width'],
                    'height': img_meta['height'],
                    'category_id': final_category_id,
                    'category_name': img_meta.get('dl_name', 'Unknown'),
                    'bbox_x': x,
                    'bbox_y': y,
                    'bbox_w': w,
                    'bbox_h': h,
                    'area': annotation.get('area', w * h),
                    'drug_N': img_meta.get('drug_N', ''),
                    'json_path': str(json_path)
                }
            }
            
        except Exception as e:
            return {
                'status': 'invalid',
                'error_type': 'parse_error',
                'data': {'json_path': str(json_path), 'error': str(e)}
            }


class DataValidator:
    """데이터 검증 및 중복 제거"""
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict], int]:
        """중복 bbox 제거"""
        if df.empty:
            return df, [], 0
        
        # bbox 튜플 생성
        df['bbox_tuple'] = list(zip(
            df['bbox_x'], df['bbox_y'], df['bbox_w'], df['bbox_h']
        ))
        
        # 중복 마스크 생성
        duplicate_mask = df.duplicated(
            subset=['file_name', 'bbox_tuple'], keep=False
        )
        
        duplicates = df[duplicate_mask]
        duplicate_details = []
        
        if not duplicates.empty:
            for (filename, bbox_tuple), group in duplicates.groupby(
                ['file_name', 'bbox_tuple']
            ):
                duplicate_details.append({
                    'file_name': filename,
                    'bbox': list(bbox_tuple),
                    'count': len(group),
                    'records': group[['drug_N', 'category_name', 'json_path']].to_dict('records')
                })
        
        # 중복 제거
        before_count = len(df)
        df_clean = df[~duplicate_mask].drop(columns=['bbox_tuple'])
        removed_count = before_count - len(df_clean)
        
        return df_clean, duplicate_details, removed_count


class DataExporter:
    """데이터 내보내기"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def export_csv(self, df: pd.DataFrame, category_map: pd.DataFrame, 
                   dataset_name: str) -> Dict[str, str]:
        """CSV 파일 내보내기"""
        files = {}
        
        # 데이터셋 CSV
        dataset_path = self.output_path / f"{dataset_name}_dataset.csv"
        df.to_csv(dataset_path, index=False, encoding='utf-8-sig')
        files['dataset_csv'] = str(dataset_path)
        
        # 카테고리 CSV
        category_path = self.output_path / f"{dataset_name}_categories.csv"
        category_map.to_csv(category_path, index=False, encoding='utf-8-sig')
        files['category_csv'] = str(category_path)
        
        return files
    
    def export_reports(self, validation_result: ValidationResult, 
                      dataset_name: str) -> Dict[str, str]:
        """보고서 파일 내보내기"""
        files = {}
        
        # 무효 레코드 보고서
        invalid_path = self.output_path / f"{dataset_name}_invalid_records.json"
        with open(invalid_path, 'w', encoding='utf-8') as f:
            json.dump(dict(validation_result.invalid_records), 
                     f, ensure_ascii=False, indent=2)
        files['invalid_report'] = str(invalid_path)
        
        # 중복 bbox 보고서
        duplicate_path = self.output_path / f"{dataset_name}_duplicate_bboxes.json"
        with open(duplicate_path, 'w', encoding='utf-8') as f:
            json.dump(validation_result.duplicate_details, 
                     f, ensure_ascii=False, indent=2)
        files['duplicate_report'] = str(duplicate_path)
        
        return files
    
    def create_yolo_format(self, df: pd.DataFrame, category_map: pd.DataFrame):
        """YOLO 형식 데이터셋 생성"""
        yolo_dir = self.output_path / "yolo_format"
        images_dir = yolo_dir / "images"
        labels_dir = yolo_dir / "labels"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        category_to_idx = {
            row['category_id']: idx 
            for idx, row in category_map.iterrows()
        }
        
        for file_name, group in df.groupby('file_name'):
            # 이미지 복사
            src_image = Path(group.iloc[0]['image_path'])
            dst_image = images_dir / file_name
            
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)
            
            # 라벨 파일 생성
            label_file = dst_image.stem + '.txt'
            label_path = labels_dir / label_file
            
            with open(label_path, 'w') as f:
                for _, row in group.iterrows():
                    # YOLO 정규화
                    img_w, img_h = row['width'], row['height']
                    x, y, w, h = row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']
                    
                    center_x = (x + w / 2) / img_w
                    center_y = (y + h / 2) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h
                    
                    class_idx = category_to_idx[row['category_id']]
                    f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} "
                           f"{norm_w:.6f} {norm_h:.6f}\n")
        
        # data.yaml 생성
        yaml_path = yolo_dir / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {yolo_dir.absolute()}\n")
            f.write("train: images\n")
            f.write("val: images\n\n")
            f.write(f"nc: {len(category_map)}\n")
            f.write("names:\n")
            for idx, row in category_map.iterrows():
                f.write(f"  {idx}: '{row['category_name']}'\n")


class DatasetGenerator:
    """메인 데이터셋 생성기"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.output_path = Path(config.output_dir) / config.dataset_name
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        self.image_indexer = ImageIndexer(config.image_root)
        self.exporter = DataExporter(self.output_path)
    
    def generate(self) -> Dict[str, Any]:
        """전체 데이터셋 생성 프로세스"""
        start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info(f"데이터셋 생성 시작: {self.config.dataset_name}")
        self.logger.info("=" * 80)
        
        try:
            # 1. 이미지 인덱싱
            image_index = self.image_indexer.build_index()
            
            # 2. JSON 파일 수집
            json_files = self._collect_json_files()
            
            # 3. 병렬 처리로 어노테이션 파싱
            validation_result = self._process_annotations_parallel(
                json_files, image_index
            )
            
            # 4. DataFrame 생성 및 중복 제거
            df = self._create_and_clean_dataframe(validation_result)
            
            # 5. 카테고리 맵 생성
            category_map = self._create_category_map(df)
            
            # 6. 파일 내보내기
            export_files = self._export_all_files(
                df, category_map, validation_result
            )
            
            # 7. 통계 생성
            stats = self._generate_final_stats(
                df, category_map, validation_result, start_time, export_files
            )
            
            self._print_summary(stats)
            return stats
            
        except Exception as e:
            self.logger.error(f"데이터셋 생성 실패: {e}")
            raise
    
    def _collect_json_files(self) -> List[Path]:
        """JSON 파일 수집"""
        self.logger.info("\n[1단계] JSON 파일 수집...")
        json_files = list(Path(self.config.label_root).rglob('*.json'))
        self.logger.info(f"발견된 JSON 파일: {len(json_files):,}개")
        return json_files
    
    def _process_annotations_parallel(self, json_files: List[Path], 
                                    image_index: Dict[str, str]) -> ValidationResult:
        """JSON 어노테이션 병렬 처리"""
        self.logger.info(f"\n[2단계] 어노테이션 병렬 처리 (워커: {self.config.max_workers}개)...")
        
        result = ValidationResult()
        total = len(json_files)
        processed = 0
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 작업 제출
            futures = {
                executor.submit(
                    AnnotationProcessor.parse_single_json,
                    json_file, self.config, image_index
                ): json_file 
                for json_file in json_files
            }
            
            # 결과 수집
            for future in as_completed(futures):
                try:
                    parsed_result = future.result()
                    
                    if parsed_result['status'] == 'valid':
                        result.valid_records.append(parsed_result['data'])
                    else:
                        error_type = parsed_result['error_type']
                        result.invalid_records[error_type].append(parsed_result['data'])
                    
                    processed += 1
                    if processed % 1000 == 0:
                        self.logger.info(f"진행률: {processed}/{total} ({processed/total*100:.1f}%)")
                        
                except Exception as e:
                    self.logger.error(f"처리 오류: {e}")
        
        self.logger.info(f"파싱 완료 - 유효: {len(result.valid_records):,}개")
        for error_type, records in result.invalid_records.items():
            if records:
                self.logger.info(f"  {error_type}: {len(records)}개")
        
        return result
    
    def _create_and_clean_dataframe(self, validation_result: ValidationResult) -> pd.DataFrame:
        """DataFrame 생성 및 중복 제거"""
        self.logger.info("\n[3단계] DataFrame 생성 및 중복 제거...")
        
        if not validation_result.valid_records:
            self.logger.warning("유효한 레코드가 없습니다.")
            return pd.DataFrame()
        
        df = pd.DataFrame(validation_result.valid_records)
        
        # 중복 제거
        df_clean, duplicate_details, removed_count = DataValidator.remove_duplicates(df)
        
        validation_result.duplicate_details = duplicate_details
        
        self.logger.info(f"중복 제거 완료 - 제거: {removed_count}개, 남은 레코드: {len(df_clean):,}개")
        
        return df_clean
    
    def _create_category_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """카테고리 맵 생성"""
        if df.empty:
            return pd.DataFrame()
        
        category_map = (df[['category_id', 'category_name']]
                       .drop_duplicates()
                       .sort_values('category_id')
                       .reset_index(drop=True))
        
        self.logger.info(f"카테고리 맵 생성 완료 - 총 {len(category_map)}개 카테고리")
        return category_map
    
    def _export_all_files(self, df: pd.DataFrame, category_map: pd.DataFrame,
                         validation_result: ValidationResult) -> Dict[str, str]:
        """모든 파일 내보내기"""
        self.logger.info("\n[4단계] 파일 내보내기...")
        
        export_files = {}
        
        # CSV 파일
        csv_files = self.exporter.export_csv(df, category_map, self.config.dataset_name)
        export_files.update(csv_files)
        
        # 보고서 파일
        report_files = self.exporter.export_reports(validation_result, self.config.dataset_name)
        export_files.update(report_files)
        
        # YOLO 형식
        if not df.empty:
            self.exporter.create_yolo_format(df, category_map)
            export_files['yolo_format'] = str(self.output_path / "yolo_format")
        
        self.logger.info("파일 내보내기 완료")
        return export_files
    
    def _generate_final_stats(self, df: pd.DataFrame, category_map: pd.DataFrame,
                            validation_result: ValidationResult, start_time: float,
                            export_files: Dict[str, str]) -> Dict[str, Any]:
        """최종 통계 생성"""
        processing_time = time.time() - start_time
        
        return {
            'dataset_name': self.config.dataset_name,
            'dataset_type': self.config.dataset_type,
            'total_valid_records': len(df),
            'total_images': df['file_name'].nunique() if not df.empty else 0,
            'total_categories': len(category_map),
            'processing_time_seconds': round(processing_time, 2),
            'invalid_counts': {
                k: len(v) for k, v in validation_result.invalid_records.items()
            },
            'duplicate_removed': len(validation_result.duplicate_details),
            'output_directory': str(self.output_path),
            'export_files': export_files
        }
    
    def _print_summary(self, stats: Dict[str, Any]):
        """최종 요약 출력"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("데이터셋 생성 완료")
        self.logger.info("=" * 80)
        
        self.logger.info(f"데이터셋 이름: {stats['dataset_name']}")
        self.logger.info(f"처리 시간: {stats['processing_time_seconds']}초")
        self.logger.info(f"유효 레코드: {stats['total_valid_records']:,}개")
        self.logger.info(f"고유 이미지: {stats['total_images']:,}개")
        self.logger.info(f"카테고리 수: {stats['total_categories']}개")
        
        if any(stats['invalid_counts'].values()):
            self.logger.info("\n제외된 레코드:")
            for error_type, count in stats['invalid_counts'].items():
                if count > 0:
                    self.logger.info(f"  - {error_type}: {count:,}개")
        
        self.logger.info(f"\n출력 디렉터리: {stats['output_directory']}")
        self.logger.info("=" * 80)


# 사용 예시
if __name__ == "__main__":
    # Windows 환경에서 multiprocessing 사용 시 필요
    from multiprocessing import freeze_support
    freeze_support()
    
    # 설정 생성
    config = DatasetConfig(
        label_root=r"E:\download\label\단일경구약제_5000종\TL_81_단일",
        image_root=r"E:\download\image\단일경구약제_5000종\TS_81_단일",
        output_dir=r"E:\download\datasets",
        dataset_name="TS_81_single",
        dataset_type="single",
        exclude_index_images=False,
        max_workers=4  # CPU 코어 수에 맞춰 조정
    )
    
    # 데이터셋 생성
    generator = DatasetGenerator(config)
    stats = generator.generate()
