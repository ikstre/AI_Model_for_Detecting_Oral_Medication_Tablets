import os
import json
import logging
from pathlib import Path
from typing import Dict, Set, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm


@dataclass
class GlobalCategoryIndex:
    """전역 카테고리 인덱스"""
    category_map: Dict[int, Dict[str, any]]  # category_id -> {name, dl_idx, count}
    index_to_id: Dict[int, int]  # global_index (0, 1, 2...) -> category_id
    id_to_index: Dict[int, int]  # category_id -> global_index
    total_categories: int
    total_annotations: int


class GCIBuilder:
    """Global Category Index 빌더"""
    
    def __init__(self, annotation_dirs: List[str], output_dir: str = "global_index"):
        """
        Args:
            annotation_dirs: 모든 annotation 디렉토리 리스트
            output_dir: GCI 저장 디렉토리
        """
        self.annotation_dirs = [Path(d) for d in annotation_dirs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def extract_category_from_json(json_path: Path) -> Tuple[int, str, str]:
        """
        단일 JSON에서 카테고리 정보 추출 (병렬 처리용)
        
        Returns:
            (category_id, category_name, dl_idx)
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_meta = data['images'][0]
            annotation = data['annotations'][0]
            
            # dl_idx 우선, 없으면 category_id 사용
            dl_idx = img_meta.get('dl_idx')
            category_id = annotation.get('category_id')
            
            if dl_idx and str(dl_idx).strip():
                final_id = int(dl_idx)
            elif category_id is not None:
                final_id = int(category_id)
            else:
                return None
            
            category_name = img_meta.get('dl_name', 'Unknown')
            
            return (final_id, category_name, str(dl_idx) if dl_idx else '')
            
        except Exception as e:
            return None
    
    def scan_all_annotations(self) -> Dict[int, Dict]:
        """모든 annotation 디렉토리 스캔"""
        
        self.logger.info("=" * 80)
        self.logger.info("전역 카테고리 인덱스 생성 시작")
        self.logger.info("=" * 80)
        
        # 1. JSON 파일 수집
        self.logger.info("\n[1단계] JSON 파일 수집...")
        all_json_files = []
        
        for ann_dir in self.annotation_dirs:
            if ann_dir.exists():
                json_files = list(ann_dir.rglob('*.json'))
                all_json_files.extend(json_files)
                self.logger.info(f"  • {ann_dir.name}: {len(json_files):,}개")
            else:
                self.logger.warning(f"디렉토리 없음: {ann_dir}")
        
        self.logger.info(f"\n총 JSON 파일: {len(all_json_files):,}개")
        
        # 2. 병렬로 카테고리 추출
        self.logger.info("\n[2단계] 카테고리 정보 추출 (병렬 처리)...")
        
        category_data = {}  # category_id -> {name, dl_idx, count}
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.extract_category_from_json, json_file): json_file
                for json_file in all_json_files
            }
            
            with tqdm(total=len(all_json_files), desc="스캔 진행") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result:
                        cat_id, cat_name, dl_idx = result
                        
                        if cat_id not in category_data:
                            category_data[cat_id] = {
                                'category_id': cat_id,
                                'category_name': cat_name,
                                'dl_idx': dl_idx,
                                'count': 0
                            }
                        
                        category_data[cat_id]['count'] += 1
                    
                    pbar.update(1)
        
        self.logger.info(f"\n발견된 고유 카테고리: {len(category_data)}개")
        
        return category_data
    
    def build_global_index(self, category_data: Dict[int, Dict]) -> GlobalCategoryIndex:
        """전역 인덱스 생성 (category_id를 0부터 시작하는 연속된 인덱스로 매핑)"""
        
        self.logger.info("\n[3단계] 전역 인덱스 매핑 생성...")
        
        # category_id 기준으로 정렬
        sorted_categories = sorted(category_data.items(), key=lambda x: x[0])
        
        index_to_id = {}
        id_to_index = {}
        
        for global_idx, (cat_id, cat_info) in enumerate(sorted_categories):
            index_to_id[global_idx] = cat_id
            id_to_index[cat_id] = global_idx
        
        total_annotations = sum(cat['count'] for cat in category_data.values())
        
        gci = GlobalCategoryIndex(
            category_map=category_data,
            index_to_id=index_to_id,
            id_to_index=id_to_index,
            total_categories=len(category_data),
            total_annotations=total_annotations
        )
        
        self.logger.info(f"전역 인덱스 생성 완료:")
        self.logger.info(f"  • 전체 카테고리: {gci.total_categories}개")
        self.logger.info(f"  • 전체 annotation: {gci.total_annotations:,}개")
        self.logger.info(f"  • 인덱스 범위: 0 ~ {gci.total_categories - 1}")
        
        return gci
    
    def save_gci(self, gci: GlobalCategoryIndex):
        """GCI를 여러 형식으로 저장"""
        
        self.logger.info("\n[4단계] GCI 저장...")
        
        # 1. CSV 형식 (category_id, global_index, name, count)
        df_data = []
        for cat_id, cat_info in sorted(gci.category_map.items()):
            df_data.append({
                'category_id': cat_id,
                'global_index': gci.id_to_index[cat_id],
                'category_name': cat_info['category_name'],
                'dl_idx': cat_info['dl_idx'],
                'sample_count': cat_info['count']
            })
        
        df = pd.DataFrame(df_data)
        csv_path = self.output_dir / "global_category_index.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"CSV: {csv_path}")
        
        # 2. JSON 형식 (전체 정보)
        json_data = {
            'metadata': {
                'total_categories': gci.total_categories,
                'total_annotations': gci.total_annotations,
                'index_range': f"0-{gci.total_categories - 1}"
            },
            'category_map': gci.category_map,
            'index_to_id': {str(k): v for k, v in gci.index_to_id.items()},
            'id_to_index': {str(k): v for k, v in gci.id_to_index.items()}
        }
        
        json_path = self.output_dir / "global_category_index.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"JSON: {json_path}")
        
        # 3. YOLO classes.txt (global_index 순서)
        classes_path = self.output_dir / "yolo_classes.txt"
        with open(classes_path, 'w', encoding='utf-8') as f:
            for idx in range(gci.total_categories):
                cat_id = gci.index_to_id[idx]
                cat_name = gci.category_map[cat_id]['category_name']
                f.write(f"{cat_name}\n")
        self.logger.info(f"YOLO classes: {classes_path}")
        
        # 4. 통계 보고서
        self._save_statistics_report(gci, df)
    
    def _save_statistics_report(self, gci: GlobalCategoryIndex, df: pd.DataFrame):
        """통계 보고서 저장"""
        
        report_path = self.output_dir / "category_statistics.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("전역 카테고리 인덱스 통계 보고서\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"전체 카테고리 수: {gci.total_categories:,}개\n")
            f.write(f"전체 annotation 수: {gci.total_annotations:,}개\n")
            f.write(f"카테고리당 평균 샘플: {gci.total_annotations / gci.total_categories:.1f}개\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("샘플 수 상위 20개 카테고리\n")
            f.write("=" * 80 + "\n")
            top_20 = df.nlargest(20, 'sample_count')
            for idx, row in top_20.iterrows():
                f.write(f"{row['global_index']:4d} | ID {row['category_id']:5d} | "
                       f"{row['category_name']:40s} | {row['sample_count']:5d}개\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("샘플 수 하위 20개 카테고리\n")
            f.write("=" * 80 + "\n")
            bottom_20 = df.nsmallest(20, 'sample_count')
            for idx, row in bottom_20.iterrows():
                f.write(f"{row['global_index']:4d} | ID {row['category_id']:5d} | "
                       f"{row['category_name']:40s} | {row['sample_count']:5d}개\n")
        
        self.logger.info(f"통계 보고서: {report_path}")
    
    def print_summary(self, gci: GlobalCategoryIndex):
        """요약 정보 출력"""
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("전역 카테고리 인덱스 생성 완료")
        self.logger.info("=" * 80)
        
        # 샘플 수 통계
        counts = [cat['count'] for cat in gci.category_map.values()]
        
        self.logger.info(f"\n 전체 통계:")
        self.logger.info(f"  • 전체 카테고리: {gci.total_categories:,}개")
        self.logger.info(f"  • 전체 annotation: {gci.total_annotations:,}개")
        self.logger.info(f"  • 평균 샘플 수: {sum(counts) / len(counts):.1f}개")
        self.logger.info(f"  • 최대 샘플 수: {max(counts):,}개")
        self.logger.info(f"  • 최소 샘플 수: {min(counts):,}개")
        
        self.logger.info(f"\n 출력 디렉토리: {self.output_dir}")
        self.logger.info("=" * 80)
    
    def run(self) -> GlobalCategoryIndex:
        """전체 프로세스 실행"""
        
        # 1. 모든 annotation 스캔
        category_data = self.scan_all_annotations()
        
        # 2. 전역 인덱스 생성
        gci = self.build_global_index(category_data)
        
        # 3. 저장
        self.save_gci(gci)
        
        # 4. 요약
        self.print_summary(gci)
        
        return gci


# 사용 예시
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    # 모든 annotation 디렉토리 경로 리스트
    annotation_dirs = [
        # 기존 train set
        r"E:\download\sprint_ai_project1_data\train_annotations",
        
        # 조합 데이터 (TL_1 ~ TL_8)
        r"E:\download\label\경구약제조합_5000종\TL_1_조합",
        r"E:\download\label\경구약제조합_5000종\TL_2_조합",
        r"E:\download\label\경구약제조합_5000종\TL_3_조합",
        r"E:\download\label\경구약제조합_5000종\TL_4_조합",
        r"E:\download\label\경구약제조합_5000종\TL_5_조합",
        r"E:\download\label\경구약제조합_5000종\TL_6_조합",
        r"E:\download\label\경구약제조합_5000종\TL_7_조합",
        r"E:\download\label\경구약제조합_5000종\TL_8_조합",
        
        # 단일 데이터 (TL_1 ~ TL_81)
        # 아래처럼 수동으로 추가하거나, 반복문으로 생성 가능
    ]
    
    # 단일 데이터 자동 추가 (TL_1 ~ TL_81)
    base_path = r"E:\download\label\단일경구약제_5000종"
    
    # TL_1 ~ TL_9
    for i in range(1, 10):
        annotation_dirs.append(f"{base_path}\\TL_{i}_단일")
    
    # TL_10 ~ TL_81
    for i in range(10, 82):
        annotation_dirs.append(f"{base_path}\\TL_{i}_단일")
    
    # GCI 빌더 생성 및 실행
    builder = GCIBuilder(
        annotation_dirs=annotation_dirs,
        output_dir=r"E:\download\global_category_index"
    )
    
    gci = builder.run()
    
    print("\nGCI 생성 완료!")
    print(f"저장 위치: {builder.output_dir}")