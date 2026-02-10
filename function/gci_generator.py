import os
import json
import logging
from pathlib import Path
from typing import Dict, Set, List, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from collections import defaultdict
from typing import Iterable, Optional
from multiprocessing import cpu_count
import pandas as pd
from tqdm import tqdm


@dataclass
class GlobalCategoryIndex:
    category_map: Dict[int, Dict[str, any]]
    index_to_id: Dict[int, int]
    id_to_index: Dict[int, int]
    total_categories: int
    total_annotations: int
    dataset_stats: Dict[str, Dict] = field(default_factory=dict)


def iter_json_files(annotation_dirs: List[Path]) -> Iterable[Tuple[Path, str]]:
    """
    (json_path, dataset_name) 을 스트리밍으로 뽑아주는 제너레이터
    """
    for ann_dir in annotation_dirs:
        if not ann_dir.exists():
            continue
        dataset_name = ann_dir.name
        for jf in ann_dir.rglob("*.json"):
            yield jf, dataset_name


def parse_one_json(args: Tuple[str, str]) -> Optional[Tuple[int, str, str, str, int]]:
    """
    단일 JSON에서:
    - final_id (dl_idx 우선)
    - category_name
    - dl_idx(str)
    - dataset_name
    - count_in_file (해당 id가 이 json에서 몇 번 나왔는지)

    반환: (cat_id, cat_name, dl_idx, dataset_name, k)
    """
    json_path, dataset_name = args
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_meta = data.get("images", [{}])[0]
        ann_list = data.get("annotations", [])

        if not ann_list:
            return None

        dl_idx = img_meta.get("dl_idx")
        # dl_idx가 있으면 그걸 id로 사용
        if dl_idx and str(dl_idx).strip():
            cat_id = int(dl_idx)
        else:
            cat_id = int(ann_list[0].get("category_id")) if ann_list[0].get("category_id") is not None else None

        if cat_id is None:
            return None

        cat_name = img_meta.get("dl_name", "Unknown")
        # 이 JSON 안에서 해당 cat_id가 몇 번 등장했는지
        k = 0
        for a in ann_list:
            k += 1

        return (cat_id, cat_name, str(dl_idx) if dl_idx else "", dataset_name, k)

    except Exception:
        return None


class GCIBuilder:
    def __init__(self, annotation_dirs: List[str], output_dir: str = "global_index"):
        self.annotation_dirs = [Path(d) for d in annotation_dirs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def scan_all_annotations(self) -> Tuple[Dict[int, Dict], Dict[str, Dict]]:
        self.logger.info("=" * 80)
        self.logger.info("전역 카테고리 인덱스 생성 시작")
        self.logger.info("=" * 80)

        # 1) 존재하는 디렉토리만 로그 출력
        self.logger.info("\n[1단계] JSON 파일 스트리밍 스캔 준비...")
        existing_dirs = [d for d in self.annotation_dirs if d.exists()]
        for d in self.annotation_dirs:
            if d.exists():
                self.logger.info(f"  • OK: {d}")
            else:
                self.logger.warning(f"  ⚠️  디렉토리 없음: {d}")

        # 2) 스트리밍으로 처리
        self.logger.info("\n[2단계] 카테고리 정보 추출 (ProcessPool.map 스트리밍)...")

        category_data = {}  # cat_id -> info
        dataset_stats = defaultdict(lambda: {
            "total_annotations": 0,
            "unique_categories": set(),
            "category_counts": defaultdict(int),
        })

        # worker 수: CPU 코어 기반으로
        workers = min(8, max(2, cpu_count() - 1))


        with ProcessPoolExecutor(max_workers=workers) as ex:
            # map에 넘길 iterable은 (str_path, dataset_name)
            it = ((str(p), ds) for p, ds in iter_json_files(existing_dirs))

            for result in tqdm(ex.map(parse_one_json, it, chunksize=200), desc="스캔 진행"):
                if not result:
                    continue

                cat_id, cat_name, dl_idx, dataset_name, k = result

                if cat_id not in category_data:
                    category_data[cat_id] = {
                        "category_id": cat_id,
                        "category_name": cat_name,
                        "dl_idx": dl_idx,
                        "count": 0,
                        "datasets": defaultdict(int),
                    }

                category_data[cat_id]["count"] += k
                category_data[cat_id]["datasets"][dataset_name] += k

                dataset_stats[dataset_name]["total_annotations"] += k
                dataset_stats[dataset_name]["unique_categories"].add(cat_id)
                dataset_stats[dataset_name]["category_counts"][cat_id] += k

        # unique_categories set -> count
        for ds_name, stats in dataset_stats.items():
            stats["unique_categories"] = len(stats["unique_categories"])
            stats["category_counts"] = dict(stats["category_counts"])

        self.logger.info(f"\n발견된 고유 카테고리: {len(category_data)}개")
        self.logger.info(f"데이터셋 수: {len(dataset_stats)}개")

        return category_data, dict(dataset_stats)
    
    def build_global_index(self, category_data: Dict[int, Dict], 
                          dataset_stats: Dict[str, Dict]) -> GlobalCategoryIndex:
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
            total_annotations=total_annotations,
            dataset_stats=dataset_stats
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
            # 데이터셋 정보를 문자열로 변환
            dataset_info = ', '.join([f"{ds}({cnt})" for ds, cnt in 
                                     sorted(cat_info['datasets'].items(), 
                                           key=lambda x: x[1], reverse=True)[:5]])  # 상위 5개만
            
            df_data.append({
                'category_id': cat_id,
                'global_index': gci.id_to_index[cat_id],
                'category_name': cat_info['category_name'],
                'dl_idx': cat_info['dl_idx'],
                'total_count': cat_info['count'],
                'num_datasets': len(cat_info['datasets']),
                'top_datasets': dataset_info
            })
        
        df = pd.DataFrame(df_data)
        csv_path = self.output_dir / "global_category_index.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"CSV: {csv_path}")
        
        # 2. 데이터셋별 통계 CSV
        self._save_dataset_statistics_csv(gci)
        
        # 3. JSON 형식 (전체 정보)
        json_data = {
            'metadata': {
                'total_categories': gci.total_categories,
                'total_annotations': gci.total_annotations,
                'total_datasets': len(gci.dataset_stats),
                'index_range': f"0-{gci.total_categories - 1}"
            },
            'category_map': {
                str(k): {
                    **v,
                    'datasets': dict(v['datasets'])  # defaultdict를 dict로 변환
                } for k, v in gci.category_map.items()
            },
            'index_to_id': {str(k): v for k, v in gci.index_to_id.items()},
            'id_to_index': {str(k): v for k, v in gci.id_to_index.items()},
            'dataset_stats': gci.dataset_stats
        }
        
        json_path = self.output_dir / "global_category_index.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"JSON: {json_path}")
        
        # 4. YOLO classes.txt (global_index 순서)
        classes_path = self.output_dir / "yolo_classes.txt"
        with open(classes_path, 'w', encoding='utf-8') as f:
            for idx in range(gci.total_categories):
                cat_id = gci.index_to_id[idx]
                cat_name = gci.category_map[cat_id]['category_name']
                f.write(f"{cat_name}\n")
        self.logger.info(f"YOLO classes: {classes_path}")
        
        # 5. 통계 보고서
        self._save_statistics_report(gci, df)
    
    def _save_dataset_statistics_csv(self, gci: GlobalCategoryIndex):
        """데이터셋별 통계를 CSV로 저장"""
        
        dataset_data = []
        for ds_name, stats in sorted(gci.dataset_stats.items()):
            dataset_data.append({
                'dataset_name': ds_name,
                'total_annotations': stats['total_annotations'],
                'unique_categories': stats['unique_categories'],
                'avg_per_category': stats['total_annotations'] / stats['unique_categories'] 
                                   if stats['unique_categories'] > 0 else 0
            })
        
        df_dataset = pd.DataFrame(dataset_data)
        csv_path = self.output_dir / "dataset_statistics.csv"
        df_dataset.to_csv(csv_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"데이터셋 통계 CSV: {csv_path}")
    
    def _save_statistics_report(self, gci: GlobalCategoryIndex, df: pd.DataFrame):
        """통계 보고서 저장"""
        
        report_path = self.output_dir / "category_statistics.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("전역 카테고리 인덱스 통계 보고서\n")
            f.write("=" * 100 + "\n\n")
            
            # 전체 통계
            f.write(f"전체 카테고리 수: {gci.total_categories:,}개\n")
            f.write(f"전체 annotation 수: {gci.total_annotations:,}개\n")
            f.write(f"전체 데이터셋 수: {len(gci.dataset_stats)}개\n")
            f.write(f"카테고리당 평균 샘플: {gci.total_annotations / gci.total_categories:.1f}개\n\n")
            
            # 데이터셋별 통계
            f.write("=" * 100 + "\n")
            f.write("데이터셋별 통계\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'데이터셋':<40} {'Annotations':>12} {'카테고리':>10} {'평균':>10}\n")
            f.write("-" * 100 + "\n")
            
            for ds_name, stats in sorted(gci.dataset_stats.items(), 
                                        key=lambda x: x[1]['total_annotations'], 
                                        reverse=True):
                avg = stats['total_annotations'] / stats['unique_categories'] if stats['unique_categories'] > 0 else 0
                f.write(f"{ds_name:<40} {stats['total_annotations']:>12,} {stats['unique_categories']:>10} {avg:>10.1f}\n")
            
            # 전체 카테고리 상위/하위
            f.write("\n" + "=" * 100 + "\n")
            f.write("샘플 수 상위 20개 카테고리 (전체)\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Idx':>4} | {'Cat ID':>6} | {'카테고리명':<40} | {'총샘플':>7} | {'데이터셋수':>10} | 주요 데이터셋\n")
            f.write("-" * 100 + "\n")
            
            top_20 = df.nlargest(20, 'total_count')
            for idx, row in top_20.iterrows():
                f.write(f"{row['global_index']:4d} | {row['category_id']:6d} | "
                       f"{row['category_name']:<40s} | {row['total_count']:7d} | "
                       f"{row['num_datasets']:10d} | {row['top_datasets']}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("샘플 수 하위 20개 카테고리 (전체)\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Idx':>4} | {'Cat ID':>6} | {'카테고리명':<40} | {'총샘플':>7} | {'데이터셋수':>10} | 주요 데이터셋\n")
            f.write("-" * 100 + "\n")
            
            bottom_20 = df.nsmallest(20, 'total_count')
            for idx, row in bottom_20.iterrows():
                f.write(f"{row['global_index']:4d} | {row['category_id']:6d} | "
                       f"{row['category_name']:<40s} | {row['total_count']:7d} | "
                       f"{row['num_datasets']:10d} | {row['top_datasets']}\n")
            
            # 데이터셋별 상위 카테고리
            f.write("\n" + "=" * 100 + "\n")
            f.write("데이터셋별 상위 10개 카테고리\n")
            f.write("=" * 100 + "\n")
            
            for ds_name, stats in sorted(gci.dataset_stats.items())[:10]:  # 처음 10개 데이터셋만
                f.write(f"\n[{ds_name}]\n")
                f.write(f"{'Rank':>4} | {'Cat ID':>6} | {'카테고리명':<40} | {'샘플수':>7}\n")
                f.write("-" * 80 + "\n")
                
                # 상위 10개
                top_cats = sorted(stats['category_counts'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]
                
                for rank, (cat_id, count) in enumerate(top_cats, 1):
                    cat_name = gci.category_map[cat_id]['category_name']
                    f.write(f"{rank:4d} | {cat_id:6d} | {cat_name:<40s} | {count:7d}\n")
        
        self.logger.info(f"통계 보고서: {report_path}")
    
    def print_summary(self, gci: GlobalCategoryIndex):
        """요약 정보 출력"""
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("전역 카테고리 인덱스 생성 완료")
        self.logger.info("=" * 80)
        
        # 샘플 수 통계
        counts = [cat['count'] for cat in gci.category_map.values()]
        
        self.logger.info(f"\n📊 전체 통계:")
        self.logger.info(f"  • 전체 카테고리: {gci.total_categories:,}개")
        self.logger.info(f"  • 전체 annotation: {gci.total_annotations:,}개")
        self.logger.info(f"  • 전체 데이터셋: {len(gci.dataset_stats)}개")
        self.logger.info(f"  • 평균 샘플 수: {sum(counts) / len(counts):.1f}개")
        self.logger.info(f"  • 최대 샘플 수: {max(counts):,}개")
        self.logger.info(f"  • 최소 샘플 수: {min(counts):,}개")
        
        # 데이터셋 통계 (상위 5개)
        self.logger.info(f"\n📁 데이터셋 통계 (상위 5개):")
        top_datasets = sorted(gci.dataset_stats.items(), 
                            key=lambda x: x[1]['total_annotations'], 
                            reverse=True)[:5]
        
        for ds_name, stats in top_datasets:
            self.logger.info(f"  • {ds_name}: {stats['total_annotations']:,}개 annotation, "
                           f"{stats['unique_categories']}개 카테고리")
        
        self.logger.info(f"\n💾 출력 디렉토리: {self.output_dir}")
        self.logger.info("=" * 80)
    
    def run(self) -> GlobalCategoryIndex:
        """전체 프로세스 실행"""
        
        # 1. 모든 annotation 스캔
        category_data, dataset_stats = self.scan_all_annotations()
        
        # 2. 전역 인덱스 생성
        gci = self.build_global_index(category_data, dataset_stats)
        
        # 3. 저장
        self.save_gci(gci)
        
        # 4. 요약
        self.print_summary(gci)
        
        return gci


def build_range_paths(base_dir, prefix, suffix, start, end, exclude=None):
    """
    TL_1_단일 같은 반복 경로 자동 생성
    exclude: 제외할 index 리스트 (ex: [2])
    """
    if exclude is None:
        exclude = set()
    else:
        exclude = set(exclude)

    paths = []

    for i in range(start, end + 1):
        if i in exclude:
            print(f"[SKIP] {prefix}_{i}_{suffix}")
            continue

        paths.append(str(base_dir / f"{prefix}_{i}_{suffix}"))

    return paths


# if __name__ == "__main__":
#     freeze_support()

#     ROOT = Path(r"E:\download")
#     TRAIN_ANN = ROOT / "sprint_ai_project1_data" / "train_annotations"
#     # COMB_BASE = ROOT / "label" / "경구약제조합_5000종"
#     # SINGLE_BASE = ROOT / "label" / "단일경구약제_5000종"

#     annotation_dirs = []
#     # annotation_dirs.append(str(TRAIN_ANN))
    
#     # # 조합
#     # annotation_dirs += build_range_paths(COMB_BASE, "TL", "조합", 1, 8, exclude=[2])
#     # annotation_dirs.append(str(COMB_BASE / "VL_1_조합"))
    
#     # # 단일
#     # annotation_dirs += build_range_paths(SINGLE_BASE, "VL", "단일", 1, 10)
#     # annotation_dirs += build_range_paths(SINGLE_BASE, "TL", "단일", 1, 81)

#     builder = GCIBuilder(
#         annotation_dirs=annotation_dirs,
#         output_dir=str(ROOT / "global_category_index(train_set_56)")
#     )

#     gci = builder.run()

#     print("\n✅ GCI 생성 완료!")
#     print(f"📁 저장 위치: {builder.output_dir}")
#     print(f"📊 총 데이터셋: {len(annotation_dirs)}개")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    ROOT = Path(r"E:\download")
    
    # 📍 TL2 조합의 정확한 경로 설정
    # 주신 경로: E:\download\download\166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터\01.데이터\1.Training\원천데이터\경구약제조합_5000종\TL_2_조합
    # (주의: 경로 내 '원천데이터'라고 적어주셨는데, GCI는 JSON 파일이 있는 '라벨링데이터' 폴더를 바라봐야 합니다!)
    
    # 만약 JSON이 들어있는 폴더가 '라벨링데이터' 하위에 있다면 경로를 그쪽으로 잡아주세요.
    TL2_PATH = ROOT / "download" / "166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터" / "01.데이터" / "1.Training" / "원천데이터" / "경구약제조합_5000종" / "TL_2_조합"

    # 만약 주신 경로(원천데이터 하위)에 JSON이 같이 있다면 아래를 사용하세요.
    # TL2_PATH = Path(r"E:\download\download\166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터\01.데이터\1.Training\원천데이터\경구약제조합_5000종\TL_2_조합")

    annotation_dirs = [str(TL2_PATH)]

    # 📁 결과 저장 폴더명 변경 (비교를 위해 구분함)
    output_folder_name = "global_category_index_TL2_Only"
    
    builder = GCIBuilder(
        annotation_dirs=annotation_dirs,
        output_dir=str(ROOT / output_folder_name)
    )

    gci = builder.run()

    print("\n✅ TL2 전용 GCI 분석 완료!")
    print(f"📁 저장 위치: {builder.output_dir}")
    print(f"📊 분석 대상: {annotation_dirs[0]}")
    
    # 🔍 결과 요약 출력 (몇 종인지 바로 확인)
    num_classes = len(gci.get('index_to_id', {}))
    print(f"🧐 TL2 내 발견된 카테고리 수: {num_classes}개")