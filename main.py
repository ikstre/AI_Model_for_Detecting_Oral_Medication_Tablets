#!/usr/bin/env python3
"""
=================================================================================
약품 식별 AI 프로젝트 - 메인 실행 파일
=================================================================================

이 파일은 데이터 전처리부터 모델 학습, 추론까지 전체 파이프라인을 순차적으로 실행합니다.

실행 순서:
1. GCI 생성 (Global Category Index)
2. 카테고리 ID Offset (+1 증가 및 병합)
3. YOLO 데이터셋 생성
4. 모델 학습
5. 추론 및 CSV 제출 파일 생성

주의사항:
- 각 단계는 순차적으로 실행되어야 합니다.
- 각 단계가 완료되면 생성된 파일을 확인한 후 다음 단계로 진행하세요.
- 파일 용량이 크므로 충분한 디스크 공간을 확보하세요.

=================================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# ==================== 설정 클래스 ====================

class ProjectConfig:
    """프로젝트 전체 설정을 관리하는 클래스"""
    
    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: 프로젝트 루트 디렉토리 (예: "E:\\download")
        """
        self.base_dir = Path(base_dir)
        
        # 기본 디렉토리
        self.label_root = self.base_dir / "label_aug"
        self.image_root = self.base_dir / "train_aug"
        self.test_images = self.base_dir / "sprint_ai_project1_data" / "test_images"
        
        # 출력 디렉토리
        self.output_dir = self.base_dir / "outputs"
        self.gci_dir = self.output_dir / "global_category_index"
        self.dataset_dir = self.output_dir / "datasets"
        self.model_dir = self.output_dir / "yolo_runs"
        self.submission_dir = self.output_dir / "submissions"
        
        # 데이터셋 설정
        self.train_annotations = self.label_root / "train_annotations"
        self.comb_base = self.label_root
        
        # 파일 경로
        self.gci_original = self.gci_dir / "global_category_index.json"
        self.gci_updated = self.gci_dir / "global_category_index_updated.json"
        
        # 학습 설정
        self.model_name = "yolov8s.pt"
        self.dataset_name = "pill_detection"
        self.project_name = "pill_baseline"
        
        # 실행 플래그 (True로 설정된 단계만 실행)
        self.run_gci_generation = True
        self.run_id_increment = True
        self.run_dataset_generation = True
        self.run_model_training = False  # 학습은 오래 걸리므로 기본 False
        self.run_inference = False  # 추론도 기본 False
        
        # 디렉토리 생성
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        for dir_path in [
            self.output_dir,
            self.gci_dir,
            self.dataset_dir,
            self.model_dir,
            self.submission_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_annotation_dirs(self):
        """어노테이션 디렉토리 리스트 반환"""
        annotation_dirs = []
        
        # train_annotations 추가
        annotation_dirs.append(str(self.train_annotations))
        
        # TL_1_조합 ~ TL_8_조합 추가 (TL_2 제외)
        for i in range(1, 9):
            if i == 2:  # TL_2는 제외
                continue
            path = self.comb_base / f"TL_{i}_조합"
            if path.exists():
                annotation_dirs.append(str(path))
        
        # VL_1_조합 추가
        vl_path = self.comb_base / "VL_1_조합"
        if vl_path.exists():
            annotation_dirs.append(str(vl_path))
        
        return annotation_dirs
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "base_dir": str(self.base_dir),
            "label_root": str(self.label_root),
            "image_root": str(self.image_root),
            "output_dir": str(self.output_dir),
            "gci_dir": str(self.gci_dir),
            "dataset_dir": str(self.dataset_dir),
            "model_dir": str(self.model_dir),
            "submission_dir": str(self.submission_dir),
            "dataset_name": self.dataset_name,
            "project_name": self.project_name,
        }


# ==================== 로깅 설정 ====================

def setup_logging(output_dir: Path):
    """로깅 설정"""
    log_file = output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


# ==================== 파이프라인 단계 함수 ====================

def step1_generate_gci(config: ProjectConfig, logger: logging.Logger) -> bool:
    """
    Step 1: Global Category Index 생성
    
    모든 어노테이션 파일을 스캔하여 전역 카테고리 인덱스를 생성합니다.
    """
    logger.info("=" * 80)
    logger.info("Step 1: Global Category Index 생성")
    logger.info("=" * 80)
    
    try:
        from function.gci_generator import GCIBuilder
        from multiprocessing import freeze_support
        freeze_support()
        
        annotation_dirs = config.get_annotation_dirs()
        
        logger.info(f"어노테이션 디렉토리: {len(annotation_dirs)}개")
        for i, dir_path in enumerate(annotation_dirs, 1):
            logger.info(f"  {i}. {dir_path}")
        
        builder = GCIBuilder(
            annotation_dirs=annotation_dirs,
            output_dir=str(config.gci_dir)
        )
        
        gci = builder.run()
        
        logger.info(f"✅ GCI 생성 완료")
        logger.info(f"   출력 파일: {config.gci_original}")
        logger.info(f"   총 카테고리: {gci.total_categories}개")
        logger.info(f"   총 어노테이션: {gci.total_annotations:,}개")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GCI 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def step2_increment_category_ids(config: ProjectConfig, logger: logging.Logger) -> bool:
    """
    Step 2: 카테고리 ID Offset 적용
    
    카테고리 ID를 +1 증가시키고 train_annotations와 병합합니다.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: 카테고리 ID Offset (+1) 및 병합")
    logger.info("=" * 80)
    
    try:
        from function.increment_category_ids import increment_category_ids
        
        if not config.gci_original.exists():
            logger.error(f"입력 파일이 없습니다: {config.gci_original}")
            logger.error("먼저 Step 1을 실행하세요.")
            return False
        
        logger.info(f"입력 파일: {config.gci_original}")
        logger.info(f"출력 파일: {config.gci_updated}")
        
        increment_category_ids(
            input_file=str(config.gci_original),
            output_file=str(config.gci_updated)
        )
        
        logger.info(f"✅ 카테고리 ID Offset 완료")
        logger.info(f"   출력 파일: {config.gci_updated}")
        
        # 결과 확인
        with open(config.gci_updated, 'r', encoding='utf-8') as f:
            updated_gci = json.load(f)
        
        logger.info(f"   업데이트된 카테고리 수: {len(updated_gci['category_map'])}개")
        logger.info(f"   Index 범위: 0 ~ {len(updated_gci['index_to_id']) - 1}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 카테고리 ID Offset 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def step3_generate_dataset(config: ProjectConfig, logger: logging.Logger) -> bool:
    """
    Step 3: YOLO 데이터셋 생성
    
    어노테이션과 이미지를 결합하여 YOLO 형식의 데이터셋을 생성합니다.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: YOLO 데이터셋 생성")
    logger.info("=" * 80)
    
    try:
        from function.dataset_generator import DatasetConfig, DatasetGenerator
        from multiprocessing import freeze_support
        freeze_support()
        
        if not config.gci_updated.exists():
            logger.error(f"GCI 파일이 없습니다: {config.gci_updated}")
            logger.error("먼저 Step 2를 실행하세요.")
            return False
        
        dataset_config = DatasetConfig(
            label_root=str(config.label_root),
            image_root=str(config.image_root),
            output_dir=str(config.dataset_dir),
            dataset_name=config.dataset_name,
            dataset_type="combination",
            gci_path=str(config.gci_updated),
            exclude_index_images=True,
            max_workers=4
        )
        
        logger.info(f"Label 디렉토리: {dataset_config.label_root}")
        logger.info(f"Image 디렉토리: {dataset_config.image_root}")
        logger.info(f"출력 디렉토리: {dataset_config.output_dir}")
        
        generator = DatasetGenerator(dataset_config)
        stats = generator.generate()
        
        logger.info(f"✅ 데이터셋 생성 완료")
        logger.info(f"   유효 레코드: {stats['total_valid_records']:,}개")
        logger.info(f"   고유 이미지: {stats['total_images']:,}개")
        logger.info(f"   카테고리 수: {stats['total_categories']}개")
        logger.info(f"   출력 디렉토리: {stats['output_directory']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 데이터셋 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def step4_train_model(config: ProjectConfig, logger: logging.Logger) -> bool:
    """
    Step 4: YOLO 모델 학습
    
    생성된 데이터셋으로 YOLOv8 모델을 학습합니다.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: YOLO 모델 학습")
    logger.info("=" * 80)
    
    try:
        from ultralytics import YOLO
        import torch
        
        logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # YOLO 데이터셋 경로 확인
        yolo_data_yaml = config.dataset_dir / config.dataset_name / "yolo_format" / "data.yaml"
        
        if not yolo_data_yaml.exists():
            logger.error(f"YOLO 데이터셋이 없습니다: {yolo_data_yaml}")
            logger.error("먼저 Step 3을 실행하세요.")
            return False
        
        logger.info(f"데이터셋 YAML: {yolo_data_yaml}")
        logger.info(f"모델: {config.model_name}")
        logger.info(f"프로젝트: {config.project_name}")
        
        # 모델 로드 및 학습
        model = YOLO(config.model_name)
        
        logger.info("모델 학습 시작...")
        logger.info("⚠️  학습은 시간이 오래 걸립니다. (GPU 사용 시 수 시간)")
        
        results = model.train(
            data=str(yolo_data_yaml),
            split=0.9,
            epochs=500,
            patience=30,
            imgsz=640,
            batch=8,
            workers=0,
            amp=True,
            optimizer="AdamW",
            lr0=0.001,
            # 증강 설정
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            fliplr=0.5,
            flipud=0.5,
            mosaic=0.1,
            mixup=0.0,
            erasing=0.1,
            device=0,
            project=str(config.model_dir),
            name=config.project_name,
            save_period=10
        )
        
        logger.info(f"✅ 모델 학습 완료")
        logger.info(f"   결과 저장: {config.model_dir / config.project_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 학습 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def step5_generate_submission(config: ProjectConfig, logger: logging.Logger) -> bool:
    """
    Step 5: 추론 및 제출 파일 생성
    
    학습된 모델로 테스트 이미지를 추론하고 Kaggle 제출용 CSV를 생성합니다.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: 추론 및 제출 파일 생성")
    logger.info("=" * 80)
    
    try:
        from function.csv_generator import run_infer_to_csv
        
        # Best 모델 경로
        best_weights = config.model_dir / config.project_name / "weights" / "best.pt"
        
        if not best_weights.exists():
            logger.error(f"학습된 모델이 없습니다: {best_weights}")
            logger.error("먼저 Step 4를 실행하세요.")
            return False
        
        if not config.test_images.exists():
            logger.error(f"테스트 이미지 디렉토리가 없습니다: {config.test_images}")
            return False
        
        # 출력 CSV 경로
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = config.submission_dir / f"submission_{timestamp}.csv"
        
        logger.info(f"모델: {best_weights}")
        logger.info(f"테스트 이미지: {config.test_images}")
        logger.info(f"출력 CSV: {output_csv}")
        
        run_infer_to_csv(
            weights_path=str(best_weights),
            test_img_dir=str(config.test_images),
            out_csv_path=str(output_csv),
            gci_json_path=str(config.gci_updated),
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det_per_image=4,
            device=0,
            pred_batch=4,
            chunk_size=200,
            half=True,
        )
        
        logger.info(f"✅ 제출 파일 생성 완료")
        logger.info(f"   출력 파일: {output_csv}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 제출 파일 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ==================== 메인 실행 ====================

def main():
    """메인 실행 함수"""
    
    # ==================== 사용자 설정 ====================
    # 프로젝트 루트 디렉토리를 설정하세요
    BASE_DIR = r"E:\download"  # 여기를 본인의 경로로 수정하세요
    # ====================================================
    
    # 설정 초기화
    config = ProjectConfig(BASE_DIR)
    
    # 로깅 설정
    logger = setup_logging(config.output_dir)
    
    logger.info("=" * 80)
    logger.info("약품 식별 AI 프로젝트 파이프라인 시작")
    logger.info("=" * 80)
    logger.info(f"프로젝트 디렉토리: {config.base_dir}")
    logger.info(f"출력 디렉토리: {config.output_dir}")
    
    # 설정 저장
    config_file = config.output_dir / "pipeline_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"설정 저장: {config_file}")
    
    # 파이프라인 실행
    results = {}
    
    # Step 1: GCI 생성
    if config.run_gci_generation:
        results['step1_gci'] = step1_generate_gci(config, logger)
        if not results['step1_gci']:
            logger.error("Step 1 실패. 파이프라인을 중단합니다.")
            return
        logger.info("\n✅ Step 1 완료. 생성된 파일을 확인하세요.")
        input("\n계속하려면 Enter를 누르세요...")
    else:
        logger.info("Step 1 건너뛰기 (run_gci_generation=False)")
    
    # Step 2: ID Offset
    if config.run_id_increment:
        results['step2_increment'] = step2_increment_category_ids(config, logger)
        if not results['step2_increment']:
            logger.error("Step 2 실패. 파이프라인을 중단합니다.")
            return
        logger.info("\n✅ Step 2 완료. 생성된 파일을 확인하세요.")
        input("\n계속하려면 Enter를 누르세요...")
    else:
        logger.info("Step 2 건너뛰기 (run_id_increment=False)")
    
    # Step 3: 데이터셋 생성
    if config.run_dataset_generation:
        results['step3_dataset'] = step3_generate_dataset(config, logger)
        if not results['step3_dataset']:
            logger.error("Step 3 실패. 파이프라인을 중단합니다.")
            return
        logger.info("\n✅ Step 3 완료. 생성된 데이터셋을 확인하세요.")
        input("\n계속하려면 Enter를 누르세요...")
    else:
        logger.info("Step 3 건너뛰기 (run_dataset_generation=False)")
    
    # Step 4: 모델 학습
    if config.run_model_training:
        results['step4_training'] = step4_train_model(config, logger)
        if not results['step4_training']:
            logger.error("Step 4 실패. 파이프라인을 중단합니다.")
            return
        logger.info("\n✅ Step 4 완료. 학습 결과를 확인하세요.")
        input("\n계속하려면 Enter를 누르세요...")
    else:
        logger.info("Step 4 건너뛰기 (run_model_training=False)")
    
    # Step 5: 추론 및 제출
    if config.run_inference:
        results['step5_submission'] = step5_generate_submission(config, logger)
        if not results['step5_submission']:
            logger.error("Step 5 실패.")
            return
        logger.info("\n✅ Step 5 완료. 제출 파일을 확인하세요.")
    else:
        logger.info("Step 5 건너뛰기 (run_inference=False)")
    
    # 최종 요약
    logger.info("\n" + "=" * 80)
    logger.info("파이프라인 실행 완료")
    logger.info("=" * 80)
    
    for step_name, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        logger.info(f"{step_name}: {status}")
    
    logger.info(f"\n📁 출력 디렉토리: {config.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
