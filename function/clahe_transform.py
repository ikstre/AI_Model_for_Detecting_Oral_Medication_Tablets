import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from pathlib import Path
from tqdm import tqdm

class MorphologyEdgeEnhance(ImageOnlyTransform):
    def __init__(self, kernel_size=3, always_apply=False, p=0.5):
        super(MorphologyEdgeEnhance, self).__init__(always_apply, p)
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def apply(self, image, **params):
        # 형태학적 그레이디언트로 경계선 추출 후 합성
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, self.kernel)
        enhanced = cv2.addWeighted(image, 0.9, gradient, 0.1, 0)
        return enhanced

class CLAHETransform:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - 알약 경계 및 내부 구분선 강조
    - 색상 보존하면서 명암 대비 향상
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), p=0.5):
        """
        Args:
            clip_limit: 대비 제한 (1.0~4.0, 높을수록 강함)
            tile_grid_size: 타일 크기 (작을수록 세밀함)
            p: 적용 확률
        """
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, 
            tileGridSize=tile_grid_size
        )
        self.p = p

    def __call__(self, image, **kwargs):
        """
        Args:
            image: RGB numpy array (H, W, 3)
        
        Returns:
            RGB numpy array
        """
        if np.random.rand() > self.p:
            return image
        
        # 🔧 수정 1: RGB → LAB 변환 (YOLOv8은 RGB 사용)
        # OpenCV는 BGR이지만, albumentations는 RGB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        l, a, b = cv2.split(lab)
        
        # L 채널에만 CLAHE 적용 (색상 보존)
        l_clahe = self.clahe.apply(l)
        
        # 채널 병합
        lab_clahe = cv2.merge((l_clahe, a, b))
        
        # LAB → RGB
        image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return image_clahe


class AdaptiveSharpen:
    """
    언샵 마스킹 - 알약 엣지 추가 강조
    """
    def __init__(self, amount=0.3, threshold=0, p=0.3):
        """
        Args:
            amount: 샤프닝 강도 (0.0~1.0)
            threshold: 적용 임계값
            p: 적용 확률
        """
        self.amount = amount
        self.threshold = threshold
        self.p = p
    
    def __call__(self, image, **kwargs):
        if np.random.rand() > self.p:
            return image
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(
            image, 1.0 + self.amount, 
            blurred, -self.amount, 
            0
        )
        
        # Threshold 적용
        if self.threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < self.threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)


def get_pill_augmentation(img_size=640, train=True):
    """
    알약 탐지 전용 증강 파이프라인
    
    Args:
        img_size: 이미지 크기
        train: 학습용(True) / 검증용(False)
    """
    
    if train:
        return A.Compose([
            # 🔧 1. 명암 대비 강화 (색상 보존)
            CLAHETransform(
                clip_limit=2.5,      # 2.0~3.0 권장
                tile_grid_size=(8, 8),
                p=0.5
            ),
            
            # 🔧 2. 엣지 샤프닝
            AdaptiveSharpen(
                amount=0.3,
                p=0.3
            ),
            
            # 🔧 3. 구분선 강조 (약하게)
            MorphologyEdgeEnhance(
                kernel_size=3,
                p=0.2
            ),
            
            # 기하학적 변환 (기존)
            A.Rotate(
                limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            
            # 🔧 4. 조명 변화 (매우 약하게)
            A.RandomBrightnessContrast(
                brightness_limit=0.1,   # ±10%
                contrast_limit=0.1,
                p=0.3
            ),
            
            # 🔧 5. 가우시안 노이즈 (현실적 노이즈)
            A.GaussNoise(
                var_limit=(5.0, 15.0),
                p=0.2
            ),
            
            # 정규화 (YOLOv8 자동 처리, 여기선 생략)
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    else:
        # 검증용: 증강 없음
        return A.Compose([
            # CLAHE만 적용 (일관성)
            CLAHETransform(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=1.0  # 항상 적용
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    



def preprocess_dataset_with_clahe(
    input_dir: str,
    output_dir: str,
    clip_limit: float = 2.5
):
    """
    데이터셋 전체에 CLAHE 전처리 적용
    
    Args:
        input_dir: 원본 이미지 디렉토리
        output_dir: 전처리된 이미지 저장 디렉토리
        clip_limit: CLAHE 강도
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )
    
    image_files = list(input_path.glob('*.png')) + \
                  list(input_path.glob('*.jpg'))
    
    for img_file in tqdm(image_files, desc="CLAHE 전처리"):
        # 이미지 로드 (RGB)
        img = cv2.imread(str(img_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # LAB 변환
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE 적용
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        
        # RGB로 다시 변환
        img_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        img_enhanced_bgr = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR)
        
        # 저장
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), img_enhanced_bgr)
    
    print(f"✅ 전처리 완료: {len(image_files)}개 → {output_path}")


# 사용법
if __name__ == '__main__':
    # Train 이미지 전처리
    preprocess_dataset_with_clahe(
        input_dir=r'E:\download\sprint_ai_project1_data\test_images',
        output_dir=r'E:\download\sprint_ai_project1_data\test_images_clahe',
        clip_limit=2.5
    )
    
    # # 라벨 파일 복사
    # import shutil
    # shutil.copytree(
    #     r'E:\download\datasets\original_trainset\yolo_format\labels',
    #     r'E:\download\datasets\original_trainset\yolo_clahe\labels'
    # )
    

# # 최종 추천 방식

# # 방법 2: 전처리 스크립트 사용
# # 1. 전체 데이터셋에 CLAHE 적용
# # 2. 새 폴더에 저장
# # 3. 그 폴더로 학습

# preprocess_dataset_with_clahe(...)
# model.train(data='clahe_data.yaml')