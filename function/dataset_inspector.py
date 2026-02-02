import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import random
from collections import Counter
import seaborn as sns
import os


# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def comprehensive_dataset_inspection(dataset_csv_path: str, category_csv_path: str):
    """
    데이터셋 전체 검증 및 시각화
    
    Args:
        dataset_csv_path: 데이터셋 CSV 파일 경로
        category_csv_path: 카테고리 CSV 파일 경로
    """
    
    print("=" * 80)
    print("데이터셋 종합 검증 시작")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(dataset_csv_path)
    categories = pd.read_csv(category_csv_path)
    
    print(f"\n✅ 데이터 로드 완료")
    print(f"   • 데이터셋 레코드: {len(df):,}개")
    print(f"   • 카테고리 수: {len(categories)}개")
    
    # 1. 기본 통계
    print_basic_statistics(df, categories)
    
    # 2. 결측값 검사
    print_missing_values(df)
    
    # 3. 중복값 검사
    print_duplicate_analysis(df)
    
    # 4. 이상치 검사
    print_outlier_analysis(df)
    
    # 5. 카테고리 분포
    print_category_distribution(df, categories)
    
    # 6. bbox 통계
    print_bbox_statistics(df)
    
    # 7. 이미지-JSON 매칭 확인
    verify_image_json_matching(df)
    
    # 8. 시각화
    visualize_dataset_overview(df, categories)
    
    return df, categories


def print_basic_statistics(df: pd.DataFrame, categories: pd.DataFrame):
    """기본 통계 정보 출력"""
    
    print("\n" + "=" * 80)
    print("📊 1. 기본 통계")
    print("=" * 80)
    
    print(f"\n[데이터셋 크기]")
    print(f"   • 전체 레코드 수: {len(df):,}개")
    print(f"   • 고유 이미지 수: {df['file_name'].nunique():,}개")
    print(f"   • 고유 약 종류: {df['category_id'].nunique()}개")
    
    print(f"\n[이미지당 객체 수]")
    objects_per_image = df.groupby('file_name').size()
    print(f"   • 평균: {objects_per_image.mean():.2f}개")
    print(f"   • 최소: {objects_per_image.min()}개")
    print(f"   • 최대: {objects_per_image.max()}개")
    print(f"   • 중앙값: {objects_per_image.median():.0f}개")
    
    print(f"\n[카테고리당 샘플 수]")
    samples_per_category = df.groupby('category_id').size()
    print(f"   • 평균: {samples_per_category.mean():.2f}개")
    print(f"   • 최소: {samples_per_category.min()}개 (ID: {samples_per_category.idxmin()})")
    print(f"   • 최대: {samples_per_category.max()}개 (ID: {samples_per_category.idxmax()})")
    
    print(f"\n[파일 경로 존재 여부]")
    existing_images = df['image_path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False).sum()
    print(f"   • 존재하는 이미지: {existing_images:,}개")
    print(f"   • 존재하지 않는 이미지: {len(df) - existing_images:,}개")


def print_missing_values(df: pd.DataFrame):
    """결측값 검사"""
    
    print("\n" + "=" * 80)
    print("🔍 2. 결측값 검사")
    print("=" * 80)
    
    missing = df.isnull().sum()
    missing_percent = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        '컬럼': missing.index,
        '결측 개수': missing.values,
        '결측 비율(%)': missing_percent.values
    })
    
    missing_df = missing_df[missing_df['결측 개수'] > 0].sort_values('결측 개수', ascending=False)
    
    if len(missing_df) > 0:
        print(f"\n⚠️  결측값이 있는 컬럼:")
        print(missing_df.to_string(index=False))
    else:
        print(f"\n✅ 결측값 없음!")
    
    # bbox 관련 결측값 상세 확인
    bbox_cols = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
    bbox_missing = df[bbox_cols].isnull().any(axis=1)
    
    if bbox_missing.sum() > 0:
        print(f"\n⚠️  bbox 결측값이 있는 레코드: {bbox_missing.sum()}개")
        print("\n상위 5개 예시:")
        print(df[bbox_missing][['file_name', 'category_name'] + bbox_cols].head().to_string(index=False))


def print_duplicate_analysis(df: pd.DataFrame):
    """중복값 분석"""
    
    print("\n" + "=" * 80)
    print("🔄 3. 중복값 분석")
    print("=" * 80)
    
    # 완전 중복 레코드
    full_duplicates = df.duplicated(subset=['file_name', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']).sum()
    print(f"\n[완전 중복 레코드]")
    print(f"   • 중복 개수: {full_duplicates}개")
    
    if full_duplicates > 0:
        dup_df = df[df.duplicated(subset=['file_name', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'], keep=False)]
        print("\n상위 5개 예시:")
        print(dup_df[['file_name', 'category_name', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].head(10).to_string(index=False))
    
    # 같은 이미지, 같은 bbox에 다른 카테고리
    df['bbox_tuple'] = df.apply(lambda x: (x['bbox_x'], x['bbox_y'], x['bbox_w'], x['bbox_h']), axis=1)
    same_bbox_diff_category = df.groupby(['file_name', 'bbox_tuple']).filter(lambda x: len(x) > 1 and x['category_id'].nunique() > 1)
    
    print(f"\n[같은 bbox, 다른 카테고리]")
    print(f"   • 문제 레코드: {len(same_bbox_diff_category)}개")
    
    if len(same_bbox_diff_category) > 0:
        print("\n상위 5개 예시:")
        print(same_bbox_diff_category[['file_name', 'category_name', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].head(10).to_string(index=False))


def print_outlier_analysis(df: pd.DataFrame):
    """이상치 분석"""
    
    print("\n" + "=" * 80)
    print("⚡ 4. 이상치 분석")
    print("=" * 80)
    
    IMG_W, IMG_H = 976, 1280
    
    # bbox 이상치
    print(f"\n[bbox 좌표 이상치]")
    
    # 음수 좌표
    negative_coords = df[(df['bbox_x'] < 0) | (df['bbox_y'] < 0)]
    print(f"   • 음수 좌표: {len(negative_coords)}개")
    
    # 0 크기
    zero_size = df[(df['bbox_w'] <= 0) | (df['bbox_h'] <= 0)]
    print(f"   • 0 또는 음수 크기: {len(zero_size)}개")
    
    # 경계 이탈
    out_of_bounds = df[(df['bbox_x'] + df['bbox_w'] > IMG_W) | (df['bbox_y'] + df['bbox_h'] > IMG_H)]
    print(f"   • 이미지 경계 이탈: {len(out_of_bounds)}개")
    
    if len(out_of_bounds) > 0:
        print("\n   경계 이탈 상위 5개:")
        oob_sample = out_of_bounds.head()
        for idx, row in oob_sample.iterrows():
            overflow_x = max(0, row['bbox_x'] + row['bbox_w'] - IMG_W)
            overflow_y = max(0, row['bbox_y'] + row['bbox_h'] - IMG_H)
            print(f"      • {row['file_name']}: X초과 {overflow_x}px, Y초과 {overflow_y}px")
    
    # bbox 크기 이상치 (IQR 방식)
    print(f"\n[bbox 크기 이상치]")
    
    Q1_w = df['bbox_w'].quantile(0.25)
    Q3_w = df['bbox_w'].quantile(0.75)
    IQR_w = Q3_w - Q1_w
    
    Q1_h = df['bbox_h'].quantile(0.25)
    Q3_h = df['bbox_h'].quantile(0.75)
    IQR_h = Q3_h - Q1_h
    
    width_outliers = df[(df['bbox_w'] < Q1_w - 1.5 * IQR_w) | (df['bbox_w'] > Q3_w + 1.5 * IQR_w)]
    height_outliers = df[(df['bbox_h'] < Q1_h - 1.5 * IQR_h) | (df['bbox_h'] > Q3_h + 1.5 * IQR_h)]
    
    print(f"   • 너비 이상치: {len(width_outliers)}개 (Q1={Q1_w:.0f}, Q3={Q3_w:.0f})")
    print(f"   • 높이 이상치: {len(height_outliers)}개 (Q1={Q1_h:.0f}, Q3={Q3_h:.0f})")
    
    # Area 이상치
    print(f"\n[면적 이상치]")
    print(f"   • 최소 면적: {df['area'].min():.0f} px²")
    print(f"   • 최대 면적: {df['area'].max():.0f} px²")
    print(f"   • 평균 면적: {df['area'].mean():.0f} px²")
    
    very_small_area = df[df['area'] < 1000]
    very_large_area = df[df['area'] > 200000]
    print(f"   • 매우 작은 면적 (<1000): {len(very_small_area)}개")
    print(f"   • 매우 큰 면적 (>200000): {len(very_large_area)}개")


def print_category_distribution(df: pd.DataFrame, categories: pd.DataFrame):
    """카테고리 분포 분석"""
    
    print("\n" + "=" * 80)
    print("📦 5. 카테고리 분포")
    print("=" * 80)
    
    category_counts = df['category_id'].value_counts().sort_values(ascending=False)
    
    print(f"\n[전체 카테고리]")
    print(f"   • 전체 카테고리 수: {len(category_counts)}개")
    print(f"   • 샘플이 가장 많은 카테고리: ID {category_counts.index[0]} ({category_counts.iloc[0]}개)")
    print(f"   • 샘플이 가장 적은 카테고리: ID {category_counts.index[-1]} ({category_counts.iloc[-1]}개)")
    
    print(f"\n[상위 10개 카테고리]")
    top_10 = category_counts.head(10)
    for cat_id, count in top_10.items():
        cat_name = categories[categories['category_id'] == cat_id]['category_name'].values
        cat_name = cat_name[0] if len(cat_name) > 0 else 'Unknown'
        print(f"   • ID {cat_id:5d} ({cat_name[:30]:30s}): {count:4d}개")
    
    print(f"\n[하위 10개 카테고리]")
    bottom_10 = category_counts.tail(10)
    for cat_id, count in bottom_10.items():
        cat_name = categories[categories['category_id'] == cat_id]['category_name'].values
        cat_name = cat_name[0] if len(cat_name) > 0 else 'Unknown'
        print(f"   • ID {cat_id:5d} ({cat_name[:30]:30s}): {count:4d}개")


def print_bbox_statistics(df: pd.DataFrame):
    """bbox 통계"""
    
    print("\n" + "=" * 80)
    print("📏 6. Bbox 통계")
    print("=" * 80)
    
    print(f"\n[bbox 위치 (x, y)]")
    print(f"   • X 좌표: 평균={df['bbox_x'].mean():.1f}, 중앙값={df['bbox_x'].median():.1f}, 범위=[{df['bbox_x'].min():.0f}, {df['bbox_x'].max():.0f}]")
    print(f"   • Y 좌표: 평균={df['bbox_y'].mean():.1f}, 중앙값={df['bbox_y'].median():.1f}, 범위=[{df['bbox_y'].min():.0f}, {df['bbox_y'].max():.0f}]")
    
    print(f"\n[bbox 크기 (w, h)]")
    print(f"   • 너비: 평균={df['bbox_w'].mean():.1f}, 중앙값={df['bbox_w'].median():.1f}, 범위=[{df['bbox_w'].min():.0f}, {df['bbox_w'].max():.0f}]")
    print(f"   • 높이: 평균={df['bbox_h'].mean():.1f}, 중앙값={df['bbox_h'].median():.1f}, 범위=[{df['bbox_h'].min():.0f}, {df['bbox_h'].max():.0f}]")
    
    print(f"\n[bbox 면적]")
    print(f"   • 평균: {df['area'].mean():.0f} px²")
    print(f"   • 중앙값: {df['area'].median():.0f} px²")
    print(f"   • 표준편차: {df['area'].std():.0f} px²")
    
    print(f"\n[bbox 종횡비 (w/h)]")
    df['aspect_ratio'] = df['bbox_w'] / df['bbox_h']
    print(f"   • 평균: {df['aspect_ratio'].mean():.2f}")
    print(f"   • 중앙값: {df['aspect_ratio'].median():.2f}")
    print(f"   • 정사각형에 가까운 비율 (0.9~1.1): {len(df[(df['aspect_ratio'] > 0.9) & (df['aspect_ratio'] < 1.1)])}개 ({len(df[(df['aspect_ratio'] > 0.9) & (df['aspect_ratio'] < 1.1)])/len(df)*100:.1f}%)")


def verify_image_json_matching(df: pd.DataFrame):
    """이미지-JSON 매칭 검증"""
    
    print("\n" + "=" * 80)
    print("🔗 7. 이미지-JSON 매칭 검증")
    print("=" * 80)
    
    print(f"\n[파일 존재 여부 확인]")
    
    # 이미지 파일 확인
    existing_images = 0
    missing_images = []
    
    for idx, row in df.iterrows():
        if pd.notna(row['image_path']) and os.path.exists(row['image_path']):
            existing_images += 1
        else:
            missing_images.append(row['file_name'])
        
        if idx > 0 and idx % 1000 == 0:
            print(f"   확인 중... {idx}/{len(df)}")
    
    print(f"\n   • 존재하는 이미지: {existing_images:,}개 ({existing_images/len(df)*100:.1f}%)")
    print(f"   • 누락된 이미지: {len(missing_images):,}개")
    
    if len(missing_images) > 0:
        print(f"\n   누락된 이미지 (상위 10개):")
        for img in missing_images[:10]:
            print(f"      • {img}")
    
    # JSON 파일 확인 (json_path 컬럼이 있는 경우)
    if 'json_path' in df.columns:
        existing_jsons = df['json_path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False).sum()
        print(f"\n   • 존재하는 JSON: {existing_jsons:,}개 ({existing_jsons/len(df)*100:.1f}%)")
        print(f"   • 누락된 JSON: {len(df) - existing_jsons:,}개")


def visualize_dataset_overview(df: pd.DataFrame, categories: pd.DataFrame):
    """데이터셋 전체 시각화"""
    
    print("\n" + "=" * 80)
    print("📈 8. 데이터셋 시각화")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('데이터셋 종합 시각화', fontsize=16, fontweight='bold')
    
    # 1. 카테고리별 샘플 수 (상위 20개)
    category_counts = df['category_id'].value_counts().head(20)
    axes[0, 0].barh(range(len(category_counts)), category_counts.values)
    axes[0, 0].set_yticks(range(len(category_counts)))
    axes[0, 0].set_yticklabels([f'ID {cat_id}' for cat_id in category_counts.index])
    axes[0, 0].set_xlabel('샘플 수')
    axes[0, 0].set_title('카테고리별 샘플 수 (상위 20개)')
    axes[0, 0].invert_yaxis()
    
    # 2. bbox 크기 분포
    axes[0, 1].hist2d(df['bbox_w'], df['bbox_h'], bins=50, cmap='Blues')
    axes[0, 1].set_xlabel('Bbox 너비')
    axes[0, 1].set_ylabel('Bbox 높이')
    axes[0, 1].set_title('Bbox 크기 분포')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. bbox 위치 분포
    axes[0, 2].scatter(df['bbox_x'] + df['bbox_w']/2, df['bbox_y'] + df['bbox_h']/2, 
                      alpha=0.1, s=1)
    axes[0, 2].set_xlim(0, 976)
    axes[0, 2].set_ylim(1280, 0)  # Y축 반전
    axes[0, 2].set_xlabel('중심 X 좌표')
    axes[0, 2].set_ylabel('중심 Y 좌표')
    axes[0, 2].set_title('Bbox 중심 위치 분포')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 면적 분포
    axes[1, 0].hist(df['area'], bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('면적 (px²)')
    axes[1, 0].set_ylabel('빈도')
    axes[1, 0].set_title('Bbox 면적 분포')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 종횡비 분포
    df['aspect_ratio'] = df['bbox_w'] / df['bbox_h']
    axes[1, 1].hist(df['aspect_ratio'], bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('종횡비 (w/h)')
    axes[1, 1].set_ylabel('빈도')
    axes[1, 1].set_title('Bbox 종횡비 분포')
    axes[1, 1].axvline(x=1.0, color='red', linestyle='--', label='정사각형')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 이미지당 객체 수
    objects_per_image = df.groupby('file_name').size()
    axes[1, 2].hist(objects_per_image, bins=range(1, objects_per_image.max()+2), 
                   edgecolor='black', align='left')
    axes[1, 2].set_xlabel('이미지당 객체 수')
    axes[1, 2].set_ylabel('이미지 수')
    axes[1, 2].set_title('이미지당 객체 수 분포')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_overview.png', dpi=150, bbox_inches='tight')
    print(f"\n   ✅ 시각화 저장: dataset_overview.png")
    plt.show()


def visualize_random_samples(df: pd.DataFrame, n_samples: int = 20, save_dir: str = "sample_visualizations"):
    """
    랜덤 샘플 이미지에 bbox 시각화
    
    Args:
        df: 데이터셋 DataFrame
        n_samples: 시각화할 샘플 수
        save_dir: 저장 디렉토리
    """
    
    print("\n" + "=" * 80)
    print(f"🖼️  랜덤 샘플 {n_samples}개 시각화")
    print("=" * 80)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 이미지가 실제로 존재하는 레코드만 필터링
    valid_df = df[df['image_path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)]
    
    if len(valid_df) == 0:
        print("\n⚠️  유효한 이미지가 없습니다!")
        return
    
    # 랜덤 샘플 선택
    sample_images = valid_df['file_name'].unique()
    n_samples = min(n_samples, len(sample_images))
    selected_images = random.sample(list(sample_images), n_samples)
    
    print(f"\n   선택된 이미지: {n_samples}개")
    
    # 각 이미지 시각화
    for idx, img_name in enumerate(selected_images, 1):
        # 해당 이미지의 모든 객체 가져오기
        img_data = valid_df[valid_df['file_name'] == img_name]
        
        if len(img_data) == 0:
            continue
        
        img_path = img_data.iloc[0]['image_path']
        
        try:
            # 이미지 로드
            img = Image.open(img_path)
            
            # 플롯 생성
            fig, ax = plt.subplots(1, 1, figsize=(12, 16))
            ax.imshow(img)
            
            # 각 객체에 bbox 그리기
            colors = plt.cm.rainbow(np.linspace(0, 1, len(img_data)))
            
            for obj_idx, (_, row) in enumerate(img_data.iterrows()):
                x, y, w, h = row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']
                category = row['category_name']
                
                # bbox 사각형
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor=colors[obj_idx],
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # 라벨
                label_text = f"{category}\nID: {row['category_id']}\n({w:.0f}x{h:.0f})"
                ax.text(
                    x, y - 10,
                    label_text,
                    fontsize=9,
                    color='white',
                    bbox=dict(boxstyle='round', facecolor=colors[obj_idx], alpha=0.8),
                    verticalalignment='bottom'
                )
            
            ax.set_title(f"샘플 {idx}: {img_name}\n객체 수: {len(img_data)}개", 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # 저장
            save_path = os.path.join(save_dir, f"sample_{idx:02d}_{img_name}")
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ [{idx}/{n_samples}] {img_name}: {len(img_data)}개 객체")
            
        except Exception as e:
            print(f"   ❌ [{idx}/{n_samples}] {img_name}: 오류 - {str(e)}")
    
    print(f"\n   💾 시각화 저장 완료: {save_dir}/")


def export_summary_report(df: pd.DataFrame, categories: pd.DataFrame, output_path: str = "dataset_summary.txt"):
    """
    종합 보고서를 텍스트 파일로 저장
    
    Args:
        df: 데이터셋 DataFrame
        categories: 카테고리 DataFrame
        output_path: 저장 경로
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("데이터셋 종합 보고서\n")
        f.write("=" * 80 + "\n\n")
        
        # 기본 정보
        f.write("1. 기본 정보\n")
        f.write("-" * 80 + "\n")
        f.write(f"전체 레코드 수: {len(df):,}개\n")
        f.write(f"고유 이미지 수: {df['file_name'].nunique():,}개\n")
        f.write(f"고유 카테고리 수: {df['category_id'].nunique()}개\n\n")
        
        # 결측값
        f.write("2. 결측값\n")
        f.write("-" * 80 + "\n")
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                f.write(f"{col}: {count}개 ({count/len(df)*100:.2f}%)\n")
        if missing.sum() == 0:
            f.write("결측값 없음\n")
        f.write("\n")
        
        # 중복값
        f.write("3. 중복값\n")
        f.write("-" * 80 + "\n")
        dup_count = df.duplicated(subset=['file_name', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']).sum()
        f.write(f"완전 중복 레코드: {dup_count}개\n\n")
        
        # bbox 통계
        f.write("4. Bbox 통계\n")
        f.write("-" * 80 + "\n")
        f.write(f"평균 크기: {df['bbox_w'].mean():.1f} x {df['bbox_h'].mean():.1f}\n")
        f.write(f"평균 면적: {df['area'].mean():.0f} px²\n")
        f.write(f"평균 종횡비: {(df['bbox_w'] / df['bbox_h']).mean():.2f}\n\n")
        
        # 카테고리 분포
        f.write("5. 카테고리 분포 (상위 20개)\n")
        f.write("-" * 80 + "\n")
        category_counts = df['category_id'].value_counts().head(20)
        for cat_id, count in category_counts.items():
            cat_name = categories[categories['category_id'] == cat_id]['category_name'].values
            cat_name = cat_name[0] if len(cat_name) > 0 else 'Unknown'
            f.write(f"ID {cat_id:5d} ({cat_name[:40]:40s}): {count:4d}개\n")
    
    print(f"\n💾 종합 보고서 저장: {output_path}")



if __name__ == "__main__":
    
    # 데이터셋 경로 설정
    DATASET_CSV = r"E:\download\datasets\TS_8_combination\TS_8_combination_dataset.csv"
    CATEGORY_CSV = r"E:\download\datasets\TS_8_combination\TS_8_combination_categories.csv"

    print("🔍 데이터셋 검증 시작\n")

    # 1. 종합 검증
    df, categories = comprehensive_dataset_inspection(DATASET_CSV, CATEGORY_CSV)

    # 2. 랜덤 샘플 시각화 (20개)
    visualize_random_samples(df, n_samples=20, save_dir="sample_visualizations")

    # 3. 종합 보고서 저장
    export_summary_report(df, categories, "dataset_summary_report.txt")

    print("\n✅ 모든 검증 완료!")