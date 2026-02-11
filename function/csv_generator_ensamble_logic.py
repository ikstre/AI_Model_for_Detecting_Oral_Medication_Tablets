import json
import csv
import re
from pathlib import Path
from ultralytics import YOLO
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def list_images(img_dir: Path):
    """이미지 디렉토리에서 모든 이미지 파일을 재귀적으로 찾아 정렬된 리스트로 반환"""
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: str(p).lower())
    return files

def xyxy_to_xywh(x1, y1, x2, y2):
    """YOLO의 xyxy 포맷을 xywh 포맷으로 변환"""
    return x1, y1, (x2 - x1), (y2 - y1)

def load_index_to_id(gci_json_path: str):
    """
    GCI JSON 파일에서 index_to_id 매핑을 로드
    각 모델마다 다른 매핑 테이블을 사용하므로 개별적으로 호출
    """
    with open(gci_json_path, "r", encoding="utf-8") as f:
        gci = json.load(f)
    
    # JSON 구조 유연성 확보
    if "index_to_id" in gci:
        idx2id_raw = gci["index_to_id"]
    else:
        idx2id_raw = gci
        
    return {int(k): int(v) for k, v in idx2id_raw.items()}

def chunked(lst, n):
    """리스트를 n개씩 나누어 배치 처리용 청크로 생성"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def extract_image_id_from_filename(p: Path) -> int:
    """파일명에서 image_id를 추출"""
    stem = p.stem
    if stem.isdigit():
        return int(stem)

    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"파일명에서 숫자를 찾을 수 없음: {p.name}")
    return int(nums[-1])

@torch.no_grad()
def run_ensemble_to_csv(
    spec_weights: str,
    gen_weights: str,
    spec_json: str,
    gen_json: str,
    test_img_dir: str,
    out_csv_path: str,
    
    # 추론 하이퍼파라미터
    imgsz: int = 640,
    spec_conf: float = 0.001,
    gen_conf: float = 0.001,
    iou_nms: float = 0.7,
    max_det_per_image: int = 4,
    device: int = 0,
    pred_batch: int = 4,
    chunk_size: int = 200,
    half: bool = True,
    
    # 앙상블 로직 하이퍼파라미터
    high_conf_threshold: float = 0.85,
    low_conf_threshold: float = 0.15,
    iou_match_threshold: float = 0.5,
):
    """
    전문가 모델과 일반 모델을 앙상블하여 추론 결과를 CSV로 저장
    
    **앙상블 로직:**
    - 전문가 모델 score >= 0.85: 전문가 결과 채택
    - 전문가 모델 score < 0.15: IOU > 0.5인 일반 모델 박스로 교체
    - 그 외 (0.15 ~ 0.85): 전문가 결과 유지
    - BBox는 항상 전문가 모델 기준 사용
    
    **OOD 대비:** 전문가 모델 미탐지 시 일반 모델 결과를 백업으로 사용
    """
    
    # 경로 설정
    test_img_dir = Path(test_img_dir)
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 각 모델의 매핑 테이블 개별 로드
    print("📋 매핑 테이블 로드 중...")
    spec_idx_to_id = load_index_to_id(spec_json)
    gen_idx_to_id = load_index_to_id(gen_json)
    print(f"  - 전문가 모델: {len(spec_idx_to_id)} 클래스")
    print(f"  - 일반 모델: {len(gen_idx_to_id)} 클래스")

    # 모델 로드
    print("🤖 모델 로드 중...")
    model_spec = YOLO(spec_weights)
    model_gen = YOLO(gen_weights)

    # 이미지 파일 목록
    img_files = list_images(test_img_dir)
    if not img_files:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {test_img_dir}")
    print(f"📸 총 {len(img_files)}개 이미지 발견")

    # 이미지 ID 중복 확인 (선택사항)
    seen = set()
    for p in img_files[:50]:
        iid = extract_image_id_from_filename(p)
        if iid in seen:
            print(f"⚠️ image_id 중복 가능성: {iid} (파일: {p.name})")
        seen.add(iid)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "annotation_id", "image_id", "category_id",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
        ])

        ann_id = 1
        total_det = 0
        no_det_both = 0
        no_det_spec = 0
        ensemble_switches = 0
        ood_backups = 0

        pbar = tqdm(total=len(img_files), desc="앙상블 추론 진행")

        # 배치 단위 처리로 성능 최적화
        for chunk in chunked(img_files, chunk_size):
            sources = [str(p) for p in chunk]

            # 두 모델에 대해 스트림 추론 수행
            spec_results_iter = model_spec.predict(
                source=sources,
                imgsz=imgsz,
                conf=spec_conf,
                iou=iou_nms,
                device=device,
                stream=True,
                batch=pred_batch,
                half=half,
                verbose=False,
                max_det=max_det_per_image,
            )

            gen_results_iter = model_gen.predict(
                source=sources,
                imgsz=imgsz,
                conf=gen_conf,
                iou=iou_nms,
                device=device,
                stream=True,
                batch=pred_batch,
                half=half,
                verbose=False,
                max_det=max_det_per_image,
            )

            # 두 모델의 결과를 동시에 처리
            for res_spec, res_gen in zip(spec_results_iter, gen_results_iter):
                img_path = Path(res_spec.path)
                img_id = extract_image_id_from_filename(img_path)

                spec_boxes = res_spec.boxes
                gen_boxes = res_gen.boxes

                # 박스 존재 여부 확인
                has_spec = spec_boxes is not None and len(spec_boxes) > 0
                has_gen = gen_boxes is not None and len(gen_boxes) > 0

                # **케이스 1: 둘 다 탐지 없음**
                if not has_spec and not has_gen:
                    no_det_both += 1
                    pbar.update(1)
                    continue

                # **케이스 2: 전문가 모델 미탐지, 일반 모델 탐지 (OOD 백업)**
                if not has_spec and has_gen:
                    no_det_spec += 1
                    gen_confs = gen_boxes.conf.cpu().numpy()
                    gen_clss = gen_boxes.cls.cpu().numpy()
                    gen_xyxy = gen_boxes.xyxy.cpu().numpy()
                    
                    # confidence 순으로 정렬하여 상위 선택
                    order_gen = gen_confs.argsort()[::-1][:max_det_per_image]
                    
                    for gi in order_gen:
                        g_cls = int(gen_clss[gi])
                        if g_cls not in gen_idx_to_id:
                            continue
                        
                        g_cat_id = gen_idx_to_id[g_cls]
                        g_score = float(gen_confs[gi])
                        
                        gx1, gy1, gx2, gy2 = gen_xyxy[gi]
                        bx, by, bw, bh = xyxy_to_xywh(gx1, gy1, gx2, gy2)
                        bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)
                        
                        if bw <= 0 or bh <= 0:
                            continue
                            
                        writer.writerow([
                            ann_id, img_id, g_cat_id,
                            bx, by, bw, bh, round(g_score, 6)
                        ])
                        ann_id += 1
                        total_det += 1
                        ood_backups += 1
                    
                    pbar.update(1)
                    continue

                # **케이스 3: 전문가 모델 기반 앙상블 로직**
                if has_spec:
                    spec_xyxy = spec_boxes.xyxy
                    spec_confs = spec_boxes.conf.cpu().numpy()
                    spec_clss = spec_boxes.cls.cpu().numpy()

                    # IOU 계산 (GPU에서 효율적으로 수행)
                    ious = None
                    if has_gen:
                        gen_xyxy = gen_boxes.xyxy
                        ious = box_iou(spec_xyxy, gen_xyxy)  # (Ns, Ng)

                    # confidence 순으로 정렬
                    order_spec = spec_confs.argsort()[::-1][:max_det_per_image]

                    for si in order_spec:
                        s_score = float(spec_confs[si])
                        s_cls = int(spec_clss[si])

                        # 전문가 매핑 테이블 확인
                        if s_cls not in spec_idx_to_id:
                            continue

                        # 기본값: 전문가 모델 결과 사용
                        final_cat_id = spec_idx_to_id[s_cls]
                        final_score = s_score
                        final_bbox_xyxy = spec_xyxy[si]

                        # **앙상블 판단 로직**
                        if s_score >= high_conf_threshold:
                            # 고신뢰도: 전문가 결과 그대로 사용
                            pass
                        elif s_score < low_conf_threshold and ious is not None:
                            # 저신뢰도: 일반 모델 결과로 교체 시도
                            iou_row = ious[si]
                            best_gen_idx = int(torch.argmax(iou_row))
                            best_iou = float(iou_row[best_gen_idx])

                            if best_iou > iou_match_threshold:
                                g_score = float(gen_boxes.conf[best_gen_idx])
                                g_cls = int(gen_boxes.cls[best_gen_idx])
                                
                                if g_cls in gen_idx_to_id:
                                    final_cat_id = gen_idx_to_id[g_cls]
                                    final_score = g_score
                                    ensemble_switches += 1
                                    # BBox는 여전히 전문가 모델 것 사용
                        else:
                            # 중간 신뢰도: 전문가 결과 유지
                            pass

                        # 최종 결과 기록
                        x1, y1, x2, y2 = final_bbox_xyxy.cpu().numpy()
                        bx, by, bw, bh = xyxy_to_xywh(x1, y1, x2, y2)
                        bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)

                        if bw <= 0 or bh <= 0:
                            continue

                        writer.writerow([
                            ann_id, img_id, final_cat_id,
                            bx, by, bw, bh, round(final_score, 6)
                        ])
                        ann_id += 1
                        total_det += 1

                pbar.update(1)

        pbar.close()

    # **결과 요약 출력**
    print("\n✅ 앙상블 추론 완료")
    print(f"{'='*60}")
    print(f"📁 저장 경로: {out_csv_path}")
    print(f"📸 처리 이미지: {len(img_files)}개")
    print(f"🔍 총 탐지 수: {total_det}개")
    print(f"❌ 둘 다 탐지 없음: {no_det_both}개")
    print(f"⚠️ 전문가만 탐지 없음: {no_det_spec}개")
    print(f"🔄 앙상블 전환: {ensemble_switches}개")
    print(f"🆘 OOD 백업 사용: {ood_backups}개")
    print(f"{'='*60}")


if __name__ == "__main__":
    # 실행 예시
    run_ensemble_to_csv(
        spec_weights=r"E:\yolo_runs\specialist\weights\best.pt",
        gen_weights=r"E:\yolo_runs\generalist\weights\best.pt",
        spec_json=r"E:\download\spec_57_mapping.json",
        gen_json=r"E:\download\gen_4565_mapping.json",
        test_img_dir=r"E:\download\sprint_ai_project1_data\test_images",
        out_csv_path=r"E:\download\submission\ensemble_result.csv",
        
        # 하이퍼파라미터 조정 가능
        imgsz=640,
        spec_conf=0.001,
        gen_conf=0.001,
        iou_nms=0.7,
        max_det_per_image=4,
        device=0,
        pred_batch=4,
        chunk_size=200,
        half=True,
        high_conf_threshold=0.85,
        low_conf_threshold=0.15,
        iou_match_threshold=0.5,
    )