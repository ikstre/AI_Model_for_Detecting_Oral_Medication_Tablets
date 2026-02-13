import json
import csv
import re
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def list_images(img_dir: Path):
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: str(p).lower())
    return files

def xyxy_to_xywh(x1, y1, x2, y2):
    return x1, y1, (x2 - x1), (y2 - y1)

def load_index_to_id(gci_json_path: str):
    with open(gci_json_path, "r", encoding="utf-8") as f:
        gci = json.load(f)
    idx2id_raw = gci["index_to_id"]
    return {int(k): int(v) for k, v in idx2id_raw.items()}

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def extract_image_id_from_filename(p: Path) -> int:
    stem = p.stem
    if stem.isdigit():
        return int(stem)

    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"파일명에서 숫자를 찾을 수 없음: {p.name}")
    return int(nums[-1])  # 필요 시 nums[0]로 변경

def run_infer_to_csv(
    weights_path: str,
    test_img_dir: str,
    out_csv_path: str,
    gci_json_path: str,
    imgsz: int = 512,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det_per_image: int = 4,
    device: int = 0,
    pred_batch: int = 4,
    chunk_size: int = 200,
    half: bool = True,
):
    test_img_dir = Path(test_img_dir)
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    index_to_id = load_index_to_id(gci_json_path)
    model = YOLO(weights_path)

    img_files = list_images(test_img_dir)
    if not img_files:
        raise FileNotFoundError(f"No images in: {test_img_dir}")

    # (선택) image_id 중복/규칙 확인용
    seen = set()
    for p in img_files[:50]:  # 앞 50개만 샘플 체크
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
        no_det = 0

        pbar = tqdm(total=len(img_files), desc="추론 진행")

        for chunk in chunked(img_files, chunk_size):
            results_iter = model.predict(
                source=[str(p) for p in chunk],
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                stream=True,
                batch=pred_batch,
                half=half,
                verbose=False,
                max_det=max_det_per_image,
            )

            for res in results_iter:
                img_path = Path(res.path)
                img_id = extract_image_id_from_filename(img_path)  # ✅ 핵심 수정

                if res.boxes is None or len(res.boxes) == 0:
                    no_det += 1
                    pbar.update(1)
                    continue

                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()

                order = confs.argsort()[::-1][:max_det_per_image]

                for i in order:
                    score = float(confs[i])
                    global_index = int(clss[i])

                    if global_index not in index_to_id:
                        continue
                    cat_id = index_to_id[global_index]

                    x1, y1, x2, y2 = xyxy[i]
                    bx, by, bw, bh = xyxy_to_xywh(x1, y1, x2, y2)

                    bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)
                    if bw <= 0 or bh <= 0:
                        continue

                    writer.writerow([ann_id, img_id, cat_id, bx, by, bw, bh, round(score, 6)])
                    ann_id += 1
                    total_det += 1

                pbar.update(1)

        pbar.close()

    print("✅ 완료")
    print(f"- 저장: {out_csv_path}")
    print(f"- 이미지: {len(img_files)}")
    print(f"- 탐지없음: {no_det}")
    print(f"- 총 탐지: {total_det}")

if __name__ == "__main__":
    WEIGHTS = r"E:\yolo_runs\pill_baseline_augdata2\weights\best.pt"
    TEST_DIR = r"E:\download\sprint_ai_project1_data\test_images"
    OUT_CSV  = r"E:\download\submission\submission(aug_sort).csv"
    GCI_JSON = r"E:\download\gci_57_MODEL_SORTED_BY_CATEGORY_ID.json"

    run_infer_to_csv(
        weights_path=WEIGHTS,
        test_img_dir=TEST_DIR,
        out_csv_path=OUT_CSV,
        gci_json_path=GCI_JSON,
        imgsz=1024,
        pred_batch=4,
        chunk_size=200,
        half=True,
        device=0,
    )