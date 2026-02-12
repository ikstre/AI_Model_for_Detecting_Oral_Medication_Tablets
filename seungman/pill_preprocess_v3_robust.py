"""
pill_preprocess_v3_robust.py  (수정 버전)

변경 사항:
  [FIX-P1] 파일 하단의 중복 `import re` / `from pathlib import Path` 제거 (L1에서 이미 import)
  [FIX-P2] parse_test_image_id: mode=="digits" 블록과 else 블록이 완전 동일 → else 제거, digits를 기본 fallback으로 통합
  [FIX-P3] resolve_image_path에 image_index(재귀 스캔 결과) 지원 추가 → 노트북에서 별도 함수 재정의할 필요 없음
"""

import os, json, re, random, shutil
from pathlib import Path

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def is_valid_bbox(b):
    if not isinstance(b, (list, tuple)):
        return False
    if len(b) != 4:
        return False
    x, y, w, h = b
    try:
        x = float(x); y = float(y); w = float(w); h = float(h)
    except Exception:
        return False
    return (w > 0) and (h > 0)

def clip_bbox(b, width, height):
    x, y, w, h = map(float, b)
    x = max(0.0, min(x, float(width)))
    y = max(0.0, min(y, float(height)))
    w = max(0.0, min(w, float(width) - x))
    h = max(0.0, min(h, float(height) - y))
    return [x, y, w, h]

def collect_json_files(train_ann_dir):
    ann_dir = Path(train_ann_dir)
    return sorted([p for p in ann_dir.rglob("*.json") if p.is_file()])

def load_json(path):
    # Windows(cp949)에서도 안전하게: utf-8로 읽고 실패 시 errors='replace'
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)

def merge_coco_from_many_json(json_paths):
    merged = {"images": [], "annotations": [], "categories": []}
    seen_img = set()
    seen_ann = set()
    seen_cat = set()

    for p in json_paths:
        obj = load_json(p)

        # categories
        for c in obj.get("categories", []) or []:
            cid = c.get("id", None)
            if cid is None or cid in seen_cat:
                continue
            merged["categories"].append(c)
            seen_cat.add(cid)

        # images
        for im in obj.get("images", []) or []:
            iid = im.get("id", None)
            if iid is None or iid in seen_img:
                continue
            merged["images"].append(im)
            seen_img.add(iid)

        # annotations
        for a in obj.get("annotations", []) or []:
            aid = a.get("id", None)
            if aid is None or aid in seen_ann:
                continue
            merged["annotations"].append(a)
            seen_ann.add(aid)

    return merged

def sanitize_coco_inplace(coco):
    # ✅ ID는 그대로 유지
    # - annotations는 "bbox 존재" + "길이 4" + w,h>0 만 통과
    # - bbox는 이미지 경계로 clip
    img_by_id = {im["id"]: im for im in coco.get("images", []) if "id" in im}

    new_anns = []
    for a in coco.get("annotations", []):
        b = a.get("bbox", None)
        if not b:
            continue
        if not is_valid_bbox(b):
            continue

        iid = a.get("image_id", None)
        im = img_by_id.get(iid, None)
        if im is None:
            continue

        w = im.get("width", None); h = im.get("height", None)
        if w is None or h is None:
            continue

        a2 = dict(a)
        a2["bbox"] = clip_bbox(b, w, h)
        a2["area"] = float(a2["bbox"][2]) * float(a2["bbox"][3])
        new_anns.append(a2)

    coco["annotations"] = new_anns
    return coco

def split_coco_by_image(coco, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    img_ids = [im["id"] for im in coco.get("images", [])]
    rng.shuffle(img_ids)

    n_val = int(len(img_ids) * float(val_ratio))
    val_ids = set(img_ids[:n_val])
    train_ids = set(img_ids[n_val:])

    train_imgs = [im for im in coco["images"] if im["id"] in train_ids]
    val_imgs   = [im for im in coco["images"] if im["id"] in val_ids]
    train_anns = [a for a in coco["annotations"] if a["image_id"] in train_ids]
    val_anns   = [a for a in coco["annotations"] if a["image_id"] in val_ids]

    train_coco = {"images": train_imgs, "annotations": train_anns, "categories": coco.get("categories", [])}
    val_coco   = {"images": val_imgs,   "annotations": val_anns,   "categories": coco.get("categories", [])}
    return train_coco, val_coco

def build_cat_maps(coco):
    # categories가 제공되면 그대로 사용, 비어 있으면 annotations의 category_id에서 유도
    cats = coco.get("categories", []) or []
    if cats:
        cat_ids = sorted([c["id"] for c in cats if "id" in c])
        catid_to_name = {c["id"]: c.get("name", str(c["id"])) for c in cats if "id" in c}
    else:
        cat_ids = sorted(set([
            a.get("category_id") for a in (coco.get("annotations", []) or [])
            if a.get("category_id") is not None
        ]))
        catid_to_name = {cid: str(cid) for cid in cat_ids}

    if not cat_ids:
        raise ValueError("categories(또는 annotations의 category_id)에서 클래스 목록을 만들 수 없습니다. JSON 구조를 확인해 주세요.")

    # ✅ 원본 category_id는 그대로 두고, YOLO 학습용 index만 별도로 매핑 (0..nc-1)
    catid_to_yolo = {cid: i for i, cid in enumerate(cat_ids)}
    yolo_to_catid = {i: cid for cid, i in catid_to_yolo.items()}
    return catid_to_yolo, yolo_to_catid, catid_to_name


# [FIX-P3] image_index 파라미터 추가 → 노트북에서 별도 resolve_train_image_path 재정의 불필요
def resolve_image_path(train_img_dir, file_name, *, image_index=None, yolo_img_dir=None):
    """
    file_name → 실제 이미지 경로를 찾습니다.
    우선순위:
      0) 이미 절대/상대 경로로 존재하면 그대로
      1) yolo_img_dir (YOLO export된 images/) 에서 탐색
      2) image_index (원본 이미지 root 재귀 스캔 dict: {basename: abs_path})
      3) train_img_dir 직하위
      4) 확장자 fallback
    """
    file_name = str(file_name).replace("\\", "/")
    p = Path(file_name)

    # 0) 이미 존재하는 경로
    if p.exists():
        return p

    basename = p.name
    stem = p.stem

    # 1) YOLO export images 폴더
    if yolo_img_dir is not None:
        cand = Path(yolo_img_dir) / basename
        if cand.exists():
            return cand

    # 2) image_index (재귀 스캔 결과)
    if isinstance(image_index, dict):
        s = image_index.get(basename)
        if s and Path(s).exists():
            return Path(s)

    # 3) train_img_dir 직하위
    cand = Path(train_img_dir) / basename
    if cand.exists():
        return cand

    # 4) 확장자 fallback
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        q = Path(train_img_dir) / (stem + ext)
        if q.exists():
            return q

    return None

def coco_bbox_to_yolo_xywhn(b, img_w, img_h, clip=True, eps=1e-6):
    """
    COCO bbox [x, y, w, h] (pixel) -> YOLO normalized [cx, cy, w, h]
    - clip=True: 이미지 경계를 벗어나는 bbox를 잘라내고, 잘린 결과가 유효하지 않으면 None 반환
    """
    if b is None or (not isinstance(b, (list, tuple))) or len(b) != 4:
        return None

    x, y, w, h = map(float, b)
    if img_w <= 0 or img_h <= 0:
        return None

    # xywh -> xyxy
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    if clip:
        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= eps or bh <= eps:
        return None

    cx = (x1 + x2) / 2.0 / float(img_w)
    cy = (y1 + y2) / 2.0 / float(img_h)
    wn = bw / float(img_w)
    hn = bh / float(img_h)

    # 최종 안전 클램프
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    wn = max(eps, min(1.0, wn))
    hn = max(eps, min(1.0, hn))
    return cx, cy, wn, hn


def export_yolo_dataset(train_img_dir, train_coco, val_coco, catid_to_yolo, out_root):
    """
    export_yolo_dataset(...)

    반환:
      train_rel_paths: ["images/train/xxx.png", ...]
      val_rel_paths  : ["images/val/yyy.png", ...]
    """
    out_root = Path(out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    def _write_split(split_name, coco_split):
        img_by_id = {im["id"]: im for im in coco_split.get("images", [])}
        anns_by_img = {}
        for a in coco_split.get("annotations", []):
            anns_by_img.setdefault(a["image_id"], []).append(a)

        rel_paths = []
        for iid, im in img_by_id.items():
            src = resolve_image_path(train_img_dir, im.get("file_name", ""))
            if src is None:
                continue

            dst_img = out_root / "images" / split_name / src.name
            if not dst_img.exists():
                shutil.copy2(src, dst_img)

            # YOLO label txt
            label_lines = []
            for a in anns_by_img.get(iid, []):
                cid = a.get("category_id", None)
                if cid not in catid_to_yolo:
                    continue
                cls = catid_to_yolo[cid]
                out = coco_bbox_to_yolo_xywhn(a["bbox"], im["width"], im["height"], clip=True)
                if out is None:
                    continue
                cx, cy, wn, hn = out
                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

            lbl_path = out_root / "labels" / split_name / (Path(im["file_name"]).stem + ".txt")
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

            rel_paths.append(str(Path("images") / split_name / src.name).replace("\\", "/"))
        return rel_paths

    train_rel = _write_split("train", train_coco)
    val_rel = _write_split("val", val_coco)
    return train_rel, val_rel

def write_split_list(out_root, split_name, rel_paths):
    """
    train/val을 txt 리스트로 넘길 때, 환경(Windows/Mac) 및 Ultralytics 버전 차이로
    상대경로 해석이 흔들릴 수 있어서 **절대경로**로 저장합니다.

    - rel_paths: export_yolo_dataset()가 반환한 "images/train/xxx.png" 같은 경로
    - 결과 txt에는 out_root 기준 절대경로가 기록됩니다.
    """
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    txt_path = out_root / f"{split_name}.txt"

    abs_lines = []
    for rp in rel_paths:
        p = Path(str(rp))
        if not p.is_absolute():
            p = (out_root / p).resolve()
        abs_lines.append(str(p).replace("\\", "/"))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(abs_lines))

    return str(txt_path)


def write_dataset_yaml(out_root, names, yaml_path, train="images/train", val="images/val", test=None):
    out_root = Path(out_root)
    data = {
        "path": str(out_root.resolve()),
        "train": train,
        "val": val,
        "names": names
    }
    if test is not None:
        data["test"] = test

    import yaml
    with open(yaml_path, "w", encoding="utf-8") as f:
        # allow_unicode=False로 저장(Windows 기본 인코딩 문제 회피)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)

def save_coco_json(coco, path, ensure_ascii=True):
    """
    pycocotools.COCO는 Windows에서 기본 인코딩(cp949)으로 읽는 경우가 있어,
    한글/비ASCII가 있으면 UnicodeDecodeError가 날 수 있습니다.

    => ensure_ascii=True로 저장하면 /uXXXX escape가 들어가서 안전합니다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=bool(ensure_ascii))
    return str(path)

# -------------------------
# 클래스 불균형(옵션): class weight -> 이미지 oversampling(train.txt)
# -------------------------
def count_class_instances(coco):
    cnt = {}
    for a in coco.get("annotations", []) or []:
        cid = a.get("category_id", None)
        if cid is None:
            continue
        cnt[cid] = cnt.get(cid, 0) + 1
    return cnt

def compute_class_weight_from_coco(coco, method="balanced"):
    """
    method='balanced' -> n_samples / (n_classes * count[class]) (sklearn과 동일한 휴리스틱)
    반환: {category_id: weight}
    """
    cnt = count_class_instances(coco)
    classes = sorted(cnt.keys())
    if not classes:
        return {}

    n_samples = float(sum(cnt.values()))
    n_classes = float(len(classes))

    w = {}
    if method == "balanced":
        for cid in classes:
            w[cid] = n_samples / (n_classes * float(cnt[cid]))
    else:
        for cid in classes:
            w[cid] = 1.0
    return w

def build_image_weight_map(coco, class_weight_map, empty_weight=1.0, reduce="mean"):
    """
    이미지(=file_name)별 가중치 계산:
      - 이미지 안에 존재하는 bbox들의 class_weight를 평균/최대 등으로 집계
    반환: {file_name: weight}
    """
    img_by_id = {im["id"]: im for im in coco.get("images", []) or []}
    cats_by_img = {}
    for a in coco.get("annotations", []) or []:
        iid = a.get("image_id", None)
        cid = a.get("category_id", None)
        if iid is None or cid is None:
            continue
        cats_by_img.setdefault(iid, []).append(cid)

    out = {}
    for iid, im in img_by_id.items():
        fn = str(im.get("file_name", ""))
        cids = cats_by_img.get(iid, [])
        if not cids:
            out[fn] = float(empty_weight)
            continue
        ws = [float(class_weight_map.get(c, 1.0)) for c in cids]
        if reduce == "max":
            out[fn] = float(max(ws))
        else:
            out[fn] = float(sum(ws) / max(1, len(ws)))
    return out

def make_weighted_train_list(train_rel_paths, train_coco, class_weight_map, extra_ratio=0.0, seed=42, power=1.0):
    """
    train_rel_paths: export_yolo_dataset에서 반환된 'images/train/xxx.png' 리스트
    - 기본: 모든 이미지를 1회씩 포함
    - extra_ratio>0: 추가로 n_extra를 가중치 기반 샘플링해서 train.txt에 덧붙임
    """
    rng = random.Random(seed)

    # file_name -> weight
    imgw = build_image_weight_map(train_coco, class_weight_map)

    weights = []
    for rp in train_rel_paths:
        fn = Path(rp).name
        w = float(imgw.get(fn, 1.0))
        weights.append(max(1e-8, w) ** float(power))

    s = float(sum(weights)) if weights else 1.0
    probs = [w / s for w in weights]

    out = list(train_rel_paths)  # each once
    n_extra = int(len(train_rel_paths) * float(extra_ratio))
    if n_extra > 0 and len(train_rel_paths) > 0:
        # weighted sampling with replacement
        idxs = list(range(len(train_rel_paths)))
        for _ in range(n_extra):
            j = rng.choices(idxs, weights=probs, k=1)[0]
            out.append(train_rel_paths[j])
    return out

# -------------------------
# 제출용 test image_id 매핑 (train과 동일 규칙)
# -------------------------
def build_train_like_image_id_map(file_names, start_id=1):
    uniq = sorted(set([str(x) for x in file_names]))
    m = {}
    cur = int(start_id)
    for fn in uniq:
        m[fn] = cur
        cur += 1
    return m

# [FIX-P1] 중복 import 제거됨 (파일 상단에서 이미 import re, from pathlib import Path)

# [FIX-P2] mode=="digits" 블록과 else fallback이 완전히 동일했으므로, digits를 기본 fallback으로 통합
def parse_test_image_id(path, mode="train_like", seq_id=None, id_map=None, start_id=1):
    path = Path(path)
    fn = path.name

    if mode == "train_like":
        if id_map is None:
            id_map = build_train_like_image_id_map([fn], start_id=start_id)
        return int(id_map.get(fn))

    if mode == "sequential":
        return int(seq_id)

    # mode == "digits" 또는 기타 모든 경우 → 파일명에서 숫자 추출
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return _safe_int(path.stem) or int(seq_id)
    return int("".join(nums))


def diagnose_yolo_dataset(data_yaml_path, n_check=5):
    """
    data.yaml을 읽어 train/val 경로를 실제로 어떻게 해석하는지 점검합니다.
    - Ultralytics는 data.yaml의 path + train/val 조합을 사용합니다.
    """
    import yaml
    p = Path(data_yaml_path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", p.parent)).expanduser()
    def _resolve(x):
        x = str(x)
        px = Path(x)
        return px if px.is_absolute() else (root / px)

    train = _resolve(data["train"])
    val = _resolve(data["val"])
    names = data.get("names", None)

    out = {
        "yaml": str(p),
        "root": str(root),
        "train": str(train),
        "val": str(val),
        "names_type": type(names).__name__,
    }

    # train이 폴더인지 txt인지
    def _count_images(pathlike):
        q = Path(pathlike)
        if q.is_file() and q.suffix.lower() == ".txt":
            lines = [ln.strip() for ln in q.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
            # 상대경로면 root 기준
            files = [Path(ln) if Path(ln).is_absolute() else (root / ln) for ln in lines]
            return len(files), files[:n_check]
        if q.is_dir():
            exts = {".png",".jpg",".jpeg",".bmp",".webp"}
            imgs = [x for x in q.rglob("*") if x.suffix.lower() in exts]
            return len(imgs), imgs[:n_check]
        return 0, []

    out["train_count"], out["train_samples"] = _count_images(train)
    out["val_count"], out["val_samples"] = _count_images(val)
    return out
