import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def load_gci(gci_json_path: str):
    gci_json_path = Path(gci_json_path)
    with open(gci_json_path, "r", encoding="utf-8") as f:
        gci = json.load(f)

    # id_to_index: {"1900": 0, ...}
    id_to_index = {int(k): int(v) for k, v in gci["id_to_index"].items()}

    # index_to_id: {"0": 1900, ...}
    index_to_id = {int(k): int(v) for k, v in gci["index_to_id"].items()}

    # category_map: {"1900": {"category_name": "...", ...}, ...}
    # 저장 구조에 따라 key가 int or str일 수 있어서 정규화
    raw_cat_map = gci["category_map"]
    category_map = {}
    for k, v in raw_cat_map.items():
        category_map[int(k)] = v

    total_categories = int(gci["metadata"]["total_categories"])

    return id_to_index, index_to_id, category_map, total_categories


def build_local_index_to_category_id(local_categories_csv: str):
    """
    로컬 YOLO 인덱스(0..K-1) -> category_id 생성
    전제: categories.csv가 drop_duplicates, sort_values('category_id'), reset_index(drop=True)로 만들어졌다는 가정.
    """
    df = pd.read_csv(local_categories_csv)

    # 로컬 인덱스는 df의 row index
    if "category_id" not in df.columns:
        raise ValueError("local_categories_csv에 'category_id' 컬럼이 없습니다.")

    local_idx_to_catid = {int(i): int(row["category_id"]) for i, row in df.iterrows()}
    return local_idx_to_catid, df


def remap_labels_to_global(labels_dir: str, local_idx_to_global_idx: dict):
    labels_dir = Path(labels_dir)
    txt_files = list(labels_dir.rglob("*.txt"))

    if not txt_files:
        print(f"[WARN] 라벨 txt가 없습니다: {labels_dir}")
        return 0

    changed = 0
    for txt_path in tqdm(txt_files, desc="Remapping labels"):
        lines_out = []
        modified = False

        lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) == 1 and lines[0] == "":
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                # YOLO 라벨은 최소 5개(class cx cy w h)
                lines_out.append(line)
                continue

            local_cls = int(float(parts[0]))  # 혹시 "0.0" 같은 케이스 방지
            if local_cls not in local_idx_to_global_idx:
                raise ValueError(
                    f"[ERROR] {txt_path}에서 local class {local_cls}를 매핑할 수 없습니다."
                )

            global_cls = local_idx_to_global_idx[local_cls]
            if global_cls != local_cls:
                modified = True

            parts[0] = str(global_cls)
            lines_out.append(" ".join(parts))

        if modified:
            txt_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
            changed += 1

    return changed


def write_gci_yaml(
    output_yaml_path: str,
    dataset_root: str,
    train_rel: str,
    val_rel: str,
    total_categories: int,
    index_to_id: dict,
    category_map: dict,
):
    """
    GCI 기준 data.yaml 작성
    names는 0..(N-1) 모두 작성 (GCI의 global_index 순서)
    """
    output_yaml_path = Path(output_yaml_path)
    dataset_root = str(Path(dataset_root))

    lines = []
    lines.append(f"path: {dataset_root}")
    lines.append(f"train: {train_rel}")
    lines.append(f"val: {val_rel}")
    lines.append("")
    lines.append(f"nc: {total_categories}")
    lines.append("names:")

    for idx in range(total_categories):
        cat_id = index_to_id.get(idx)
        if cat_id is None:
            # 혹시 빠진 인덱스가 있으면 빈 이름 처리
            name = "Unknown"
        else:
            name = category_map.get(cat_id, {}).get("category_name", "Unknown")
            # YAML에서 따옴표 깨지는 것 방지
            name = str(name).replace("'", "''")

        lines.append(f"  {idx}: '{name}'")

    output_yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return str(output_yaml_path)


def main(
    gci_json_path: str,
    local_categories_csv: str,
    yolo_dir: str,
    output_yaml_name: str = "data_gci.yaml",
    train_rel: str = "images",
    val_rel: str = "images",
):
    """
    yolo_dir 구조 예:
      yolo_dir/
        images/
        labels/
        data.yaml  (기존)
    """
    yolo_dir = Path(yolo_dir)
    labels_dir = yolo_dir / "labels"

    id_to_index, index_to_id, category_map, total_categories = load_gci(gci_json_path)
    local_idx_to_catid, local_df = build_local_index_to_category_id(local_categories_csv)

    # 로컬 인덱스 -> 전역 인덱스
    local_idx_to_global_idx = {}
    for local_idx, cat_id in local_idx_to_catid.items():
        if cat_id not in id_to_index:
            raise ValueError(
                f"[ERROR] category_id={cat_id}가 GCI에 없습니다. (local_idx={local_idx})"
            )
        local_idx_to_global_idx[local_idx] = id_to_index[cat_id]

    # 1) 라벨 txt 치환
    changed_files = remap_labels_to_global(str(labels_dir), local_idx_to_global_idx)
    print(f"[OK] 수정된 라벨 파일 수: {changed_files:,} / labels 총 파일 수: {len(list(labels_dir.rglob('*.txt'))):,}")

    # 2) YAML 생성 (GCI 전체 names)
    out_yaml_path = yolo_dir / output_yaml_name
    saved_yaml = write_gci_yaml(
        output_yaml_path=str(out_yaml_path),
        dataset_root=str(yolo_dir),
        train_rel=train_rel,
        val_rel=val_rel,
        total_categories=total_categories,
        index_to_id=index_to_id,
        category_map=category_map,
    )
    print(f"[OK] GCI data.yaml 생성: {saved_yaml}")
    print(f"[INFO] nc={total_categories} (전역 클래스 수)")


if __name__ == "__main__":
    # ====== 환경에 맞게 수정 ======
    GCI_JSON = r"E:\download\global_category_index\global_category_index.json"
    LOCAL_CSV = r"E:\download\datasets\original_trainset\original_trainset_categories.csv"
    YOLO_DIR = r"E:\download\datasets\original_trainset\yolo_format"

    main(GCI_JSON, LOCAL_CSV, YOLO_DIR)