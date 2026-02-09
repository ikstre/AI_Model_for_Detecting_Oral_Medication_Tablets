import json

src = r"E:\download\gci_57.json"
dst = r"E:\download\gci_57_MODEL_SORTED_BY_CATEGORY_ID.json"

with open(src, "r", encoding="utf-8") as f:
    gci = json.load(f)

# 57개 category_id
ids = sorted(int(k) for k in gci["id_to_index"].keys())

# 오름차순 기준으로 새 매핑 생성
index_to_id = {str(i): ids[i] for i in range(len(ids))}
id_to_index = {str(ids[i]): i for i in range(len(ids))}

gci2 = dict(gci)
gci2["index_to_id"] = index_to_id
gci2["id_to_index"] = id_to_index

# 메타도 맞춰주기
gci2["metadata"] = dict(gci.get("metadata", {}))
gci2["metadata"]["total_categories"] = len(ids)
gci2["metadata"]["index_range"] = f"0-{len(ids)-1}"

with open(dst, "w", encoding="utf-8") as f:
    json.dump(gci2, f, ensure_ascii=False, indent=2)

print("saved:", dst)