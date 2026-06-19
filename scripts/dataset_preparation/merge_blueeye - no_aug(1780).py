from __future__ import annotations

import json
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# 1) BASE PATH
# ============================================================
BASE_DIR = Path(r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD\ToMerge\Blueeye")
OUT_DIR = BASE_DIR / "BLUEEYE_MERGED.v2"

# ============================================================
# 2) FOLDERI I SPLITOVI KOJI ULAZE U BLUEYE MERGE
# ============================================================
SOURCES = [
    {
        "name": "root_folder_a",
        "root": BASE_DIR,
        "splits": ["test", "valid"],
    },
    {
        "name": "masksea_v1i",
        "root": BASE_DIR / "MaskSea.v1i.coco",
        "splits": ["test", "valid"],
    },
    {
        "name": "robots_are_pain_v1i",
        "root": BASE_DIR / "Robots are pain.v1i.coco",
        "splits": ["test", "valid"],
    },
    {
        "name": "pain_2",
        "root": BASE_DIR / "pain_2",
        "splits": ["valid"],
    },
    {
        "name": "bts_v1i",
        "root": BASE_DIR / "BTS V1.v1i.coco (1)",
        "splits": ["test", "valid"],
    },
    {
        "name": "mask_v4i",
        "root": BASE_DIR / "MASK.v4i.coco",
        "splits": ["test", "valid"],
    },
    {
        "name": "mask_jd_v1i",
        "root": BASE_DIR / "MASK-JD.v1i.coco",
        "splits": ["train", "test", "valid"],
    },
]

# ============================================================
# 3) FINALNE KATEGORIJE
# ============================================================
# kako si tražio:
# 0 = bottle
# 1 = robot
# 2 = mask
# ============================================================
FINAL_CATEGORIES = [
    {"id": 0, "name": "bottle", "supercategory": "none"},
    {"id": 1, "name": "robot", "supercategory": "none"},
    {"id": 2, "name": "mask", "supercategory": "none"},
]

FINAL_CAT_ID = {
    "bottle": 0,
    "robot": 1,
    "mask": 2,
}


def ensure_dirs() -> None:
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_old_to_new_category_map(coco: dict) -> dict[int, int | None]:
    """
    Mapira stari category_id u novi finalni category_id.
    Očekuje da su imena klasa već sređena na:
    bottle / robot / mask
    Sve ostalo ignorira.
    """
    result = {}
    for cat in coco.get("categories", []):
        old_id = cat["id"]
        old_name = cat["name"].strip().lower()

        if old_name in FINAL_CAT_ID:
            result[old_id] = FINAL_CAT_ID[old_name]
        else:
            result[old_id] = None

    return result


def collect_annotations_by_image(coco: dict, old_to_new_cat: dict[int, int | None]) -> dict[int, list]:
    anns_by_image = defaultdict(list)

    for ann in coco.get("annotations", []):
        old_cat_id = ann["category_id"]
        new_cat_id = old_to_new_cat.get(old_cat_id, None)

        if new_cat_id is None:
            continue

        anns_by_image[ann["image_id"]].append((ann, new_cat_id))

    return anns_by_image


def copy_and_merge():
    ensure_dirs()

    merged_images = []
    merged_annotations = []

    next_image_id = 0
    next_annotation_id = 0

    copied_count = 0
    ann_count = 0
    missing_images = 0

    for source in SOURCES:
        source_name = source["name"]
        source_root = source["root"]
        splits = source["splits"]

        if not source_root.exists():
            print(f"[SKIP] Folder ne postoji: {source_root}")
            continue

        for split in splits:
            split_dir = source_root / split
            ann_path = split_dir / "_annotations.coco.json"

            if not split_dir.exists():
                print(f"[SKIP] Split folder ne postoji: {split_dir}")
                continue

            if not ann_path.exists():
                print(f"[SKIP] Nema annotations file: {ann_path}")
                continue

            print(f"[PROCESS] {source_name} / {split}")

            coco = load_json(ann_path)
            old_to_new_cat = build_old_to_new_category_map(coco)
            anns_by_image = collect_annotations_by_image(coco, old_to_new_cat)

            for img in coco.get("images", []):
                old_img_id = img["id"]
                old_file_name = img["file_name"]

                src_img_path = split_dir / old_file_name
                if not src_img_path.exists():
                    print(f"  [MISSING IMAGE] {src_img_path}")
                    missing_images += 1
                    continue

                # da ne dođe do sudara imena među datasetovima
                new_file_name = f"{source_name}__{split}__{old_file_name}"
                dst_img_path = OUT_DIR / "images" / new_file_name

                shutil.copy2(src_img_path, dst_img_path)

                merged_images.append({
                    "id": next_image_id,
                    "license": img.get("license", 1),
                    "file_name": new_file_name,
                    "height": img["height"],
                    "width": img["width"],
                    "date_captured": img.get("date_captured", ""),
                })
                copied_count += 1

                for old_ann, new_cat_id in anns_by_image.get(old_img_id, []):
                    merged_annotations.append({
                        "id": next_annotation_id,
                        "image_id": next_image_id,
                        "category_id": new_cat_id,
                        "bbox": old_ann["bbox"],
                        "area": old_ann.get("area", old_ann["bbox"][2] * old_ann["bbox"][3]),
                        "segmentation": old_ann.get("segmentation", []),
                        "iscrowd": old_ann.get("iscrowd", 0),
                    })
                    next_annotation_id += 1
                    ann_count += 1

                next_image_id += 1

    merged_coco = {
        "info": {
            "description": "Merged Blueeye dataset",
            "version": "1.0",
            "year": 2026,
            "contributor": "Dario",
            "date_created": "",
        },
        "licenses": [
            {
                "id": 1,
                "url": "",
                "name": "Unknown",
            }
        ],
        "categories": FINAL_CATEGORIES,
        "images": merged_images,
        "annotations": merged_annotations,
    }

    save_json(OUT_DIR / "annotations.json", merged_coco)

    print("\n=== GOTOVO ===")
    print(f"Output folder: {OUT_DIR}")
    print(f"Broj kopiranih slika: {copied_count}")
    print(f"Broj anotacija: {ann_count}")
    print(f"Missing images: {missing_images}")


if __name__ == "__main__":
    copy_and_merge()