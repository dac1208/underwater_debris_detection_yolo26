from pathlib import Path
import json
import random
import shutil

# =========================
# PATHS
# =========================

src = Path(r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD\MUZZA\dataset_muzza")
dst = Path(r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD\MUZZA\muzza_dataset_yolo_6classes")

images_dir = src / "images" / "default"
annotations_json = src / "annotations" / "instances_default.json"

# =========================
# CLASS MAPPING
# =========================

# New 6-class YOLO setup:
# 0 animal
# 1 trash_plastic
# 2 trash_other
# 3 nature
# 4 rov
# 5 unknown

new_names = {
    0: "animal",
    1: "trash_plastic",
    2: "trash_other",
    3: "nature",
    4: "rov",
    5: "unknown",
}

# COCO category name -> new YOLO class id
# Your current MUZZA dataset has only "mask",
# and we map it to trash_other.
category_name_to_new_id = {
    "mask": 2,
}

random_seed = 42
train_ratio = 0.8

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo_bbox_from_coco_bbox(coco_bbox, img_w, img_h):
    """
    COCO bbox format:
      [x_min, y_min, width, height]

    YOLO bbox format:
      x_center y_center width height
    normalized between 0 and 1.
    """
    x_min, y_min, box_w, box_h = coco_bbox

    x_center = x_min + box_w / 2
    y_center = y_min + box_h / 2

    return [
        x_center / img_w,
        y_center / img_h,
        box_w / img_w,
        box_h / img_h,
    ]


def safe_float(x):
    return f"{x:.6f}"


def main():
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if not annotations_json.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {annotations_json}")

    print(f"[INFO] Source dataset: {src}")
    print(f"[INFO] Output dataset: {dst}")

    with open(annotations_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # COCO category id -> category name
    category_id_to_name = {}
    for cat in coco.get("categories", []):
        category_id_to_name[cat["id"]] = cat["name"]

    print("[INFO] Categories found in JSON:")
    for cid, cname in category_id_to_name.items():
        print(f"  {cid}: {cname}")

    # COCO image id -> image info
    image_id_to_info = {}
    for img in coco.get("images", []):
        image_id_to_info[img["id"]] = img

    # Group annotations by image_id
    annotations_by_image_id = {img_id: [] for img_id in image_id_to_info.keys()}

    for ann in coco.get("annotations", []):
        image_id = ann["image_id"]
        category_id = ann["category_id"]

        if category_id not in category_id_to_name:
            print(f"[WARN] Unknown category_id {category_id}, skipping annotation.")
            continue

        category_name = category_id_to_name[category_id]

        if category_name not in category_name_to_new_id:
            print(f"[WARN] Category '{category_name}' not mapped, skipping annotation.")
            continue

        annotations_by_image_id.setdefault(image_id, []).append(ann)

    # Collect image entries that actually exist on disk
    items = []
    missing = 0

    for image_id, img_info in image_id_to_info.items():
        file_name = img_info["file_name"]

        img_path = images_dir / file_name

        if not img_path.exists():
            missing += 1
            print(f"[WARN] Missing image file: {img_path}")
            continue

        if img_path.suffix.lower() not in image_extensions:
            print(f"[WARN] Unsupported image extension: {img_path}")
            continue

        items.append((image_id, img_path, img_info))

    if not items:
        raise RuntimeError("No images found. Check paths and JSON file names.")

    random.seed(random_seed)
    random.shuffle(items)

    train_count = int(len(items) * train_ratio)
    train_items = items[:train_count]
    val_items = items[train_count:]

    print(f"\n[INFO] Total images: {len(items)}")
    print(f"[INFO] Train images: {len(train_items)}")
    print(f"[INFO] Val images: {len(val_items)}")
    print(f"[INFO] Missing images listed in JSON: {missing}")

    # Create output folders
    for split in ["train", "val"]:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)

    def process_split(split_name, split_items):
        for image_id, img_path, img_info in split_items:
            out_img_path = dst / split_name / "images" / img_path.name
            out_label_path = dst / split_name / "labels" / f"{img_path.stem}.txt"

            shutil.copy2(img_path, out_img_path)

            img_w = img_info["width"]
            img_h = img_info["height"]

            yolo_lines = []

            for ann in annotations_by_image_id.get(image_id, []):
                category_name = category_id_to_name[ann["category_id"]]
                new_class_id = category_name_to_new_id[category_name]

                bbox = ann.get("bbox", None)

                if bbox is None:
                    print(f"[WARN] Annotation without bbox for image {img_path.name}, skipping.")
                    continue

                x, y, w, h = yolo_bbox_from_coco_bbox(bbox, img_w, img_h)

                # Optional safety clamp
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                yolo_lines.append(
                    f"{new_class_id} {safe_float(x)} {safe_float(y)} {safe_float(w)} {safe_float(h)}"
                )

            # Empty txt is okay for images without objects
            out_label_path.write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                encoding="utf-8"
            )

    process_split("train", train_items)
    process_split("val", val_items)

    # Write data.yaml
    yaml_text = f"""path: {dst.as_posix()}
train: train/images
val: val/images

names:
  0: animal
  1: trash_plastic
  2: trash_other
  3: nature
  4: rov
  5: unknown
"""

    (dst / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("\nDone.")
    print(f"New YOLO dataset created here:\n{dst}")
    print("\nUse this YAML for training:")
    print(dst / "data.yaml")


if __name__ == "__main__":
    main()