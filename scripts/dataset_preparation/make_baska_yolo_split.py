import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# =========================
# CONFIG (edit if needed)
# =========================
SEED = 0
SPLIT = (0.7, 0.2, 0.1)   # train/val/test for small dataset; you can use (0.8,0.1,0.1)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


DATASET_ROOT = Path(r"baska_dataset")  
COCO_JSON = DATASET_ROOT / "annotations" / "instances_default.json"
IMAGES_DIR = DATASET_ROOT / "images"

# Output YOLO dataset root
OUT_ROOT = Path(r"baska_dataset_yolo")

# =========================
# Helpers
# =========================
def clamp01(x):
    return max(0.0, min(1.0, x))

def coco_bbox_to_yolo(bbox, W, H):
    # COCO bbox: [x_min, y_min, width, height]
    x, y, w, h = bbox
    xc = (x + w / 2) / W
    yc = (y + h / 2) / H
    ww = w / W
    hh = h / H
    return clamp01(xc), clamp01(yc), clamp01(ww), clamp01(hh)

def ensure_dirs():
    for split in ["train", "val", "test"]:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

def write_yaml(names):
    # YOLO dataset YAML
    yaml_path = OUT_ROOT / "baska.yaml"
    # Simple YAML writing without pyyaml dependency:
    lines = []
    lines.append(f"path: {OUT_ROOT.as_posix()}")
    lines.append("train: train/images")
    lines.append("val: val/images")
    lines.append("test: test/images")
    lines.append("names:")
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print("✅ Wrote YAML:", yaml_path)

def main():
    random.seed(SEED)
    ensure_dirs()

    coco = json.loads(COCO_JSON.read_text(encoding="utf-8"))

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    if not images or not cats:
        raise RuntimeError("COCO JSON seems incomplete (missing images or categories).")

    # Map category_id -> contiguous 0..nc-1
    cats_sorted = sorted(cats, key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats_sorted)}
    names = [c.get("name", f"class_{c['id']}") for c in cats_sorted]

    # Group annotations by image_id
    anns_by_img = defaultdict(list)
    for a in anns:
        if "bbox" in a and "image_id" in a and "category_id" in a:
            anns_by_img[a["image_id"]].append(a)

    # Build list of image records with existing files
    valid_images = []
    for im in images:
        fn = im.get("file_name")
        if not fn:
            continue
        src = IMAGES_DIR / fn
        if src.suffix.lower() not in IMG_EXTS:
            continue
        if not src.exists():
            # Some exports store different relative paths - you can adjust here if needed.
            continue
        valid_images.append(im)

    if len(valid_images) == 0:
        raise RuntimeError("No valid images found. Check IMAGES_DIR and file_name paths in JSON.")

    # Shuffle and split
    random.shuffle(valid_images)
    n = len(valid_images)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])
    n_test = n - n_train - n_val

    train_set = valid_images[:n_train]
    val_set = valid_images[n_train:n_train + n_val]
    test_set = valid_images[n_train + n_val:]

    split_map = [("train", train_set), ("val", val_set), ("test", test_set)]
    print(f"Split sizes: train={len(train_set)}, val={len(val_set)}, test={len(test_set)} (total={n})")

    # Convert each split
    for split_name, split_images in split_map:
        for im in split_images:
            img_id = im["id"]
            fn = im["file_name"]
            W = im.get("width")
            H = im.get("height")
            if not W or not H:
                # width/height should exist in COCO; if missing, you can add PIL reading here
                raise RuntimeError(f"Missing width/height for image {fn} in COCO JSON.")

            src_img = IMAGES_DIR / fn
            dst_img = OUT_ROOT / split_name / "images" / src_img.name
            shutil.copy2(src_img, dst_img)

            # Write YOLO label file
            label_lines = []
            for a in anns_by_img.get(img_id, []):
                cid = a["category_id"]
                if cid not in cat_id_to_idx:
                    continue
                cls = cat_id_to_idx[cid]
                xc, yc, ww, hh = coco_bbox_to_yolo(a["bbox"], W, H)
                if ww <= 0 or hh <= 0:
                    continue
                label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

            dst_lbl = OUT_ROOT / split_name / "labels" / (Path(fn).stem + ".txt")
            dst_lbl.write_text("\n".join(label_lines), encoding="utf-8")

    write_yaml(names)
    print("✅ Done. YOLO dataset created at:", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()