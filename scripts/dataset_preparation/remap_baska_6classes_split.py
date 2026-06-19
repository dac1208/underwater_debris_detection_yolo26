from pathlib import Path
import shutil
import random
import re

# === OLD AND NEW DATASET PATHS ===
src = Path(r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD\BASKA\baska_dataset_yolo")
dst = Path(r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD\BASKA\baska_dataset_yolo_6classes_80_20")

# Original class name -> new class id
name_to_new_id = {
    "bottle": 1,  # trash_plastic
    "mask": 2,    # trash_other
    "robot": 4,   # rov
}

new_names = {
    0: "animal",
    1: "trash_plastic",
    2: "trash_other",
    3: "nature",
    4: "rov",
    5: "unknown",
}

image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def parse_names_from_yaml(yaml_path: Path):
    """
    Reads class names from simple YOLO yaml formats, e.g.

    names:
      0: bottle
      1: mask
      2: robot

    or

    names: ["bottle", "mask", "robot"]
    """
    text = yaml_path.read_text(encoding="utf-8")

    old_id_to_name = {}

    # Case 1: names as list
    list_match = re.search(r"names\s*:\s*\[(.*?)\]", text, re.DOTALL)
    if list_match:
        items = list_match.group(1).split(",")
        for i, item in enumerate(items):
            name = item.strip().strip("'").strip('"')
            old_id_to_name[i] = name
        return old_id_to_name

    # Case 2: names as dictionary
    in_names = False
    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("names:"):
            in_names = True
            continue

        if in_names:
            if not stripped:
                continue

            match = re.match(r"(\d+)\s*:\s*(.+)", stripped)
            if match:
                class_id = int(match.group(1))
                name = match.group(2).strip().strip("'").strip('"')
                old_id_to_name[class_id] = name
            elif not line.startswith(" ") and not line.startswith("\t"):
                break

    return old_id_to_name


def collect_images():
    """
    Collects all images from existing train/val/test folders.
    We will ignore the old split and create a new 80/20 train/val split.
    """
    items = []

    for split in ["train", "val", "test"]:
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"

        if not img_dir.exists():
            continue

        for img_path in img_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                label_path = lbl_dir / f"{img_path.stem}.txt"
                items.append((img_path, label_path))

    return items


def remap_label_file(src_label: Path, dst_label: Path, old_id_to_name: dict):
    """
    Creates a new YOLO label file with remapped class ids.
    If the source label does not exist, creates an empty label file.
    """
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    if not src_label.exists():
        dst_label.write_text("", encoding="utf-8")
        return

    new_lines = []

    lines = src_label.read_text(encoding="utf-8").splitlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()
        old_id = int(parts[0])

        if old_id not in old_id_to_name:
            print(f"[WARN] Unknown old class id {old_id} in {src_label}")
            continue

        old_name = old_id_to_name[old_id]

        if old_name not in name_to_new_id:
            print(f"[WARN] Class '{old_name}' not mapped in {src_label}, skipping.")
            continue

        new_id = name_to_new_id[old_name]
        parts[0] = str(new_id)

        new_lines.append(" ".join(parts))

    dst_label.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")


def copy_split(items, split_name, old_id_to_name):
    img_out = dst / split_name / "images"
    lbl_out = dst / split_name / "labels"

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in items:
        new_img_path = img_out / img_path.name
        new_label_path = lbl_out / f"{img_path.stem}.txt"

        shutil.copy2(img_path, new_img_path)
        remap_label_file(label_path, new_label_path, old_id_to_name)


def write_yaml():
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


def main():
    yaml_candidates = [
        src / "baska.yaml",
        src / "data.yaml",
    ]

    yaml_path = None
    for candidate in yaml_candidates:
        if candidate.exists():
            yaml_path = candidate
            break

    if yaml_path is None:
        raise FileNotFoundError("Could not find baska.yaml or data.yaml in source dataset folder.")

    old_id_to_name = parse_names_from_yaml(yaml_path)

    print("[INFO] Old classes found:")
    for class_id, name in old_id_to_name.items():
        print(f"  {class_id}: {name}")

    all_items = collect_images()

    if not all_items:
        raise RuntimeError("No images found in train/val/test folders.")

    random.seed(42)
    random.shuffle(all_items)

    train_count = int(len(all_items) * 0.8)

    train_items = all_items[:train_count]
    val_items = all_items[train_count:]

    print(f"\n[INFO] Total images: {len(all_items)}")
    print(f"[INFO] Train images: {len(train_items)}")
    print(f"[INFO] Val images: {len(val_items)}")

    copy_split(train_items, "train", old_id_to_name)
    copy_split(val_items, "val", old_id_to_name)
    write_yaml()

    print("\nDone.")
    print(f"New dataset created here:\n{dst}")
    print("\nNew class mapping:")
    print("bottle -> trash_plastic")
    print("mask   -> trash_other")
    print("robot  -> rov")


if __name__ == "__main__":
    main()