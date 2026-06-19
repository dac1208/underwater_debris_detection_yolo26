from pathlib import Path
import shutil

# === PROMIJENI OVO AKO TREBA ===
src = Path(r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD\BIOGRAD(blueye+fifish)\merged_blueye_fifish")
dst = Path(r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD\BIOGRAD(blueye+fifish)\merged_blueye_fifish_remapped")

# old class id -> new class id
class_map = {
    0: 1,  # bottle -> trash_plastic
    1: 4,  # robot  -> rov
    2: 2,  # mask   -> trash_other
}

new_names = {
    0: "animal",
    1: "trash_plastic",
    2: "trash_other",
    3: "nature",
    4: "rov",
    5: "unknown",
}


def copy_images(split: str):
    src_img_dir = src / split / "images"
    dst_img_dir = dst / split / "images"
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    if not src_img_dir.exists():
        print(f"[WARN] Missing folder: {src_img_dir}")
        return

    for file in src_img_dir.iterdir():
        if file.is_file():
            shutil.copy2(file, dst_img_dir / file.name)


def remap_labels(split: str):
    src_lbl_dir = src / split / "labels"
    dst_lbl_dir = dst / split / "labels"
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not src_lbl_dir.exists():
        print(f"[WARN] Missing folder: {src_lbl_dir}")
        return

    for label_file in src_lbl_dir.glob("*.txt"):
        new_lines = []

        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            old_class = int(parts[0])

            if old_class not in class_map:
                print(f"[WARN] Unknown old class {old_class} in {label_file.name}, skipping line")
                continue

            new_class = class_map[old_class]
            parts[0] = str(new_class)

            new_lines.append(" ".join(parts))

        with open(dst_lbl_dir / label_file.name, "w", encoding="utf-8") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")
            else:
                f.write("")


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

    with open(dst / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)


def main():
    if dst.exists():
        print(f"[INFO] Destination already exists: {dst}")
        print("[INFO] Files may be overwritten/updated.")

    for split in ["train", "val"]:
        print(f"[INFO] Processing {split}...")
        copy_images(split)
        remap_labels(split)

    write_yaml()

    print("\nDone.")
    print(f"New dataset created here:\n{dst}")
    print("\nClass mapping:")
    print("bottle -> trash_plastic")
    print("robot  -> rov")
    print("mask   -> trash_other")


if __name__ == "__main__":
    main()