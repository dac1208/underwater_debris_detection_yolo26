import os
import shutil
import random

RANDOM_SEED = 42
TRAIN_RATIO = 0.30
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASE_DIR = r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD\NEW 5-80"
SRC_ROOT = os.path.join(BASE_DIR, "merged_blueye_fifish_remapped_6cls")
DST_ROOT = os.path.join(BASE_DIR, "merged_blueye_fifish_30train_70test")


def ensure_structure(root):
    for split in ["train", "test"]:
        os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "labels"), exist_ok=True)


def collect_samples(src_root):
    samples = []

    for old_split in ["train", "val"]:
        img_dir = os.path.join(src_root, old_split, "images")
        lbl_dir = os.path.join(src_root, old_split, "labels")

        if not os.path.exists(img_dir):
            continue

        for fname in os.listdir(img_dir):
            img_path = os.path.join(img_dir, fname)

            if not os.path.isfile(img_path):
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS:
                continue

            base = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")

            samples.append({
                "image_name": fname,
                "image_path": img_path,
                "label_path": lbl_path
            })

    return samples


def copy_sample(sample, dst_root, split):
    dst_img = os.path.join(dst_root, split, "images", sample["image_name"])
    dst_lbl = os.path.join(
        dst_root,
        split,
        "labels",
        os.path.splitext(sample["image_name"])[0] + ".txt"
    )

    shutil.copy2(sample["image_path"], dst_img)

    if os.path.exists(sample["label_path"]):
        shutil.copy2(sample["label_path"], dst_lbl)
    else:
        open(dst_lbl, "w", encoding="utf-8").close()


def copy_data_yaml(src_root, dst_root):
    src_yaml = os.path.join(src_root, "data.yaml")
    dst_yaml = os.path.join(dst_root, "data.yaml")

    if not os.path.exists(src_yaml):
        print("Upozorenje: data.yaml ne postoji, preskačem kopiranje.")
        return

    with open(src_yaml, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    new_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("path:"):
            new_lines.append(f"path: {dst_root.replace(os.sep, '/')}")
        elif stripped.startswith("train:"):
            new_lines.append("train: train/images")
        elif stripped.startswith("val:"):
            new_lines.append("test: test/images")
        else:
            new_lines.append(line)

    # ako u starom yaml-u nije bilo val, ipak dodaj train/test
    text_joined = "\n".join(new_lines)
    if "train:" not in text_joined:
        new_lines.insert(1, "train: train/images")
    if "test:" not in text_joined:
        new_lines.insert(2, "test: test/images")

    with open(dst_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def main():
    random.seed(RANDOM_SEED)
    ensure_structure(DST_ROOT)

    samples = collect_samples(SRC_ROOT)

    if not samples:
        raise ValueError("Nisam našao nijednu sliku u source datasetu.")

    random.shuffle(samples)

    n_train = int(len(samples) * TRAIN_RATIO)
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    for sample in train_samples:
        copy_sample(sample, DST_ROOT, "train")

    for sample in test_samples:
        copy_sample(sample, DST_ROOT, "test")

    copy_data_yaml(SRC_ROOT, DST_ROOT)

    print("Gotovo.")
    print("Source:", SRC_ROOT)
    print("Output:", DST_ROOT)
    print("Train images:", len(train_samples))
    print("Test images:", len(test_samples))


if __name__ == "__main__":
    main()