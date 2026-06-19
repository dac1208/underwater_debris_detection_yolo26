import os
import shutil
import random

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASE_DIR = r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD\NEW 5-80"

BASKA_ROOT = os.path.join(BASE_DIR, "baska_dataset_yolo_6classes")
MUZZA_ROOT = os.path.join(BASE_DIR, "muzza_dataset_yolo_6classes")

OUTPUT_ROOT = os.path.join(BASE_DIR, "merged_baska_muzza_80train_20val")


def parse_names_from_yaml(yaml_path):
    names = {}
    in_names = False

    with open(yaml_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped == "names:":
                in_names = True
                continue

            if not in_names:
                continue

            if ":" in stripped:
                left, right = stripped.split(":", 1)
                left = left.strip()
                right = right.strip()

                if left.isdigit():
                    names[int(left)] = right
                else:
                    break
            else:
                break

    if not names:
        raise ValueError(f"Ne mogu pročitati names iz: {yaml_path}")

    return dict(sorted(names.items()))


def ensure_structure(root):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "labels"), exist_ok=True)


def collect_samples(dataset_root):
    samples = []

    for old_split in ["train", "val"]:
        img_dir = os.path.join(dataset_root, old_split, "images")
        lbl_dir = os.path.join(dataset_root, old_split, "labels")

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


def copy_sample(sample, dst_root, split, prefix):
    new_img_name = f"{prefix}_{sample['image_name']}"
    new_lbl_name = f"{prefix}_{os.path.splitext(sample['image_name'])[0]}.txt"

    dst_img = os.path.join(dst_root, split, "images", new_img_name)
    dst_lbl = os.path.join(dst_root, split, "labels", new_lbl_name)

    shutil.copy2(sample["image_path"], dst_img)

    if os.path.exists(sample["label_path"]):
        shutil.copy2(sample["label_path"], dst_lbl)
    else:
        open(dst_lbl, "w", encoding="utf-8").close()


def split_dataset_samples(samples, train_ratio, seed):
    random.seed(seed)
    samples = samples[:]
    random.shuffle(samples)

    n_train = int(len(samples) * train_ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    return train_samples, val_samples


def write_data_yaml(dst_root, names_dict):
    yaml_path = os.path.join(dst_root, "data.yaml")

    lines = [
        f"path: {dst_root.replace(os.sep, '/')}",
        "train: train/images",
        "val: val/images",
        "",
        "names:"
    ]

    for idx, name in sorted(names_dict.items()):
        lines.append(f"  {idx}: {name}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    # provjera yaml klasa
    baska_yaml = os.path.join(BASKA_ROOT, "data.yaml")
    muzza_yaml = os.path.join(MUZZA_ROOT, "data.yaml")

    baska_names = parse_names_from_yaml(baska_yaml)
    muzza_names = parse_names_from_yaml(muzza_yaml)

    if baska_names != muzza_names:
        print("GREŠKA: Baska i Muzza nemaju isti redoslijed klasa u data.yaml.")
        print("\nBASKA:")
        for k, v in baska_names.items():
            print(f"  {k}: {v}")
        print("\nMUZZA:")
        for k, v in muzza_names.items():
            print(f"  {k}: {v}")
        raise ValueError("Prvo uskladi klase.")

    ensure_structure(OUTPUT_ROOT)

    # skupi sve slike iz oba dataseta
    baska_samples = collect_samples(BASKA_ROOT)
    muzza_samples = collect_samples(MUZZA_ROOT)

    if not baska_samples:
        raise ValueError("Nisam našao nijednu sliku u Baska datasetu.")
    if not muzza_samples:
        raise ValueError("Nisam našao nijednu sliku u Muzza datasetu.")

    # splitaj svaki dataset zasebno => održava omjer po datasetu
    baska_train, baska_val = split_dataset_samples(baska_samples, TRAIN_RATIO, RANDOM_SEED)
    muzza_train, muzza_val = split_dataset_samples(muzza_samples, TRAIN_RATIO, RANDOM_SEED + 1)

    # kopiraj u novi merged dataset
    for sample in baska_train:
        copy_sample(sample, OUTPUT_ROOT, "train", "baska")
    for sample in baska_val:
        copy_sample(sample, OUTPUT_ROOT, "val", "baska")

    for sample in muzza_train:
        copy_sample(sample, OUTPUT_ROOT, "train", "muzza")
    for sample in muzza_val:
        copy_sample(sample, OUTPUT_ROOT, "val", "muzza")

    write_data_yaml(OUTPUT_ROOT, baska_names)

    print("Gotovo.")
    print("Output:", OUTPUT_ROOT)
    print()
    print("BASKA")
    print("  train:", len(baska_train))
    print("  val:  ", len(baska_val))
    print()
    print("MUZZA")
    print("  train:", len(muzza_train))
    print("  val:  ", len(muzza_val))
    print()
    print("TOTAL")
    print("  train:", len(baska_train) + len(muzza_train))
    print("  val:  ", len(baska_val) + len(muzza_val))


if __name__ == "__main__":
    main()