import os
import shutil
import random
import hashlib

RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASE_DIR = r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD\NEW 5-80"

BASE_DATASET_ROOT = os.path.join(
    BASE_DIR,
    "adding_10_pct_progressive_6cls",
    "st_plus_bm_train_40pct_6cls"
)

BIOGRAD_ROOT = os.path.join(
    BASE_DIR,
    "merged_blueye_fifish_30train_70test_6cls"
)

OUTPUT_BASE = os.path.join(
    BASE_DIR,
    "biograd_added_to_st_bm_40pct_6cls"
)


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


def verify_same_classes(root_a, root_b):
    names_a = parse_names_from_yaml(os.path.join(root_a, "data.yaml"))
    names_b = parse_names_from_yaml(os.path.join(root_b, "data.yaml"))

    if names_a != names_b:
        print("GREŠKA: klase nisu iste.")
        print("\nBASE DATASET:")
        for k, v in names_a.items():
            print(f"  {k}: {v}")

        print("\nBIOGRAD:")
        for k, v in names_b.items():
            print(f"  {k}: {v}")

        raise ValueError("data.yaml names nisu isti.")

    return names_a


def ensure_structure(root):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "labels"), exist_ok=True)


def clear_and_recreate_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def collect_split_samples(dataset_root, split_name):
    samples = []

    img_dir = os.path.join(dataset_root, split_name, "images")
    lbl_dir = os.path.join(dataset_root, split_name, "labels")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Ne postoji: {img_dir}")
    if not os.path.exists(lbl_dir):
        raise FileNotFoundError(f"Ne postoji: {lbl_dir}")

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
            "label_path": lbl_path,
        })

    return samples


def make_short_name(original_name, prefix=None):
    base, ext = os.path.splitext(original_name)

    short_base = base[:40]
    digest = hashlib.md5(original_name.encode("utf-8")).hexdigest()[:10]

    if prefix:
        return f"{prefix}_{short_base}_{digest}{ext}"
    return f"{short_base}_{digest}{ext}"


def copy_sample(sample, dst_root, split, prefix=None):
    if prefix:
        new_img_name = make_short_name(sample["image_name"], prefix=prefix)
    else:
        new_img_name = sample["image_name"]

    new_lbl_name = os.path.splitext(new_img_name)[0] + ".txt"

    dst_img = os.path.join(dst_root, split, "images", new_img_name)
    dst_lbl = os.path.join(dst_root, split, "labels", new_lbl_name)

    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)

    shutil.copy2(sample["image_path"], dst_img)

    if os.path.exists(sample["label_path"]):
        shutil.copy2(sample["label_path"], dst_lbl)
    else:
        open(dst_lbl, "w", encoding="utf-8").close()


def copy_entire_split(samples, dst_root, split, prefix=None):
    count = 0
    for s in samples:
        copy_sample(s, dst_root, split, prefix=prefix)
        count += 1
    return count


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


def compute_step_sizes(total, n_steps):
    base = total // n_steps
    remainder = total % n_steps

    sizes = []
    for i in range(n_steps):
        size = base + (1 if i < remainder else 0)
        sizes.append(size)

    return sizes


def main():
    class_names = verify_same_classes(BASE_DATASET_ROOT, BIOGRAD_ROOT)

    base_train = collect_split_samples(BASE_DATASET_ROOT, "train")
    base_val = collect_split_samples(BASE_DATASET_ROOT, "val")

    biograd_train = collect_split_samples(BIOGRAD_ROOT, "train")

    if not biograd_train:
        raise ValueError("Biograd train je prazan.")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # jedan fiksni random redoslijed Biograd train slika
    biograd_ordered = biograd_train[:]
    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(biograd_ordered)

    step_sizes = compute_step_sizes(len(biograd_ordered), 6)

    cumulative_counts = []
    running = 0
    for s in step_sizes:
        running += s
        cumulative_counts.append(running)

    print("Ukupno Biograd train slika:", len(biograd_ordered))
    print("Koraci po treninzima:", step_sizes)
    print("Kumulativno:", cumulative_counts)

    for step_idx, cumulative_n in enumerate(cumulative_counts, start=1):
        dataset_name = f"st_bm_40pct_plus_biograd_step{step_idx}_6cls"
        dst_root = os.path.join(OUTPUT_BASE, dataset_name)

        clear_and_recreate_folder(dst_root)
        ensure_structure(dst_root)

        # 1) base dataset train
        base_train_count = copy_entire_split(base_train, dst_root, "train")

        # 2) isti base val
        base_val_count = copy_entire_split(base_val, dst_root, "val")

        # 3) kumulativno dodavanje biograda u train
        biograd_subset = biograd_ordered[:cumulative_n]
        biograd_added_count = copy_entire_split(
            biograd_subset,
            dst_root,
            "train",
            prefix="biograd"
        )

        write_data_yaml(dst_root, class_names)

        print("\n----------------------------------------")
        print(f"Napravljen dataset: {dataset_name}")
        print(f"  Base train:       {base_train_count}")
        print(f"  Added biograd:    {biograd_added_count}")
        print(f"  TOTAL train:      {base_train_count + biograd_added_count}")
        print(f"  VAL (same base):  {base_val_count}")

    print("\nSve gotovo.")
    print("Output folder:", OUTPUT_BASE)
    print("Biograd 70% test nismo dirali.")


if __name__ == "__main__":
    main()