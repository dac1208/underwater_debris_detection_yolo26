import os
import shutil
import random

RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASE_DIR = r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD\NEW 5-80"

ST_ROOT = os.path.join(BASE_DIR, "Merged_dataset(SeaClear_TrashCan)_6cls")
BM_ROOT = os.path.join(BASE_DIR, "merged_baska_muzza_80train_20val_6cls")
BIOGRAD_ROOT = os.path.join(BASE_DIR, "merged_blueye_fifish_30train_70test_6cls")

OUTPUT_BASE = os.path.join(BASE_DIR, "task4_new_5_80_progressive_6cls")


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


def verify_same_classes(dataset_roots):
    ref_names = None
    ref_root = None

    for root in dataset_roots:
        yaml_path = os.path.join(root, "data.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Fali data.yaml: {yaml_path}")

        names = parse_names_from_yaml(yaml_path)

        if ref_names is None:
            ref_names = names
            ref_root = root
        elif names != ref_names:
            print("\nGREŠKA: datasetovi nemaju isti redoslijed klasa.\n")
            print("REFERENCE:", ref_root)
            for k, v in ref_names.items():
                print(f"  {k}: {v}")

            print("\nPROBLEM:", root)
            for k, v in names.items():
                print(f"  {k}: {v}")

            raise ValueError("Klase nisu iste u data.yaml.")

    return ref_names


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


def sample_percentage(samples, percentage, seed):
    samples_copy = samples[:]
    rnd = random.Random(seed)
    rnd.shuffle(samples_copy)

    n = int(len(samples_copy) * (percentage / 100.0))
    if percentage > 0 and n == 0 and len(samples_copy) > 0:
        n = 1

    return samples_copy[:n]


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


def copy_entire_split(samples, dst_root, split, prefix):
    count = 0
    for s in samples:
        copy_sample(s, dst_root, split, prefix)
        count += 1
    return count


def main():
    class_names = verify_same_classes([ST_ROOT, BM_ROOT, BIOGRAD_ROOT])

    st_train = collect_split_samples(ST_ROOT, "train")
    st_val = collect_split_samples(ST_ROOT, "val")

    bm_train = collect_split_samples(BM_ROOT, "train")
    bm_val = collect_split_samples(BM_ROOT, "val")

    biograd_test = collect_split_samples(BIOGRAD_ROOT, "test")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # zajednički val za svih 10 datasetova
    shared_val_root = os.path.join(OUTPUT_BASE, "_shared_val_temp")
    clear_and_recreate_folder(shared_val_root)
    ensure_structure(shared_val_root)

    st_val_count = copy_entire_split(st_val, shared_val_root, "val", "st")
    bm_val_count = copy_entire_split(bm_val, shared_val_root, "val", "bm")
    biograd_val_count = copy_entire_split(biograd_test, shared_val_root, "val", "biograd")

    shared_val_img_dir = os.path.join(shared_val_root, "val", "images")
    shared_val_lbl_dir = os.path.join(shared_val_root, "val", "labels")

    total_shared_val = st_val_count + bm_val_count + biograd_val_count

    print("Zajednički val napravljen:")
    print(f"  ST val:       {st_val_count}")
    print(f"  BM val:       {bm_val_count}")
    print(f"  Biograd test: {biograd_val_count}")
    print(f"  TOTAL val:    {total_shared_val}")

    for pct in range(10, 101, 10):
        dataset_name = f"st_plus_bm_train_{pct}pct_6cls"
        dst_root = os.path.join(OUTPUT_BASE, dataset_name)

        clear_and_recreate_folder(dst_root)
        ensure_structure(dst_root)

        # cijeli ST train
        st_train_count = copy_entire_split(st_train, dst_root, "train", "st")

        # X% BM train
        bm_subset = sample_percentage(bm_train, pct, RANDOM_SEED + pct)
        bm_subset_count = copy_entire_split(bm_subset, dst_root, "train", "bm")

        # isti val za svih 10
        for fname in os.listdir(shared_val_img_dir):
            shutil.copy2(
                os.path.join(shared_val_img_dir, fname),
                os.path.join(dst_root, "val", "images", fname)
            )

        for fname in os.listdir(shared_val_lbl_dir):
            shutil.copy2(
                os.path.join(shared_val_lbl_dir, fname),
                os.path.join(dst_root, "val", "labels", fname)
            )

        write_data_yaml(dst_root, class_names)

        print("\n----------------------------------------")
        print(f"Napravljen dataset: {dataset_name}")
        print(f"  ST train:        {st_train_count}")
        print(f"  BM {pct}%:       {bm_subset_count} / {len(bm_train)}")
        print(f"  TOTAL train:     {st_train_count + bm_subset_count}")
        print(f"  TOTAL val:       {total_shared_val}")

    print("\nSve gotovo.")
    print("Output folder:", OUTPUT_BASE)


if __name__ == "__main__":
    main()