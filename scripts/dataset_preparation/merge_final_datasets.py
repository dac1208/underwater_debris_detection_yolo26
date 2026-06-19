import os
import shutil

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASE_DIR = r"C:\Users\Dario\OneDrive\DIPLOMSKI RAD"

# Ulazni datasetovi
SEACLEAR_TRASHCAN_ROOT = os.path.join(
    BASE_DIR,
    "SeaClear&TrashCan_merged",
    "Merged_dataset(SeaClear_TrashCan)_6cls"
)

BASKA_ROOT = os.path.join(
    BASE_DIR,
    "BASKA",
    "baska_dataset_yolo_6classes"
)

MUZZA_ROOT = os.path.join(
    BASE_DIR,
    "MUZZA",
    "muzza_dataset_yolo_6classes"
)

BIOGRAD_ROOT = os.path.join(
    BASE_DIR,
    "BIOGRAD(blueye+fifish)",
    "merged_blueye_fifish_remapped_6cls"
)

# Izlazni datasetovi
MERGE_ST_BM_ROOT = os.path.join(
    BASE_DIR,
    "FINAL_MERGED_ST_BASKA_MUZZA_6cls"
)

MERGE_ALL_ROOT = os.path.join(
    BASE_DIR,
    "FINAL_MERGED_ALL_6cls"
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


def ensure_structure(root):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "labels"), exist_ok=True)


def check_dataset_structure(dataset_root):
    required = [
        os.path.join(dataset_root, "train", "images"),
        os.path.join(dataset_root, "train", "labels"),
        os.path.join(dataset_root, "val", "images"),
        os.path.join(dataset_root, "val", "labels"),
        os.path.join(dataset_root, "data.yaml"),
    ]

    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fali: {path}")


def copy_split(src_root, dst_root, split, prefix):
    src_images = os.path.join(src_root, split, "images")
    src_labels = os.path.join(src_root, split, "labels")

    dst_images = os.path.join(dst_root, split, "images")
    dst_labels = os.path.join(dst_root, split, "labels")

    img_count = 0
    lbl_count = 0

    for fname in os.listdir(src_images):
        src_img = os.path.join(src_images, fname)

        if not os.path.isfile(src_img):
            continue

        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTS:
            continue

        base = os.path.splitext(fname)[0]

        new_img_name = f"{prefix}_{fname}"
        new_lbl_name = f"{prefix}_{base}.txt"

        dst_img = os.path.join(dst_images, new_img_name)
        dst_lbl = os.path.join(dst_labels, new_lbl_name)

        shutil.copy2(src_img, dst_img)
        img_count += 1

        src_lbl = os.path.join(src_labels, base + ".txt")
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            open(dst_lbl, "w", encoding="utf-8").close()

        lbl_count += 1

    return img_count, lbl_count


def write_data_yaml(root, names_dict):
    yaml_path = os.path.join(root, "data.yaml")

    lines = [
        f"path: {root.replace(os.sep, '/')}",
        "train: train/images",
        "val: val/images",
        "",
        "names:"
    ]

    for idx, name in sorted(names_dict.items()):
        lines.append(f"  {idx}: {name}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def verify_same_classes(dataset_roots):
    all_names = []

    for root in dataset_roots:
        yaml_path = os.path.join(root, "data.yaml")
        names = parse_names_from_yaml(yaml_path)
        all_names.append((root, names))

    ref_root, ref_names = all_names[0]

    for root, names in all_names[1:]:
        if names != ref_names:
            print("\nGREŠKA: datasetovi nemaju isti names redoslijed.\n")
            print("REFERENCE:", ref_root)
            for k, v in ref_names.items():
                print(f"  {k}: {v}")

            print("\nPROBLEM DATASET:", root)
            for k, v in names.items():
                print(f"  {k}: {v}")

            raise ValueError("Names u data.yaml nisu isti za sve datasetove.")

    return ref_names


def merge_group(output_root, datasets):
    """
    datasets = [
        ("prefix", "dataset_root"),
        ...
    ]
    """
    for _, root in datasets:
        check_dataset_structure(root)

    class_names = verify_same_classes([root for _, root in datasets])

    ensure_structure(output_root)

    summary = {}

    for prefix, root in datasets:
        tr_i, tr_l = copy_split(root, output_root, "train", prefix)
        va_i, va_l = copy_split(root, output_root, "val", prefix)

        summary[prefix] = {
            "train_images": tr_i,
            "train_labels": tr_l,
            "val_images": va_i,
            "val_labels": va_l,
        }

    write_data_yaml(output_root, class_names)

    total_train = sum(v["train_images"] for v in summary.values())
    total_val = sum(v["val_images"] for v in summary.values())

    print("\n====================================")
    print("GOTOV MERGE:", output_root)
    print("====================================")

    for prefix, stats in summary.items():
        print(f"\n{prefix}")
        print(f"  train images: {stats['train_images']}")
        print(f"  val images:   {stats['val_images']}")

    print(f"\nTOTAL train images: {total_train}")
    print(f"TOTAL val images:   {total_val}")
    print(f"data.yaml: {os.path.join(output_root, 'data.yaml')}")


def main():
    # 1) SeaClear+TrashCan + Baska + Muzza
    merge_group(
        MERGE_ST_BM_ROOT,
        [
            ("st", SEACLEAR_TRASHCAN_ROOT),
            ("baska", BASKA_ROOT),
            ("muzza", MUZZA_ROOT),
        ]
    )

    # 2) SeaClear+TrashCan + Baska + Muzza + Biograd
    merge_group(
        MERGE_ALL_ROOT,
        [
            ("st", SEACLEAR_TRASHCAN_ROOT),
            ("baska", BASKA_ROOT),
            ("muzza", MUZZA_ROOT),
            ("biograd", BIOGRAD_ROOT),
        ]
    )

    print("\nSve gotovo.")


if __name__ == "__main__":
    main()