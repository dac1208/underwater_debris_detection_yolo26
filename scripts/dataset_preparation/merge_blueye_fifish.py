import os
import shutil

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BLUEYE_ROOT = os.path.join(SCRIPT_DIR, "blueye_yolo")
FIFISH_ROOT = os.path.join(SCRIPT_DIR, "fifish_yolo")
MERGED_ROOT = os.path.join(SCRIPT_DIR, "merged_blueye_fifish")


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


def copy_split(src_root, dst_root, split, prefix):
    src_images = os.path.join(src_root, split, "images")
    src_labels = os.path.join(src_root, split, "labels")

    dst_images = os.path.join(dst_root, split, "images")
    dst_labels = os.path.join(dst_root, split, "labels")

    if not os.path.exists(src_images):
        raise FileNotFoundError(f"Ne postoji: {src_images}")
    if not os.path.exists(src_labels):
        raise FileNotFoundError(f"Ne postoji: {src_labels}")

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


def main():
    blueye_yaml = os.path.join(BLUEYE_ROOT, "data.yaml")
    fifish_yaml = os.path.join(FIFISH_ROOT, "data.yaml")

    if not os.path.exists(blueye_yaml):
        raise FileNotFoundError(f"Ne nalazim: {blueye_yaml}")
    if not os.path.exists(fifish_yaml):
        raise FileNotFoundError(f"Ne nalazim: {fifish_yaml}")

    blueye_names = parse_names_from_yaml(blueye_yaml)
    fifish_names = parse_names_from_yaml(fifish_yaml)

    if blueye_names != fifish_names:
        print("GREŠKA: BLUEYE i FIFISH nemaju isti redoslijed klasa u data.yaml.")
        print("\nBLUEYE names:")
        for k, v in blueye_names.items():
            print(f"  {k}: {v}")
        print("\nFIFISH names:")
        for k, v in fifish_names.items():
            print(f"  {k}: {v}")
        raise ValueError("Prvo uskladi klase pa tek onda merge.")

    ensure_structure(MERGED_ROOT)

    print("Spajam BLUEYE...")
    by_train_imgs, _ = copy_split(BLUEYE_ROOT, MERGED_ROOT, "train", "blueye")
    by_val_imgs, _ = copy_split(BLUEYE_ROOT, MERGED_ROOT, "val", "blueye")

    print("Spajam FIFISH...")
    ff_train_imgs, _ = copy_split(FIFISH_ROOT, MERGED_ROOT, "train", "fifish")
    ff_val_imgs, _ = copy_split(FIFISH_ROOT, MERGED_ROOT, "val", "fifish")

    write_data_yaml(MERGED_ROOT, blueye_names)

    print("\nGOTOVO")
    print("Merged dataset:", MERGED_ROOT)

    print("\nTRAIN")
    print(f"  BLUEYE images: {by_train_imgs}")
    print(f"  FIFISH images: {ff_train_imgs}")
    print(f"  TOTAL images:  {by_train_imgs + ff_train_imgs}")

    print("\nVAL")
    print(f"  BLUEYE images: {by_val_imgs}")
    print(f"  FIFISH images: {ff_val_imgs}")
    print(f"  TOTAL images:  {by_val_imgs + ff_val_imgs}")

    print("\ndata.yaml napravljen.")


if __name__ == "__main__":
    main()