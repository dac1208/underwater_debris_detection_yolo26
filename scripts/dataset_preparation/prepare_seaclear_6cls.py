import os
import json
import random
import shutil
from collections import defaultdict

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Novi superclass ID-jevi:
# 0 = animal
# 1 = trash_plastic
# 2 = trash_other
# 3 = nature
# 4 = rov
# 5 = unknown
SUPERCLASS_NAMES = {
    0: "animal",
    1: "trash_plastic",
    2: "trash_other",
    3: "nature",
    4: "rov",
    5: "unknown",
}

# SeaClear 40 -> tvojih 6
CLASS_MAP = {
    # animal
    "animal_etc": 0,
    "animal_fish": 0,
    "animal_shells": 0,
    "animal_sponge": 0,
    "animal_starfish": 0,
    "animal_urchin": 0,

    # trash_plastic
    "bag_plastic": 1,
    "bottle_plastic": 1,
    "container_plastic": 1,
    "cup_plastic": 1,
    "lid_plastic": 1,
    "net_plastic": 1,
    "pipe_plastic": 1,
    "rope_plastic": 1,
    "sanitaries_plastic": 1,
    "snack_wrapper_plastic": 1,
    "tarp_plastic": 1,

    # trash_other
    "boot_rubber": 2,
    "bottle_glass": 2,
    "brick_clay": 2,
    "cable_metal": 2,
    "can_metal": 2,
    "cardboard_paper": 2,
    "clothing_fiber": 2,
    "container_middle_size_metal": 2,
    "cup_ceramic": 2,
    "furniture_wood": 2,
    "jar_glass": 2,
    "rope_fiber": 2,
    "snack_wrapper_paper": 2,
    "tire_rubber": 2,
    "tube_cement": 2,
    "wreckage_metal": 2,

    # nature
    "plant": 3,
    "branch_wood": 3,

    # rov
    "rov_bluerov": 4,
    "rov_cable": 4,
    "rov_tortuga": 4,
    "rov_vehicle_leg": 4,

    # unknown
    "unknown_instance": 5,
}


def norm_path(p):
    return p.replace("\\", "/").lstrip("./")


def build_image_index(dataset_root, ignore_dirs=None):
    ignore_dirs = set(ignore_dirs or [])
    rel_index = {}
    base_index = defaultdict(list)

    for dirpath, dirnames, filenames in os.walk(dataset_root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS:
                continue

            full_path = os.path.join(dirpath, fname)
            rel_path = norm_path(os.path.relpath(full_path, dataset_root))

            rel_index[rel_path] = full_path
            base_index[os.path.basename(fname)].append(full_path)

    return rel_index, base_index


def resolve_image_path(file_name, dataset_root, rel_index, base_index):
    file_name = norm_path(file_name)

    # 1) direktan relative path iz json-a
    if file_name in rel_index:
        return rel_index[file_name]

    # 2) suffix match
    suffix_matches = [full for rel, full in rel_index.items() if rel.endswith(file_name)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    # 3) basename match
    base = os.path.basename(file_name)
    candidates = base_index.get(base, [])

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        refined = [
            c for c in candidates
            if norm_path(os.path.relpath(c, dataset_root)).endswith(file_name)
        ]
        if len(refined) == 1:
            return refined[0]

        raise FileNotFoundError(
            f"Više kandidata za '{file_name}'. "
            f"Ručno provjeri duplikate naziva. Primjeri: {candidates[:5]}"
        )

    raise FileNotFoundError(f"Ne mogu naći sliku za file_name='{file_name}'")


def make_flat_output_name(image_path, dataset_root):
    rel_path = norm_path(os.path.relpath(image_path, dataset_root))
    return rel_path.replace("/", "__")


def ensure_dirs(root_out):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(root_out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root_out, split, "labels"), exist_ok=True)


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w = w / img_w
    h = h / img_h
    return x_center, y_center, w, h


def write_data_yaml(root_out):
    yaml_path = os.path.join(root_out, "data.yaml")
    content = f"""path: {root_out.replace(os.sep, '/')}
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
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = script_dir
    json_path = os.path.join(dataset_root, "dataset.json")
    output_root = os.path.join(dataset_root, "seaclear_6cls")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Ne nalazim dataset.json u: {json_path}")

    random.seed(RANDOM_SEED)
    ensure_dirs(output_root)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Provjera da su sve klase pokrivene mapiranjem
    dataset_class_names = set(categories.values())
    unmapped = sorted(dataset_class_names - set(CLASS_MAP.keys()))
    if unmapped:
        print("Ove klase nisu mapirane:")
        for c in unmapped:
            print(" -", c)
        raise ValueError("Dopuni CLASS_MAP pa pokreni opet.")

    # Index svih slika kroz sve podfoldere
    rel_index, base_index = build_image_index(
        dataset_root,
        ignore_dirs={"seaclear_6cls", "__pycache__"}
    )

    # Grupiraj anotacije po slici
    annotations_per_image = defaultdict(list)
    class_counts_total = defaultdict(int)

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        cat_name = categories[cat_id]
        new_class_id = CLASS_MAP[cat_name]

        img_w = images[img_id]["width"]
        img_h = images[img_id]["height"]

        x_center, y_center, w, h = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)

        annotations_per_image[img_id].append(
            f"{new_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        )
        class_counts_total[new_class_id] += 1

    # 80/20 split po slikama
    image_ids = list(images.keys())
    random.shuffle(image_ids)

    n_train = int(len(image_ids) * TRAIN_RATIO)
    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train:])

    split_image_counts = {"train": 0, "val": 0}
    split_object_counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
    }

    missing_images = []

    for img_id in image_ids:
        split = "train" if img_id in train_ids else "val"
        img_info = images[img_id]

        try:
            src_img = resolve_image_path(
                img_info["file_name"],
                dataset_root,
                rel_index,
                base_index
            )
        except FileNotFoundError as e:
            missing_images.append(str(e))
            continue

        out_img_name = make_flat_output_name(src_img, dataset_root)
        dst_img = os.path.join(output_root, split, "images", out_img_name)
        dst_lbl = os.path.join(
            output_root, split, "labels",
            os.path.splitext(out_img_name)[0] + ".txt"
        )

        shutil.copy2(src_img, dst_img)

        label_lines = annotations_per_image.get(img_id, [])
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        split_image_counts[split] += 1

        for line in label_lines:
            cls_id = int(line.split()[0])
            split_object_counts[split][cls_id] += 1

    write_data_yaml(output_root)

    print("\nGotovo.")
    print("Output:", output_root)
    print("data.yaml:", os.path.join(output_root, "data.yaml"))

    print("\nBroj slika:")
    print("train:", split_image_counts["train"])
    print("val  :", split_image_counts["val"])

    print("\nBroj objekata po superklasi:")
    for cls_id in range(6):
        print(
            f"{cls_id} ({SUPERCLASS_NAMES[cls_id]}): "
            f"train={split_object_counts['train'][cls_id]} | "
            f"val={split_object_counts['val'][cls_id]} | "
            f"total={class_counts_total[cls_id]}"
        )

    if missing_images:
        print("\nUPOZORENJE - neke slike nisu nađene:")
        for msg in missing_images[:20]:
            print(msg)
        if len(missing_images) > 20:
            print(f"... i još {len(missing_images) - 20} kom.")


if __name__ == "__main__":
    main()