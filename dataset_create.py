
import os
import zipfile
import shutil
import random
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
base_dir = Path("/home/sd6701/datasets/fall2025_deeplearning/cc3m_96px_vicreg_final_exp")
train_dir = base_dir / "train" / "class1"
val_dir = base_dir / "val" / "class1"

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)
print(f"✅ Folders created at {base_dir}")

# ZIP files
cc3m_zips = [
    "/home/sd6701/datasets/fall2025_deeplearning/cc3m_96px_part1.zip",
    "/home/sd6701/datasets/fall2025_deeplearning/cc3m_96px_part2.zip",
    "/home/sd6701/datasets/fall2025_deeplearning/cc3m_96px_part3.zip",
    "/home/sd6701/datasets/fall2025_deeplearning/cc3m_96px_part4.zip",
    "/home/sd6701/datasets/fall2025_deeplearning/cc3m_96px_part5.zip"
]

ys2krqw_zip = "/home/sd6701/datasets/fall2025_deeplearning/trImgs/ys2krqw.zip"

# -----------------------------
# Function to unzip and move files
# -----------------------------
def unzip_to_temp(zip_path, temp_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

# -----------------------------
# Function to move files safely
# -----------------------------
def move_files_safe(src_dir, dst_dir):
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff")
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        if not os.path.isfile(src_file):
            continue
        if not filename.lower().endswith(image_exts):
            continue
        
        dst_file = os.path.join(dst_dir, filename)
        if os.path.exists(dst_file):
            dst_file = os.path.join(dst_dir, f"xyz-{filename}")
        
        shutil.move(src_file, dst_file)

# -----------------------------
# Step 1: Process CC3M zips
# -----------------------------
temp_cc3m_dir = Path("/tmp/cc3m_unzip_temp")
temp_cc3m_dir.mkdir(parents=True, exist_ok=True)

print("🔹 Unzipping CC3M files...")
for zip_file in cc3m_zips:
    print(f"Extracting {zip_file}...")
    unzip_to_temp(zip_file, temp_cc3m_dir)

# -----------------------------
# Step 2: Process ys2krqw.zip
# -----------------------------
temp_ys2krqw_dir = Path("/tmp/ys2krqw_unzip_temp")
temp_ys2krqw_dir.mkdir(parents=True, exist_ok=True)

print("🔹 Extracting ys2krqw.zip...")
unzip_to_temp(ys2krqw_zip, temp_ys2krqw_dir)

# -----------------------------
# Step 3: Combine all images into one list
# -----------------------------
all_images = []

for temp_dir in [temp_cc3m_dir, temp_ys2krqw_dir]:
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append(file_path)

print(f"Total images collected: {len(all_images)}")

# -----------------------------
# Step 4: Shuffle and split into train/val
# -----------------------------
random.shuffle(all_images)
split_ratio = 0.9
split_index = int(len(all_images) * split_ratio)
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Move files
print("🔹 Moving images to train directory...")
for img in train_images:
    move_files_safe(os.path.dirname(img), train_dir)

print("🔹 Moving images to val directory...")
for img in val_images:
    move_files_safe(os.path.dirname(img), val_dir)

print("✅ All images moved and split into train/val successfully.")




# # prepare_vicreg_first1000_local.py
# import shutil
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# from PIL import Image

# print("🚀 Starting VicReg dataset preparation")

# # ----------------------------
# # 1️⃣ Load images from local folder
# # ----------------------------
# dataset_dir = Path("/home/sd6701/datasets/fall2025_deeplearning/cc3m_96px")

# # Collect all image files (jpg, png)
# all_images = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in [".jpg", ".png"]][:100000]


# print(f"✅ Total images selected: {len(all_images)}")

# if len(all_images) == 0:
#     raise ValueError(f"No images found in {dataset_dir}!")

# # ----------------------------
# # 2️⃣ Split train/val (80/20)
# # ----------------------------
# train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
# print(f"✅ Train images: {len(train_imgs)}, Validation images: {len(val_imgs)}")

# # ----------------------------
# # 3️⃣ Create VicReg folder structure
# # ----------------------------
# base_dir = Path("/home/sd6701/datasets/fall2025_deeplearning/cc3m_96px_vicreg_final_exp")
# train_dir = base_dir / "train" / "class1"
# val_dir = base_dir / "val" / "class1"
# train_dir.mkdir(parents=True, exist_ok=True)
# val_dir.mkdir(parents=True, exist_ok=True)
# print(f"✅ Folders created at {base_dir}")

# # ----------------------------
# # 4️⃣ Copy images to folders
# # ----------------------------
# def save_images(img_list, folder):
#     print(f"5️⃣ Copying {len(img_list)} images to {folder}...")
#     for idx, img_path in enumerate(tqdm(img_list, desc=f"Copying to {folder}")):
#         img = Image.open(img_path)
#         img.save(folder / f"{idx}.jpg")
#     print(f"✅ Finished copying images to {folder}")

# save_images(train_imgs, train_dir)
# save_images(val_imgs, val_dir)

# print("🎉 VicReg-ready dataset creation complete!")
# print(f"Dataset location: {base_dir}")


# python main_vicreg.py \
#   --data-dir "/Users/samprasmanueldsouza/Desktop/Home Work/datasets/mydataset_small" \
#   --exp-dir "/Users/samprasmanueldsouza/Desktop/Home Work/experiments/vicreg_run" \
#   --arch resnet50 \
#   --epochs 100 \
#   --batch-size 32 \
#   --base-lr 0.03 \
#   --device cpu \
#   --world-size 1