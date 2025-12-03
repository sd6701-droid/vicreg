# prepare_vicreg_first1000_local.py
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

print("🚀 Starting VicReg dataset preparation")

# ----------------------------
# 1️⃣ Load images from local folder
# ----------------------------
dataset_dir = Path("/home/sd6701/datasets/fall2025_deeplearning/cc3m_96px")

# Collect all image files (jpg, png)
all_images = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in [".jpg", ".png"]]


print(f"✅ Total images selected: {len(all_images)}")

if len(all_images) == 0:
    raise ValueError(f"No images found in {dataset_dir}!")

# ----------------------------
# 2️⃣ Split train/val (80/20)
# ----------------------------
train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
print(f"✅ Train images: {len(train_imgs)}, Validation images: {len(val_imgs)}")

# ----------------------------
# 3️⃣ Create VicReg folder structure
# ----------------------------
base_dir = Path("/home/sd6701/datasets/fall2025_deeplearning/cc3m_96px_vicreg_final_exp")
train_dir = base_dir / "train" / "class1"
val_dir = base_dir / "val" / "class1"
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)
print(f"✅ Folders created at {base_dir}")

# ----------------------------
# 4️⃣ Copy images to folders
# ----------------------------
def save_images(img_list, folder):
    print(f"5️⃣ Copying {len(img_list)} images to {folder}...")
    for idx, img_path in enumerate(tqdm(img_list, desc=f"Copying to {folder}")):
        img = Image.open(img_path)
        img.save(folder / f"{idx}.jpg")
    print(f"✅ Finished copying images to {folder}")

save_images(train_imgs, train_dir)
save_images(val_imgs, val_dir)

print("🎉 VicReg-ready dataset creation complete!")
print(f"Dataset location: {base_dir}")


# python main_vicreg.py \
#   --data-dir "/Users/samprasmanueldsouza/Desktop/Home Work/datasets/mydataset_small" \
#   --exp-dir "/Users/samprasmanueldsouza/Desktop/Home Work/experiments/vicreg_run" \
#   --arch resnet50 \
#   --epochs 100 \
#   --batch-size 32 \
#   --base-lr 0.03 \
#   --device cpu \
#   --world-size 1