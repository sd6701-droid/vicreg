# prepare_vicreg_first200.py
import datasets
import torchvision.datasets as datasets
from datasets import load_dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("🚀 Starting VicReg dataset preparation")

# ----------------------------
# 1️⃣ Load dataset from cache
# ----------------------------
print("1️⃣ Loading dataset from Hugging Face cache...")
dataset = load_dataset(
    "tsbpp/fall2025_deeplearning",
    split="train",
    download_mode="reuse_cache_if_exists"
)
print(f"✅ Dataset loaded. Total samples available: {len(dataset)}")

# ----------------------------
# 2️⃣ Take first 20000 images
# ----------------------------
print("2️⃣ Selecting first 5000 images...")
subset_images = [dataset[i]['image'] for i in range(5000)]
print(f"✅ Selected {len(subset_images)} images")

# ----------------------------
# 3️⃣ Split train/val (80/20)
# ----------------------------
print("3️⃣ Splitting into train and validation sets (80/20)...")
train_imgs, val_imgs = train_test_split(subset_images, test_size=0.2, random_state=42)
print(f"✅ Train images: {len(train_imgs)}, Validation images: {len(val_imgs)}")

# ----------------------------
# 4️⃣ Create VicReg folder structure
# ----------------------------
print("4️⃣ Creating folder structure for VicReg...")
base_dir = Path("/Users/samprasmanueldsouza/Desktop/Home Work/datasets/mydataset_small_5000")
train_dir = base_dir / "train" / "class1"
val_dir = base_dir / "val" / "class1"
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)
print(f"✅ Folders created at {base_dir}")

# ----------------------------
# 5️⃣ Save images
# ----------------------------
def save_images(img_list, folder):
    print(f"5️⃣ Saving {len(img_list)} images to {folder}...")
    for idx, img in enumerate(tqdm(img_list, desc=f"Saving to {folder}")):
        img.save(folder / f"{idx}.jpg")
    print(f"✅ Finished saving images to {folder}")

save_images(train_imgs, train_dir)
save_images(val_imgs, val_dir)

print("🎉 VicReg-ready dataset creation complete!")
print(f"Dataset location: {base_dir}")
