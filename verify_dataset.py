import torch
from hf_dataset import HFDataset
from torchvision import transforms

def test_dataset():
    print("Testing HFDataset...")
    # Use a simple transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Test with just one dataset first to be quick, or both if possible
    # Note: This requires internet access to download/load the dataset metadata
    try:
        ds = HFDataset(["tsbpp/fall2025_deeplearning"], transform=transform)
        print(f"Successfully loaded dataset. Size: {len(ds)}")
        
        img, target = ds[0]
        print(f"Sample 0 shape: {img.shape}, Target: {target}")
        
        assert img.shape == (3, 224, 224)
        print("Shape check passed.")
        
    except Exception as e:
        print(f"Dataset loading failed: {e}")

if __name__ == "__main__":
    test_dataset()
