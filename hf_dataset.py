import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import io

class HFDataset(Dataset):
    def __init__(self, dataset_names, transform=None):
        """
        Args:
            dataset_names (list of str): List of dataset names or paths to load.
                                         Can include split info like 'dataset:split=train'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.datasets = []
        
        for name in dataset_names:
            if ":" in name:
                d_name, split = name.split(":")
                # Handle "split=train" format
                if "=" in split:
                    _, split_name = split.split("=")
                else:
                    split_name = split
            else:
                d_name = name
                split_name = "train" # Default to train if not specified
            
            print(f"Loading {d_name} split {split_name}...")
            ds = load_dataset(d_name, split=split_name)
            
            # Ensure we only keep the 'image' column to avoid schema mismatches if concatenating
            if 'image' in ds.column_names:
                ds = ds.select_columns(['image'])
            
            self.datasets.append(ds)

        if len(self.datasets) > 1:
            self.dataset = concatenate_datasets(self.datasets)
        else:
            self.dataset = self.datasets[0]

        print(f"Total samples: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image']
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
            
        target = 0 # Dummy target as in the original FlatImageFolder
        return img, target
