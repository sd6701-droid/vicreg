"""
Create Kaggle Submission with Custom DINOv2/VICReg Model (Linear Probe + TTA)
=============================================================================

This script generates a submission using a pretrained DINOv2 or VICReg model.
It improves upon the standard linear probe by adding Test Time Augmentation (TTA).
Features are extracted from 10 views (5 crops + flips) of each image and averaged.
Then a Linear Probe is trained using PyTorch with hyperparameter tuning.

Usage:
    python create_submission_linear_tta.py \
        --data_dir ./data \
        --checkpoint logs/dinov2_vits14_ep100/checkpoint.pth \
        --output submission_linear_tta.csv \
        --tune_hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os
import random
import time

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

def exclude_bias_and_norm(p):
    return p.ndim == 1

# Import DINOv2 model definitions
try:
    from dinov2.models import vision_transformer as vits
except ImportError:
    # It's okay if we are just using VICReg or if dinov2 is not in path but in src
    pass

# ============================================================================
#                          MODEL SECTION
# ============================================================================

import torchvision.models as models
from torchvision.models.resnet import Bottleneck, ResNet

class ResNet50x2(ResNet):
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 6, 3])
        self.inplanes = 128
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(Bottleneck, 128, 3)
        self.layer2 = self._make_layer(Bottleneck, 256, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 512, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 1024, 3, stride=2)
        self.fc = nn.Identity()

class FeatureExtractor:
    def __init__(self, checkpoint_path, arch='vit_small', patch_size=14, device='cuda', image_size=224):
        self.device = device
        self.image_size = image_size
        self.arch = arch
        
        print(f"Initializing model {arch}...")
        if arch == 'vicreg_resnet50x2':
            self.model = ResNet50x2()
        elif arch in vits.__dict__:
            self.model = vits.__dict__[arch](
                patch_size=patch_size,
                img_size=518,
                init_values=1.0,
                block_chunks=0
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Handle pointer files
        try:
            with open(checkpoint_path, 'rb') as f:
                header = f.read(4)
            if header != b'PK\x03\x04' and header != b'\x80\x02\x8a\n':
                with open(checkpoint_path, 'r') as f:
                    content = f.read().strip()
                if len(content) < 256 and '\n' not in content:
                    parent_dir = os.path.dirname(checkpoint_path)
                    resolved_path = os.path.join(parent_dir, content)
                    if os.path.exists(resolved_path):
                        checkpoint_path = resolved_path
        except Exception:
            pass
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'teacher' in checkpoint: state_dict = checkpoint['teacher']
        elif 'student' in checkpoint: state_dict = checkpoint['student']
        elif 'model' in checkpoint: state_dict = checkpoint['model']
        else: state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '').replace('backbone.', '')
            new_state_dict[k] = v
            
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Model loaded with message: {msg}")
        
        self.model.to(device)
        self.model.eval()
        
        # Standard Transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # TTA Transforms
        # Resize to slightly larger then 5 crop
        resize_size = int(image_size / 0.875) 
        self.resize_tta = transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.BICUBIC)
        self.five_crop = transforms.FiveCrop(image_size)
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.norm = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        
    def extract_batch_features(self, images, use_tta=False):
        if not use_tta:
            batch_tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
            with torch.no_grad():
                if self.arch == 'vicreg_resnet50x2':
                    features = self.model(batch_tensors)
                else:
                    outputs = self.model.forward_features(batch_tensors)
                    features = outputs["x_norm_clstoken"]
            return features.cpu().numpy()
        else:
            # TTA: 10 crops per image
            batch_crops = []
            for img in images:
                # Resize
                img_resized = self.resize_tta(img)
                # 5 Crops
                crops = self.five_crop(img_resized) # tuple of 5 PIL images
                # Flips
                flipped_crops = [F.hflip(c) for c in crops]
                # Combine
                all_crops = list(crops) + flipped_crops
                # To Tensor & Norm
                all_crops = [self.norm(F.to_tensor(c)) for c in all_crops]
                # Stack: (10, C, H, W)
                batch_crops.append(torch.stack(all_crops))
            
            # Stack Batch: (B, 10, C, H, W)
            batch_crops = torch.stack(batch_crops).to(self.device)
            B, n_crops, C, H, W = batch_crops.shape
            
            # Flatten: (B*10, C, H, W)
            inputs = batch_crops.view(-1, C, H, W)
            
            with torch.no_grad():
                if self.arch == 'vicreg_resnet50x2':
                    features = self.model(inputs)
                else:
                    outputs = self.model.forward_features(inputs)
                    features = outputs["x_norm_clstoken"]
            
            # Reshape: (B, 10, D)
            features = features.view(B, n_crops, -1)
            
            # Mean Pool over crops
            features = features.mean(dim=1)
            
            return features.cpu().numpy()

# ============================================================================
#                          DATA SECTION
# ============================================================================

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_list, labels=None):
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        image = Image.open(img_path).convert('RGB')
        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name

def collate_fn(batch):
    if len(batch[0]) == 3:
        return [item[0] for item in batch], [item[1] for item in batch], [item[2] for item in batch]
    else:
        return [item[0] for item in batch], [item[1] for item in batch]

def extract_features_from_dataloader(feature_extractor, dataloader, split_name='train', use_tta=False):
    all_features, all_labels, all_filenames = [], [], []
    print(f"\nExtracting features from {split_name} set (TTA={use_tta})...")
    for batch in tqdm(dataloader, desc=f"{split_name} features"):
        if len(batch) == 3:
            images, labels, filenames = batch
            all_labels.extend(labels)
        else:
            images, filenames = batch
        features = feature_extractor.extract_batch_features(images, use_tta=use_tta)
        all_features.append(features)
        all_filenames.extend(filenames)
    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels) if all_labels else None
    return features, labels, all_filenames

# ============================================================================
#                          LINEAR PROBE (PyTorch)
# ============================================================================

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def train_linear_probe(train_features, train_labels, val_features, val_labels, 
                       batch_size=64, lr=1e-3, weight_decay=1e-4, 
                       epochs=50, optimizer_name='adamw', scheduler_name='cosine', 
                       device='cuda', verbose=True):
    
    input_dim = train_features.shape[1]
    num_classes = len(np.unique(train_labels))
    
    model = LinearProbe(input_dim, num_classes).to(device)
    
    # Create DataLoaders
    train_ds = TensorDataset(torch.from_numpy(train_features).float(), torch.from_numpy(train_labels).long())
    val_ds = TensorDataset(torch.from_numpy(val_features).float(), torch.from_numpy(val_labels).long())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
    # Scheduler
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*0.6), gamma=0.1)
    else:
        scheduler = None
        
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
        if scheduler:
            scheduler.step()
            
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        val_acc = correct / total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Val Acc: {val_acc:.4f}")
            
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_acc

def tune_linear_probe(train_features, train_labels, val_features, val_labels, n_trials=20, device='cuda'):
    print(f"\nStarting Hyperparameter Tuning ({n_trials} trials)...")
    
    best_acc = 0.0
    best_params = {}
    best_model = None
    
    # Hyperparameter distributions
    param_dist = {
        'batch_size': [32, 64, 128, 256, 512],
        'lr': lambda: 10 ** random.uniform(-4, -1), # 1e-4 to 1e-1
        'weight_decay': lambda: 10 ** random.uniform(-6, -3), # 1e-6 to 1e-3
        'epochs': [20, 30, 50, 100],
        'optimizer_name': ['sgd', 'adamw'],
        'scheduler_name': ['cosine', 'step']
    }
    
    for i in range(n_trials):
        # Sample parameters
        params = {
            'batch_size': random.choice(param_dist['batch_size']),
            'lr': param_dist['lr'](),
            'weight_decay': param_dist['weight_decay'](),
            'epochs': random.choice(param_dist['epochs']),
            'optimizer_name': random.choice(param_dist['optimizer_name']),
            'scheduler_name': random.choice(param_dist['scheduler_name']),
            'device': device,
            'verbose': False
        }
        
        print(f"\nTrial {i+1}/{n_trials}: {params}")
        
        model, val_acc = train_linear_probe(
            train_features, train_labels, val_features, val_labels, **params
        )
        
        print(f"  -> Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            best_model = model
            print(f"  *** New Best! ***")
            
    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    print(f"Best Hyperparameters: {best_params}")
    
    return best_model, best_params

# ============================================================================
#                          SUBMISSION
# ============================================================================

def create_submission(test_features, test_filenames, model, output_path, device='cuda'):
    print("\nGenerating predictions on test set...")
    model.eval()
    
    test_ds = TensorDataset(torch.from_numpy(test_features).float())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for x in test_loader:
            x = x[0].to(device)
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'class_id': all_preds
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

# ============================================================================
#                          MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Linear Probe Submission + TTA')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='submission_linear_tta.csv')
    parser.add_argument('--arch', type=str, default='vit_small')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=96)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_tta', action='store_true', help='Disable TTA')
    
    # Tuning args
    parser.add_argument('--tune_hyperparameters', action='store_true')
    parser.add_argument('--n_trials', type=int, default=20)
    
    # Fixed training args (if not tuning)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='cosine')
    
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load Data
    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')
    
    # Datasets & Loaders
    train_loader = DataLoader(ImageDataset(data_dir/'train', train_df['filename'].tolist(), train_df['class_id'].tolist()), 
                            batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(ImageDataset(data_dir/'val', val_df['filename'].tolist(), val_df['class_id'].tolist()), 
                          batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(ImageDataset(data_dir/'test', test_df['filename'].tolist()), 
                           batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn)
    
    # Feature Extraction
    extractor = FeatureExtractor(args.checkpoint, args.arch, args.patch_size, device, args.image_size)
    
    use_tta = not args.no_tta
    train_feats, train_lbls, _ = extract_features_from_dataloader(extractor, train_loader, 'train', use_tta=use_tta)
    val_feats, val_lbls, _ = extract_features_from_dataloader(extractor, val_loader, 'val', use_tta=use_tta)
    test_feats, _, test_names = extract_features_from_dataloader(extractor, test_loader, 'test', use_tta=use_tta)
    
    # Train/Tune
    if args.tune_hyperparameters:
        model, _ = tune_linear_probe(train_feats, train_lbls, val_feats, val_lbls, args.n_trials, device)
    else:
        model, _ = train_linear_probe(
            train_feats, train_lbls, val_feats, val_lbls,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            epochs=args.epochs, optimizer_name=args.optimizer, scheduler_name=args.scheduler,
            device=device
        )
        
    # Submission
    create_submission(test_feats, test_names, model, args.output, device)

if __name__ == "__main__":
    main()