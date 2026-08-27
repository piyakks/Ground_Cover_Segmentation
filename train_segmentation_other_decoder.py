"""
Segmentation Training Script
Converted from train_mask.ipynb
Trains a segmentation head on top of DINOv2 backbone
"""

from random import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision
from tqdm import tqdm

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    Image.fromarray(img.astype(np.uint8)).save(filename)  # PIL already imported



# ============================================================================
# Bucket Batch Sampler
# ============================================================================

class BucketBatchSampler(Sampler):
    """Groups samples by image size so each batch contains similarly-sized
    images, minimising padding waste in pad_collate."""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Record snapped (H, W) for every sample without loading pixel data
        sizes = []
        for fname in dataset.data_ids:
            path = os.path.join(dataset.image_dir, fname)
            with Image.open(path) as img:
                w, h = img.size
            snapped_h = (h // 14) * 14
            snapped_w = (w // 14) * 14
            sizes.append((snapped_h, snapped_w))

        # Sort indices by (H, W) so similar sizes end up in the same bucket
        self.sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i])

    def __iter__(self):
        indices = self.sorted_indices.copy()
        # Shuffle within each bucket (groups of batch_size) to add variety
        # while keeping sizes similar
        if self.shuffle:
            buckets = [indices[i:i + self.batch_size]
                       for i in range(0, len(indices), self.batch_size)]
            import random
            random.shuffle(buckets)
            indices = [idx for bucket in buckets for idx in bucket]
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """

        num_classes = logits.shape[1]

        # softmax
        probs = torch.softmax(logits, dim=1)

        # one-hot encoding
        targets_onehot = F.one_hot(
            targets.clamp(0, num_classes - 1),
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_onehot, dims)
        union = torch.sum(probs + targets_onehot, dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits : (B, C, H, W)
        targets: (B, H, W)
        """

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction='none',
            ignore_index=self.ignore_index
        )

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            focal_loss = focal_loss[valid]

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    255 : 0,    # blank
    10000: 0,     # other
    6000: 1,     # buildings
    2: 0,  # other/Impervious Surfaces NONE PRESENT
    11000 : 2, # water
    400: 0,      # other/ Prairie/Grassland/Natural Ground Cover
    300: 0,      # other/ Prairie/Grassland/Natural Ground Cover
    600: 0,      # other/ Prairie/Grassland/Natural Ground Cover
    700: 0,      # other/ Prairie/Grassland/Natural Ground Cover
    100 : 3, #tree coverage
    200 : 3, #tree coverage
    0 : 0,   #other / Turf/Irrigated Land
    7100 : 0,   #other / Turf/Irrigated Land
    800: 0,   # other/ Barren Land
    300: 0,    #other/dry grass
    400 : 0,  # other/lush grass
    550: 0,    # other/ground clutter
    900: 0,    # other/man made items
    1000: 0,    # vehicles
    7000: 0,    #other/roads
    7100: 0,    #other/roads
}
n_classes = len(set(value_map.values()))

def get_filepaths():
    train_folder = "./dataset/train"  # Enter train folder here
    val_folder = "./dataset/val"             # Enter val folder here
    return train_folder, val_folder


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random
class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        # Both color images and masks are .png files with same name
        mask_path = os.path.join(self.masks_dir, data_id)
        # print(mask_path)
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
     
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
       
        mask = convert_mask(mask)
        if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # angle = random.uniform(-15, 15)
        # image = TF.rotate(
        # image,
        # angle,
        # interpolation=InterpolationMode.BILINEAR
        # )
        # mask = TF.rotate(
        # mask,
        # angle,
        # interpolation=InterpolationMode.NEAREST
        # )
           
        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255
            
        return image, mask


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style)
# ============================================================================
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()

        # 1x1 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 3x3 atrous conv
        self.convs = nn.ModuleList()
        for rate in atrous_rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=1, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # image-level pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(atrous_rates)), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        features = [self.conv1(x)]
        for conv in self.convs:
            features.append(conv(x))
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
        features.append(gp)
        x = torch.cat(features, dim=1)
        return self.project(x)
    
class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=16,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x) #그냥 conv 3*3 2번 진행 함 
        # res = self.body(x).mul(self.res_scale)
        res += x #원본 이미지에 채널 
        return res
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=14):
        super().__init__()
        self.patch_size = patch_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        # self.rcab=RCAB(128,kernel_size=7,reduction=4)
        
        
        self.stem2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.rcab2=RCAB(64,kernel_size=3,reduction=4)
        
        self.aspp=ASPP(64,64)
        
        self.block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, padding=3, groups=64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.GELU(),
        )
        
        self.classifier = nn.Conv2d(32, out_channels, 1)
       
    def forward(self, x, img_h, img_w):
        """
        x: (B, N, C) patch tokens from DINOv2
        img_h, img_w: original image spatial dimensions (used to derive token grid shape)
        """
        B, N, C = x.shape
        token_h = img_h // self.patch_size
        token_w = img_w // self.patch_size
        x = x.reshape(B, token_h, token_w, C).permute(0, 3, 1, 2)
        x = self.stem(x) #128 36
        # x = self.rcab(x)
        x=F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #128 72    
        x = self.stem2(x) #128 36
        x = self.rcab2(x)
        x=F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #128 72   
        x=self.aspp(x)+x
        x = self.block(x)
        
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class)


def compute_dice(pred, target, num_classes=10, smooth=1e-6, ignore_index=255):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    valid = target != ignore_index
    pred, target = pred[valid], target[valid]

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    """Evaluate all metrics on a dataset."""
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []

    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            output = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(output.to(device), imgs.shape[2], imgs.shape[3])
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            iou = compute_iou(outputs, labels, num_classes=num_classes)
            dice = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)

    model.train()
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies)


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    print(f"Saved training curves to '{output_dir}/training_curves.png'")

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.title('Train IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()
    print(f"Saved IoU curves to '{output_dir}/iou_curves.png'")

    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()
    print(f"Saved Dice curves to '{output_dir}/dice_curves.png'")

    # Plot 4: Combined metrics plot
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train')
    plt.plot(history['val_iou'], label='val')
    plt.title('IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train')
    plt.plot(history['val_dice'], label='val')
    plt.title('Dice Score vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i]
            ))

    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 8
    patch_size = 14  # DINOv2 patch size
    lr = 1e-4
    n_epochs = 80

    # Output directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    def snap_to_patch(img, patch=patch_size):
        """Crop image dimensions down to the nearest multiple of patch_size."""
        w, h = img.size
        new_w = (w // patch) * patch
        new_h = (h // patch) * patch
        return img.crop((0, 0, new_w, new_h))

    # Transforms — no fixed Resize; images are snapped to patch-grid alignment at load time
    transform = transforms.Compose([
        transforms.Lambda(snap_to_patch),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Lambda(snap_to_patch),
        transforms.ToTensor(),
    ])

    train_folder, val_folder = get_filepaths()
    # Dataset paths (relative to script location)
    data_dir = train_folder  # os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = val_folder     # os.path.join(script_dir,  'Offroad_Segmentation_Training_Dataset','val')

    def pad_collate(batch):
        """Pad images and masks in a batch to the largest H and W, filling
        mask padding with 255 so CrossEntropyLoss ignores those pixels."""
        imgs, masks = zip(*batch)
        max_h = max(img.shape[1] for img in imgs)
        max_w = max(img.shape[2] for img in imgs)

        padded_imgs, padded_masks = [], []
        for img, mask in zip(imgs, masks):
            h, w = img.shape[1], img.shape[2]
            pad_h, pad_w = max_h - h, max_w - w
            padded_imgs.append(F.pad(img, (0, pad_w, 0, pad_h), value=0))
            padded_masks.append(F.pad(mask, (0, pad_w, 0, pad_h), value=255))

        return torch.stack(padded_imgs), torch.stack(padded_masks)

    # Create datasets
    trainset = MaskDataset(data_dir=data_dir, transform=transform, mask_transform=mask_transform)
    train_sampler = BucketBatchSampler(trainset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(trainset, batch_sampler=train_sampler, collate_fn=pad_collate)

    valset = MaskDataset(data_dir=val_dir, transform=transform, mask_transform=mask_transform)
    val_sampler = BucketBatchSampler(valset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(valset, batch_sampler=val_sampler, collate_fn=pad_collate)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # Load DINOv2 backbone
    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "large"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    # Get embedding dimension from backbone
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")
    print(f"Patch tokens shape: {output.shape}")

    # Create segmentation head
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
    )
    classifier = classifier.to(device)
    
    # Loss and optimizer
    # weights = torch.tensor([1.0, 3.0, 4.0, 2.0]).cuda()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=255)
    dice_loss=DiceLoss()
    
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.1)
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
        'train_pixel_acc': [],
        'val_pixel_acc': []
    }

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # Training phase
        classifier.train()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", 
                          leave=False, unit="batch")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            # print(imgs.shape)
            with torch.no_grad():
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

            logits = classifier(output.to(device), imgs.shape[2], imgs.shape[3])
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            loss = loss_fct(outputs, labels)
            # vals, counts = torch.unique(labels, return_counts=True)
            


            loss2= dice_loss(outputs, labels)
            total_loss=loss+loss2
            total_loss=loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(total_loss.item())
            train_pbar.set_postfix(loss=f"{total_loss.item():.4f}")
        if epoch==60:
            lr_scheduler.step()
        # Validation phase
        classifier.eval()
        val_losses = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", 
                        leave=False, unit="batch")
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)

                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(output.to(device), imgs.shape[2], imgs.shape[3])
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                labels = labels.squeeze(dim=1).long()
                loss = loss_fct(outputs, labels)
                val_losses.append(loss.item())
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Calculate metrics
        train_iou, train_dice, train_pixel_acc = evaluate_metrics(
            classifier, backbone_model, train_loader, device, num_classes=n_classes
        )
        val_iou, val_dice, val_pixel_acc = evaluate_metrics(
            classifier, backbone_model, val_loader, device, num_classes=n_classes
        )

        # Store history
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_pixel_acc'].append(val_pixel_acc)

        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix(
            train_loss=f"{epoch_train_loss:.3f}",
            val_loss=f"{epoch_val_loss:.3f}",
            val_iou=f"{val_iou:.3f}",
            val_acc=f"{val_pixel_acc:.3f}"
        )
        if epoch>49 and epoch%5==0:
            torch.save(classifier.state_dict(), os.path.join(output_dir, f"model_{epoch}.pth"))
            print(f"Saved best model at epoch {epoch+1} with Val IoU: {val_iou:.4f}")

    # Save plots
    print("\nSaving training curves...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    # Save model (in scripts directory)
    model_path = os.path.join(script_dir, "segmentation_head.pth")
    torch.save(classifier.state_dict(), model_path)
    print(f"Saved model to '{model_path}'")

    # Final evaluation
    print("\nFinal evaluation results:")
    print(f"  Final Val Loss:     {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_pixel_acc'][-1]:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()