"""
Segmentation Inference + Submission Script

Runs a trained segmentation head on color images, saves predictions,
and generates a Kaggle submission CSV.

Outputs:
  <OUTPUT_DIR>/predictions/   — raw class-ID mask (uint8 PNG) for every image
  <OUTPUT_DIR>/comparisons/   — NUM_VISUALIZATION_SETS random side-by-side pairs
  <OUTPUT_DIR>/submission.csv — Kaggle RLE submission
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import cv2
import os
from tqdm import tqdm

plt.switch_backend('Agg')


# ============================================================================
# Configuration — edit these before running
# ============================================================================

IMAGE_DIR   = "./testing_images/Color_Images"   # folder of color input images
# IMAGE_DIR   = "./dataset/Color_Images"
MODEL_PATH  = "./segmentation_head.pth"
OUTPUT_DIR  = "./predictions"
BATCH_SIZE  = 200

# Number of random side-by-side comparison images to save
# (color input | colorized prediction)
NUM_VISUALIZATION_SETS = 200


# ============================================================================
# Class / palette configuration
# ============================================================================


CLASS_NAMES = ['other', 'buildings', 'water', 'trees']
N_CLASSES   = len(CLASS_NAMES)  # 4

# Fixed color palette — deterministic across all runs.
# Index = collapsed class ID.
COLOR_PALETTE = np.array([
    [200, 200, 200],   # 0 - other
    [255,   0,   0],   # 1 - buildings
    [  0,   0, 255],   # 2 - water
    [  0, 255,   0],   # 3 - trees
], dtype=np.uint8)


# ============================================================================
# Mask utilities
# ============================================================================

def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert integer class mask (H, W) to RGB image using COLOR_PALETTE."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(N_CLASSES):
        color_mask[mask == class_id] = COLOR_PALETTE[class_id]
    return color_mask


def mask_to_rle(mask: np.ndarray) -> str:
    """Convert a binary mask to a Kaggle RLE string."""
    if mask.sum() == 0:
        return " "
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# ============================================================================
# Dataset — color images only
# ============================================================================

def snap_to_patch(size: int, patch: int = 14) -> int:
    """Round size down to the nearest multiple of patch size."""
    return (size // patch) * patch


class ImageDataset(Dataset):
    """
    Loads images without a fixed resize. Each image is snapped to the nearest
    multiple of 14 in both dimensions (required by DINOv2 patch size), preserving
    the original aspect ratio as closely as possible.
    Returns the original (W, H) so the prediction can be resized back.
    """
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.data_ids = sorted(os.listdir(image_dir))
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size  # PIL size is (W, H)

        # Snap each dimension down to the nearest multiple of 14
        new_w = max(14, snap_to_patch(orig_w))
        new_h = max(14, snap_to_patch(orig_h))
        image = image.resize((new_w, new_h), Image.BILINEAR)

        return self.to_tensor(image), data_id, orig_w, orig_h


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style) — must match training
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x, tokenH, tokenW):
        """
        x      : (B, N, C) patch token tensor from DINOv2.
        tokenH, tokenW: token grid for the current input resolution (input_h // 14, input_w // 14).
        Passed dynamically so any image size works without reloading the model.
        """
        B, N, C = x.shape
        x = x.reshape(B, tokenH, tokenW, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# Visualization
# ============================================================================

def save_comparison(img_tensor, pred_mask_np, output_path, data_id):
    """Save side-by-side: color input | colorized prediction."""
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1) * std + mean
    img = np.clip(img, 0, 1)

    pred_color = mask_to_color(pred_mask_np)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    axes[1].imshow(pred_color)
    axes[1].set_title('Predicted Segmentation')
    axes[1].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Submission generation
# ============================================================================

def generate_submission(preds_dir: str, output_csv: str):
    """
    Read raw class-ID masks from preds_dir and write a Kaggle RLE submission CSV.
    Each image produces N_CLASSES rows, one per class.
    """
    mask_paths = sorted(Path(preds_dir).glob("*.png"))
    if not mask_paths:
        raise ValueError(f"No PNG masks found in {preds_dir}")

    rows = []
    for mask_path in mask_paths:
        image_id = mask_path.stem
        mask = np.array(Image.open(mask_path))
        for cls in range(N_CLASSES):
            binary_mask = (mask == cls).astype(np.uint8)
            rle = mask_to_rle(binary_mask)
            rows.append({
                "id":    f"{image_id}_{cls}",
                # "id":        image_id,
                "class":     cls,
                "predicted": rle,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}  ({len(df)} rows)")


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset and loader
    # Images are processed one at a time because each may be a different size.
    # Each image is snapped to its own nearest-multiple-of-14 dimensions.
    print(f"Loading images from {IMAGE_DIR}...")
    dataset = ImageDataset(image_dir=IMAGE_DIR)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(dataset)} images")

    # Load DINOv2 backbone
    print("Loading DINOv2 backbone...")
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded!")

    # Probe embedding dimension using the first image
    sample_img, _, _, _ = dataset[0]
    with torch.no_grad():
        probe = backbone_model.forward_features(
            sample_img.unsqueeze(0).to(device)
        )["x_norm_patchtokens"]
    n_embedding = probe.shape[2]
    print(f"Embedding dimension: {n_embedding}")

    # Load classifier — infer out_channels from checkpoint
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    checkpoint_out_channels = checkpoint['classifier.weight'].shape[0]
    print(f"Checkpoint output channels: {checkpoint_out_channels}")
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=checkpoint_out_channels,
    )
    classifier.load_state_dict(checkpoint)
    classifier.to(device)
    classifier.eval()
    print("Model loaded!")

    # Output folders
    preds_dir       = os.path.join(OUTPUT_DIR, 'predictions')
    comparisons_dir = os.path.join(OUTPUT_DIR, 'comparisons')
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # Pre-select random indices for visualization
    num_viz     = min(NUM_VISUALIZATION_SETS, len(dataset))
    viz_indices = set(np.random.choice(len(dataset), size=num_viz, replace=False).tolist())
    print(f"/nWill save {num_viz} random comparison images")
    
    # -------------------------------------------------------------------------
    # Inference loop
    # -------------------------------------------------------------------------
    print(f"/nRunning inference on {len(dataset)} images...")
    sample_count = 0
    viz_count    = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Processing", unit="image")
        for imgs, data_ids, orig_ws, orig_hs in pbar:
            # batch_size=1, so unpack the single item
            img        = imgs[0].unsqueeze(0).to(device)   # (1, C, H, W)
            data_id    = data_ids[0]
            orig_w, orig_h = int(orig_ws[0]), int(orig_hs[0])

            _, _, input_h, input_w = img.shape
            tokenH = input_h // 14
            tokenW = input_w // 14
            
            patch_tokens = backbone_model.forward_features(img)["x_norm_patchtokens"]
            logits  = classifier(patch_tokens, tokenH=tokenH, tokenW=tokenW)
            # Upsample to the snapped input resolution
            outputs = F.interpolate(logits, size=(input_h, input_w), mode="bilinear", align_corners=False)

            predicted_mask_raw = torch.argmax(outputs, dim=1)  # (1, H, W)
            predicted_mask = predicted_mask_raw[0]
            pred_mask = predicted_mask.cpu().numpy().astype(np.uint8)

            # Resize prediction back to original image dimensions
            pred_mask_orig = np.array(
                Image.fromarray(pred_mask).resize((orig_w, orig_h), Image.NEAREST)
            )

            base_name = os.path.splitext(data_id)[0]

            # Save raw class-ID mask at original resolution (used by submission generator)
            Image.fromarray(pred_mask_orig).save(
                os.path.join(preds_dir, f'{base_name}.png')
            )

            # Save comparison for randomly selected samples
            if sample_count in viz_indices:
                save_comparison(
                    imgs[0], pred_mask,
                    os.path.join(comparisons_dir, f'comparison_{viz_count:03d}_{base_name}.png'),
                    data_id
                )
                viz_count += 1

            sample_count += 1

    print(f"/nDone! Processed {sample_count} images.")

    # -------------------------------------------------------------------------
    # Generate submission CSV from the saved raw masks
    # -------------------------------------------------------------------------
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    generate_submission(preds_dir=preds_dir, output_csv=submission_path)

    print(f"/nAll outputs saved to {OUTPUT_DIR}/")
    print(f"  predictions/   -- raw class-ID mask for every image")
    print(f"  comparisons/   -- {num_viz} random side-by-side pairs")
    print(f"  submission.csv -- Kaggle RLE submission")


if __name__ == "__main__":
    main()