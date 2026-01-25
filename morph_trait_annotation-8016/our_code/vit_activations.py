"""
Extract activations from DINOv2 Vision Transformer and save to computer. These activations will be fed into the SAE model.
"""
import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm


# ============== CONFIG ==============
DATA_ROOT = "/Users/danielpanariti/Documents/universite/m2_2025-2026_MIND/s3/deep_l/projet/bioscan_5m/bioscan5m/images/cropped_256"
OUTPUT_DIR = "./activations"
DEVICE = "cpu"                      
BATCH_SIZE = 32
NUM_WORKERS = 4

# DINOv2 ViT-B/14 settings
VIT_CHECKPOINT = "dinov2_vitb14"
D_VIT = 768
N_PATCHES = 256
LAYERS_TO_EXTRACT = [-2]  # List of layers (original supports multiple)
IMGS_PER_SHARD = 10000


# ============== DINOV2 WRAPPER ==============
class DinoV2Wrapper(nn.Module):
    def __init__(self, checkpoint="dinov2_vitb14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", checkpoint)
        
    def get_blocks(self):
        return self.model.blocks
    
    def get_patch_indices(self, n_patches):
        n_reg = self.model.num_register_tokens
        indices = torch.cat([
            torch.tensor([0]),
            torch.arange(n_reg + 1, n_reg + 1 + n_patches),
        ])
        return indices
    
    def forward(self, x):
        out = self.model.forward_features(x)
        features = torch.cat([
            out["x_norm_clstoken"][:, None, :],
            out["x_norm_patchtokens"]
        ], dim=1)
        return features


# ============== ACTIVATION RECORDER ==============
class ActivationRecorder(nn.Module):
    """Records activations from multiple layers."""
    
    def __init__(self, vit, n_patches, layer_indices):
        super().__init__()
        self.vit = vit
        self.n_patches = n_patches
        self.patch_indices = vit.get_patch_indices(n_patches)
        self.layer_indices = layer_indices
        
        # Storage for each layer's activations
        self._activations = {}
        
        # Register hooks on all requested layers
        blocks = vit.get_blocks()
        for layer_idx in layer_indices:
            blocks[layer_idx].register_forward_hook(
                lambda module, inp, out, idx=layer_idx: self._hook(idx, out)
            )
    
    def _hook(self, layer_idx, output):
        self._activations[layer_idx] = output[:, self.patch_indices, :].detach()
    
    def forward(self, x):
        self._activations = {}
        _ = self.vit(x)
        
        # Stack layers in order: [batch, n_layers, n_patches+1, d_vit]
        stacked = torch.stack(
            [self._activations[idx] for idx in self.layer_indices],
            dim=1
        )
        return stacked.cpu()


# ============== IMAGE TRANSFORMS ==============
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ============== MAIN EXTRACTION ==============
def extract_activations():
    print(f"Loading DINOv2 model...")
    vit = DinoV2Wrapper(VIT_CHECKPOINT)
    recorder = ActivationRecorder(vit, N_PATCHES, LAYERS_TO_EXTRACT).to(DEVICE)
    
    print(f"Loading images from {DATA_ROOT}")
    dataset = datasets.ImageFolder(DATA_ROOT, transform=get_transform())
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    n_imgs = len(dataset)
    print(f"Found {n_imgs} images")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Calculate shard parameters (matching original logic)
    n_layers = len(LAYERS_TO_EXTRACT)
    n_patches_with_cls = N_PATCHES + 1
    n_patches_per_shard = IMGS_PER_SHARD * n_layers * n_patches_with_cls
    
    # Shape for each shard file
    shard_shape = (IMGS_PER_SHARD, n_layers, n_patches_with_cls, D_VIT)
    
    all_activations = []
    shard_idx = 0
    
    with torch.inference_mode():
        for images, _ in tqdm(dataloader, desc="Extracting"):
            images = images.to(DEVICE)
            # Shape: [batch, n_layers, n_patches+1, d_vit]
            acts = recorder(images)
            all_activations.append(acts.numpy())
            
            total_imgs = sum(len(a) for a in all_activations)
            if total_imgs >= IMGS_PER_SHARD:
                save_shard(all_activations, shard_idx, shard_shape)
                all_activations = []
                shard_idx += 1
    
    # Save remaining (may be smaller than IMGS_PER_SHARD)
    if all_activations:
        remaining_imgs = sum(len(a) for a in all_activations)
        final_shape = (remaining_imgs, n_layers, n_patches_with_cls, D_VIT)
        save_shard(all_activations, shard_idx, final_shape)
    
    # Save metadata.json
    metadata = {
        "vit_family": "dinov2",
        "vit_ckpt": VIT_CHECKPOINT,
        "layers": LAYERS_TO_EXTRACT,
        "n_patches_per_img": N_PATCHES,
        "cls_token": True,
        "d_vit": D_VIT,
        "seed": 42,
        "n_imgs": n_imgs,
        "n_patches_per_shard": n_patches_per_shard,
        "data": f"ImageFolderDataset(root='{DATA_ROOT}')"
    }
    
    metadata_path = Path(OUTPUT_DIR) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")
    
    print(f"Done! Saved to {OUTPUT_DIR}/")


def save_shard(activations_list, shard_idx, shape):
    """Save activations to a .bin file with correct naming."""
    combined = np.concatenate(activations_list, axis=0).astype(np.float32)
    
    # Pad if necessary to match expected shape (for non-final shards)
    if combined.shape[0] < shape[0]:
        # This is the final shard - use actual shape
        pass
    
    # Use original naming convention: acts000000.bin
    filepath = Path(OUTPUT_DIR) / f"acts{shard_idx:06d}.bin"
    combined.tofile(filepath)
    print(f"Saved {len(combined)} images to {filepath}")


if __name__ == "__main__":
    extract_activations()