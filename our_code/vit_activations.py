"""
Extract activations from DINOv2 Vision Transformer and save to disk!
We save as .bin files containing [n_images, n_layers, n_patches + 1, d_vit] tensors.
"""
import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm


# ============== CONFIG ==============
# folder with class subfolders, change accordingly
DATA_ROOT = "/Users/danielpanariti/Documents/universite/m2_2025-2026_MIND/s3/deep_l/projet/bioscan_5m/bioscan5m/images/cropped_256"
OUTPUT_DIR = "./activations"        # where to save .bin files
DEVICE = "cpu"                      
BATCH_SIZE = 32
NUM_WORKERS = 4

# DINOv2 ViT-B/14 settings
VIT_CHECKPOINT = "dinov2_vitb14"
D_VIT = 768                         # hidden dimension for ViT-Base
N_PATCHES = 256                     # 224/14 = 16, so 16x16 = 256 patches
LAYER_TO_EXTRACT = -2               # second to last layer 
IMGS_PER_FILE = 10000               # how many images per .bin file


# ============== DINOV2 WRAPPER ==============
class DinoV2Wrapper(nn.Module):
    """
    Wrapper for DINOv2 model.
    
    The input sequence to ViT looks like:
    [CLS, reg1, reg2, ..., regN, patch1, patch2, ..., patch256]
    
    - CLS: special token at index 0, represents whole image
    - reg tokens: DINOv2 specific, we skip these
    - patch tokens: the actual image patches we care about
    """
    
    def __init__(self, checkpoint="dinov2_vitb14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", checkpoint)
        
    def get_blocks(self):
        # transformer blocks are stored in model.blocks
        return self.model.blocks
    
    def get_patch_indices(self, n_patches):
        """
        Returns indices to select CLS + patch tokens, skipping register tokens.
        E.g for 4 register tokens, we'd want indices [0, 5, 6, 7, ..., 260]
        """
        n_reg = self.model.num_register_tokens
        indices = torch.cat([
            torch.tensor([0]),  # CLS is always first
            torch.arange(n_reg + 1, n_reg + 1 + n_patches),  # patches come after registers
        ])
        return indices
    
    def forward(self, x):
        out = self.model.forward_features(x)
        # combine CLS and patch tokens into one tensor
        features = torch.cat([
            out["x_norm_clstoken"][:, None, :],
            out["x_norm_patchtokens"]
        ], dim=1)
        return features


# ============== ACTIVATION RECORDER ==============
class ActivationRecorder(nn.Module):
    """
    Uses PyTorch hooks to grab activations from a specific layer.
    
    Hooks are callbacks that get called automatically during forward pass.
    We register a hook on the layer we want, and it saves the output.
    """
    
    def __init__(self, vit, n_patches, layer_idx):
        super().__init__()
        self.vit = vit
        self.n_patches = n_patches
        self.patch_indices = vit.get_patch_indices(n_patches)
        
        # this will store the activations when hook is called
        self._activations = None
        
        # register hook on the layer we want
        blocks = vit.get_blocks()
        blocks[layer_idx].register_forward_hook(self._hook)
    
    def _hook(self, module, input, output):
        """
        This gets called automatically when the layer runs.
        output shape: [batch, num_tokens, hidden_dim]
        """
        # grab only CLS + patches, skip register tokens
        self._activations = output[:, self.patch_indices, :].detach()
    
    def forward(self, x):
        self._activations = None
        _ = self.vit(x)  # this triggers the hook
        return self._activations.cpu()


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

# ============== SAVE ACTS TO .BIN ==============
def save_activations(activations_list, file_idx):
    """Save list of activation arrays to a .bin file."""
    combined = np.concatenate(activations_list, axis=0)
    filepath = Path(OUTPUT_DIR) / f"activations_{file_idx:04d}.bin"
    combined.tofile(filepath)
    print(f"Saved {len(combined)} images to {filepath}")


# ============== MAIN EXTRACTION ==============
def extract_activations():
    print(f"Loading DINOv2 model...")
    vit = DinoV2Wrapper(VIT_CHECKPOINT)
    recorder = ActivationRecorder(vit, N_PATCHES, LAYER_TO_EXTRACT).to(DEVICE)
    
    print(f"Loading images from {DATA_ROOT}")
    dataset = datasets.ImageFolder(DATA_ROOT, transform=get_transform())
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    print(f"Found {len(dataset)} images")
    
    # create output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # we'll collect activations and save in chunks
    all_activations = []
    file_idx = 0
    
    with torch.inference_mode():
        for images, _ in tqdm(dataloader, desc="Extracting"):
            images = images.to(DEVICE)
            acts = recorder(images)  # records activations for this batch of images, shape: [batch, 257, 768]
            all_activations.append(acts.numpy()) # append to final list of activations
            
            # save to file when we have enough
            total_so_far = sum(len(a) for a in all_activations)
            if total_so_far >= IMGS_PER_FILE:
                save_activations(all_activations, file_idx)
                all_activations = []
                file_idx += 1
    
    # save any remaining
    if all_activations:
        save_activations(all_activations, file_idx)
    
    print(f"Done! Saved to {OUTPUT_DIR}/")
    
# ============== RUN ==============
if __name__ == "__main__":
    extract_activations()