"""
Simple script to find the top-k images that maximally activate each SAE neuron.
Patch mode: stores patch activations during top-k computation for fast heatmap generation.

Variables ending with:
    _im = refers to whole images
    _p = refers to patches
"""

import os
import json
import random
import torch
import einops
import io
from our_code.sae_model import SparseAutoencoder
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.cm as cm


# ============ HARDCODED CONFIG - CHANGE THESE ============
SAE_CHECKPOINT = "../checkpoints/sae.pt"
ACTIVATIONS_PATH = "../activations_REAL"
OUTPUT_DIR = "../output/neuron_visuals"
IMAGE_DATASET_PATH = "/Users/danielpanariti/Documents/universite/m2_2025-2026_MIND/s3/deep_l/projet/bioscan_5m/bioscan5m/images/cropped_256"

TOP_K = 32  # how many top images to keep per neuron
BATCH_SIZE = 16384  # should be multiple of n_patches_per_img (256)
SAE_BATCH_SIZE = 4096  # for SAE forward pass
NUM_WORKERS = 0
DEVICE = "cpu"  # "cuda" if available

# filtering params - only visualize neurons in this range
MIN_LOG_FREQ = -6.0
MAX_LOG_FREQ = -2.0
MIN_LOG_VALUE = -1.0
MAX_LOG_VALUE = 1.0

# how many random neurons to visualize
NUM_NEURONS_TO_VIS = 100
SEED = 42

# Image processing
RESIZE_SIZE = (512, 512)
CROP_SIZE = (448, 448)
# =========================================================


def load_sae(ckpt_path):
    """Load the sparse autoencoder."""
    with open(ckpt_path, "rb") as f:
        config_line = f.readline().decode()
        sae_config = json.loads(config_line)
        state_dict_bytes = f.read()
        buffer = io.BytesIO(state_dict_bytes)
    
    d_vit = sae_config["d_vit"]
    exp_factor = sae_config["exp_factor"]
    d_sae = d_vit * exp_factor
    
    print(f"  d_vit={d_vit}, exp_factor={exp_factor}, d_sae={d_sae}")
    sae = SparseAutoencoder(d_vit, expansion_factor=exp_factor)
    state_dict = torch.load(buffer, weights_only=True, map_location='cpu')
    sae.load_state_dict(state_dict)
    
    return sae.eval(), d_sae


class PatchActivationsDataset(Dataset):
    """
    Loads pre-computed ViT activations from .bin files.
    Returns individual patches (not CLS tokens) for patch-mode top-k.
    """
    
    def __init__(self, shard_root):
        self.shard_root = Path(shard_root)
        
        with open(self.shard_root / "metadata.json") as f:
            self.metadata = json.load(f)
        
        self.d_vit = self.metadata["d_vit"]
        self.n_layers = len(self.metadata["layers"])
        self.n_patches_per_img = self.metadata["n_patches_per_img"]  # 256
        self.n_patches_with_cls = self.n_patches_per_img + 1  # 257
        self.n_imgs = self.metadata["n_imgs"]
        
        self.shard_files = sorted(self.shard_root.glob("acts*.bin"))
        self.bytes_per_img = self.n_layers * self.n_patches_with_cls * self.d_vit * 4
        
        # Build shard index
        self.shard_boundaries = []
        current_idx = 0
        for shard_path in self.shard_files:
            file_size = shard_path.stat().st_size
            n_imgs_in_shard = file_size // self.bytes_per_img
            self.shard_boundaries.append((current_idx, current_idx + n_imgs_in_shard, shard_path))
            current_idx += n_imgs_in_shard
        
        self.total_imgs = current_idx
        
        # Total patches (excluding CLS)
        self.total_patches = self.total_imgs * self.n_patches_per_img
        
        self._cached_shard = None
        self._cached_shard_idx = -1
        
        print(f"  Found {len(self.shard_files)} shards, {self.total_imgs} images, {self.total_patches} patches")
    
    def __len__(self):
        return self.total_patches
    
    def _load_shard(self, shard_idx):
        start, end, path = self.shard_boundaries[shard_idx]
        n_imgs = end - start
        data = np.fromfile(path, dtype=np.float32)
        data = data.reshape(n_imgs, self.n_layers, self.n_patches_with_cls, self.d_vit)
        return data
    
    def _find_shard_for_img(self, img_idx):
        for shard_idx, (start, end, _) in enumerate(self.shard_boundaries):
            if start <= img_idx < end:
                return shard_idx, img_idx - start
        raise IndexError(f"Image index {img_idx} out of range")
    
    def _ensure_shard_loaded(self, shard_idx):
        if self._cached_shard_idx != shard_idx:
            self._cached_shard = self._load_shard(shard_idx)
            self._cached_shard_idx = shard_idx
    
    def __getitem__(self, idx):
        """Returns a single patch activation."""
        # idx = img_idx * n_patches_per_img + patch_idx
        img_idx = idx // self.n_patches_per_img
        patch_idx = idx % self.n_patches_per_img
        
        shard_idx, local_idx = self._find_shard_for_img(img_idx)
        self._ensure_shard_loaded(shard_idx)
        
        # Patches are at positions 1 to n_patches (0 is CLS)
        patch_act = self._cached_shard[local_idx, 0, patch_idx + 1, :]
        
        return {
            "act": torch.from_numpy(patch_act.copy()),
            "image_i": img_idx,
        }


def load_image_dataset(path):
    """Load the original images dataset."""
    dataset = datasets.ImageFolder(path, transform=None)
    
    class ImageDatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset
            self.classes = dataset.classes
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, label_idx = self.dataset[idx]
            return {
                "image": img,
                "label": self.classes[label_idx],
            }
    
    return ImageDatasetWrapper(dataset)


def get_sae_acts(vit_acts, sae):
    """Get SAE activations in batches to avoid memory issues."""
    sae_acts = []
    for start in range(0, len(vit_acts), SAE_BATCH_SIZE):
        end = min(start + SAE_BATCH_SIZE, len(vit_acts))
        with torch.no_grad():
            _, f_x, _ = sae(vit_acts[start:end].to(DEVICE))
        sae_acts.append(f_x)
    return torch.cat(sae_acts, dim=0)


def gather_batched(values, indices):
    """
    Gather values along dim 1 using batched indices.
    values: [d_sae, n, n_patches]
    indices: [d_sae, k]
    returns: [d_sae, k, n_patches]
    """
    d_sae, n, n_patches = values.shape
    k = indices.shape[1]
    batch_idx = torch.arange(d_sae, device=values.device)[:, None].expand(-1, k)
    return values[batch_idx, indices]


@torch.inference_mode()
def get_topk_patch(sae, d_sae, dataset, dataloader, n_patches_per_img):
    """
    Get top-k images for each neuron using patch activations.
    Stores all patch values for heatmap generation.
    
    Returns:
        top_values_p: [d_sae, top_k, n_patches] - patch activations for top images
        top_i_im: [d_sae, top_k] - image indices
        sparsity: [d_sae] - fraction of patches where neuron fires
        mean_values: [d_sae] - mean activation
    """
    # Storage for top-k with all patch values
    top_values_p = torch.full((d_sae, TOP_K, n_patches_per_img), -1.0, device=DEVICE)
    top_i_im = torch.zeros((d_sae, TOP_K), dtype=torch.long, device=DEVICE)
    
    sparsity = torch.zeros(d_sae, device=DEVICE)
    mean_values = torch.zeros(d_sae, device=DEVICE)
    
    # Adjust batch size to be multiple of n_patches_per_img
    effective_batch_size = (BATCH_SIZE // n_patches_per_img) * n_patches_per_img
    n_imgs_per_batch = effective_batch_size // n_patches_per_img
    
    print(f"  Effective batch size: {effective_batch_size} patches ({n_imgs_per_batch} images)")
    
    for batch in tqdm(dataloader, desc="Finding top-k patches"):
        vit_acts = batch["act"]  # [batch, d_vit]
        image_indices = batch["image_i"]  # [batch]
        
        actual_batch = vit_acts.shape[0]
        
        # Get SAE activations
        sae_acts = get_sae_acts(vit_acts, sae)  # [batch, d_sae]
        
        # Transpose to [d_sae, batch]
        sae_acts_t = einops.rearrange(sae_acts, "batch d_sae -> d_sae batch")
        
        # Update stats
        sparsity += (sae_acts_t > 0).sum(dim=1)
        mean_values += sae_acts_t.sum(dim=1)
        
        # Get unique images in this batch
        i_im = torch.sort(torch.unique(image_indices)).values
        n_imgs = len(i_im)
        
        # Check if we have complete images
        if actual_batch % n_patches_per_img != 0:
            # Skip incomplete batch at end
            continue
        
        # Reshape to [d_sae, n_imgs, n_patches]
        values_p = sae_acts_t.view(d_sae, n_imgs, n_patches_per_img)
        
        # Get top-k from this batch based on max patch activation
        # First get top-k patch indices
        k = min(TOP_K, n_imgs)
        _, topk_idx = torch.topk(sae_acts_t, k=TOP_K, dim=1)
        
        # Convert patch indices to image indices
        k_im = topk_idx // n_patches_per_img
        
        # Gather the full patch values for these images
        batch_values_p = gather_batched(values_p, k_im)  # [d_sae, TOP_K, n_patches]
        batch_i_im = i_im.to(DEVICE)[k_im]  # [d_sae, TOP_K]
        
        # Merge with running top-k
        all_values_p = torch.cat([top_values_p, batch_values_p], dim=1)  # [d_sae, 2*TOP_K, n_patches]
        all_i_im = torch.cat([top_i_im, batch_i_im], dim=1)  # [d_sae, 2*TOP_K]
        
        # Select new top-k based on max patch value per image
        max_per_img = all_values_p.max(dim=-1).values  # [d_sae, 2*TOP_K]
        _, selection = torch.topk(max_per_img, k=TOP_K, dim=1)
        
        top_values_p = gather_batched(all_values_p, selection)
        top_i_im = torch.gather(all_i_im, 1, selection)
    
    # Finalize stats
    total_patches = len(dataset)
    mean_values = mean_values / sparsity
    sparsity = sparsity / total_patches
    
    return (
        top_values_p.cpu(),
        top_i_im.cpu(),
        sparsity.cpu(),
        mean_values.cpu()
    )


def filter_neurons(sparsity, mean_values, d_sae):
    """Filter neurons based on frequency and value ranges."""
    log_freq = torch.log10(sparsity + 1e-10)
    log_val = torch.log10(mean_values + 1e-10)
    
    mask = (
        (log_freq > MIN_LOG_FREQ) & 
        (log_freq < MAX_LOG_FREQ) &
        (log_val > MIN_LOG_VALUE) & 
        (log_val < MAX_LOG_VALUE) &
        ~torch.isnan(log_val)
    )
    
    return torch.arange(d_sae)[mask].tolist()


def add_highlights(img, patch_values, upper=None):
    """
    Add heatmap overlay to image based on patch activations.
    Similar to original imaging.add_highlights.
    """
    n_patches = len(patch_values)
    grid_size = int(np.sqrt(n_patches))
    
    # Handle empty patches (CLS mode)
    if n_patches == 0:
        return img
    
    heatmap = patch_values.reshape(grid_size, grid_size)
    
    # Normalize
    if upper is not None and upper > 0:
        heatmap = np.clip(heatmap / upper, 0, 1)
    elif heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)
    
    # Apply colormap
    colormap = cm.get_cmap('plasma')
    heatmap_colored = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    
    # Resize to image size
    img_rgb = img.convert('RGB')
    img_array = np.array(img_rgb)
    h, w = img_array.shape[:2]
    
    heatmap_img = Image.fromarray(heatmap_colored).resize((w, h), Image.BILINEAR)
    heatmap_array = np.array(heatmap_img)
    
    # Create alpha mask based on activation strength
    alpha_map = np.array(Image.fromarray(
        (heatmap * 255).astype(np.uint8)
    ).resize((w, h), Image.BILINEAR)) / 255.0
    
    # Blend - higher activation = more heatmap visible
    alpha = alpha_map[:, :, np.newaxis] * 0.7  # max 70% opacity
    blended = (1 - alpha) * img_array + alpha * heatmap_array
    
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def make_img(img, patch_values, upper=None):
    """
    Process image: resize, crop, add heatmap.
    Matches original make_img behavior.
    """
    # Resize and crop
    resize_w, resize_h = RESIZE_SIZE
    crop_w, crop_h = CROP_SIZE
    crop_coords = (
        (resize_w - crop_w) // 2,
        (resize_h - crop_h) // 2,
        (resize_w + crop_w) // 2,
        (resize_h + crop_h) // 2,
    )
    
    img = img.resize(RESIZE_SIZE).crop(crop_coords)
    
    # Add heatmap if we have patch values
    if len(patch_values) > 0:
        img = add_highlights(img, patch_values, upper=upper)
    
    return img


def save_neuron_visuals(neuron_idx, top_i_im, top_values_p, image_dataset, sparsity, mean_values):
    """
    Save top images for a neuron with heatmaps.
    Uses pre-computed patch values from top_values_p.
    """
    neuron_dir = os.path.join(OUTPUT_DIR, "neurons", str(neuron_idx))
    os.makedirs(neuron_dir, exist_ok=True)
    
    # Get upper bound for consistent heatmap scaling across all images
    upper = None
    if top_values_p[neuron_idx].numel() > 0:
        upper = top_values_p[neuron_idx].max().item()
    
    seen = set()
    saved_count = 0
    
    # Iterate through top images and their patch values
    for i_im, values_p in zip(top_i_im[neuron_idx].tolist(), top_values_p[neuron_idx]):
        if i_im < 0 or i_im in seen:
            continue
        seen.add(i_im)
        
        # Load image
        try:
            example = image_dataset[i_im]
            img = example["image"]
            label = example["label"]
        except Exception as e:
            print(f"  Error loading image {i_im}: {e}")
            continue
        
        # Make image with heatmap (values_p already contains patch activations)
        img = make_img(img, values_p.numpy(), upper=upper)
        
        # Save
        img.save(os.path.join(neuron_dir, f"{saved_count}.png"))
        with open(os.path.join(neuron_dir, f"{saved_count}.txt"), "w") as f:
            f.write(label + "\n")
        
        saved_count += 1
        if saved_count >= TOP_K:
            break
    
    # Save metadata
    metadata = {
        "neuron": neuron_idx,
        "log10_freq": torch.log10(sparsity[neuron_idx] + 1e-10).item(),
        "log10_value": torch.log10(mean_values[neuron_idx] + 1e-10).item(),
        "num_images_saved": saved_count
    }
    with open(os.path.join(neuron_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    print("=" * 50)
    print("SAE Neuron Visualization (Patch Mode)")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Top K: {TOP_K}")
    print()
    
    # Load SAE
    print("Loading SAE...")
    sae, d_sae = load_sae(SAE_CHECKPOINT)
    print(f"  SAE has {d_sae} neurons")
    
    # Load patch activations dataset
    print("\nLoading activations dataset (patch mode)...")
    act_dataset = PatchActivationsDataset(ACTIVATIONS_PATH)
    n_patches_per_img = act_dataset.n_patches_per_img
    
    # Load image dataset
    print("\nLoading image dataset...")
    img_dataset = load_image_dataset(IMAGE_DATASET_PATH)
    print(f"  Found {len(img_dataset)} images")
    
    # Create dataloader with batch size multiple of n_patches
    effective_batch_size = (BATCH_SIZE // n_patches_per_img) * n_patches_per_img
    
    dataloader = DataLoader(
        act_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=True  # Drop incomplete batches
    )
    
    # Find top-k with patch values
    print("\n" + "=" * 50)
    print("Finding top-k images per neuron...")
    top_values_p, top_i_im, sparsity, mean_values = get_topk_patch(
        sae, d_sae, act_dataset, dataloader, n_patches_per_img
    )
    
    print(f"\nTop-k computation complete!")
    print(f"  top_values_p shape: {top_values_p.shape}")  # [d_sae, TOP_K, n_patches]
    print(f"  top_i_im shape: {top_i_im.shape}")  # [d_sae, TOP_K]
    
    # Save tensors
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(top_values_p, os.path.join(OUTPUT_DIR, "top_values.pt"))
    torch.save(top_i_im, os.path.join(OUTPUT_DIR, "top_img_i.pt"))
    torch.save(sparsity, os.path.join(OUTPUT_DIR, "sparsity.pt"))
    torch.save(mean_values, os.path.join(OUTPUT_DIR, "mean_values.pt"))
    print(f"  Saved tensors to {OUTPUT_DIR}")
    
    # Filter neurons
    valid_neurons = filter_neurons(sparsity, mean_values, d_sae)
    print(f"\n{len(valid_neurons)} neurons pass the frequency/value filter")
    
    # Randomly select subset
    random.seed(SEED)
    random.shuffle(valid_neurons)
    neurons_to_vis = valid_neurons[:NUM_NEURONS_TO_VIS]
    print(f"Will visualize {len(neurons_to_vis)} randomly selected neurons")
    
    # Save visualizations (fast - uses pre-computed patch values)
    print("\nSaving neuron visualizations...")
    for neuron_idx in tqdm(neurons_to_vis, desc="Saving neurons"):
        save_neuron_visuals(
            neuron_idx,
            top_i_im,
            top_values_p,
            img_dataset,
            sparsity,
            mean_values
        )
    
    print("\nDone!")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()