"""
This script serves to visualize where the latents (hidden neurons in SAE) activate. To do this we get the top-k 
images for each neuron using patch activations. Using this data for each neuron we get the images that they 
activate the most on, showing their activation heatmap imposed onto the image. This script also saves these 
images to an output directory. 
"""

import os
import json
import random
import torch
import einops
import io
from source.sae_model import SparseAutoencoder
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.cm as cm


# CONFIGURATION
SAE_CHECKPOINT = "./checkpoints/sae.pt"
ACTIVATIONS_PATH = "./activations"
OUTPUT_DIR = "./outputs"
IMAGE_DATASET_PATH = "/Users/danielpanariti/Documents/universite/m2_2025-2026_MIND/s3/deep_l/projet/BIOSCAN-5M/bioscan5m/images/cropped_256"

TOP_K = 32
BATCH_SIZE = 16384
SAE_BATCH_SIZE = 4096
NUM_WORKERS = 0
DEVICE = "cpu"

VIT_LAYER_IDX = 0  # basically which layer we get from saved ViT layers, we only saved 1 so we set to 0

# metrics we use to filter neurons
# sparsity: must not fire too rarely or too often! 
MIN_LOG_FREQ = -8.0
MAX_LOG_FREQ = -1.0
# avg activation: must not too low or too high of a signal! 
MIN_LOG_VALUE = -3.0
MAX_LOG_VALUE = 3.0

NUM_NEURONS_TO_VIS = 100
SEED = 42

# reshaping image params
RESIZE_SIZE = (512, 512)
CROP_SIZE = (448, 448)



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
    
    layer = sae_config.get("layer", VIT_LAYER_IDX)
    print(f"  d_vit={d_vit}, exp_factor={exp_factor}, d_sae={d_sae}, layer={layer}")
    
    sae = SparseAutoencoder(d_vit, expansion_factor=exp_factor)
    state_dict = torch.load(buffer, weights_only=True, map_location='cpu')
    sae.load_state_dict(state_dict)
    
    return sae.eval(), d_sae, layer


class PatchActivationsDataset(Dataset):
    """
    Loads pre-computed ViT activations from .bin files.
    """
    
    def __init__(self, shard_root, layer_idx=0): # by default only one layer saved
        self.shard_root = Path(shard_root)
        self.layer_idx = layer_idx  
        
        with open(self.shard_root / "metadata.json") as f:
            self.metadata = json.load(f)
        
        self.d_vit = self.metadata["d_vit"]
        self.n_layers = len(self.metadata["layers"])
        self.n_patches_per_img = self.metadata["n_patches_per_img"]
        self.n_patches_with_cls = self.n_patches_per_img + 1
        self.n_imgs = self.metadata["n_imgs"]

        # weird issue...
        if self.layer_idx >= self.n_layers:
            raise ValueError(f"layer_idx {self.layer_idx} >= n_layers {self.n_layers}")
        print(f"  Using layer {self.layer_idx} (available: {self.metadata['layers']})")
        
        self.shard_files = sorted(self.shard_root.glob("acts*.bin"))
        self.bytes_per_img = self.n_layers * self.n_patches_with_cls * self.d_vit * 4
        
        self.shard_boundaries = []
        current_idx = 0
        for shard_path in self.shard_files:
            file_size = shard_path.stat().st_size
            n_imgs_in_shard = file_size // self.bytes_per_img
            self.shard_boundaries.append((current_idx, current_idx + n_imgs_in_shard, shard_path))
            current_idx += n_imgs_in_shard
        
        self.total_imgs = current_idx
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
        img_idx = idx // self.n_patches_per_img
        patch_idx = idx % self.n_patches_per_img
        
        shard_idx, local_idx = self._find_shard_for_img(img_idx)
        self._ensure_shard_loaded(shard_idx)
        
        patch_act = self._cached_shard[local_idx, self.layer_idx, patch_idx + 1, :]
        
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
    """Get SAE activations in batches."""
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
    """
    top_values_p = torch.full((d_sae, TOP_K, n_patches_per_img), -1.0, device=DEVICE)
    top_i_im = torch.zeros((d_sae, TOP_K), dtype=torch.long, device=DEVICE)
    
    sparsity = torch.zeros(d_sae, device=DEVICE)
    mean_values = torch.zeros(d_sae, device=DEVICE)
    
    effective_batch_size = (BATCH_SIZE // n_patches_per_img) * n_patches_per_img
    n_imgs_per_batch = effective_batch_size // n_patches_per_img
    
    print(f"  Effective batch size: {effective_batch_size} patches ({n_imgs_per_batch} images)")
    
    for batch in tqdm(dataloader, desc="Finding top-k patches"):
        vit_acts = batch["act"]
        image_indices = batch["image_i"]
        
        actual_batch = vit_acts.shape[0]
        
        # Skip incomplete batches
        if actual_batch % n_patches_per_img != 0:
            continue
        
        sae_acts = get_sae_acts(vit_acts, sae)
        sae_acts_t = einops.rearrange(sae_acts, "batch d_sae -> d_sae batch")
        
        # stats update...
        sparsity += (sae_acts_t > 0).sum(dim=1)
        mean_values += sae_acts_t.sum(dim=1)
        
        # get unique images
        i_im = torch.sort(torch.unique(image_indices)).values
        n_imgs = len(i_im)
        
        # In case of mismatch
        assert n_imgs == actual_batch // n_patches_per_img, \
            f"Batch has interleaved images! Expected {actual_batch // n_patches_per_img} images, got {n_imgs}"
        
        # Reshape to [d_sae, n_imgs, n_patches]
        values_p = sae_acts_t.view(d_sae, n_imgs, n_patches_per_img)
        
        # Get top-k patches
        _, topk_idx = torch.topk(sae_acts_t, k=TOP_K, dim=1)
        
        # Convert to image indices within this batch
        k_im = topk_idx // n_patches_per_img
        
        # clamp k_im to valid range
        k_im = k_im.clamp(0, n_imgs - 1)
        
        # Gather patch values for top images
        batch_values_p = gather_batched(values_p, k_im)
        batch_i_im = i_im.to(DEVICE)[k_im]
        
        # Merge with running top-k
        all_values_p = torch.cat([top_values_p, batch_values_p], dim=1)
        all_i_im = torch.cat([top_i_im, batch_i_im], dim=1)
        
        # Select new top-k based on max patch value
        max_per_img = all_values_p.max(dim=-1).values
        _, selection = torch.topk(max_per_img, k=TOP_K, dim=1)
        
        top_values_p = gather_batched(all_values_p, selection)
        top_i_im = torch.gather(all_i_im, 1, selection)
    
    # Final stats
    total_patches = len(dataset)
    mean_values = mean_values / (sparsity + 1e-10) 
    sparsity = sparsity / total_patches
    
    return (
        top_values_p.cpu(),
        top_i_im.cpu(),
        sparsity.cpu(),
        mean_values.cpu()
    )


def add_highlights(img, patch_values, upper=None):
    """
    Add heatmap overlay to image based on patch activations.
    """
    n_patches = len(patch_values)
    if n_patches == 0:
        return img
    
    grid_size = int(np.sqrt(n_patches))
    heatmap = patch_values.reshape(grid_size, grid_size)
    
    # Normalize - handle the case where all values are the same!!
    if upper is not None and upper > 0:
        heatmap_norm = np.clip(heatmap / upper, 0, 1)
    elif heatmap.max() > heatmap.min():
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap_norm = np.zeros_like(heatmap)
    
    # Apply colormap
    colormap = cm.get_cmap('plasma')
    heatmap_colored = (colormap(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
    
    # Resize to image size
    img_rgb = img.convert('RGB')
    img_array = np.array(img_rgb)
    h, w = img_array.shape[:2]
    
    heatmap_img = Image.fromarray(heatmap_colored).resize((w, h), Image.NEAREST) # nearest for visuals similar to their paper
    heatmap_array = np.array(heatmap_img)
    
    alpha_2d = np.array(Image.fromarray(
        (heatmap_norm * 255).astype(np.uint8)
    ).resize((w, h), Image.NEAREST)) / 255.0
    
    # blend with max 70% opacity where activation is highest
    alpha = alpha_2d[:, :, np.newaxis] * 0.7
    blended = (1 - alpha) * img_array + alpha * heatmap_array
    
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def make_img(img, patch_values, upper=None):
    """Process image: resize, crop, add heatmap."""
    resize_w, resize_h = RESIZE_SIZE
    crop_w, crop_h = CROP_SIZE
    crop_coords = (
        (resize_w - crop_w) // 2,
        (resize_h - crop_h) // 2,
        (resize_w + crop_w) // 2,
        (resize_h + crop_h) // 2,
    )
    
    img = img.resize(RESIZE_SIZE).crop(crop_coords)
    
    if len(patch_values) > 0:
        img = add_highlights(img, patch_values, upper=upper)
    
    return img


def filter_neurons(sparsity, mean_values, d_sae):
    """Filter neurons based on frequency and value ranges."""
    log_freq = torch.log10(sparsity + 1e-10)
    log_val = torch.log10(mean_values + 1e-10)
    
    mask = (
        (log_freq > MIN_LOG_FREQ) & 
        (log_freq < MAX_LOG_FREQ) &
        (log_val > MIN_LOG_VALUE) & 
        (log_val < MAX_LOG_VALUE) &
        ~torch.isnan(log_val) &
        ~torch.isinf(log_val)
    )
    
    return torch.arange(d_sae)[mask].tolist()


def save_neuron_visuals(neuron_idx, top_i_im, top_values_p, image_dataset, sparsity, mean_values):
    """Save top images for a neuron with heatmaps."""
    neuron_dir = os.path.join(OUTPUT_DIR, "neurons", str(neuron_idx))
    os.makedirs(neuron_dir, exist_ok=True)
    
    # Get upper bound for consistent scaling
    upper = None
    if top_values_p[neuron_idx].numel() > 0:
        upper = top_values_p[neuron_idx].max().item()
    
    seen = set()
    saved_count = 0
    
    for i_im, values_p in zip(top_i_im[neuron_idx].tolist(), top_values_p[neuron_idx]):
        if i_im < 0 or i_im in seen:
            continue
        seen.add(i_im)
        
        try:
            example = image_dataset[i_im]
            img = example["image"]
            label = example["label"]
        except Exception as e:
            print(f"  Error loading image {i_im}: {e}")
            continue
        
        img = make_img(img, values_p.numpy(), upper=upper)
        
        img.save(os.path.join(neuron_dir, f"{saved_count}.png"))
        with open(os.path.join(neuron_dir, f"{saved_count}.txt"), "w") as f:
            f.write(label + "\n")
        
        saved_count += 1
        if saved_count >= TOP_K:
            break
    
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
    print("SAE Neuron Visualization (Fixed)")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Top K: {TOP_K}")
    print()
    
    # Load SAE - now returns layer info
    print("Loading SAE...")
    sae, d_sae, layer_idx = load_sae(SAE_CHECKPOINT)
    print(f"  SAE has {d_sae} neurons, trained on layer {layer_idx}")
    
    # pass layer_idx to dataset
    print("\nLoading activations dataset...")
    act_dataset = PatchActivationsDataset(ACTIVATIONS_PATH, layer_idx=layer_idx)
    n_patches_per_img = act_dataset.n_patches_per_img
    
    print("\nLoading image dataset...")
    img_dataset = load_image_dataset(IMAGE_DATASET_PATH)
    print(f"  Found {len(img_dataset)} images")
    
    effective_batch_size = (BATCH_SIZE // n_patches_per_img) * n_patches_per_img
    
    dataloader = DataLoader(
        act_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=True
    )
    
    # Check for cached results
    cache_exists = all(
        os.path.exists(os.path.join(OUTPUT_DIR, f))
        for f in ["top_values.pt", "top_img_i.pt", "sparsity.pt", "mean_values.pt"]
    )
    
    if not cache_exists:
        print("\n" + "=" * 50)
        print("Finding top-k images per neuron...")
        top_values_p, top_i_im, sparsity, mean_values = get_topk_patch(
            sae, d_sae, act_dataset, dataloader, n_patches_per_img
        )
        
        print(f"\nTop-k computation complete!")
        print(f"  top_values_p shape: {top_values_p.shape}")
        print(f"  top_i_im shape: {top_i_im.shape}")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(top_values_p, os.path.join(OUTPUT_DIR, "top_values.pt"))
        torch.save(top_i_im, os.path.join(OUTPUT_DIR, "top_img_i.pt"))
        torch.save(sparsity, os.path.join(OUTPUT_DIR, "sparsity.pt"))
        torch.save(mean_values, os.path.join(OUTPUT_DIR, "mean_values.pt"))
        print(f"  Saved tensors to {OUTPUT_DIR}")
    else:
        print("\nLoading cached results...")
        top_values_p = torch.load(os.path.join(OUTPUT_DIR, "top_values.pt"),map_location=torch.device(DEVICE))
        top_i_im = torch.load(os.path.join(OUTPUT_DIR, "top_img_i.pt"), map_location=torch.device(DEVICE))
        sparsity = torch.load(os.path.join(OUTPUT_DIR, "sparsity.pt"), map_location=torch.device(DEVICE))
        mean_values = torch.load(os.path.join(OUTPUT_DIR, "mean_values.pt"), map_location=torch.device(DEVICE))
    
    valid_neurons = filter_neurons(sparsity, mean_values, d_sae)
    print(f"\n{len(valid_neurons)} neurons pass the frequency/value filter")
    
    random.seed(SEED)
    random.shuffle(valid_neurons)
    neurons_to_vis = valid_neurons[:NUM_NEURONS_TO_VIS]
    print(f"Will visualize {len(neurons_to_vis)} randomly selected neurons")
    
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