"""
Evaluate token-level background bias in ViT representations by ablating background patch tokens and analyzing 
the resulting changes in Sparse Autoencoder latent activations.
"""

import torch
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import numpy as np
from sae_model import load_sae
from vit_activations import DinoV2Wrapper

# CONFIGURATION, CHANGE PATHS AS NEEDED
SAE_CHECKPOINT_PATH = "/Users/chris/Downloads/deepL_run/deepL_run/local_checkpoints/checkpoints/sae.pt"
DATASET_ROOT = "/Users/chris/Datasets/bioscan-5m/bioscan5m/images/cropped_256"
HISTOGRAM_SAVE_PATH = "ratio_histogram.png"
DELTA_BG_SAVE_PATH = "delta_background_sensitive.png"
DELTA_STABLE_SAVE_PATH = "delta_stable.png"

N_SAMPLE = 100
FG_RATIO = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Image preprocessing
base_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


def sample_indices(n_total, n_sample, seed=42):
    """Randomly sample indices from a dataset with a fixed seed for reproducibility."""
    random.seed(seed)
    return random.sample(range(n_total), n_sample)


def denormalize(img):
    """Denormalize the image for visualization."""
    img = img.clone()
    img[0] = img[0] * 0.229 + 0.485
    img[1] = img[1] * 0.224 + 0.456
    img[2] = img[2] * 0.225 + 0.406
    return img.clamp(0, 1)


# Patch-level background ablation
def center_patch_mask(grid_size=16, fg_ratio=0.5, device="cpu"):
    """
    Create a binary mask over a ViT patch grid where:
    - central patches = foreground (1)
    - outer patches = background (0)
    """
    fg_size = int(grid_size * fg_ratio)
    start = (grid_size - fg_size) // 2
    end = start + fg_size

    mask = torch.zeros((grid_size, grid_size), device=device)
    mask[start:end, start:end] = 1.0
    return mask


# Latent extraction
@torch.no_grad()
def get_latents_cls(sae, vit, images):
    """Extract SAE latents from the CLS token of the ViT."""
    acts = vit(images)[:, 0, :]  # CLS token
    _, f, _ = sae(acts)
    return f


# Latent extraction with token-level background ablation
@torch.no_grad()
def get_latents_cls_token_ablation(sae, vit, images, fg_ratio=0.5):
    """
    Perform background ablation in token space by:
    - masking background patch tokens
    - averaging foreground patch embeddings
    - passing the result through the SAE
    """
    B = images.size(0)
    outputs = vit(images)

    patches = outputs[:, 1:, :]
    D = patches.size(-1)
    patches = patches.view(B, 16, 16, D)

    mask = center_patch_mask(fg_ratio=fg_ratio, device=patches.device)

    patches_fg = patches * mask[None, :, :, None]
    patches_fg = patches_fg.view(B, 256, D)

    cls_fg = patches_fg.mean(dim=1)

    _, f, _ = sae(cls_fg)
    return f


# Latent heatmaps (original vs background-ablated)
@torch.no_grad()
def get_latent_heatmaps_delta(sae, vit, image, latent_idx, fg_ratio=0.5):
    """
    Compute patch-level latent activation heatmaps for original image and background-ablated tokens.
    """
    image = image.unsqueeze(0).to(DEVICE)

    acts = vit(image)[:, 1:, :]  # patch tokens
    D = acts.size(-1)
    acts = acts.view(1, 16, 16, D)

    mask = center_patch_mask(fg_ratio=fg_ratio, device=acts.device)

    # Original
    acts_flat = acts.view(-1, D)
    _, f_orig, _ = sae(acts_flat)
    heat_orig = f_orig[:, latent_idx].view(16, 16)

    # Background ablated
    acts_fg = acts * mask[None, :, :, None]
    acts_fg_flat = acts_fg.view(-1, D)

    _, f_fg, _ = sae(acts_fg_flat)
    heat_fg = f_fg[:, latent_idx].view(16, 16)

    # Subtract mean activation on masked patches
    bg_mask = (mask == 0)
    bg_mean = heat_fg[bg_mask].mean()
    heat_fg = heat_fg - bg_mean
    heat_fg = torch.clamp(heat_fg, min=0.0)

    return (heat_orig.cpu().numpy(), heat_fg.cpu().numpy())


# Visualization
def visualize_delta_heatmap(sae, vit, img, latent_idx, fg_ratio=0.5, save_path="delta_heatmap.png"):
    img_orig_dn = denormalize(img).permute(1, 2, 0).numpy()
    img_ab_dn = visualize_token_level_ablation_image(img, fg_ratio=fg_ratio)

    heat_orig, heat_fg = get_latent_heatmaps_delta(sae, vit, img, latent_idx, fg_ratio)

    # Shared scales
    vmin = 0.0
    vmax = max(heat_orig.max(), heat_fg.max())

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))

    # Images
    axes[0].imshow(img_orig_dn)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(img_ab_dn)
    axes[1].set_title("Token-level bg ablation")
    axes[1].axis("off")

    # Heatmaps
    im1 = axes[2].imshow(heat_orig, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[2].set_title("Latent (orig)")
    axes[2].axis("off")

    im2 = axes[3].imshow(heat_fg, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[3].set_title("Latent (bg ablated)")
    axes[3].axis("off")

    fig.colorbar(im2, ax=axes[2:4], location="right", shrink=0.9, pad=0.02)

    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] {save_path}")


def visualize_token_level_ablation_image(img, fg_ratio=0.5, patch_size=14):
    """
    Create a visualization showing which ViT patches are masked. This is just for visualization, 
    this is NOT the image used as model input.
    """
    img_dn = denormalize(img)
    C, H, W = img_dn.shape

    grid_size = H // patch_size  # 16
    fg_size = int(grid_size * fg_ratio)
    start = (grid_size - fg_size) // 2
    end = start + fg_size

    img_ab = img_dn.clone()

    for i in range(grid_size):
        for j in range(grid_size):
            if not (start <= i < end and start <= j < end):
                y0 = i * patch_size
                y1 = y0 + patch_size
                x0 = j * patch_size
                x1 = x0 + patch_size

                # Gray out background patches
                img_ab[:, y0:y1, x0:x1] = 0.5

    return img_ab.permute(1, 2, 0).numpy()


# Histogram
def plot_ratio_histogram(ratio, mean_orig, save_path="ratio_histogram.png"):
    active_mask = mean_orig > 1e-3
    ratio_active = ratio[active_mask].cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(ratio_active, bins=100, color="darkgreen")
    plt.xlim(0, 5)
    plt.axvline(0.2, color="red", linestyle="--", label="background-sensitive")
    plt.axvline(0.8, color="limegreen", linestyle="--", label="stable")
    plt.xlabel("mean activation (bg ablated) / mean activation (original)", fontsize=13)
    plt.ylabel("Number of latents", fontsize=13)
    plt.title("Token-level background bias (active latents)", fontsize=15, fontweight="bold")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] {save_path}")


# MAIN
def main():
    sae, _ = load_sae(SAE_CHECKPOINT_PATH, device=DEVICE)
    sae.eval().to(DEVICE)

    vit = DinoV2Wrapper("dinov2_vitb14").to(DEVICE)
    vit.eval()

    dataset = datasets.ImageFolder(DATASET_ROOT, transform=base_transform)

    indices = sample_indices(len(dataset), n_sample=N_SAMPLE)
    imgs = torch.stack([dataset[i][0] for i in indices]).to(DEVICE)

    f_orig = get_latents_cls(sae, vit, imgs)
    f_bg = get_latents_cls_token_ablation(sae, vit, imgs, fg_ratio=FG_RATIO)

    mean_orig = f_orig.mean(dim=0)
    mean_bg = f_bg.mean(dim=0)

    ratio = (mean_bg + 1e-6) / (mean_orig + 1e-6)

    plot_ratio_histogram(ratio, mean_orig, HISTOGRAM_SAVE_PATH)

    bg_latents = torch.where(ratio < 0.2)[0]
    stable_latents = torch.where(ratio > 0.8)[0]

    print(f"Images tested: {len(indices)}")
    print(f"Background-sensitive latents: {len(bg_latents)}")
    print(f"Stable latents: {len(stable_latents)}")

    if len(bg_latents) == 0 or len(stable_latents) == 0:
        print("Not enough latents to visualize.")
        return

    visualize_delta_heatmap(sae, vit, img=imgs[0].cpu(), latent_idx=int(bg_latents[0]), fg_ratio=FG_RATIO, save_path=DELTA_BG_SAVE_PATH)
    visualize_delta_heatmap(sae, vit, img=imgs[0].cpu(), latent_idx=int(stable_latents[0]), fg_ratio=FG_RATIO, save_path=DELTA_STABLE_SAVE_PATH)


if __name__ == "__main__":
    main()