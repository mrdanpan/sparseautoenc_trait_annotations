"""
This script loads the patches (DINOv2 activations) from ./activations for training, defines an SAE model
as well as a training loop. 
"""
import os
import io
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging 

# Logging details
LOG_DIR = "../logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler(os.path.join(LOG_DIR, "training.log"))  # Save to file
    ]
)
logger = logging.getLogger("sae_training")

# Config details
ACTIVATIONS_DIR = "../activations"
CHECKPOINT_DIR = "../checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAE hyperparameters (matching Table D.5)
EXPANSION_FACTOR = 32     # -> 768 * 32 = 24,576 hidden width
SPARSITY_COEFF = 4e-4  # Paper uses {2e-4, 4e-4, 8e-4}, pick middle
SPARSITY_WARMUP_STEPS = 500  
BATCH_SIZE = 4096  # Paper uses 16,384
LR = 1e-3     # Paper uses {5e-4, 1e-3}
LR_WARMUP_STEPS = 500    
ACTIVATION_THRESHOLD = 0.9   # For later use in feature selection
N_EPOCHS = 1
# Training length
N_PATCHES_TARGET = 43_507_609  # Original default: 100 million patches


# Loading Patches into dataset
class ActivationsDataset(Dataset):
    """
    Loads the ViT activations from .bin files.
    Each .bin file has shape [n_imgs, n_layers, n_patches+1, d_vit].
    We flatten to individual patch activations of shape [d_vit].
    """
    
    def __init__(self, activations_dir, metadata, layer_idx=0, use_patches=True):
        self.activations_dir = activations_dir
        self.layer_idx = layer_idx
        self.use_patches = use_patches
        
        # Load metadata
        self.metadata = metadata
        
        self.d_vit = self.metadata["d_vit"]
        self.n_patches = self.metadata["n_patches_per_img"]
        self.n_imgs = self.metadata["n_imgs"]
        self.n_layers = len(self.metadata["layers"])
        
        # Find all shard files
        self.shard_files = sorted([
            f for f in os.listdir(activations_dir) 
            if f.startswith("acts") and f.endswith(".bin")
        ])
        
        # Calculate images per shard from metadata
        n_patches_with_cls = self.n_patches + 1
        self.imgs_per_shard = (
            self.metadata["n_patches_per_shard"] 
            // self.n_layers 
            // n_patches_with_cls
        )
        
        # Total number of examples depends on just using patches or just CLS
        if self.use_patches:
            # Each image gives us n_patches examples (excluding CLS)
            self.n_examples = self.n_imgs * self.n_patches
        else:
            # Each image gives us 1 CLS token
            self.n_examples = self.n_imgs
        
        # Cache for currently loaded shard
        self._cached_shard_idx = -1
        self._cached_data = None
    
    def _load_shard(self, shard_idx):
        """Load a shard file into memory."""
        if shard_idx == self._cached_shard_idx:
            return self._cached_data
        
        shard_path = os.path.join(self.activations_dir, self.shard_files[shard_idx])
        
        # Determine shape - last shard might be smaller
        if shard_idx == len(self.shard_files) - 1:
            # Last shard: calculate remaining images
            imgs_in_previous = shard_idx * self.imgs_per_shard
            imgs_in_this_shard = self.n_imgs - imgs_in_previous
        else:
            imgs_in_this_shard = self.imgs_per_shard
        
        shape = (imgs_in_this_shard, self.n_layers, self.n_patches + 1, self.d_vit)
        
        # Memory-map the file (efficient for large files)
        data = np.memmap(shard_path, dtype=np.float32, mode='r', shape=shape)
        
        self._cached_shard_idx = shard_idx
        self._cached_data = data
        return data
    
    def __len__(self):
        return self.n_examples
    
    def __getitem__(self, idx):
        if self.use_patches:
            # idx = image_idx * n_patches + patch_idx
            img_idx = idx // self.n_patches
            patch_idx = idx % self.n_patches
            
            # Which shard and position within shard?
            shard_idx = img_idx // self.imgs_per_shard
            pos_in_shard = img_idx % self.imgs_per_shard
            
            data = self._load_shard(shard_idx)
            
            # Get activation: [layer_idx, patch_idx+1, :] (+1 because CLS is at index 0)
            act = data[pos_in_shard, self.layer_idx, patch_idx + 1, :]
        else:
            # Just use CLS token
            img_idx = idx
            shard_idx = img_idx // self.imgs_per_shard
            pos_in_shard = img_idx % self.imgs_per_shard
            
            data = self._load_shard(shard_idx)
            act = data[pos_in_shard, self.layer_idx, 0, :]  # CLS is at index 0
        
        return torch.from_numpy(act.copy())


# ============== SPARSE AUTOENCODER ==============

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder matching the original saev implementation.
    """
    
    def __init__(self, d_input, expansion_factor=32, sparsity_coeff=4e-4):
        super().__init__()
        
        d_hidden = d_input * expansion_factor
        
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.sparsity_coeff = sparsity_coeff  # mutable for warmup
        
        # Encoder weights and bias
        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(d_input, d_hidden))
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        
        # Decoder weights and bias
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(d_hidden, d_input))
        )
        self.b_dec = nn.Parameter(torch.zeros(d_input))
    
    def encode(self, x):
        """Encode input to sparse codes."""
        # Subtract decoder bias before encoding (as per Anthropic)
        h_pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        f_x = torch.relu(h_pre)
        return f_x
    
    def decode(self, f_x):
        """Decode sparse codes to reconstruction."""
        x_hat = f_x @ self.W_dec + self.b_dec
        return x_hat
    
    def forward(self, x):
        """Full forward pass, returns reconstruction, codes, and loss dict."""
        f_x = self.encode(x)
        x_hat = self.decode(f_x)
        
        # MSE loss
        mse_loss = ((x_hat - x) ** 2).mean()
        
        # L0 and L1
        l0 = (f_x > 0).float().sum(dim=1).mean()
        l1 = f_x.sum(dim=1).mean()  # sum over features, mean over batch
        
        # Sparsity loss (original style)
        sparsity_loss = self.sparsity_coeff * l1
        
        total_loss = mse_loss + sparsity_loss
        
        return x_hat, f_x, {
            "loss": total_loss,
            "mse": mse_loss,
            "sparsity": sparsity_loss,
            "l0": l0,
            "l1": l1
        }
    
    @torch.no_grad()
    def normalize_w_dec(self):
        """Set W_dec rows to unit norm."""
        norms = torch.norm(self.W_dec, dim=1, keepdim=True)
        self.W_dec.data = self.W_dec.data / (norms + 1e-8)
    
    @torch.no_grad()
    def remove_parallel_grads(self):
        """Remove gradient components parallel to W_dec rows."""
        if self.W_dec.grad is None:
            return
        # W_dec shape: [d_hidden, d_input]
        # For each row, remove component parallel to the row itself
        W_dec_normed = self.W_dec / (torch.norm(self.W_dec, dim=1, keepdim=True) + 1e-8)
        parallel = (self.W_dec.grad * W_dec_normed).sum(dim=1, keepdim=True) * W_dec_normed
        self.W_dec.grad = self.W_dec.grad - parallel


# ============== TRAINING ==============
def compute_loss(x, x_recon, codes, sparsity_coeff):
    """
    Compute SAE loss:
    - MSE reconstruction loss
    - L1 sparsity penalty on codes
    """
    recon_loss = ((x - x_recon) ** 2).mean()
    sparsity_loss = codes.abs().mean()
    total_loss = recon_loss + sparsity_coeff * sparsity_loss
    return total_loss, recon_loss, sparsity_loss

def get_lr(step, warmup_steps, base_lr):
    """Linear warmup then constant."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def get_sparsity_coeff(step, warmup_steps, base_coeff):
    """Linear warmup then constant."""
    if step < warmup_steps:
        return base_coeff * (step + 1) / warmup_steps
    return base_coeff



# def train():
#     logger.info(f"Using device: {DEVICE}")
    
#     # Metadata
#     ex_shard_path = "./activations/acts000000.bin"
#     file_size = os.path.getsize(ex_shard_path)
#     bytes_per_img = 1 * 257 * 768 * 4
#     n_imgs = file_size // bytes_per_img

#     logger.info(f"File size: {file_size / (1024**2):.1f} MB")
#     logger.info(f"Number of images: {n_imgs}")

    
#     metadata = {
#         "vit_family": "dinov2",
#         "vit_ckpt": "dinov2_vitb14",
#         "layers": [-2],
#         "n_patches_per_img": 256,
#         "cls_token": True,
#         "d_vit": 768,
#         "seed": 42,
#         "n_imgs": n_imgs,
#         "n_patches_per_shard": n_imgs * 1 * 257,
#         "data": "ImageFolderDataset(...)"
#     }
#     # Load dataset
#     logger.info("Loading activations dataset...")
#     dataset = ActivationsDataset(
#         ACTIVATIONS_DIR, 
#         layer_idx=0,
#         use_patches=True,
#         metadata=metadata
#     )
#     logger.info(f"Dataset size: {len(dataset)} examples")
#     logger.info(f"Activation dimension: {dataset.d_vit}")
    
    
    
#     dataloader = DataLoader(
#         dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=0,
#         drop_last=True
#     )
    
#     # Create SAE
#     sae = SparseAutoencoder(
#         d_input=dataset.d_vit,
#         expansion_factor=EXPANSION_FACTOR,
#         sparsity_coeff=SPARSITY_COEFF
#     ).to(DEVICE)
    
#     logger.info(f"SAE hidden dim: {sae.d_hidden}")

#     # Optimizer
#     optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    
#     # Training loop
#     os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#     step = 0
#     total_steps = len(dataloader) * N_EPOCHS
    
#     for epoch in range(N_EPOCHS):
#         logger.info(f"Epoch {epoch + 1}/{N_EPOCHS}")
        
#         pbar = tqdm(dataloader, desc="Training")
#         for batch in pbar:
#             batch = batch.to(DEVICE)
            
#             # Warmup: update LR and sparsity coeff
#             current_lr = get_lr(step, LR_WARMUP_STEPS, LR)
#             current_sparsity = get_sparsity_coeff(step, SPARSITY_WARMUP_STEPS, SPARSITY_COEFF)
            
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = current_lr
#             sae.sparsity_coeff = current_sparsity
            
#             # Normalize decoder weights before forward pass
#             sae.normalize_w_dec()
            
#             # Forward pass
#             x_recon, codes, losses = sae(batch)
            
#             # Backward pass
#             optimizer.zero_grad()
#             losses["loss"].backward()
            
#             # Remove parallel gradients before step
#             sae.remove_parallel_grads()
            
#             optimizer.step()
            
#             # Logging
#             step += 1
#             if step % 25 == 0:
#                 logger.info(
#                     "step: %d/%d, loss: %.5f, mse: %.5f, sparsity: %.5f, L0: %.1f, L1: %.1f, lr: %.2e, α: %.2e",
#                     step,
#                     total_steps,
#                     losses["loss"].item(),
#                     losses["mse"].item(),
#                     losses["sparsity"].item(),
#                     losses["l0"].item(),
#                     losses["l1"].item(),
#                     current_lr,
#                     current_sparsity
#                 )
    
#     # Save checkpoint
#     ckpt_path = os.path.join(CHECKPOINT_DIR, "sae.pt")
#     torch.save({
#         "model_state_dict": sae.state_dict(),
#         "d_input": sae.d_input,
#         "d_hidden": sae.d_hidden,
#         "expansion_factor": EXPANSION_FACTOR,
#     }, ckpt_path)
#     logger.info(f"Saved checkpoint to {ckpt_path}")
    
#     return sae

def train():
    logger.info(f"Using device: {DEVICE}")
    
    # Metadata
    ex_shard_path = "../activations/acts000000.bin"
    file_size = os.path.getsize(ex_shard_path)
    bytes_per_img = 1 * 257 * 768 * 4
    n_imgs = file_size // bytes_per_img

    logger.info(f"File size: {file_size / (1024**2):.1f} MB")
    logger.info(f"Number of images: {n_imgs}")

    metadata = {
        "vit_family": "dinov2",
        "vit_ckpt": "dinov2_vitb14",
        "layers": [-2],
        "n_patches_per_img": 256,
        "cls_token": True,
        "d_vit": 768,
        "seed": 42,
        "n_imgs": n_imgs,
        "n_patches_per_shard": n_imgs * 1 * 257,
        "data": "ImageFolderDataset(...)"
    }
    
    # Load dataset
    logger.info("Loading activations dataset...")
    dataset = ActivationsDataset(
        ACTIVATIONS_DIR, 
        layer_idx=0,
        use_patches=True,
        metadata=metadata
    )
    logger.info(f"Dataset size: {len(dataset)} examples")
    logger.info(f"Activation dimension: {dataset.d_vit}")
    
    # Calculate number of epochs to match N_PATCHES_TARGET
    patches_per_epoch = len(dataset)
    n_epochs = max(1, N_PATCHES_TARGET // patches_per_epoch)
    total_steps = (patches_per_epoch // BATCH_SIZE) * n_epochs
    
    logger.info(f"Patches per epoch: {patches_per_epoch:,}")
    logger.info(f"Target patches: {N_PATCHES_TARGET:,}")
    logger.info(f"Calculated epochs: {n_epochs}")
    logger.info(f"Total steps: {total_steps:,}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    # Create SAE
    sae = SparseAutoencoder(
        d_input=dataset.d_vit,
        expansion_factor=EXPANSION_FACTOR,
        sparsity_coeff=SPARSITY_COEFF
    ).to(DEVICE)
    
    logger.info(f"SAE hidden dim: {sae.d_hidden}")

    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    
    # Training loop
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    step = 0
    n_patches_seen = 0
    
    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch + 1}/{n_epochs}")
        
        for batch in dataloader:
            batch = batch.to(DEVICE)
            
            # Warmup: update LR and sparsity coeff
            current_lr = get_lr(step, LR_WARMUP_STEPS, LR)
            current_sparsity = get_sparsity_coeff(step, SPARSITY_WARMUP_STEPS, SPARSITY_COEFF)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            sae.sparsity_coeff = current_sparsity
            
            # Normalize decoder weights before forward pass
            sae.normalize_w_dec()
            
            # Forward pass
            x_recon, codes, losses = sae(batch)
            
            # Backward pass
            optimizer.zero_grad()
            losses["loss"].backward()
            
            # Remove parallel gradients before step
            sae.remove_parallel_grads()
            
            optimizer.step()
            
            # Track progress
            step += 1
            n_patches_seen += len(batch)
            
            # Logging
            if step % 25 == 0:
                logger.info(
                    "step: %d/%d, patches: %d/%d, loss: %.5f, mse: %.5f, sparsity: %.5f, L0: %.1f, L1: %.1f, lr: %.2e, α: %.2e",
                    step,
                    total_steps,
                    n_patches_seen,
                    N_PATCHES_TARGET,
                    losses["loss"].item(),
                    losses["mse"].item(),
                    losses["sparsity"].item(),
                    losses["l0"].item(),
                    losses["l1"].item(),
                    current_lr,
                    current_sparsity
                )
    
    logger.info(f"Training complete. Total patches seen: {n_patches_seen:,}")
    
    # Save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "sae.pt")
    save_sae(ckpt_path, sae, cfg_dict)
    logger.info(f"Saved checkpoint to {ckpt_path}")
    
    return sae

def save_sae(fpath, sae, cfg_dict):
    """Save SAE in original format."""
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as fd:
        # First line: JSON config
        cfg_str = json.dumps(cfg_dict)
        fd.write((cfg_str + "\n").encode("utf-8"))
        # Rest: state_dict
        torch.save(sae.state_dict(), fd)


def load_sae(fpath, device="cpu"):
    """Load SAE from original format."""
    with open(fpath, "rb") as fd:
        cfg = json.loads(fd.readline().decode())
        buffer = io.BytesIO(fd.read())
    
    sae = SparseAutoencoder(
        d_input=cfg["d_vit"],
        expansion_factor=cfg["exp_factor"],
        sparsity_coeff=cfg["sparsity_coeff"]
    )
    state_dict = torch.load(buffer, weights_only=True, map_location=device)
    sae.load_state_dict(state_dict)
    return sae, cfg


cfg_dict = {
    "d_vit": 768,
    "exp_factor": EXPANSION_FACTOR,
    "sparsity_coeff": SPARSITY_COEFF,
    "n_reinit_samples": 0,
    "ghost_grads": False,
    "remove_parallel_grads": True,
    "normalize_w_dec": True,
    "seed": 42
}

# MAIN EXEC
if __name__ == "__main__":
    sae = train()
    save_sae(os.path.join(CHECKPOINT_DIR, "sae.pt"), sae, cfg_dict)
    
    