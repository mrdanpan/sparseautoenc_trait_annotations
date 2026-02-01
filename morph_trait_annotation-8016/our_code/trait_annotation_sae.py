"""Trait annotation generator using our trained SAE (found in checkpoints/sae.pt and an MLLM
(QWEN-7B to run locally, GPT-5-mini to run with API) with the algorithm 1 Salient Trait Extraction from
Sparse Autoencoder Activations taken from the article
"""

import os
import json
import random
import argparse
from collections import Counter, defaultdict
import io
from io import BytesIO
import base64

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

CONFIG = {
    "data_dir": "/workspace/Datasets/bioscan-5m/bioscan5m/images/cropped_256/train",
    "sae_checkpoint": "/workspace/checkpoints/sae.pt",
    "output_dir": "/workspace/trait_output",
    "vit_checkpoint": "dinov2_vitb14",
    "layer_id": 10,
    "n_patches": 256,
    "mllm_backend": "qwen",
    "openai_model": "gpt-5-mini",
    "activation_thresh": 0.1,
    "trait_thresh": 1e-6,
    "debug": True, # True if we want to use a subset of data (true for now)
    "debug_size": 100,  # no of images in debug mode
    "max_traits_per_species": 10,
    "seed": 42,
}

PATCH_SIZE = 14 # from dinov2
NUM_PATCHES_PER_ROW = 16 # 224 (img size) / 14
IMAGE_SIZE = 224

# DINOv2 + SAE

class DinoV2Wrapper(nn.Module):
    """Wrapper for DINOv2 for feature extraction"""
    def __init__(self, checkpoint="dinov2_vitb14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", checkpoint)

    def get_blocks(self):
        return self.model.blocks

    def get_patch_indices(self, n_patches):
        """indices for patch tokens + CLS"""
        n_reg = self.model.num_register_tokens
        # CLS token + patch tokens
        indices = torch.cat([torch.tensor([0]), torch.arange(n_reg + 1, n_reg + n_patches + 1)])
        return indices

    def forward(self, x):
        out = self.model.forward_features(x)
        features = torch.cat([
            out["x_norm_clstoken"][:, None, :],
            out["x_norm_patchtokens"]
        ], dim=1)
        return features

class ActivationRecorder(nn.Module):
    """Recording activations from a specific ViT layer with hooks"""
    def __init__(self, vit, n_patches, layer_indices):
        super().__init__()
        self.vit = vit
        self.n_patches = n_patches
        self.patch_indices = self.get_patch_indices(n_patches)
        self.layer_indices = layer_indices
        self._activations = {}
        # hooks for the specified layers
        blocks = self.get_blocks()
        for layer_index in layer_indices:
            block = blocks[layer_index].register_forward_hook(lambda module, inp, out, idx = layer_index: self._hook(idx, out))

    def _hook(self, idx, out):
        self._activations[idx] = out[:, self.patch_indices, :].detach()

    def forward(self, x):
        self._activations = {}
        _ = self.vit(x)
        stacked = torch.stack([self._activations[idx] for idx in self.layer_indices], dim=1)
        return stacked

class SparseAutoencoder(nn.Module):
    """SAE for feature extraction"""
    def __init__(self, d_input, expansion_factor=32, sparsity_coeff=4e-4):
        super().__init__()
        d_hidden = d_input * expansion_factor
        self.d_hidden = d_hidden
        self.d_input = d_input
        self.sparsity_coeff = sparsity_coeff
        # encoder
        self.W_enc = nn.Parameter(torch.empty(d_input, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        # decoder
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_input))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        # weights, we use Kaiming He init
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.b_enc)

    def encode(self, x):
        h_pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        f_x = torch.relu(h_pre)
        return f_x

    def decode(self, f_x):
        x_hat = f_x @ self.W_dec + self.b_dec
        return x_hat

    def forward(self, x):
        f_x = self.encode(x)
        x_hat = self.decode(f_x)
        return x_hat, f_x, {}

def load_sae_checkpoint(checkpoint_path, device="cuda"):
    """Load pretrained SAE from our checkpoint"""
    with open(checkpoint_path, "rb") as f:
        cfg = json.loads(f.readline().decode()) # config from the first line
        buffer = io.BytesIO(f.read()) # state dict

    sae = SparseAutoencoder(d_input=cfg["d_vit"], expansion_factor=cfg["exp_factor"], sparsity_coeff=cfg.get("sparsity_coeff", 4e-4))
    state_dict = torch.load(buffer, weights_only = True, map_location=device)
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()
    return sae, cfg

# image preprocessing utils

def get_image_transform():
    """Standard DINOv2 img preprocessing"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_display_transform():
    """Transform for displaying, no normalization"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

def patch_index_to_coordinates(patch_indices, patch_size=14, patches_per_row=16):
    """Convert patch indices (x,y) to pixel coordinates
       Returns : List of (x, y, width, height) for each patch"""
    coords = []
    for idx in patch_indices:
        row = idx // patches_per_row
        col = idx % patches_per_row
        x = col * patch_size
        y = row * patch_size
        coords.append((x, y, patch_size, patch_size))
    return coords

def draw_bounding_boxes(image, patch_coords, label=None, color="red", width=2):
    """Drawing bounding boxes on given image
       Returns: annotated PIL image"""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for (x, y, w, h) in patch_coords:
        draw.rectangle((x, y, x + w, y + h), outline=color, width=width)
    if label and patch_coords:
    # if there is a label, we draw it
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        x, y, w, h = patch_coords[0]
        text_pos = (x, max(0, y - 15))
        draw.text(text_pos, label, font=font, fill=color)
    return image

# data loading

def load_image_dataset(data_dir, batch_size=32, subset_size=None):
    """Load ImageFolder dataset
       Returns: dataset, dataloader"""
    dataset = datasets.ImageFolder(root=data_dir) # we load without transforms

    if subset_size:
        indices =list(range(min(subset_size, len(dataset))))
        dataset = Subset(dataset, indices)
        dataset.classes = dataset.ImageFolder(root=data_dir).classes

    def collate_fn(batch):
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataset, dataloader

# feature extraction

def extract_sae_features(dataloader, vit_recorder, sae, transform, device="cuda"):
    """Extract SAE features for all images
       Returns:
           all_features: List of [n_patches, sae_dim] tensors per image
           all_labels: List of int labels"""
    all_features = []
    all_labels = []
    vit_recorder.eval()
    sae.eval()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting SAE features"):
            images_t = torch.stack([transform(img) for img in images]).to(device) # transform to each img
            vit_acts = vit_recorder(images_t) # vit activations
            vit_acts = vit_acts[:, 0, 1, :] # select layer and remove CLS
            _, f_x, _ = sae(vit_acts)
            for i in range(f_x.shape[0]):
                all_features.append(f_x[i].cpu())
            all_labels.extend(labels.tolist())

    return all_features, all_labels

# prominent trait detection

def find_prominent_traits(features, labels, class_names, activation_tresh = 0.9, trait_tresh = 1e-4):
    """SAE latents that are prominent at species level. Prominent for a species if:
        1. it activates above the threshold for that species
        2. activation frequency is higher for the species than for the genus
        Returns:
        prominent_traits: Dict mapping species -> list of trait info dicts
        latent_to_patch_map: Dict mapping image_idx -> {latent_idx: [patch_indices]}"""
    species_label = [' '.join(class_names[l].split('_')) for l in labels]
    genus_labels = [class_names[l].split('_')[0] for l in labels]
    unique_species = set(species_label)
    unique_genus = set(genus_labels)
    print("Unique species labels:", len(unique_species), "Unique genus labels:", len(unique_genus))
    # we count the latent activations for both levels
    species_counter = {s: Counter() for s in unique_species}
    genus_counter = {g: Counter() for g in unique_genus}
    latent_to_patch_map = {} # which patches activate which latents

    print("Analysing laten activations:")
    for i, feat in enumerate(tqdm(features, desc="Processing SAE features")):
        species = species_label[i]
        genus = genus_labels[i]
        active = (feat > activation_tresh).numpy().astype(int)
        patch_indices, latent_indices = np.nonzero(active) # patch, latent idx pairs where they are active

        latent_to_patch_map[i] = defaultdict(list)
        for patch_id, latent_id in zip(patch_indices, latent_indices):
            latent_to_patch_map[i][int(latent_id)].append(int(patch_id))
        # we update the frequency counters
        active_latents = set(latent_indices.tolist())
        species_counter[species].update(active_latents)
        genus_counter[genus].update(active_latents)

    prominent_traits = defaultdict(list)

    print("Finding prominent traits at species level:")
    for i, class_idx in enumerate(tqdm(labels, desc="Filtering traits")):
        species = species_label[i]
        genus = genus_labels[i]
        sp_counts = species_counter[species]
        gen_counts = genus_counter[genus]
        sp_total = sum(sp_counts.values())
        gen_total = sum(gen_counts.values())

        if sp_total == 0 or gen_total == 0:
            continue

        for latent_id in latent_to_patch_map[i]:
            sp_freq = sp_counts[latent_id]
            gen_freq = gen_counts[latent_id]
            sp_ratio = sp_freq / sp_total
            gen_ratio = gen_freq / gen_total

            if sp_ratio > trait_tresh and gen_ratio > trait_tresh and sp_ratio > gen_ratio:
                prominent_traits[species].append({
                    'latent_idx': latent_id,
                    'ex_id': i,
                    'patch_idx': latent_to_patch_map[i][latent_id],
                    'species_ratio': sp_ratio,
                    'genus_ratio': gen_ratio,
                })

    print(f"Found traits for {len(prominent_traits)} species")

    return prominent_traits, latent_to_patch_map

# MLLM backends

class MLLMBackend:
    def describe_region(self, image, prompt):
        """Ask the Vision language model to describe a region in an img"""
        raise NotImplementedError

class QwenLocalBackend(MLLMBackend):
    """Local Qwen2.5-VL-7B backend using HuggingFace Transformers"""
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        print(f"Loading model {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.device = device
        print("Qwen model loaded")

    def describe_region(self, image, prompt):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]
        # chat template
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        # process inputs
        inputs = self.processor(text=text, images=[image], padding=True, return_tensors="pt").to(self.device)
        # generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        # decode
        output_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return response

class OpenAIBackend(MLLMBackend):
    """OpenAI API backend using gpt-5-mini (vision capable).
    Requires OPENAI_API_KEY environment variable."""
    def __init__(self, model="gpt-5-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        print(f"OpenAI model loaded {model}")

    def describe_region(self, image, prompt):
        # we convert PIL image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }],
            max_tokens=200,
            temperature=0.1
        )

        return response.choices[0].message.content

def get_mllm_backend(backend_name, **kwargs):
    if backend_name == "qwen":
        return QwenLocalBackend(**kwargs)
    elif backend_name == "openai":
        return OpenAIBackend(**kwargs)
    else:
        raise ValueError(f"Unsupported backend {backend_name}")

# trait verbalization

DEFAULT_PROMPT = """You are given an image of an insect with red bounding boxes highlighting a specific region.

1. Determine whether the highlighted region contains a visible body part of the insect or only background. If it appears to be background, respond with "background".

2. If it contains a visible body part, identify which part it is (e.g., wing, leg, antenna, thorax, abdomen, head, eye).

3. Briefly describe the observable morphological features—such as shape, size, color, texture, or distinct markings—based solely on what is visible in the image.

IMPORTANT: Do not infer or assume information that is not directly observable. Keep your response under 100 words."""

def verbalize_traits(dataset, prominent_traits, mllm_backend, output_dir, display_transform, prompt=DEFAULT_PROMPT):
    """Use MLLM to describe the boxed regions for prominent traits"""
    os.makedirs(output_dir, exist_ok=True)
    response_file = os.path.join(output_dir, "trait_descriptions.jsonl")

    with open(response_file, "w") as f:
        for species, traits in tqdm(prominent_traits.items(), desc="Processing species"):
            species_dir = os.path.join(output_dir, species.replace(' ', '_'))
            os.makedirs(species_dir, exist_ok=True)
            for trait in tqdm(traits, desc=f"  {species}", leave=False):
                ex_id = trait['ex_id']
                latent_idx = trait['latent_idx']
                patch_indices = trait['patch_idx']
                # the original img
                image, _ = dataset[ex_id]
                image_display = display_transform(image)
                # converting patch idx to coords
                patch_coords = patch_index_to_coordinates(patch_indices)
                # bounding boxes
                annotated_image = draw_bounding_boxes(image_display, patch_coords, label=f"Latent {latent_idx}")
                img_path = os.path.join(species_dir, f"latent_{latent_idx}_ex_{ex_id}.png")
                annotated_image.save(img_path)
                try:
                    response = mllm_backend.describe_region(annotated_image, prompt)
                except Exception as e:
                    print(f"MLLM error for {species}, latent {latent_idx}: {e}")
                    response = f"ERROR: {str(e)}"
                result = {
                    'species': species,
                    'ex_id': ex_id,
                    'latent_idx': latent_idx,
                    'patch_indices': patch_indices,
                    'response': response
                }
                f.write(json.dumps(result) + '\n')
                f.flush()

                print(f"\n[{species}] Latent {latent_idx}:")
                print(f"  Response: {response[:200]}...")

# main

def main():
    print("Current config:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading DINOv2...")
    vit = DinoV2Wrapper(CONFIG["vit_checkpoint"])
    vit_recorder = ActivationRecorder(vit, CONFIG["n_patches"], [CONFIG["layer_id"]])
    vit_recorder = vit_recorder.to(device)
    vit_recorder.eval()

    print(f"Loading SAE from {CONFIG['sae_checkpoint']}:")
    sae, sae_cfg = load_sae_checkpoint(CONFIG["sae_checkpoint"], device)
    print(f"SAE config: d_vit={sae_cfg['d_vit']}, expansion={sae_cfg['exp_factor']}")

    print("Loading the dataset:")
    subset_size = CONFIG["debug_size"] if CONFIG["debug"] else None
    dataset, dataloader = load_image_dataset(
        CONFIG["data_dir"],
        batch_size=32,
        subset_size=subset_size
    )

    if hasattr(dataset, 'dataset'):
        class_names = dataset.dataset.classes
    else:
        class_names = dataset.classes

    print(f"Loaded {len(dataset)} images from {len(class_names)} classes")
    print("Extracting SAE features:")
    transform = get_image_transform()
    features, labels = extract_sae_features(
        dataloader, vit_recorder, sae, transform, device
    )

    print(f"Extracted features for {len(features)} images")
    print(f"Feature shape per image: {features[0].shape}")

    print("\nActivation statistics:")
    sample_features = torch.stack(features[:min(100, len(features))])
    all_acts = sample_features.flatten()
    nonzero_acts = all_acts[all_acts > 0]

    if len(nonzero_acts) > 0:
        print(f"  Total activations sampled: {len(all_acts):,}")
        print(f"  Non-zero activations: {len(nonzero_acts):,} ({100 * len(nonzero_acts) / len(all_acts):.2f}%)")
        print(f"  Non-zero activation stats:")
        print(f"    Min:    {nonzero_acts.min().item():.6f}")
        print(f"    Max:    {nonzero_acts.max().item():.6f}")
        print(f"    Mean:   {nonzero_acts.mean().item():.6f}")
        print(f"    Median: {nonzero_acts.median().item():.6f}")

        for p in [50, 75, 90, 95, 99]:
            val = torch.quantile(nonzero_acts, p / 100).item()
            print(f"    {p}th percentile: {val:.6f}")

        print(f"\n  Current activation threshold: {CONFIG['activation_thresh']}")
        pct_above = (nonzero_acts > CONFIG['activation_thresh']).float().mean().item() * 100
        print(f"  % of non-zero activations above threshold: {pct_above:.2f}%")

        if pct_above < 1:
            print(f"  WARNING: Very few activations above threshold!")
            print(f"  Consider lowering activation_thresh in CONFIG")
    else:
        print("  WARNING: All activations are zero! Check your SAE checkpoint.")

    print("Finding prominent traits:")
    prominent_traits, latent_map = find_prominent_traits(
        features, labels, class_names,
        activation_thresh=CONFIG["activation_thresh"],
        trait_thresh=CONFIG["trait_thresh"]
    )

    if CONFIG["max_traits_per_species"]:
        for species in prominent_traits:
            if len(prominent_traits[species]) > CONFIG["max_traits_per_species"]:
                prominent_traits[species] = random.sample(
                    prominent_traits[species],
                    CONFIG["max_traits_per_species"]
                )

    total_traits = sum(len(v) for v in prominent_traits.values())
    print(f"Total traits to process: {total_traits}")

    if total_traits == 0:
        print("WARNING: No prominent traits found!")
        return

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    with open(os.path.join(CONFIG["output_dir"], 'latent_to_patch_map.json'), 'w') as f:
        json.dump({k: dict(v) for k, v in latent_map.items()}, f, indent=2)

    print("Caption creation:")
    if CONFIG["mllm_backend"] == "qwen":
        mllm = get_mllm_backend("qwen", device=device)
    else:
        mllm = get_mllm_backend("openai", model=CONFIG["openai_model"])

    display_transform = get_display_transform()
    verbalize_traits(dataset, prominent_traits, mllm, CONFIG["output_dir"], display_transform)

    print(f"Results saved to: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()