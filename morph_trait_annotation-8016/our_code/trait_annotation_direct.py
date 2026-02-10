"""Trait annotation generator using our trained SAE (found in checkpoints/sae.pt and an MLLM
(QWEN-7B to run locally, GPT-5-mini to run with API) with the algorithm 1 Salient Trait Extraction from
Sparse Autoencoder Activations taken from the article by sending the whole image
"""
import os
import json
import random
from collections import Counter, defaultdict
from io import BytesIO
import base64

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import io

CONFIG = {
    "data_dir": "/workspace/Datasets/bioscan-5m/bioscan5m/images/cropped_256/train",
    "sae_checkpoint": "/workspace/checkpoints/sae.pt",
    "output_dir": "/workspace/trait_output",
    "use_saved_activations": True,
    "activations_dir": "/workspace/Datasets/bioscan_activations/7826a159f94da805f7a22de978079f6e0bdf25d1871e3e60dcf48905950cd053",
    "vit_checkpoint": "dinov2_vitb14",
    "layer_id": 10,
    "n_patches": 256,
    "mllm_backend": "qwen",
    "openai_model": "gpt-5-mini",
    "n_images": 1,
    "activation_thresh": 0.05,
    "trait_thresh": 1e-8,
    "debug": True, # True if we want to use a subset of data (true for now)
    "debug_size": 500,  # no of images in debug mode
    "max_traits_per_species": 10,
    "seed": 42,
}
PATCH_SIZE = 14
NUM_PATCHES_PER_ROW = 16
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
        nn.init.zeros_(self.b_enc)

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


def upscale_image(image, scale=2):
    """Upscale PIL image for better MLLM processing"""
    new_size = (image.width * scale, image.height * scale)
    return image.resize(new_size, resample=Image.BICUBIC)


# data loading

class SavedActivationsDataset:
    """Load pre-saved ViT activations from .bin files"""
    def __init__(self, activations_dir, layer_idx=0):
        self.activations_dir = activations_dir
        self.layer_idx = layer_idx
        metadata_path = os.path.join(activations_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.d_vit = self.metadata["d_vit"]
        self.n_patches = self.metadata["n_patches_per_img"]
        self.n_imgs = self.metadata["n_imgs"]
        self.n_layers = len(self.metadata["layers"])
        self.shard_files = sorted([
            f for f in os.listdir(activations_dir)
            if f.startswith("acts") and f.endswith(".bin")
        ])
        n_patches_with_cls = self.n_patches + 1
        self.imgs_per_shard = (
                self.metadata["n_patches_per_shard"]
                // self.n_layers
                // n_patches_with_cls
        )
        self._cached_shard_idx = -1
        self._cached_data = None

        print(f"Loaded activation metadata: {self.n_imgs} images, {self.d_vit}d, {self.n_patches} patches")

    def _load_shard(self, shard_idx):
        """Load a shard file into memory"""
        if shard_idx == self._cached_shard_idx:
            return self._cached_data
        shard_path = os.path.join(self.activations_dir, self.shard_files[shard_idx])
        if shard_idx == len(self.shard_files) - 1:
            imgs_in_previous = shard_idx * self.imgs_per_shard
            imgs_in_this_shard = self.n_imgs - imgs_in_previous
        else:
            imgs_in_this_shard = self.imgs_per_shard

        shape = (imgs_in_this_shard, self.n_layers, self.n_patches + 1, self.d_vit)
        data = np.memmap(shard_path, dtype=np.float32, mode='r', shape=shape)

        self._cached_shard_idx = shard_idx
        self._cached_data = data
        return data

    def get_image_activations(self, img_idx):
        """Get activations for a single image"""
        shard_idx = img_idx // self.imgs_per_shard
        pos_in_shard = img_idx % self.imgs_per_shard

        data = self._load_shard(shard_idx)
        acts = data[pos_in_shard, self.layer_idx, 1:, :]
        return torch.from_numpy(acts.copy())

    def __len__(self):
        return self.n_imgs


def load_sae_features_from_saved(activations_dir, sae, device="cuda", layer_idx=0, subset_indices=None):
    """Load pre-saved activations and run them through the SAE"""
    act_dataset = SavedActivationsDataset(activations_dir, layer_idx=layer_idx)
    all_features = []
    if subset_indices is not None:
        indices = subset_indices
    else:
        indices = range(len(act_dataset))
    sae.eval()
    batch_size = 32
    with torch.no_grad():
        for start in tqdm(range(0, len(indices), batch_size), desc="Loading saved activations"):
            end = min(start + batch_size, len(indices))
            batch_indices = [indices[i] for i in range(start, end)]
            batch_acts = []
            for idx in batch_indices:
                acts = act_dataset.get_image_activations(idx)
                batch_acts.append(acts)

            batch_acts = torch.stack(batch_acts).to(device)  # [batch, n_patches, d_vit]
            _, f_x, _ = sae(batch_acts)  # [batch, n_patches, sae_dim]

            for i in range(f_x.shape[0]):
                all_features.append(f_x[i].cpu())

    return all_features, act_dataset.metadata


def load_image_dataset(data_dir, batch_size=32, subset_size=None):
    """Load ImageFolder dataset
       Returns: dataset, dataloader"""
    dataset = datasets.ImageFolder(root=data_dir) # we load without transforms

    if subset_size:
        indices =list(range(min(subset_size, len(dataset))))
        dataset = Subset(dataset, indices)
        dataset.classes = datasets.ImageFolder(root=data_dir).classes

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

def find_prominent_traits(features, labels, class_names, activation_thresh=0.9, trait_thresh=1e-4):
    """Find SAE latents that are prominent at the species level. This version does NOT track patch indices since we're showing
    whole images without bounding boxes."""
    species_labels = [' '.join(class_names[l].split('_')) for l in labels]
    genus_labels = [class_names[l].split('_')[0] for l in labels]
    unique_species = set(species_labels)
    unique_genus = set(genus_labels)
    print(f"Found {len(unique_species)} species in {len(unique_genus)} genera")

    species_counter = {s: Counter() for s in unique_species}
    genus_counter = {g: Counter() for g in unique_genus}

    print("Analysing laten activations:")
    for i, feat in enumerate(tqdm(features, desc="Processing features")):
        species = species_labels[i]
        genus = genus_labels[i]
        active = (feat > activation_thresh).numpy().astype(int)
        active_latents = set(np.nonzero(active)[1].tolist())
        species_counter[species].update(active_latents)
        genus_counter[genus].update(active_latents)

    prominent_traits = defaultdict(list)

    print("Finding prominent traits at species level:")
    for i, class_idx in enumerate(tqdm(labels, desc="Filtering traits")):
        species = species_labels[i]
        genus = genus_labels[i]
        sp_counts = species_counter[species]
        gen_counts = genus_counter[genus]
        sp_total = sum(sp_counts.values())
        gen_total = sum(gen_counts.values())

        if sp_total == 0 or gen_total == 0:
            continue

        feat = features[i]
        active = (feat > activation_thresh).numpy().astype(int)
        active_latents = set(np.nonzero(active)[1].tolist())

        for latent_idx in active_latents:
            sp_freq = sp_counts[latent_idx]
            gen_freq = gen_counts[latent_idx]

            sp_ratio = sp_freq / sp_total
            gen_ratio = gen_freq / gen_total

            if sp_ratio > trait_thresh and gen_ratio > trait_thresh and sp_ratio > gen_ratio:
                prominent_traits[species].append({
                    'latent_idx': latent_idx,
                    'ex_id': i,
                    'species_ratio': sp_ratio,
                    'genus_ratio': gen_ratio,
                })

    print(f"Found prominent traits for {len(prominent_traits)} species")

    return prominent_traits

# MLLM backends

class MLLMBackend:
    def describe_image(self, image, prompt):
        raise NotImplementedError

    def describe_images(self, images, prompt):
        raise NotImplementedError


class QwenLocalBackend(MLLMBackend):
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"Loading {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)
        self.device = device
        print("Qwen model loaded!")

    def describe_image(self, image, prompt):
        return self.describe_images([image], prompt)

    def describe_images(self, images, prompt):
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        max_tokens = 256 if len(images) > 1 else 200
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        output_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        return response


class OpenAIBackend(MLLMBackend):
    def __init__(self, model="gpt-5-mini"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        print(f"Using OpenAI model: {model}")

    def describe_image(self, image, prompt):
        return self.describe_images([image], prompt)

    def describe_images(self, images, prompt):
        content = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
        content.append({
            "type": "text",
            "text": prompt
        })
        max_tokens = 256 if len(images) > 1 else 200
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": content
            }],
            max_tokens=max_tokens,
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


# trait captioning, full image

PROMPT_SINGLE_IMAGE = """You are given an image of an insect specimen. Your task is to visually examine the insect and describe its observable morphological traits.

1. Identify the visible body parts of the insect (e.g., head, thorax, abdomen, legs, wings, antennae).
2. For each part, briefly describe the observable morphological features—such as shape, size, color, texture, or distinct markings—*based solely on what is visible in the image*.

*IMPORTANT*:
1. Do not infer or assume information that is not directly observable. Avoid adding external knowledge.
2. Keep your response concise and under 200 tokens."""

PROMPT_MULTI_IMAGE = """You are given three images of insects. Your task is to visually analyze them and extract observable morphological traits.

1. Identify the visible body parts of the insect (e.g., head, thorax, abdomen, legs, wings, antennae), *common in all three images*.
2. For each part, identify its morphological features — such as shape, size, color, texture, or distinct markings.
3. After analyzing all three images individually, list the morphological traits that are *common across all three insects*. *Only output traits that are visibly consistent across all images*.

*IMPORTANT*: Do not infer or assume information that is not directly observable. Avoid adding external knowledge."""


def verbalize_traits(dataset, prominent_traits, mllm_backend, output_dir,
                     display_transform, n_images=1):
    """Use MLLM to describe the full image for prominent traits"""
    os.makedirs(output_dir, exist_ok=True)
    if n_images == 3:
        prompt = PROMPT_MULTI_IMAGE
    else:
        prompt = PROMPT_SINGLE_IMAGE
    response_file = os.path.join(output_dir, "trait_descriptions.jsonl")

    with open(response_file, 'w') as f:
        for species, traits in tqdm(prominent_traits.items(), desc="Processing species"):
            species_dir = os.path.join(output_dir, species.replace(' ', '_'))
            os.makedirs(species_dir, exist_ok=True)
            unique_ex_ids = list(set([t['ex_id'] for t in traits]))

            if n_images == 3:
                if len(unique_ex_ids) < 3:
                    print(f"  Skipping {species}: only {len(unique_ex_ids)} examples (need 3)")
                    continue
                max_groups = CONFIG["max_traits_per_species"] or 999
                for group_idx in range(min(max_groups, len(unique_ex_ids) // 3)):
                    if len(unique_ex_ids) >= 3:
                        chosen_ids = random.sample(unique_ex_ids, 3)
                    else:
                        continue

                    images = []
                    for i, ex_id in enumerate(chosen_ids):
                        image, _ = dataset[ex_id]
                        image_display = display_transform(image)
                        image_upscaled = upscale_image(image_display, scale=2)
                        images.append(image_upscaled)

                        img_path = os.path.join(species_dir, f"group_{group_idx}_img{i + 1}_ex_{ex_id}.png")
                        image_upscaled.save(img_path)

                    try:
                        response = mllm_backend.describe_images(images, prompt)
                    except Exception as e:
                        print(f"MLLM error for {species}: {e}")
                        response = f"ERROR: {str(e)}"

                    result = {
                        'species': species,
                        'ex_ids': chosen_ids,
                        'group_idx': group_idx,
                        'response': response,
                        'n_images': 3
                    }
                    f.write(json.dumps(result) + '\n')
                    f.flush()

                    print(f"\n[{species}] Group {group_idx} (3 images):")
                    print(f"  Response: {response[:200]}...")

                    for eid in chosen_ids:
                        if eid in unique_ex_ids:
                            unique_ex_ids.remove(eid)

            else: # single image
                max_per_species = CONFIG["max_traits_per_species"] or len(traits)
                for trait_idx, trait in enumerate(tqdm(traits[:max_per_species],
                                                       desc=f"  {species}", leave=False)):
                    ex_id = trait['ex_id']
                    latent_idx = trait['latent_idx']

                    image, _ = dataset[ex_id]
                    image_display = display_transform(image)
                    image_upscaled = upscale_image(image_display, scale=2)
                    img_path = os.path.join(species_dir,f"ex_{ex_id}_latent_{latent_idx}.png")
                    image_upscaled.save(img_path)

                    try:
                        response = mllm_backend.describe_image(image_upscaled, prompt)
                    except Exception as e:
                        print(f"MLLM error for {species}, ex {ex_id}: {e}")
                        response = f"ERROR: {str(e)}"

                    result = {
                        'species': species,
                        'ex_id': ex_id,
                        'latent_idx': latent_idx,
                        'response': response,
                        'n_images': 1
                    }
                    f.write(json.dumps(result) + '\n')
                    f.flush()

                    print(f"\n[{species}] Example {ex_id}:")
                    print(f"  Response: {response[:200]}...")


# main

def main():
    print("Config:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading SAE from {CONFIG['sae_checkpoint']}...")
    sae, sae_cfg = load_sae_checkpoint(CONFIG["sae_checkpoint"], device)
    print(f"SAE config: d_vit={sae_cfg['d_vit']}, expansion={sae_cfg['exp_factor']}")

    print("loading data:")
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

    print("Extracting sae features")
    if CONFIG["use_saved_activations"]:
        print(f"Loading pre-saved activations from: {CONFIG['activations_dir']}")

        subset_indices = None
        if CONFIG["debug"]:
            subset_indices = list(range(CONFIG["debug_size"]))

        features, act_metadata = load_sae_features_from_saved(
            CONFIG["activations_dir"],
            sae,
            device=device,
            layer_idx=0,
            subset_indices=subset_indices
        )

        if subset_indices:
            labels = [dataset[i][1] for i in subset_indices]
        else:
            labels = [dataset[i][1] for i in range(len(dataset))]
    else:
        print("Extracting activations from images...")

        print("Loading DINOv2...")
        vit = DinoV2Wrapper(CONFIG["vit_checkpoint"])
        vit_recorder = ActivationRecorder(vit, CONFIG["n_patches"], [CONFIG["layer_id"]])
        vit_recorder = vit_recorder.to(device)
        vit_recorder.eval()

        transform = get_image_transform()
        features, labels = extract_sae_features(
            dataloader, vit_recorder, sae, transform, device
        )

    print(f"Extracted features for {len(features)} images")
    print(f"Feature shape per image: {features[0].shape}")

    print("Activation statistics:")

    sample_features = torch.stack(features[:min(100, len(features))])
    all_acts = sample_features.flatten()
    nonzero_acts = all_acts[all_acts > 0]

    if len(nonzero_acts) > 0:
        print(f"  Non-zero activations: {len(nonzero_acts):,} ({100 * len(nonzero_acts) / len(all_acts):.2f}%)")
        print(f"  Mean: {nonzero_acts.mean().item():.6f}")
        print(f"  Max:  {nonzero_acts.max().item():.6f}")
        for p in [90, 95, 99]:
            val = torch.quantile(nonzero_acts, p / 100).item()
            print(f"  {p}th percentile: {val:.6f}")

        pct_above = (nonzero_acts > CONFIG['activation_thresh']).float().mean().item() * 100
        print(f"\n  Current threshold: {CONFIG['activation_thresh']}")
        print(f"  % above threshold: {pct_above:.2f}%")


    print("Prominent traits:")

    prominent_traits = find_prominent_traits(
        features, labels, class_names,
        activation_thresh=CONFIG["activation_thresh"],
        trait_thresh=CONFIG["trait_thresh"]
    )

    total_traits = sum(len(v) for v in prominent_traits.values())
    print(f"Total images with prominent traits: {total_traits}")

    if total_traits == 0:
        print("WARNING: No prominent traits found!")
        return

    print("Caption generation:")
    if CONFIG["mllm_backend"] == "qwen":
        mllm = get_mllm_backend("qwen", device=device)
    else:
        mllm = get_mllm_backend("openai", model=CONFIG["openai_model"])

    display_transform = get_display_transform()

    verbalize_traits(
        dataset,
        prominent_traits,
        mllm,
        CONFIG["output_dir"],
        display_transform,
        n_images=CONFIG["n_images"]
    )
    print(f"Results saved to: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
