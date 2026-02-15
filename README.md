# Project : Re-implementation of **"Automatic Image-Level Morphological Trait Annotation for Organismal Images" (ICLR 2026)**

**"Automatic Image-Level Morphological Trait Annotation for Organismal Images"** is a paper accepted at ICLR 2026, studying the use of sparse autoencoders (SAEs) on improving trait descriptions of biological datasets. Their work finds that training SAEs on vision-transformer (ViT) activations (otherwise called patches) create latents with monosemantic features, identifying meaningful properties of the image on a species level. They use this to leberage multi-modal LLMs (MLLMs) to help generate interpretable trait descriptions on BIOSCAN-5M, a dataset of insect images.

In this repository we implement code allowing us to pre-process the dataset so as to organize the images by species, to obtain the ViT patches necessary for training the SAE, the code defining the SAE and training it, as well as code to visualize the latents produced by the SAE and code to generate trait descriptions with an MLLM.

All the work has been reimplemented based on our interpretations from the original paper. We have also added extra analysis of the background ablation to the code and the poster.

## Students

Daniel PANARITI, Defne OZGUVEN, Christine ANTON

---

## Installation

### 1. Install and download BIOSCAN-5M dataset

Before running the preprocessing scripts, install the BIOSCAN dataset package:

```bash
uv pip install bioscan-dataset
```

You can download and initialize the dataset using the following code by adapting the path:

```python
from bioscan_dataset import BIOSCAN5M
ds = BIOSCAN5M("~/Documents/universite/m2_2025-2026_MIND/s3/deep_l/projet/BIOSCAN-5M", download=True)
```

### 2. Preprocess BIOSCAN-5M dataset

Make sure you are outside the source directory when running this in the terminal.
```bash
uv run python -m source.preprocess_data
```

### 3. Obtain the ViT patches with DiNO

```bash
uv run python -m source.vit_activations
```

### 4. Train the Sparse Autoencoder (SAE) model

```bash
uv run python -m source.sae_model
```

Alternatively, to allow for WandDB tracking:

```bash
uv run python -m source.wandb_sae_train
```

### 4. Visualize the heatmap of SAE activations on the dataset

```bash
uv run python -m source.visuals_neurons
```

### 5. Trait Annotation (Direct or SAE)

Direct:
```bash
uv run python -m source.trait_annotation_direct
```

With the SAE:
```bash
uv run python -m source.trait_annotation_sae
```

### 6. Background Bias Analysis (our additional analysis)

```bash
uv run python -m source.background_bias
```
