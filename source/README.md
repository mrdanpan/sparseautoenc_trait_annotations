# Project : Re-implementation of **"Automatic Image-Level Morphological Trait Annotation for Organismal Images" (ICLR 2026)**

## Students

Daniel PANARITI, Defne OZGUVEN, Christine ANTON

---

## Installation

### 1. Install and download BIOSCAN-5M dataset

Before running the preprocessing scripts, install the BIOSCAN dataset package:

```bash
pip install bioscan-dataset
```

You can download and initialize the dataset using the following code by adapting the path:

```python
from bioscan_dataset import BIOSCAN5M
ds = BIOSCAN5M("~/Datasets/bioscan-5m", download=True)
```

### 2. Preprocess BIOSCAN-5M dataset

```bash
uv run python -m source.preprocess_data.py
```


### 3. Train the Sparse Autoencoder (SAE) model

```bash
uv run python -m source.sae_model.py
```

Alternatively, to allow for WandDB tracking:

```bash
uv run python -m source.wandb_sae_train.py
```

### 4. Visualize the heatmap of SAE activations on the dataset

```bash
uv run python -m source.visuals_neurons.py
```

### 5. Trait Annotation (Direct or SAE)

Direct:
```bash
uv run python -m source.trait_annotation_direct.py
```

With the SAE:
```bash
uv run python -m source.trait_annotation_sae.py
```

### 6. Background Bias Analysis

```bash
uv run python -m source.background_bias
```
