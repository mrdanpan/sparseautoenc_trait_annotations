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
python our_code/preprocess_data.py
```

### 3. Get the ViT activations

```bash
python our_code/vit_activations.py
```
