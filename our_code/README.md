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
python preprocess_bioscan.py --csv-file /path/to/csv/file/in/bioscan-5m/ --image-dir /path/to/images/dir/in/bioscan-5m/
```

# Ou bien (à voir en fonction de notre fichier preprocess.py et de ce qu'il prend en arguments) : python preprocess_bioscan.py --csv-file /path/to/metadata.csv --image-dir /path/to/images/ --out-dir /path/to/output/
