# sparseautoenc_trait_annotations
Re-implementation of "Automatic Image-Level Morphological Trait Annotation for Organismal Images" from ICLR 2026. 



**Step 1**: Download BIOSCAN-5M
# train split of cropped 256px images (~289K images)
pip install bioscan-dataset
from bioscan_dataset import BIOSCAN5M
ds = BIOSCAN5M("~/Datasets/bioscan-5m", download=True)

**Step 2**: Preprocess BIOSCAN-5M
sefpfeff

**Step 3**: Get the ViT activations
python our_code/vit_activations.py

