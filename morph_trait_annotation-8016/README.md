# Code for submission: Automatic Image-Level Morphological Trait Annotation for Organismal Images

**Note**: We use the SAE training code from the publicly available [SAEV repository](https://github.com/OSU-NLP-Group/saev). The same is copied here for convenience.

## Installation
Installation will be done automatically by uv package manager when the python scripts are run.

**Step 1**: Download BIOSCAN-5M
# train split of cropped 256px images (~289K images)
```
pip install bioscan-dataset
```

```
from bioscan_dataset import BIOSCAN5M
ds = BIOSCAN5M("~/Datasets/bioscan-5m", download=True)
```

**Step 2**: Preprocess BIOSCAN-5M
```
/opt/miniconda3/envs/bmcb/bin/python morph_trait_annotation-8016/preprocess_bioscan.py \
    --csv-file bioscan_5m/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv \
    --image-dir bioscan_5m/bioscan5m/images/cropped_256/train \
    --out-dir ./organized_species \
    --split train
```
python process_bioscan.py --csv-file /path/to/csv/file/in/bioscan-5m/ --image-dir /path/to/images/dir/in/bioscan-5m/
```

**Step 3**: Dump DINOv2 activations for BIOSCAN-5M.
```
/opt/miniconda3/envs/bmcb/bin/python
```
```
uv run python -m saev activations \
  --vit-family dinov2 \
  --vit-ckpt dinov2_vitb14 \
  --vit-batch-size 1024 \
  --d-vit 768 \
  --n-patches-per-img 256 \
  --vit-layers -2 \
  --dump-to /path/to/activations/dir \
  --n-patches-per-shard 2_4000_000 \
  data:image-folder-dataset \
  --data.root /path/to/processed/bioscan/
```

**Step 4**: Train a sparse autoencoder 
```
uv run python -m saev train \
  --data.shard-root /path/to/activations/dir \
  --data.layer -2 \
  --data.patches patches \
  --data.scale-mean False \
  --data.scale-norm False \
  --sae.d-vit 768 \
  --sae.exp-factor 32 \
  --ckpt-path /path/to/sae/ckpt/ \
  --lr 1e-3 > LOG.txt 2>&1
```

**Step 5**: Start Qwen2.5-VL-72B server
```
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port <PORT>
```

**Step 6**: Dataset Generation

### MLLM + SAE

#### Setting: single image, Qwen 2.5 VL 72B, t_freq=3e-3
```
uv run python -u create_trait_dataset_mllm_sae.py --data-dir /PATH/TO/BIOSCAN/DATASET/ --sae-ckpt-path /PATH/TO-SAE/CKPT --thresh 0.9 --out-dir OUTPUT_DIR/ --trait-thresh 3e-3 --serve-choice qwen_72b --api-url http://0.0.0.0:<PORT>/v1/chat/completions --debug > LOG.txt 2>&1
```

#### Setting: multiple images, Qwen 2.5 VL 72B, t_freq=3e-3
```
uv run python -u create_trait_dataset_mllm_sae.py --data-dir /PATH/TO/BIOSCAN/DATASET/ --sae-ckpt-path /PATH/TO-SAE/CKPT --thresh 0.9 --out-dir OUTPUT_DIR/ --trait-thresh 3e-3 --serve-choice qwen_72b --api-url http://0.0.0.0:<PORT>/v1/chat/completions --debug --n-img-input 3 > LOG.txt 2>&1
```

### MLLM

#### Setting: single image, Qwen 2.5 VL 72B (generate corresponding to all samples in MLLM+SAE with trait_thresh=3e-3)
```
uv run python -u create_trait_dataset_mllm.py --data-dir /PATH/TO/BIOSCAN/DATASET/ --sae-ckpt-path /PATH/TO-SAE/CKPT --thresh 0.9 --out-dir OUTPUT_DIR/ --trait-thresh 3e-3 --serve-choice qwen_72b --api-url http://0.0.0.0:<PORT>/v1/chat/completions --debug > LOG.txt 2>&1
```

#### Setting: multiple images, Qwen 2.5 VL 72B (generate corresponding to all samples in MLLM+SAE with trait_thresh=3e-3)
```
uv run python -u create_trait_dataset_mllm.py --data-dir /PATH/TO/BIOSCAN/DATASET/ --sae-ckpt-path /PATH/TO-SAE/CKPT --thresh 0.9 --out-dir OUTPUT_DIR/ --trait-thresh 3e-3 --serve-choice qwen_72b --api-url http://0.0.0.0:<PORT>/v1/chat/completions  --debug --n-img-input 3 > LOG.txt 2>&1
```

### Fine-tuning Experiments

We build on the publicly available [BioCLIP](https://github.com/Imageomics/bioclip) training and evaluation framework. For dataset pre-processing, we rely on the provided utility scripts:
- `utils/create_train_json.py` – to construct training JSON files from CSV inputs.
- `utils/convert_trait_wds.py` – to convert trait annotations into WebDataset (WDS) format for efficient loading.

These scripts generate the input data in the required format prior to launching training runs, ensuring full compatibility with the BioCLIP repo.
