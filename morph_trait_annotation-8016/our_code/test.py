import os
import json
import io
import torch

ckpt_path = "../checkpoints/sae.pt"

print("Step 1: Reading file...")
with open(ckpt_path, "rb") as f:
    config_line = f.readline().decode()
    print(f"  Config: {config_line[:500]}...")
    state_dict_bytes = f.read()
    print(f"  State dict size: {len(state_dict_bytes) / 1e6:.1f} MB")

print("Step 2: Parsing config...")
sae_config = json.loads(config_line)
d_vit = sae_config["d_vit"]
d_sae = d_vit * sae_config["exp_factor"]
print(f"  d_vit={d_vit}, d_sae={d_sae}")

print("Step 3: Creating empty SAE...")
# Don't import your class yet, just check the sizes
print(f"  W_enc would be: {d_vit * d_sae * 4 / 1e6:.1f} MB")
print(f"  W_dec would be: {d_sae * d_vit * 4 / 1e6:.1f} MB")

print("Step 4: Loading state dict...")
buffer = io.BytesIO(state_dict_bytes)
state_dict = torch.load(buffer, weights_only=True, map_location='cpu')
print(f"  Keys: {list(state_dict.keys())}")

print("Done - no crash!")