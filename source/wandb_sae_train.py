"""
Wrapper to run sae_model.py with WandB tracking enabled. Otherwise doese the same thing as sae_model.py
"""
import sys
import os
import argparse
import re
import wandb
sys.path.insert(0, os.path.dirname(__file__))
import sae_model

# we parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--track", action="store_true", help="Enable WandB tracking")
parser.add_argument("--wandb-project", type=str, default="sae-bioscan", help="WandB project name")
parser.add_argument("--tag", type=str, default="", help="Run tag")
args = parser.parse_args()

# initialize wandb for tracking
if args.track:
    wandb.init(
        project=args.wandb_project,
        config={
            "expansion_factor": sae_model.EXPANSION_FACTOR,
            "sparsity_coeff": sae_model.SPARSITY_COEFF,
            "batch_size": sae_model.BATCH_SIZE,
            "lr": sae_model.LR,
            "n_patches_target": sae_model.N_PATCHES_TARGET,
            "device": sae_model.DEVICE,
        },
        tags=[args.tag] if args.tag else [],
    )
    print(f"wandb tracking enabled: {args.wandb_project}")

    original_info = sae_model.logger.info


    def wandb_logger_info(msg, *args, **kwargs):
        original_info(msg, *args, **kwargs)
        if "step:" in msg and "loss:" in msg:
            try:
                match = re.search(
                    r'step: (\d+)/(\d+), patches: (\d+)/(\d+), '
                    r'loss: ([\d.]+), mse: ([\d.]+), sparsity: ([\d.]+), '
                    r'L0: ([\d.]+), L1: ([\d.]+), lr: ([\de.+-]+)',
                    msg % args if args else msg
                )
                if match:
                    wandb.log({
                        "step": int(match.group(1)),
                        "total_steps": int(match.group(2)),
                        "patches_seen": int(match.group(3)),
                        "patches_target": int(match.group(4)),
                        "loss": float(match.group(5)),
                        "mse": float(match.group(6)),
                        "sparsity": float(match.group(7)),
                        "l0": float(match.group(8)),
                        "l1": float(match.group(9)),
                        "lr": float(match.group(10)),
                    }, step=int(match.group(1)))
            except:
                pass


    sae_model.logger.info = wandb_logger_info
else:
    print("wandb tracking disabled")

if __name__ == "__main__":
    sae = sae_model.train()
    sae_model.save_sae(
        os.path.join(sae_model.CHECKPOINT_DIR, "sae.pt"),
        sae,
        sae_model.cfg_dict
    )

    if wandb.run is not None:
        wandb.finish()

    print("Training complete!")