# AI Generated Script to inspect raw olmo checkpoints
"""Print what's inside a raw OLMo checkpoint folder: the model config and every weight's name and shape.

We run this before writing the converter, so we know exactly how OLMo stores its weights
(for example, whether query/key/value are packed into one matrix).

Usage:
    python scripts/inspect_checkpoint.py checkpoints/olmo-60m-sam
"""

import sys
from pathlib import Path

import torch
import yaml

ckpt_dir = Path(sys.argv[1])

# config.yaml records the architecture the model was trained with.
# OLMo keeps the architecture under a "model" section; fall back to the whole file if not.
config = yaml.safe_load((ckpt_dir / "config.yaml").read_text())
model_cfg = config.get("model", config)
print("Model config:")
for key, value in model_cfg.items():
    print(f"  {key}: {value}")

# weights_only=True loads plain tensors without running any code stored in the file.
# That's the safe way to open a .pt file downloaded from the internet.
state_dict = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)

print("\nWeights (name, shape, dtype):")
total = 0
for name, tensor in state_dict.items():
    print(f"  {name:45s} {str(tuple(tensor.shape)):18s} {tensor.dtype}")
    total += tensor.numel()
print(f"\nTotal parameters: {total:,}")
