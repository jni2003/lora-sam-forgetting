# LoRA Forgetting and Sharpness-Aware Pretraining

Tests whether models pretrained with sharpness-aware minimization (SAM) forget less after LoRA fine-tuning, and whether that happens through fewer intruder dimensions or through flatter curvature along them. Building on Watts et al. (2026), *Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting*, and Shuttleworth et al. (2025), *LoRA vs Full Fine-tuning: An Illusion of Equivalence*,
I explore:
1. **Does SAM pretraining reduce forgetting after LoRA?**
    - Watts et al. show SAM pretraining reduces forgetting after full finetuning.
    - Does that result hold for LoRA (rank 8 and 64) on the same SAM/AdamW checkpoints and data?
      - Compare at matched finetuning loss, with full finetuning across the same learning rates as reference
2. **Does SAM pretraining change intruder dimensions?**
    - Do LoRA updates on SAM pretrained weight matrices create fewer or weaker intruder dimensions?
3. **Are SAM's intruders in flatter directions?**
    - For SAM and AdamW, compare curvature of the pretraining loss along each intruder direction (with Hessian-vector products) and compare with random directions.
4. **What mechanisms impact forgetting?**
    - less or weaker intruders, the same intruders in flatter directions, both?
### Central Hypothesis:

The forgetting caused by an intruder dimension grows with the curvature of the pretraining loss along its direction.

---
### Setup
(Python 3.10)
#### Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
```
Login to Hugging Face + Accept Terms for the [StarCoder dataset](https://huggingface.co/datasets/bigcode/starcoderdata).
#### Download Model Checkpoints
Raw OLMo 60M SAM and AdamW checkpoints from Watts et al.
```bash
hf download WattsIshaan/OLMo-60m-192B-sam-cosine config.yaml model.pt --local-dir checkpoints/olmo-60m-sam
hf download WattsIshaan/OLMo-60m-192B-adamw-cosine config.yaml model.pt --local-dir checkpoints/olmo-60m-adamw
```
(Experiments convert to `transformers` Llama models)
#### Build Data
Tokenize Hugging Face data with `allenai/dolma2-tokenizer` (same as Watts et al.) and save 1024-token rows to `data/` as numpy files with JSON descriptions.
```bash
# 20M held-out DCLM tokens from 100 random files for measuring forgetting
python -m scripts.make_dclm_heldout

# 10M train + 1M val tokens of StarCoder Python for finetuning
python -m scripts.make_starcoder
```
### Run tests
```bash
python -m pytest -v
```

---
### Joel Indurkar
### Science of Deep Learning (CSCI-GA.3033)
### NYU Courant - Fall 2026
