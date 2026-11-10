# LoRA Forgetting and Sharpness-Aware Pretraining

Science of Deep Learning (CSCI-GA.3033), NYU Courant, Fall 2026. Joel Indurkar.

Tests whether models pretrained with sharpness-aware minimization (SAM) forget less after LoRA fine-tuning, and whether that happens through fewer intruder dimensions or through flatter curvature along them. It builds on Watts et al. (2026), *Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting*, and Shuttleworth et al. (2025), *LoRA vs Full Fine-tuning: An Illusion of Equivalence*.
