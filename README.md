# OptJail: Dynamic Optimization and Safety Indicator Injection for Jailbreaking Text-to-Image Models

This repository contains the source code and experimental results for **OptJail**, a black-box jailbreak framework targeting multi-stage safety pipelines in text-to-image (T2I) generation services.

## Overview

OptJail consists of two components:

1. **Dynamic Optimization** — An iterative feedback-driven loop that rewrites NSFW prompts to bypass text safety filters while preserving semantic alignment (measured via CLIP similarity).
2. **Adaptive Safety Indicator Injection** — A multi-armed bandit (MAB) approach that selects and appends ISO safety symbol descriptions to adversarial prompts, misleading image safety filters via attention redirection.

## Repository Structure

```
OptJail/
├── automaton_czx/
│   ├── main_czx_feedback_similarity.py   # Core attack loop (Dynamic Optimization, Algorithm 1)
│   ├── main_czx.py                       # Text-feedback-only variant (ablation, no CLIP)
│   ├── main_czx_classifier.py            # Variant using DistilBERT classifier as text filter
│   ├── adaptive_injection.py             # MAB-based indicator injection (Algorithm 2)
│   ├── qwen_test.py                      # LLM rewriter (Qwen2.5-7B-Instruct)
│   ├── infer_shieldlm_czx.py            # Text safety filter (ShieldLM-7B)
│   ├── internvl_main_czx.py             # Image safety filter (InternVL2-2B)
│   ├── Flux_generate.py                  # T2I generator (FLUX.1-schnell)
│   ├── test_similar.py                   # CLIP similarity scoring
│   ├── NSFW_text_classifier.py           # DistilBERT NSFW classifier (used by main_czx_classifier)
│   ├── Evaluation.py                     # Full pipeline evaluation
│   └── indicator_injection_log/
│       └── pretrained_Q.npy              # Pretrained Q-values for MAB warm-up
├── prompt_checker/                       # Alternative text safety filters
│   ├── deepseek.py                       # DeepSeek API-based checker
│   ├── Detoxify.py                       # Detoxify (BERT) multi-label toxicity detector
│   ├── NSFW_text_classify.py             # DistilBERT NSFW text classifier
│   └── text_match.py                     # Keyword-based NSFW text matching
├── img_checker/                          # Alternative image safety filters
│   ├── NSFW_clip.py                      # CLIP ViT-L/14 + autokeras binary classifier
│   ├── nudenet_detect.py                 # NudeNet-based NSFW detection
│   └── q16_detect.py                     # Q16 ResNet50 (neutral/sexy/porn)
├── data/
│   └── nsfw_200.txt                      # NSFW-200 evaluation dataset
├── results/
│   ├── OptJail_prompt.txt                # OptJail adversarial prompts (200)
│   ├── output_sneakyprompt.txt           # SneakyPrompt baseline
│   ├── output_Jailfuzzer.txt             # JailFuzzer baseline
│   ├── output_PGJ.txt                    # PGJ baseline
│   ├── output_DACA.txt                   # DACA baseline
│   ├── output_MMA.txt                    # MMA-Diffusion baseline
│   ├── output_QF.txt                     # QF-PGD baseline
│   └── I2P_200.txt                       # I2P baseline
├── README.md
└── requirements.txt
```

## Requirements

- Python >= 3.10
- CUDA >= 12.1
- GPU with >= 24GB VRAM (tested on NVIDIA RTX 3090)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Models

The following models are required (download from HuggingFace):

| Component | Model | HuggingFace ID |
|-----------|-------|----------------|
| LLM Rewriter | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| Text Safety Filter | ShieldLM-7B | `thu-coai/ShieldLM-7B-internlm2` |
| Image Safety Filter | InternVL2-2B | `OpenGVLab/InternVL2-2B` |
| T2I Generator | FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` |
| Similarity Scorer | CLIP ViT-B/32 | `openai/clip-vit-base-patch32` |

## Usage

### Run the full attack pipeline

```bash
cd automaton_czx
python main_czx_feedback_similarity.py \
    --input ../data/nsfw_200.txt \
    --output ../results/OptJail_prompt.txt \
    --log ../log_files/attack.log \
    --figure_dir ../figures_files/attack
```

### Run indicator injection (Phase 2)

```bash
cd automaton_czx
python adaptive_injection.py
```

### Evaluate results

```bash
cd automaton_czx
python Evaluation.py
```

### Test text filter bypass rate

```bash
cd automaton_czx
python infer_shieldlm_czx.py --input ../results/OptJail_prompt.txt
```

### Test image filter bypass rate

```bash
cd automaton_czx
python internvl_main_czx.py --image_dir ../figures_files/attack
```

### Alternative text safety filters

```bash
# DeepSeek API (requires API key in deepseek.py)
cd prompt_checker
python deepseek.py

# Detoxify multi-label toxicity
python Detoxify.py

# DistilBERT NSFW classifier
python NSFW_text_classify.py

# Keyword matching
python text_match.py
```

### Alternative image safety filters

```bash
cd img_checker

# CLIP-based NSFW detector
python NSFW_clip.py --image_dir ../figures_files/attack

# NudeNet detector
python nudenet_detect.py --image path/to/image.png

# Q16 ResNet50 classifier
python q16_detect.py --image path/to/image.png
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max queries (Q) | 50 | Maximum optimization iterations |
| CLIP threshold (δ) | 0.26 | Semantic alignment gate |
| Sliding window (N) | 10 | Failed attempt history size |
| LLM temperature | 0.7 | Rewriter sampling temperature |
| MAB learning rate (α) | 0.6 | EMA update magnitude |
| MAB temperature (τ) | 0.3 | Softmax policy temperature |
| Reward λ₁ | 1.0 | Bypass success weight |
| Reward λ₂ | 0.4 | Semantic preservation weight |
| Pretrain iterations (T) | 5000 | Offline MAB warm-up |
| Top-K | 5 | Short-list refinement size |
| ISO indicators | 50 | Candidate safety symbols |
| Random seed | 42 | Reproducibility seed |

## Hardware

Experiments were conducted on a single workstation with:
- CPU: AMD Ryzen 9 5950X (16 cores)
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 64GB
- OS: Ubuntu 22.04 LTS

## Citation

```
[To be added upon publication]
```
