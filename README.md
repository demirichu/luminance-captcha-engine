# luminance-captcha-engine: Universal Captcha AI Pipeline & Defense Framework

> **Production-grade deep learning pipeline to solve heavily obfuscated text CAPTCHAs.**  
> Features PGD adversarial training, knowledge distillation, and dynamic hard mining to produce ultra-compact INT8 ONNX engines achieving **99.94% accuracy** and **0.12ms CPU inference latency**. Fully profile-driven and adaptable to any format.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-INT8%20Quantized-005CED.svg?logo=onnx)](https://onnxruntime.ai/)
[![Kornia](https://img.shields.io/badge/GPU%20Augmentation-Kornia-blue.svg)](https://kornia.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Responsible Disclosure & Anti-Abuse Notice

> [!IMPORTANT]
> **Ethical Research & Anti-Abuse Statement:**  
> This project is designed exclusively for academic, defensive deep learning, and AI security research. To comply with online service Terms of Service (ToS), uphold responsible disclosure standards, and **prevent automated account generation or botting on live game servers and production gaming ecosystems**:
> - **Target-Specific Network Packet Adapters & Decompression Loaders Are Withheld:** Proprietary network interception scripts, packet decoders, and raw decompression adapters are intentionally excluded from this public release.
> - **Pre-Trained Production Weights & Raw Datasets Are Withheld:** Pre-trained model checkpoints (`.pth`, `.onnx`) and scraped raw challenge datasets (`.npz`, `.dat`) are not distributed.
> - **Target-Specific Production Profiles Are Withheld:** Live target-specific dimensions, salt-and-pepper noise thresholds, and exact character sets are omitted in favor of a clean, generic benchmark profile.
>
> ### Why Are These Artifacts Withheld?
> 1. **Prevention of Automated Abuse & Sybil Attacks:** Distributing turnkey production weights or live game packet loaders would allow malicious actors to automate account registration or spam at scale without computational investment, violating game platform Terms of Service.
> 2. **Focus on General-Purpose Deep Learning & Defense:** The primary objective and scientific contribution of this repository is the **architectural defense framework**—combining GPU-accelerated PGD adversarial training, Curriculum Knowledge Distillation, Monte Carlo TTA triage, and INT8 single-file quantization.
> 3. **Turnkey Training Pipeline Provided:** By providing the full, profile-driven training and distillation engines (`train.py`, `knowledge_distillation.py`, `quantize_model.py`, `dataset_formatter.py`), researchers and defense teams can effortlessly train and benchmark models on their own legitimate datasets by simply setting their profile in `config.py`.
>
> This repository is published strictly as a **general-purpose framework**. Users are expected to define their own parameters in `config.py` and supply their own labeled training images via the generic `dataset_formatter.py` utility.

---

## Overview

This repository provides an end-to-end, industrial-grade pipeline for training, distilling, and deploying ultra-fast, adversarial-resistant deep learning models capable of solving complex, noisy, distorted, single-channel (L-channel / grayscale) text CAPTCHAs.

The architecture is **100% profile-driven**: by configuring the parameters in `config.py` (dimensions, character count, charset), the models (`models.py`), training engine (`train.py`), distillation pipeline (`knowledge_distillation.py`), and quantization tools (`quantize_model.py`) automatically adapt their topology to fit any target CAPTCHA format without requiring code modifications.

---

## Demos

### 1. Training Pipeline (`train.py`)
> GPU-accelerated adversarial training (PGD), dynamic hard sample mining, and real-time loss tracking.

https://github.com/user-attachments/assets/de078950-75fb-453b-8b6a-6384e14f5217


### 2. Terminal CAPTCHA Inspector (`view_captchas.py`)
> ANSI 24-bit TrueColor terminal visualizer with live model inference and character-level confidence scores.

https://github.com/user-attachments/assets/2fb3cab5-ee95-4dd0-8a4d-c7ae945fb52c


## Key Features & Advanced ML Techniques

- **Profile-Driven Core (`config.py` & `models.py`):** Model dimensions, input channels, attention heads, sequence pooling, and classification layers dynamically derive their topology from the active configuration profile.
- **GPU-Native Pipeline & Mixed Precision (AMP):** Completely eliminates CPU data-loading bottlenecks by executing all spatial and intensity augmentations directly on the GPU using **Kornia**. Combined with PyTorch Automatic Mixed Precision (FP16), training achieves maximum hardware saturation.
- **Advanced Adversarial Training (PGD):** Integrates an iterative **Projected Gradient Descent (PGD)** multi-restart attack mechanism into the training loop, forcing the model to build decision boundaries impervious to heavy distortion, salt-and-pepper noise, and line interference.
- **Model EMA (Exponential Moving Average):** Smooths weight updates during training to avoid catastrophic forgetting, stabilize gradients, and guarantee superior validation generalization.
- **Curriculum-Based Knowledge Distillation:** Transfers rich representations from a heavy Teacher CRNN (Attention + BiGRU) to an ultra-lightweight Student CNN using a dynamic soft-to-hard temperature schedule.
- **Dynamic Hard Sample Mining:** Continuously tracks prediction confidence per sample during training and dynamically weights high-loss samples with exponential decay.
- **Monte Carlo Triage Engine (`labeler_mct.py`):** Automatically purifies noisy, web-scraped datasets into Golden, Risky, and Quarantine subsets using test-time dropout (TTA + uncertainty/entropy analysis).
- **INT8 Single-Piece Welding (`quantize_model.py`):** Quantizes FP32 student models into an optimized INT8 single-file ONNX engine via `onnxruntime` QDQ format and `onnxsim`, achieving microsecond inference latencies with zero external dependencies.

---

## Reference Case Study: Hardened Alpha-Channel Obfuscation in Real-Time Multiplayer Platforms

As a real-world validation of its architectural robustness, this pipeline was evaluated against a hardened, production authentication challenge deployed across high-concurrency multiplayer gaming services—an adversarial system featuring complex alpha-channel mask manipulation, variable stream-level scaling, and non-solid contour glyphs:

> [!NOTE]
> **Target Anonymization & Responsible Disclosure:**  
> The specific production game service and infrastructure provider analyzed in this case study are deliberately anonymized. Naming the live target is withheld to prevent targeted exploitation against production authentication endpoints, mitigate automated platform abuse, and maintain strict alignment with academic defensive security standards.

### The Obfuscation Mechanics
1. **Zero-Luminance Base Canvas:** The raw payload contains a uniform zero-value color space (`RGB = (0, 0, 0)`), encoding all spatial character data exclusively across the alpha/luminance channel.
2. **3-Band Quantized Noise Floor:** Ambient noise is injected across three discrete alpha density bands (`~18`, `~38`, and `~77`), completely defeating naive Otsu thresholding and standard binarization filters.
3. **Hollow Aura Glyphs:** Characters are not solid strokes; they are rendered as non-solid contour boundaries ("hollow auras") where interior glyph regions match the background noise distribution.
4. **Dynamic LEB128 Stream Multiplier:** Network packets supply a variable intensity coefficient (1x–4x) encoded via variable-length integers (LEB128) that dynamically scales rendered alpha intensity per authentication challenge.
5. **Adversarial Boundary Perturbation:** Aggressive kerning, rotational shear, and partial aura erosion actively fuse character perimeters into ambient noise thresholds.

### Verified Benchmark Results
Evaluated on an internal verified split of **51,413 samples** (17x60 single-channel bitmaps) on a standard CPU execution provider:

| Model Architecture | Precision | Clean Accuracy | Char Accuracy | Confidence | PGD Robustness | Model Size | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher CRNN (Attention)** | FP32 | 99.74% | 99.93% | 97.27% | 90.40% | ~2.76 MB | ~2.95 ms |
| **Student CNN** | FP32 | 99.96% | 99.99% | 98.24% | 95.60% | ~1.13 MB | ~0.77 ms |
| **Student CNN (Quantized)** | INT8 | **99.94%** | **99.98%** | **98.16%** | **96.30%** | **~313.8 KB** | **~0.12 ms** |

> **Evaluation Notes:**
> - **PGD Attack:** $\epsilon = 0.03$, $\alpha = 0.008$, 5 iterations.
> - **Inference Latency:** Benchmarked on single-thread CPU execution (Batch Size = 1) across 80 iterations with 15 warmup cycles.
> - *Notice:* In accordance with responsible disclosure, the specific network payload extraction script, production profiles, and trained weights for this target are strictly withheld.

---

## Adapting to ANY CAPTCHA in 3 Steps

You can adapt this entire pipeline to any custom single-channel (L-channel) CAPTCHA in minutes:

### 1. Configure Profile in `config.py`
Set your target image dimensions, character length, and character alphabet:
```python
CAPTCHA_NAME = "my_custom_captcha"
IMG_WIDTH = 120
IMG_HEIGHT = 40
IMG_CHANNELS = 1          # 1 for Grayscale / L-channel
CAPTCHA_LENGTH = 5
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

### 2. Format Your Dataset
Place your labeled images (e.g., `A9BK3.png`, `X8P12_01.jpg`) into a folder and format them into the canonical training archive:
```bash
python dataset_formatter.py --input ./raw_images --output ./files/master_dataset.npz
```

### 3. Train & Deploy
Run the end-to-end training and quantization pipeline:
```bash
# 1. Train Teacher CRNN with PGD adversarial defense
python train.py

# 2. (Optional) Triage noisy labels with Monte Carlo Dropout
python labeler_mct.py

# 3. Distill into lightweight Student CNN
python knowledge_distillation.py

# 4. Quantize and weld into single-file INT8 ONNX
python quantize_model.py

# 5. Benchmark all models
python benchmark.py
```

---

## File Structure

```text
luminance-captcha-engine/
├── config.py                 # Centralized configuration profiles, hyperparameters, paths
├── models.py                 # Profile-driven architectures (Teacher CRNN & Student CNN)
│
├── dataset_formatter.py      # Generic dataset builder for standard images (.png, .jpg, .bmp)
├── merge_datasets.py         # Utility to merge and concatenate dataset partitions
├── labeler_mct.py            # Monte Carlo Dropout triage (Golden / Risky / Quarantine)
├── view_captchas.py          # Terminal TrueColor CAPTCHA visualizer & live inference inspector
│
├── train.py                  # Teacher CRNN training with PGD adversarial defense & EMA
├── knowledge_distillation.py # Student distillation using Curriculum Learning
├── quantize_model.py         # Quantizes and welds FP32 model into single-piece INT8 ONNX
├── pgd_attack.py             # Adversarial evaluation and attack perturbation visualizer
├── benchmark.py              # Comprehensive benchmark engine (Accuracy, Latency, PGD, Size)
│
├── files/                    # Model weights, datasets (.npz), and exported ONNX models
└── requirements.txt          # Python dependencies
```

---

## Requirements & Installation

1. **Install PyTorch with CUDA/ROCm acceleration:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Legal & Ethical Disclaimer

> **Educational & Security Research Purposes Only:** This repository is developed strictly for academic research, reverse engineering analysis, and defensive AI security benchmarking. The authors do not endorse, encourage, or support unauthorized automation, scraping, or Terms of Service violations. Pre-trained production weights and target-specific network captures are withheld to prevent unauthorized exploitation.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
