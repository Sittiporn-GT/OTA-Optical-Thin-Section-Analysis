# OTA — Optical Thin-Section Analysis

**Plutonic Mineral Grain Segmentation in Thin-Section Images Using a Shifted Dual-Modal Vision Transformer**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

OTA is a dual-modal semantic segmentation framework that automatically delineates and
classifies mineral grains in petrographic thin sections by jointly exploiting
**plane-polarized light (PPL)** and **cross-polarized light (XPL)** micrographs.

Built on top of the CMX (Cross-Modal Fusion for RGB-X Semantic Segmentation) framework,
OTA introduces **Deformable Cross-Attention (DeformCA)**, **Shifted Window Attention (SWA)**,
a **UPerNet** multi-scale decoder, and an **Imbalance-Aware Weighted Cross-Entropy (IA-WCE)**
loss to handle the extreme class imbalance typical of modal mineralogy.

---

## Highlights

- **Dual-modal input** — PPL and XPL image pairs are fused rather than processed separately.
- **DeformCA** — adaptive, geometry-aware cross-modal spatial fusion that tolerates slight
  misregistration between the PPL and XPL captures.
- **SWA** — shifted window attention for efficient long-range context at high resolution.
- **UPerNet decoder** — multi-scale feature aggregation for both large phenocrysts and fine
  accessory grains.
- **IA-WCE loss** — reweights rare mineral classes (e.g. topaz, spinel, tourmaline).
- **State of the art** — **82.50 % mIoU** and **91.48 % mPA**, outperforming the CMX baseline,
  U-Net, DeepLabV3+, and the AMS-ppl / AMS-xpl single-modal models.

---

## Results

| Model | Modality | mIoU (%) | mPA (%) |
|---|---|---|---|
| U-Net | single | 61.35 | 69.36 |
| DeepLabV3+ | single | 59.74 | 68.44 |
| AMS-ppl | XPL | 76.82 | 84.23 |
| AMS-xpl | PPL + XPL | 79.65 | 87.83 |
| CMX (baseline) | PPL + XPL | 73.10 | 82.76 |
| **OTA (ours)** | **PPL + XPL** | **82.50** | **91.48** |

---

## Dataset

| Property | Value |
|---|---|
| Image pairs | 2,090 (PPL + XPL) |
| Source | Three granitic belts, Thailand |
| Rock types | 15 plutonic rock types |
| Mineral classes | 14 |

**Mineral classes:** quartz, K-feldspar, plagioclase, biotite, hornblende, clinopyroxene,
orthopyroxene, olivine, muscovite, leucite, opaque minerals, tourmaline, topaz, spinel.

### Expected directory layout

```
data/
└── OTA/
    ├── PPL/          # plane-polarized light images
    │   ├── 0001.png
    │   └── ...
    ├── XPL/          # cross-polarized light images (same filenames as PPL)
    │   ├── 0001.png
    │   └── ...
    ├── Label/        # single-channel label masks, values 0..13 (255 = ignore)
    │   ├── 0001.png
    │   └── ...
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Each `*.txt` file lists one sample stem per line. PPL, XPL, and label files **must share the
same filename**.

---

## Installation

```bash
git clone https://github.com/Sittiporn-GT/OTA-Optical-Thin-Section-Analysis.git
cd OTA-Optical-Thin-Section-Analysis

conda create -n ota python=3.9 -y
conda activate ota

# install PyTorch matching your CUDA version — see https://pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.13 with CUDA
- CUDA-capable GPU (≥ 16 GB VRAM recommended for batch size 8)
- numpy, opencv-python, einops, timm, tensorboard, tqdm, scipy

---

## Pretrained weights

| Checkpoint | Backbone | mIoU | Link |
|---|---|---|---|
| `ota_mit_b2.pth` | MiT-B2 | 82.50 | _add release link_ |

Place downloaded weights in `checkpoints/`.

---

## Usage

### Training

```bash
python train.py \
    --config configs/ota_config.py \
    --data-root data/OTA \
    --gpus 1
```

### Evaluation

```bash
python eval.py \
    --config configs/ota_config.py \
    --checkpoint checkpoints/ota_mit_b2.pth \
    --split test
```

### Inference on a single PPL/XPL pair

```bash
python predict.py \
    --checkpoint checkpoints/ota_mit_b2.pth \
    --ppl samples/sample_ppl.png \
    --xpl samples/sample_xpl.png \
    --out results/sample_pred.png
```

---

## Training configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 4 × 10⁻⁵ |
| Weight decay | 0.01 |
| Batch size | 8 |
| Epochs | 300 |
| Loss | Imbalance-Aware Weighted Cross-Entropy (IA-WCE) |
| Augmentation | horizontal flip, multi-scale resize, random crop |
| Framework | PyTorch + CUDA |

---

## Architecture

```
PPL ──► Encoder (SWA / Transformer)  ─┐
                                      ├──► DeformCA fusion ──► UPerNet decoder ──► IA-WCE ──► mask
XPL ──► Encoder (SWA / Transformer)  ─┘
```

1. **Dual encoders** extract hierarchical features from PPL and XPL independently.
2. **DeformCA** learns sampling offsets so each modality attends to geometrically
   corresponding regions of the other.
3. **SWA** provides efficient global context across shifted windows.
4. **UPerNet** aggregates the fused multi-scale features into the final segmentation map.

---

## Repository structure

```
.
├── configs/           # experiment configuration files
├── models/
│   ├── encoders/      # SWA / transformer backbones
│   ├── fusion/        # DeformCA module
│   └── decoders/      # UPerNet decoder
├── datasets/          # dataset + augmentation pipeline
├── losses/            # IA-WCE implementation
├── tools/             # visualization and metric utilities
├── train.py
├── eval.py
├── predict.py
└── requirements.txt
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{kongsukho2025ota,
  title   = {Plutonic Mineral Grain Segmentation in Thin-Section Images Using a
             Shifted Dual-Modal Vision Transformer},
  author  = {Kongsukho, Sittiporn and Maneerat, Warunee and Owada, Narihiro and
             Adachi, Tsuyoshi and Vateekul, Peerapon and Sutthirat, Chakkaphan},
  journal = {},
  year    = {},
  doi     = {}
}
```

---

## Acknowledgements

This work builds on [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation),
[UPerNet](https://github.com/CSAILVision/unifiedparsing), and
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
We thank the contributors of these projects.

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Sittiporn Kongsukho — open an [issue](https://github.com/Sittiporn-GT/OTA-Optical-Thin-Section-Analysis/issues)
for questions or bug reports.
