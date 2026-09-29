# OTA — Optical Thin-Section Analysis

**Plutonic Mineral Grain Segmentation in Thin-Section Images Using a Shifted Dual-Modal Vision Transformer**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/Weights-Zenodo-1682D4.svg)](https://doi.org/10.5281/zenodo.23027695)

OTA is a dual-modal semantic segmentation framework that automatically delineates and
classifies mineral grains in thin-section images by jointly exploiting
**plane-polarized light (PPL)** and **cross-polarized light (XPL)** micrographs.

Built on top of the CMX (Cross-Modal Fusion for RGB-X Semantic Segmentation) framework,
OTA introduces **Deformable Cross-Attention (DeformCA)**, **Shifted Window Attention (SWA)**,
a **UPerNet** multi-scale decoder, and an **Imbalance-Aware Weighted Cross-Entropy (IA-WCE)**
loss to handle the extreme class imbalance.

---

## Table of contents

- [Highlights](#highlights)
- [Results](#results)
- [Ablation study and pretrained weights](#ablation-study-and-pretrained-weights)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training configuration](#training-configuration)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

---

## Highlights

- **Dual-modal input** — PPL and XPL image pairs are fused rather than processed separately.
- **DeformCA** — adaptive, geometry-aware cross-modal spatial fusion that tolerates slight
  misregistration between the PPL and XPL captures.
- **SWA** — shifted window attention for efficient long-range context at high resolution.
- **UPerNet decoder** — multi-scale feature aggregation for both large phenocrysts and fine
  accessory grains.
- **IA-WCE loss** — reweights rare mineral classes such as topaz, spinel and tourmaline.
- **State of the art** — **82.50 % mIoU** and **91.48 % mPA**, outperforming the CMX baseline,
  U-Net, DeepLabV3+, and the AMS-ppl / AMS-xpl single-modal models.

---

## Results

| Model | Modality | mIoU (%) | mPA (%) |
|---|---|---|---|
| U-Net | PPL only | 61.35 | 69.36 |
| DeepLabV3+ | PPL only | 59.74 | 68.44 |
| AMS | PPL only | 76.82 | 84.23 |
| AMS | PPL + XPL | 79.65 | 87.83 |
| CMX (baseline) | PPL + XPL | 73.98 | 82.76 |
| **OTA (ours)** | **PPL + XPL** | **82.50** | **91.48** |

OTA improves over the CMX baseline by **+8.52 pp mIoU** and over the strongest dual-modal
model (AMS-p/xpl) by **+2.85 pp mIoU**, while also raising mean pixel accuracy above 91 %.

---

## Ablation study and pretrained weights

Each row adds one component on top of the previous configuration, so the table doubles as the
ablation study reported in the paper.

| Configuration | Backbone | mIoU (%) | Gain | Download |
|---|---|---|---|---|
| CMX (baseline) | MiT-B2 | 73.98 | — | [Checkpoint](https://doi.org/10.5281/zenodo.23027010) |
| + UPerNet | MiT-B2 | 74.48 | +0.50 | [Checkpoint](https://doi.org/10.5281/zenodo.23027435) |
| + IA-WCE | MiT-B2 | 76.21 | +1.73 | [Checkpoint](https://doi.org/10.5281/zenodo.23027569) |
| + DeformCA | MiT-B2 | 80.13 | +3.92 | [Checkpoint](https://doi.org/10.5281/zenodo.23027621) |
| **+ SWA (= OTA)** | **MiT-B2** | **82.50** | **+2.37** | [**Checkpoint**](https://doi.org/10.5281/zenodo.23027695) |

Place the downloaded `.pth` files in `checkpoints/` before running evaluation.

---

## Dataset

| Property | Value |
|---|---|
| Thin-Section Image pairs | 2,090 (PPL + XPL) |
| Source | Three granitic belts, Thailand |
| Rock types | 15 plutonic rock types |
| Mineral classes | 14 |

### Class index

| ID | Mineral | ID | Mineral |
|---|---|---|---|
| 1 | quartz | 8 | orthopyroxene |
| 2 | K-feldspar | 9 | olivine |
| 3 | plagioclase | 10 | muscovite |
| 4 | biotite | 11 | leucite |
| 5 | hornblende | 12 | opaque minerals |
| 6 | clinopyroxene | 13 | tourmaline |
| 7 | topaz | 14 | spinel |

Label masks are single-channel PNG with values `1–14`; `0` is background.

### Expected directory layout

```
data/
└── OTA/
    ├── PPL/          
    │   ├── 01_granite.png
    │   └── ...
    ├── XPL/          
    │   ├── 01_granite.png
    │   └── ...
    ├── Label/       
    │   ├── 01_granite.png
    │   └── ...
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Each `*.txt` file lists one sample stem per line. PPL, XPL, and label files **must share the
same filename**.

### Download

(Demo test) Dataset archived on Zenodo: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

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

## Usage

### Training

```bash
python train.py -d=0
```

### Evaluation

```bash
python eval.py -d=0 -e=ota -p/results/ota/
```

The script reports per-class IoU, mIoU, per-class pixel accuracy and mPA over the 14 mineral
classes.

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

## Related work

- [PViT — Petrographic Vision Transformer](https://github.com/Sittiporn-GT/PViT-Petrographic-Vision-Transformer)
  — classification of 15 plutonic rock types from thin-section images, a complementary task
  to the grain-level segmentation performed here.

---

## Acknowledgements

This research was supported by the **Development and Promotion of Science and Technology
Talented Project (DPST)**, the **Institute for the Promotion of Teaching Science and
Technology (IPST)**, and the **90th Anniversary of Chulalongkorn University Scholarship**
under the Ratchadapisek Somphot Endowment Fund.

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Sittiporn Kongsukho — open an [issue](https://github.com/Sittiporn-GT/OTA-Optical-Thin-Section-Analysis/issues)
for questions or bug reports.
