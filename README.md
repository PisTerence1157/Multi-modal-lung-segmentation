<img width="622" height="379" alt="Screenshot 2026-04-24 at 12 03 53" src="https://github.com/user-attachments/assets/928b2ded-8a6b-4a6e-84f1-17eb45f84af7" />
# Multi-modal Lung Segmentation (CXR + CT)

> MSc dissertation — a PyTorch + MONAI pipeline for lung segmentation across 2D Chest X-Ray and 3D CT modalities, benchmarking **six segmentation architectures** with rigorous multi-seed evaluation.

![CT 2D vs 3D Lung Segmentation](dissertation_figures/fig_ct_2d_vs_3d_1_3_6_1_4_1_14519_5_2_1_6279_6001_105756.png)

*Qualitative comparison on a LUNA16 CT volume — original slice, ground truth, 2D U-Net, and 3D U-Net predictions across 5 representative slices.*

## Highlights

- **6 architectures** implemented from scratch: U-Net, Attention U-Net, SE-U-Net, CBAM-U-Net, SegFormer, 3D U-Net
- **Multi-modal** pipeline: 2D CXR (1,408 images) + 3D CT volumes (356 cases, LUNA16)
- **Rigorous evaluation**: 3 random seeds per experiment, mean ± std reported, paired significance tests
- **Critical finding**: 3D U-Net underperforms 2D U-Net on LUNA16 (0.9418 vs 0.9716 Dice) under matched compute, despite 10× longer training
- **Reproducible**: YAML-driven configs, end-to-end Jupyter notebooks, fixed seeds

## Results

### CXR Lung Field Segmentation — Validation Dice (3 seeds, mean ± std)

| Model               | Darwin              | Montgomery          | Shenzhen            | Avg        |
| ------------------- | ------------------- | ------------------- | ------------------- | ---------- |
| U-Net               | 0.9438 ± 0.0288     | 0.9636 ± 0.0074     | 0.9568 ± 0.0013     | 0.9548     |
| Attention U-Net     | 0.9613 ± 0.0007     | 0.9570 ± 0.0053     | 0.9535 ± 0.0020     | 0.9573     |
| **SE-U-Net**        | **0.9627 ± 0.0005** | **0.9666 ± 0.0005** | 0.9525 ± 0.0028     | **0.9606** |
| CBAM-U-Net          | 0.9617 ± 0.0015     | 0.9633 ± 0.0055     | 0.9541 ± 0.0014     | 0.9597     |

### LUNA16 CT — 2D vs 3D U-Net

| Model    | Best Val Dice | Epochs to converge |
| -------- | ------------- | ------------------ |
| 2D U-Net | **0.9716**    | 3                  |
| 3D U-Net | 0.9418        | 28                 |

### Model Complexity

All four CXR variants share comparable size (~13.4M params, ~51 MB) — performance gains come from the **attention modules**, not capacity.

## Datasets

| Dataset                | Modality | Cases | Split (Train/Val/Test) |
| ---------------------- | -------- | ----- | ---------------------- |
| Kaggle CXR             | 2D X-Ray | 704   | 492 / 105 / 107        |
| Shenzhen Hospital      | 2D X-Ray | 566   | 396 / 84 / 86          |
| Montgomery County      | 2D X-Ray | 138   | 96 / 20 / 22           |
| LUNA16                 | 3D CT    | 356   | 249 / 53 / 54          |

Public datasets — links: [Montgomery & Shenzhen](https://openi.nlm.nih.gov/), [LUNA16](https://luna16.grand-challenge.org/).

## Project Structure

```
.
├── models/        # U-Net, Attention U-Net, SE, CBAM, SegFormer, 3D U-Net
├── engine/        # Training loop
├── scripts/       # Train / preprocess / evaluate / visualize entry points
├── utils/         # Loss functions (Dice + BCE), metrics
├── configs/       # YAML configurations per model
├── datasets/      # Dataset classes (CXR + LUNA16 CT)
├── notebooks/     # End-to-end walkthroughs (data prep → training → evaluation)
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

python scripts/preprocess_cxr.py
python scripts/train.py --config configs/se_unet.yaml
python scripts/evaluate.py --config configs/se_unet.yaml
```

For 3D CT:

```bash
python scripts/preprocess_luna16.py
python scripts/train_ct.py --config configs/unet3d.yaml
```

## Tech Stack

PyTorch · MONAI · Albumentations · OpenCV · NumPy · Matplotlib · TensorBoard

## Author

Shuai — MSc Dissertation, 2026
