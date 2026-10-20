# Enhancing Cervical Abnormality Classification Performance via StyleGAN2 Synthetic Image Augmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-ff6f00.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Bahasa Indonesia](https://img.shields.io/badge/Language-Bahasa%20Indonesia-green.svg)](README_ID.md)

> 📖 **Language**: **English** | [Bahasa Indonesia Version](README_ID.md)

---

## 📌 Research Metadata

- **Research Title**: Enhancing Cervical Abnormality Classification Performance Using StyleGAN2 Synthetic Image Augmentation Techniques (*Peningkatan Kinerja Klasifikasi Abnormalitas Citra Serviks Menggunakan Teknik Augmentasi Citra Sintetis StyleGAN2*)
- **Author**: **M. Athallah Rafif Aldera**
- **Degree**: B.S. in Computer Engineering, Faculty of Computer Science, **Sriwijaya University**
- **Research Group**: **ISys Research Group**
- **Domain**: Medical Image Processing, Generative Adversarial Networks (GANs), Computer Vision, Deep Learning

---

## 📄 Abstract & Research Background

Early detection of cervical cancer via **Visual Inspection with Acetic Acid (VIA)** image analysis frequently faces severe challenges due to extreme **class imbalance** and scarce annotated medical image datasets. Collecting high-quality annotated medical images is inherently constrained by patient privacy protocols, the rarity of abnormal tissue samples, and the high cost of expert pathological annotations.

This undergraduate thesis introduces an end-to-end framework leveraging **StyleGAN2 with Adaptive Data Augmentation (ADA)** to synthesize high-resolution ($128 \times 128$ pixels) abnormal cervical tissue images. The synthetic images serve as targeted data augmentation to boost the classification accuracy of a **DenseNet121** Convolutional Neural Network (CNN).

### Key Contributions:
1. **Multi-Stage Generative Benchmark**: Conducted rigorous comparative analysis of visual quality and **Fréchet Inception Distance (FID)** score convergence across three generative architectures: **StyleGAN2-ADA** (achieving SOTA FID of **49.62** at 15,240 Kimg), **StyleGAN3** (FID **62.66** at 2,000 Kimg), and conventional **DCGAN** (FID **210.01**).
2. **Synthetic Data Efficacy in Limited Data Regime**: Evaluated whether 100% synthetic substitution of positive clinical samples can train a high-performing classifier without discriminator overfitting on a small dataset (666 real clinical images).
3. **CNN Classification Evaluation (DenseNet121)**: Formulated a strict **Zero Data-Leakage Protocol** (80:10:10 Train:Val:Test split) on 997 total clinical images. DenseNet121 achieved **95.00% test accuracy**, **96% F1-score**, and **100.00% positive precision / specificity** under 100% synthetic substitution (Scenario B), compared to **97.00% test accuracy** on the all-real baseline (Scenario A).
4. **Clinical Domain Gap Analysis**: Documented failure modes and subtle macro-geometric distortions (such as speculum discontinuities and cervical os boundary artifacts) that account for the 2.0% diagnostic performance delta.

---

## 🏗️ System Architecture & Workflow

```
+-------------------+     +------------------------+     +------------------------+
|  Original Cervical| --> |  StyleGAN2-ADA         | --> |  Synthetic Image       |
|  Dataset (666 VIA)|     |  Adaptive Training     |     |  Extraction (N = 666)  |
+-------------------+     +------------------------+     +------------------------+
                                                                     |
                                                                     v
+-------------------+     +------------------------+     +------------------------+
|  Test Evaluation  | <-- |  DenseNet121 CNN       | <-- |  Experimental Split    |
| (Unseen Real Test)|     |  Classifier Training   |     | (Scenario A vs B)      |
+-------------------+     +------------------------+     +------------------------+
```

### 1. StyleGAN2-ADA (Generative Synthesis)
- **Mapping Network ($F$)**: 8-layer Fully Connected Network mapping latent noise $z \in \mathcal{Z} \to w \in \mathcal{W}$ to achieve disentangled latent representations.
- **Synthesis Network ($G$)**: Weight demodulation operations eliminate water-droplet artifacts while retaining smooth spatial details.
- **Adaptive Data Augmentation (ADA)**: Dynamically adjusts augmentation probability ($p$) based on discriminator overfitting heuristics, maintaining non-leaking augmentations.

### 2. DenseNet121 Classifier (Deep Learning Architecture)
- **Backbone**: ImageNet pretrained DenseNet121 feature extractor (frozen during transfer learning).
- **Custom Classification Head**: `GlobalAveragePooling2D` -> `Dropout(0.5)` -> `Dense(128, activation='relu')` -> `Dense(1, activation='sigmoid')`.

---

## 📊 Empirical Results & Benchmarks

### 1. Generative Architecture Comparison (Image Fidelity)

| Model Architecture | Mapping Layers | Optimizer | Best FID Score | Convergence Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **DCGAN (Baseline)** | 0 (Direct $z$) | Adam ($\alpha=5\times 10^{-5}$) | **210.01** | Severe checkerboard artifacts, saturated after epoch 250 |
| **StyleGAN3** | 2 | Adam ($\alpha=0.0025$) | **62.66** (Kimg 2,000) | Diverged after 2,200 Kimg due to unaligned micro-textures |
| **StyleGAN2-ADA** | 8 | Adam ($\alpha=0.0025$, $R_1=10$) | 🏆 **49.62** (Kimg 15,240) | Highly stable convergence without mode collapse |

### 2. CNN Classification Performance on Unseen Real Test Set ($N=102$)

| Scenario | Training Set Composition | Test Set | Test Accuracy | Precision (Pos) | Recall (Pos) | Specificity | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Scenario A (Baseline)** | 264 Real Normal + 531 Real Positive | 34 Normal + 68 Positive (Real) | **97.00%** | 98.51% | 97.06% | 97.06% | 0.98 |
| **Scenario B (Synthetic Efficacy)** | 264 Real Normal + **666 Synthetic Positive** | 34 Normal + 68 Positive (Real) | **95.00%** | **100.00%** | 92.65% | **100.00%** | 0.96 |

---

## 📁 Repository Structure

```text
cervical-cancer-stylegan2-densenet121/
├── .gitignore                          # Strict security ignore rules (Weights/Datasets)
├── README.md                           # Main English Repository Documentation
├── README_ID.md                        # Indonesian Language Version Documentation
├── requirements.txt                    # Python Dependencies List
├── infer.py                            # CLI Inference Tool for DenseNet121 Evaluation
├── test_infer.py                       # Self-contained Assert-based Verification Suite
├── docs/                               # Academic Research Documentation (Pure Markdown)
│   ├── abstrak_skripsi.md
│   ├── bab1_pendahuluan.md
│   ├── bab4_hasil_dan_pembahasan.md
│   ├── bab5_kesimpulan_dan_saran.md
│   ├── laporan_metodologi_eksperimen.md
│   └── panduan_mendeley.md
├── notebooks/                          # Experimental Jupyter Notebooks
│   ├── 01_stylegan2_ada_training.ipynb
│   ├── 02_stylegan2_dashboard_evaluation.ipynb
│   ├── 03_stylegan3_comparison.ipynb
│   ├── 04_stylegan3_dashboard_visualization.ipynb
│   ├── 05_densenet121_classification.ipynb
│   └── 06_kaggle_sota_visualization.ipynb
├── src/                                # Modular Python Source Code
│   ├── models/                         # Model Architectures & Generators
│   │   ├── fix_checkpoint.py
│   │   ├── generate_tahap5.py
│   │   └── generator.py
│   └── visualization/                  # Dashboards & Plotting Scripts
│       ├── build_dashboard.py
│       ├── create_dcgan_progression.py
│       ├── run_visualisasi.py
│       └── visualisasi_sg3.py
└── figures/                            # High-Res Workflow Diagrams & Benchmark Plots
    ├── classification_training_curves.png
    ├── confusion_matrix_comparison.png
    ├── diagram_alur_penelitian.png
    ├── stylegan2_ada_pipeline.png
    └── stylegan2_awal_akhir_comparison.png
```

---

## 🚀 Quick Start & CLI Usage

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/aral3000/cervical-cancer-stylegan2-densenet121.git
cd cervical-cancer-stylegan2-densenet121

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Single-Image CLI Inference (`infer.py`)
Run real-time abnormality evaluation on any cervical VIA image:

```bash
# Production evaluation with trained weights
python infer.py --image path/to/cervical_sample.jpg --weights checkpoints/densenet121_best.h5

# Quick smoke-test / deterministic evaluation mode
python infer.py --image path/to/sample.jpg --mock

# JSON output mode (for API integration or automated pipelines)
python infer.py --image path/to/sample.jpg --mock --json
```

Sample JSON response:
```json
{
  "status": "success",
  "mode": "production_densenet121",
  "image_path": "path/to/sample.jpg",
  "original_resolution": "640x480",
  "input_resolution": "128x128",
  "probability": 0.8942,
  "threshold": 0.5,
  "prediction": "Positive (Abnormal Cervical Lesion)",
  "class_id": 1,
  "confidence": 0.8942,
  "latency_ms": 14.8
}
```

### 3. Run Verification Test Suite
```bash
python test_infer.py
```

### 4. Interactive Jupyter Notebooks
To run the full CNN training and ablation pipeline:
```bash
jupyter notebook notebooks/05_densenet121_classification.ipynb
```

---

## 🔒 Data Privacy & Model Weight Policy

In compliance with medical research data governance standards:
- **Raw Medical Datasets** (`.zip`, raw image archives) and **Heavy Model Snapshots** (`.pkl`, `.h5`, `.pth` weights >100MB) are **strictly excluded** from GitHub tracking via `.gitignore`.
- Trained model checkpoints and synthetic dataset samples are available upon academic research request to the author.

---

## 📚 Citation

If you use this research codebase or methodology in your research, please cite:

```bibtex
@thesis{aldera2026stylegan2cervical,
  author       = {M. Athallah Rafif Aldera},
  title        = {Enhancing Cervical Abnormality Classification Performance Using StyleGAN2 Synthetic Image Augmentation Techniques},
  school       = {Department of Computer Engineering, Sriwijaya University},
  year         = {2026},
  note         = {ISys Research Group}
}
```

---

## ✉️ Contact & License

- **Author**: M. Athallah Rafif Aldera
- **Affiliation**: Sriwijaya University & ISys Research Group
- **License**: Released under the [MIT License](LICENSE).
