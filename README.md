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
1. **Multi-Stage StyleGAN2-ADA Training**: Implemented progressive multi-stage training in a limited data regime (666 original medical images) without incurring discriminator overfitting or collapse.
2. **Generative Model Benchmark**: Conducted rigorous comparative analysis of visual quality and **Fréchet Inception Distance (FID)** score convergence between **StyleGAN2-ADA** (achieving a peak FID of **59.49**) and **StyleGAN3**.
3. **CNN Classification Evaluation (DenseNet121)**: Formulated a strict **Zero Data-Leakage Protocol** (80:10:10 Train:Val:Test split) to evaluate Baseline (Scenario A) vs. Synthetic-Augmented (Scenario B) classification performance.

---

## 🏗️ System Architecture & Workflow

```
+-------------------+     +------------------------+     +------------------------+
|  Original Cervical| --> |  StyleGAN2-ADA         | --> |  Synthetic Image       |
|  Dataset (666 VIA)|     |  Multi-Stage Training  |     |  Extraction (N = 666)  |
+-------------------+     +------------------------+     +------------------------+
                                                                     |
                                                                     v
+-------------------+     +------------------------+     +------------------------+
|  Test Evaluation  | <-- |  DenseNet121 CNN       | <-- |  Experimental Split    |
| (Zero Leakage)    |     |  Classifier Training   |     | (Scenario A vs B)      |
+-------------------+     +------------------------+     +------------------------+
```

### 1. StyleGAN2-ADA (Generative Synthesis)
- **Mapping Network ($F$)**: 8-layer Fully Connected Network mapping latent noise $z \in \mathcal{Z} \to w \in \mathcal{W}$ to achieve disentangled latent representations.
- **Synthesis Network ($G$)**: Weight demodulation operations eliminate water-droplet artifacts while retaining smooth spatial details.
- **Adaptive Data Augmentation (ADA)**: Dynamically adjusts augmentation probability ($p$) based on discriminator overfitting heuristics, maintaining non-leaking augmentations.

### 2. DenseNet121 Classifier (Deep Learning Architecture)
- **Backbone**: ImageNet pretrained DenseNet121 feature extractor.
- **Custom Classification Head**: `GlobalAveragePooling2D` $\to$ `Dropout(0.5)` $\to$ `Dense(128, activation='relu')` $\to$ `Dense(1, activation='sigmoid')`.

---

## 📊 Empirical Results & Benchmarks

| Model Architecture | Evaluation Metric | Result / Score | Research Insights |
| :--- | :--- | :--- | :--- |
| **StyleGAN2-ADA** | Best FID Score | **59.49** (kimg 4780) | Consistent improvement from Stage 1 FID of 89.23 |
| **StyleGAN3** | Evaluation FID | 110.2 - 145.8 | Less stable on small data regimes without ADA |
| **DenseNet121 (Scenario A)** | Baseline Test Accuracy | **82.5%** | Trained solely on original limited images |
| **DenseNet121 (Scenario B)** | Augmented Test Accuracy | **91.2%** | Injected with StyleGAN2-ADA synthetic images |

---

## 📁 Repository Structure

```text
cervical-cancer-stylegan2-densenet121/
├── .gitignore                          # Strict security ignore rules (Weights/Datasets)
├── README.md                           # Main English Repository Documentation
├── README_ID.md                        # Indonesian Language Version Documentation
├── requirements.txt                    # Python Dependencies List
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

## 🚀 Quick Start & Installation Guide

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/aral3000/cervical-cancer-stylegan2-densenet121.git
cd cervical-cancer-stylegan2-densenet121

# Create virtual environment
python -m venv venv
# Activate environment (Windows)
venv\Scripts\activate
# Activate environment (Linux/macOS)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute DenseNet121 Classification Notebook
To run the CNN baseline vs. augmented evaluation pipeline:
```bash
jupyter notebook notebooks/05_densenet121_classification.ipynb
```

### 3. Generate StyleGAN2 Evaluation Dashboard
```bash
python src/visualization/build_dashboard.py
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
