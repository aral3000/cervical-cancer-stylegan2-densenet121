# Peningkatan Kinerja Klasifikasi Abnormalitas Citra Serviks Menggunakan Teknik Augmentasi Citra Sintetis StyleGAN2

[![English README](https://img.shields.io/badge/Language-English-blue.svg)](README.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-ff6f00.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📖 **Bahasa / Language**: [English Version (Main README)](README.md) | **Bahasa Indonesia**

---

## 📌 Metadata Penelitian

- **Judul Riset**: Peningkatan Kinerja Klasifikasi Abnormalitas Citra Serviks Menggunakan Teknik Augmentasi Citra Sintetis StyleGAN2
- **Penulis**: **M. Athallah Rafif Aldera**
- **Program Studi**: S1 Sistem Komputer, Fakultas Ilmu Komputer, Universitas Sriwijaya
- **Kelompok Riset**: **ISys Research Group**
- **Domain**: Medical Image Processing, Generative Adversarial Networks (GANs), Computer Vision

---

## 📄 Abstrak & Latar Belakang

Deteksi dini kanker serviks melalui citra medis **Inspeksi Visual Asam Asetat (VIA)** sering kali mengalami tantangan berupa ketidakseimbangan kelas (*class imbalance*) dan keterbatasan jumlah data medis yang teranotasi secara akurat. Pengumpulan dataset citra medis berkualitas tinggi terhambat oleh masalah privasi pasien, kelangkaan sampel abnormal, serta biaya anotasi spesialis yang tinggi.

Penelitian skripsi ini menerapkan **StyleGAN2 dengan Adaptive Data Augmentation (ADA)** untuk memproduksi citra sintetis serviks abnormal beresolusi $128 \times 128$ piksel berkualitas tinggi guna memperkuat data latih model klasifikasi Convolutional Neural Network (CNN).

### Kontribusi Utama:
1. **Multi-Stage Training StyleGAN2-ADA**: Melatih generator generatif secara bertahap pada *limited data regime* (666 sampel citra medis) tanpa mengalami kolaps diskriminator (*discriminator overfitting*).
2. **Komparasi Generatif**: Membandingkan kualitas visual dan konvergensi metrik **Fréchet Inception Distance (FID)** antara **StyleGAN2-ADA** (FID terbaik **59.49**) dan **StyleGAN3**.
3. **Pengujian Klasifikasi CNN (DenseNet121)**: Menerapkan protokol **Zero Data-Leakage** (Split 80:10:10) untuk menguji dampak augmentasi citra sintetis pada skenario *Baseline* (Skenario A) vs *Augmented* (Skenario B).

---

## 🏗️ Arsitektur Sistem & Metodologi

```
+------------------+     +-----------------------+     +-----------------------+
|  Dataset Serviks | --> | Pelatihan StyleGAN2-  | --> | Ekstraksi Citra       |
|  Asli (666 VIA)  |     | ADA (Multi-Stage)     |     | Sintetis (Target 666) |
+------------------+     +-----------------------+     +-----------------------+
                                                                   |
                                                                   v
+------------------+     +-----------------------+     +-----------------------+
| Evaluasi Test Set| <-- | Pelatihan DenseNet121 | <-- | Pembagian Skenario    |
| (Zero Leakage)   |     | CNN Classifier        |     | A (Base) vs B (Aug)   |
+------------------+     +-----------------------+     +-----------------------+
```

### 1. StyleGAN2-ADA
- **Mapping Network**: 8-layer Fully Connected Network memetakan $z \in \mathcal{Z} \to w \in \mathcal{W}$.
- **Synthesis Network**: Operasi *weight demodulation* untuk eliminasi artefak tetesan air (*droplet artifacts*).
- **ADA Mechanism**: Pengaturan dinamis probabilitas augmentasi ($p$) untuk proteksi diskriminator pada data terbatas.

### 2. DenseNet121 Classifier
- **Feature Extractor**: Backbone DenseNet121 (*pretrained ImageNet*).
- **Custom Classification Head**: `GlobalAveragePooling2D` $\to$ `Dropout(0.5)` $\to$ `Dense(128, activation='relu')` $\to$ `Dense(1, activation='sigmoid')`.

---

## 📊 Hasil Eksperimen Utama

| Evaluasi Model | Metrik Utama | Nilai / Hasil | Keterangan |
| :--- | :--- | :--- | :--- |
| **StyleGAN2-ADA** | FID Terbaik | **59.49** (kimg 4780) | Penurunan FID konsisten dari 89.23 (Tahap 1) |
| **StyleGAN3** | FID Evaluasi | 110.2 - 145.8 | Kurang stabil pada data terbatas tanpa ADA |
| **DenseNet121 (Skenario A)**| Baseline Accuracy | ~82.5% | Latih pada citra asli terbatas |
| **DenseNet121 (Skenario B)**| Augmented Accuracy | **~91.2%** | Disuntikkan citra sintetis StyleGAN2-ADA |

---

## 📁 Struktur Repositori

```text
cervical-cancer-stylegan2-densenet121/
├── .gitignore                          # Pengaturan Pengecualian Git (Keamanan Data)
├── README.md                           # Utama (Bahasa Inggris)
├── README_ID.md                        # Versi Bahasa Indonesia
├── requirements.txt                    # Dependensi Python
├── docs/                               # Naskah & Laporan Akademik (Markdown)
│   ├── bab1_pendahuluan.md
│   ├── bab4_hasil_dan_pembahasan.md
│   ├── bab5_kesimpulan_dan_saran.md
│   ├── laporan_metodologi_eksperimen.md
│   └── panduan_mendeley.md
├── notebooks/                          # Jupyter Notebooks Eksperimen
│   ├── 01_stylegan2_ada_training.ipynb
│   ├── 02_stylegan2_dashboard_evaluation.ipynb
│   ├── 03_stylegan3_comparison.ipynb
│   ├── 04_stylegan3_dashboard_visualization.ipynb
│   ├── 05_densenet121_classification.ipynb
│   └── 06_kaggle_sota_visualization.ipynb
├── src/                                # Source Code Python Modular
│   ├── models/
│   │   ├── fix_checkpoint.py
│   │   ├── generate_tahap5.py
│   │   └── generator.py
│   └── visualization/
│       ├── build_dashboard.py
│       ├── create_dcgan_progression.py
│       ├── run_visualisasi.py
│       └── visualisasi_sg3.py
└── figures/                            # Diagram & Visualisasi Hasil Eksperimen
    ├── diagram_alur_penelitian.png
    ├── classification_training_curves.png
    ├── confusion_matrix_comparison.png
    ├── stylegan2_ada_pipeline.png
    └── stylegan2_awal_akhir_comparison.png
```

---

## ✉️ Kontak & Lisensi

- **Penulis**: M. Athallah Rafif Aldera
- **Afiliasi**: Universitas Sriwijaya & ISys Research Group
- **Lisensi**: Proyek ini dilindungi di bawah lisensi MIT.
