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
1. **Benchmark Tiga Model Generatif**: Melakukan komparasi komprehensif antara **StyleGAN2-ADA** (mencapai skor FID terbaik **49.62** pada 15.240 Kimg), **StyleGAN3** (FID **62.66** pada 2.000 Kimg), dan model konvensional **DCGAN** (FID **210.01**).
2. **Uji Efikasi Data Sintetis pada Data Terbatas**: Mengevaluasi transfer fitur diagnostik saat 100% citra abnormal serviks pada data latih disubstitusi dengan citra sintetis buatan generator StyleGAN2-ADA.
3. **Pengujian Klasifikasi CNN (DenseNet121)**: Menerapkan protokol **Zero Data-Leakage** (Split 80:10:10) pada 997 citra klinis. Model Skenario B (100% substitusi sintetis) mencapai akurasi uji **95.00%**, F1-score **96%**, serta **presisi positif dan spesifisitas 100.00%** (dibandingkan baseline data asli Skenario A dengan akurasi **97.00%**).
4. **Analisis Kausalitas Domain Gap Medis**: Mengidentifikasi kegagalan visual spesifik pada data generatif (seperti bentuk spekulum terdistorsi dan batas mulut rahim kabur) yang memicu selisih performa 2.0%.

---

## 🏗️ Arsitektur Sistem & Metodologi

```
+------------------+     +-----------------------+     +-----------------------+
|  Dataset Serviks | --> | Pelatihan StyleGAN2-  | --> | Ekstraksi Citra       |
|  Asli (666 VIA)  |     | ADA (Multi-Stage)     |     | Sintetis (N = 666)    |
+------------------+     +-----------------------+     +-----------------------+
                                                                   |
                                                                   v
+------------------+     +-----------------------+     +-----------------------+
| Evaluasi Test Set| <-- | Pelatihan DenseNet121 | <-- | Pembagian Skenario    |
| (102 Citra Asli) |     | CNN Classifier        |     | Skenario A vs B       |
+------------------+     +-----------------------+     +-----------------------+
```

### 1. StyleGAN2-ADA
- **Mapping Network**: 8-layer Fully Connected Network memetakan $z \in \mathcal{Z} \to w \in \mathcal{W}$.
- **Synthesis Network**: Operasi *weight demodulation* untuk eliminasi artefak tetesan air (*droplet artifacts*).
- **ADA Mechanism**: Pengaturan dinamis probabilitas augmentasi ($p$) untuk proteksi diskriminator pada data terbatas.

### 2. DenseNet121 Classifier
- **Feature Extractor**: Backbone DenseNet121 (*pretrained ImageNet*, bobot awal dibekukan / frozen).
- **Custom Classification Head**: `GlobalAveragePooling2D` -> `Dropout(0.5)` -> `Dense(128, activation='relu')` -> `Dense(1, activation='sigmoid')`.

---

## 📊 Hasil Eksperimen Utama

### 1. Kualitas Generatif & Nilai FID

| Arsitektur Model | Layer Mapping | Optimizer | Skor FID Terbaik | Catatan Konvergensi |
| :--- | :---: | :---: | :---: | :--- |
| **DCGAN (Baseline)** | 0 (Vektor $z$ langsung) | Adam ($\alpha=5\times 10^{-5}$) | **210.01** | Mengalami *mode collapse* dan artefak kotak-kotak |
| **StyleGAN3** | 2 | Adam ($\alpha=0.0025$) | **62.66** (Kimg 2.000) | Divergen setelah 2.200 Kimg karena lesi mikro tidak selaras |
| **StyleGAN2-ADA** | 8 | Adam ($\alpha=0.0025$, $R_1=10$) | 🏆 **49.62** (Kimg 15.240) | Konvergen stabil tanpa overfitting berkat ADA |

### 2. Kinerja Klasifikasi DenseNet121 pada Test Set Asli ($N=102$)

| Skenario | Komposisi Data Latih | Test Set Asli | Akurasi | Presisi (Positif) | Recall (Positif) | Spesifisitas | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Skenario A (Baseline)** | 264 Asli Normal + 531 Asli Positif | 34 Normal + 68 Positif | **97.00%** | 98.51% | 97.06% | 97.06% | 0.98 |
| **Skenario B (Data Sintetis)** | 264 Asli Normal + **666 Sintetis Positif** | 34 Normal + 68 Positif | **95.00%** | **100.00%** | 92.65% | **100.00%** | 0.96 |

---

## 📁 Struktur Repositori

```text
cervical-cancer-stylegan2-densenet121/
├── .gitignore                          # Pengaturan Pengecualian Git (Keamanan Data)
├── README.md                           # Utama (Bahasa Inggris)
├── README_ID.md                        # Versi Bahasa Indonesia
├── requirements.txt                    # Dependensi Python
├── infer.py                            # Alat Inferensi CLI DenseNet121
├── test_infer.py                       # Pengujian Otomatis Mandiri (Self-Check)
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
└── figures/                            # Gambar & Kurva Pelatihan
```

---

## 🚀 Panduan Eksekusi CLI & Inferensi

```bash
# 1. Jalankan inferensi cepat / smoke test (JSON output)
python infer.py --image path/ke/citra_serviks.jpg --mock --json

# 2. Jalankan pengujian verifikasi mandiri
python test_infer.py
```
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
