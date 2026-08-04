import os, glob, json, re, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from PIL import Image

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
OUT_DIR = r'c:\new era code\GAN_Kompre\Workspace_StyleGAN2\output grafik'
os.makedirs(OUT_DIR, exist_ok=True)

WORKSPACE = r'c:\new era code\GAN_Kompre\Workspace_StyleGAN2\output stylegan2'
tahap_folders = sorted(glob.glob(os.path.join(WORKSPACE, 'tahap *')))
print(f'Tahap ditemukan: {len(tahap_folders)}')

# ─── HARVEST LOG ───────────────────────────────────────────────────────────────
def parse_plain(line):
    d = {}
    for k, p in [('kimg', r'kimg\s+([\d\.]+)'), ('sec/kimg', r'sec/kimg\s+([\d\.]+)'),
                  ('augment', r'augment\s+([\d\.]+)'), ('cpumem', r'cpumem\s+([\d\.]+)'),
                  ('gpumem', r'gpumem\s+([\d\.]+)')]:
        m = re.search(p, line)
        if m: d[k] = float(m.group(1))
    return d if 'kimg' in d else None

rows, fid_rows = [], []
kimg_off = 0.0
for folder in tahap_folders:
    lf = os.path.join(folder, 'log.txt')
    local_max = 0.0
    if os.path.exists(lf):
        with open(lf) as f:
            for raw in f:
                line = raw.strip()
                if line.startswith('{"tick"'):
                    try:
                        d = json.loads(line)
                        k = float(d.get('kimg', 0)) + kimg_off
                        rows.append({'Kimg': k,
                            'Loss_Fake': d.get('Loss/scores/fake'),
                            'Loss_Real': d.get('Loss/scores/real'),
                            'Sec': d.get('Timing/sec_per_kimg', d.get('sec/kimg', 0)),
                            'Aug': d.get('Progress/augment', d.get('Augment/p', 0)),
                            'Sign_Real': d.get('Loss/signs/real'),
                            'Sign_Fake': d.get('Loss/signs/fake'),
                            'R1': d.get('Loss/r1_penalty'),
                            'PL': d.get('Loss/pl_penalty'),
                            'CPU': d.get('Resources/cpu_mem_gb', 0),
                            'GPU': d.get('Resources/peak_gpu_mem_gb', 0)})
                        local_max = max(local_max, float(d.get('kimg', 0)))
                    except: pass
                elif line.startswith('tick '):
                    d = parse_plain(line)
                    if d:
                        k = d['kimg'] + kimg_off
                        rows.append({'Kimg': k, 'Loss_Fake': None, 'Loss_Real': None,
                            'Sec': d.get('sec/kimg', 0), 'Aug': d.get('augment', 0),
                            'Sign_Real': None, 'Sign_Fake': None, 'R1': None, 'PL': None,
                            'CPU': d.get('cpumem', 0), 'GPU': d.get('gpumem', 0)})
                        local_max = max(local_max, d['kimg'])
        if rows: kimg_off = rows[-1]['Kimg']

    ff = os.path.join(folder, 'metric-fid50k_full.jsonl')
    if os.path.exists(ff):
        with open(ff) as f:
            for line in f:
                d = json.loads(line)
                if 'results' in d and 'fid50k_full' in d['results']:
                    pkl = d.get('snapshot_pkl', '')
                    try: k = float(pkl.split('-')[-1].split('.')[0])
                    except: k = float(len(fid_rows) * 25)
                    fid_rows.append({'Kimg': k, 'FID': d['results']['fid50k_full']})

df = pd.DataFrame(rows)
df_fid = pd.DataFrame(fid_rows) if fid_rows else pd.DataFrame(columns=['Kimg','FID'])
print(f'Log entries: {len(df)} | FID entries: {len(df_fid)}')
if len(df_fid): print(f'Best FID: {df_fid.FID.min():.2f}')

all_fakes = []
for folder in tahap_folders:
    all_fakes.extend(sorted(glob.glob(os.path.join(folder, 'fakes*.png'))))

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')

# ─── VIS 1: Loss ───────────────────────────────────────────────────────────────
print('\n[VIS 1] Loss Discriminator vs Generator')
has_loss = df['Loss_Fake'].notna().any()
if has_loss:
    fig, ax = plt.subplots(figsize=(14,5))
    df_v = df.dropna(subset=['Loss_Fake','Loss_Real'])
    ax.plot(df_v.Kimg, df_v.Loss_Real, label='Discriminator (Real)', color='#1f77b4', lw=2, alpha=0.8)
    ax.plot(df_v.Kimg, df_v.Loss_Fake, label='Generator (Fake)', color='#ff7f0e', lw=2, alpha=0.8)
    ax.set_title('Vis 1: Kurva Loss — Discriminator vs Generator (Nash Equilibrium)')
    ax.set_xlabel('Kimg'); ax.set_ylabel('Loss Score')
    ax.legend(); fig.tight_layout()
    save(fig, 'vis1_loss.png')
else:
    print('  [SKIP] Data loss tidak tersedia (format log lama)')

# ─── VIS 2: FID ────────────────────────────────────────────────────────────────
print('[VIS 2] FID Score Progression')
if len(df_fid):
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(df_fid.Kimg, df_fid.FID, marker='o', color='#2ca02c', lw=2.5, markersize=8)
    ax.axhline(50, color='red', linestyle='--', lw=2, alpha=0.7, label='Target FID < 50')
    best_idx = df_fid.FID.idxmin()
    bk, bf = df_fid.iloc[best_idx].Kimg, df_fid.iloc[best_idx].FID
    ax.annotate(f'Terbaik: {bf:.2f}', xy=(bk, bf), xytext=(bk, bf+8),
        arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, fontweight='bold', ha='center')
    ax.set_title('Vis 2: Progres Skor FID (Semakin Rendah = Semakin Baik)')
    ax.set_xlabel('Kimg'); ax.set_ylabel('FID Score')
    ax.legend(); fig.tight_layout()
    save(fig, 'vis2_fid.png')
else:
    print('  [SKIP] Data FID tidak tersedia')

# ─── VIS 3 & 4: ADA & Efficiency ───────────────────────────────────────────────
print('[VIS 3-4] ADA & Efficiency')
if len(df):
    fig, axes = plt.subplots(1, 2, figsize=(16,5))
    axes[0].fill_between(df.Kimg, df.Aug, alpha=0.3, color='#9467bd')
    axes[0].plot(df.Kimg, df.Aug, color='#9467bd', lw=2)
    axes[0].set_title('Vis 3: Intensitas Augmentasi ADA (p-value)')
    axes[0].set_xlabel('Kimg'); axes[0].set_ylabel('Probabilitas Augmentasi')
    axes[1].plot(df.Kimg, df.Sec, color='#8c564b', lw=2)
    axes[1].axhline(df.Sec.mean(), color='black', linestyle=':', label=f'Rata-rata: {df.Sec.mean():.1f}s')
    axes[1].set_title('Vis 4: Efisiensi Komputasi (Detik/Kimg)')
    axes[1].set_xlabel('Kimg'); axes[1].set_ylabel('Detik per Kimg')
    axes[1].legend()
    fig.tight_layout()
    save(fig, 'vis34_ada_efficiency.png')

# ─── VIS 5: Komparasi Awal vs Akhir ────────────────────────────────────────────
print('[VIS 5] Komparasi Gambar Awal vs Akhir')
if len(all_fakes) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(18,9))
    axes[0].imshow(mpimg.imread(all_fakes[0])); axes[0].axis('off')
    axes[0].set_title('Kimg Awal — Noise Acak', fontsize=16)
    axes[1].imshow(mpimg.imread(all_fakes[-1])); axes[1].axis('off')
    axes[1].set_title('Kimg Akhir — Struktur Medis Jelas', fontsize=16, fontweight='bold')
    fig.suptitle('Vis 5: Komparasi Kualitas Generator (Awal vs Akhir)', fontsize=18, fontweight='bold')
    fig.tight_layout()
    save(fig, 'vis5_komparasi.png')
elif len(all_fakes) == 1:
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(mpimg.imread(all_fakes[0])); ax.axis('off')
    ax.set_title('Vis 5: Hasil Generator Terbaru'); fig.tight_layout()
    save(fig, 'vis5_komparasi.png')
else:
    print('  [SKIP] Gambar fakes tidak ditemukan')

# ─── VIS 6: RGB Distribution ───────────────────────────────────────────────────
print('[VIS 6] Distribusi Warna RGB')
if all_fakes:
    img_arr = np.array(Image.open(all_fakes[-1]).convert('RGB'))
    pixels = img_arr.reshape(-1, 3)
    sample = pixels[np.random.choice(len(pixels), size=min(10000, len(pixels)), replace=False)]
    fig, ax = plt.subplots(figsize=(12,5))
    for i, (col, label) in enumerate(zip(['red','green','blue'], ['R','G','B'])):
        sns.kdeplot(sample[:, i], label=f'Channel {label}', color=col, fill=True, alpha=0.3, ax=ax)
    ax.set_title('Vis 6: Distribusi Intensitas RGB Gambar Generator (Fake)')
    ax.set_xlabel('Intensitas Piksel (0-255)'); ax.set_ylabel('Kepadatan')
    ax.legend(); fig.tight_layout()
    save(fig, 'vis6_rgb.png')

# ─── VIS 7: Discriminator Sign ─────────────────────────────────────────────────
print('[VIS 7] Discriminator Sign Accuracy')
df_sign = df.dropna(subset=['Sign_Real', 'Sign_Fake'])
if len(df_sign):
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(df_sign.Kimg, df_sign.Sign_Real, label='Real Sign', color='green', lw=2, alpha=0.8)
    ax.plot(df_sign.Kimg, df_sign.Sign_Fake, label='Fake Sign', color='red', lw=2, alpha=0.8)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax.set_title('Vis 7: Akurasi Deteksi Diskriminator (Real vs Fake)')
    ax.set_xlabel('Kimg'); ax.set_ylabel('Sign Confidence')
    ax.legend(); fig.tight_layout()
    save(fig, 'vis7_sign.png')
else:
    print('  [SKIP] Metrik Sign tidak tersedia (format log lama)')

# ─── VIS 8: R1 & PL Penalty ────────────────────────────────────────────────────
print('[VIS 8] R1 & Path Length Penalty')
df_reg = df.dropna(subset=['R1', 'PL'])
if len(df_reg):
    fig, axes = plt.subplots(1, 2, figsize=(16,5))
    axes[0].plot(df_reg.Kimg, df_reg.R1, color='purple', lw=2)
    axes[0].set_title('Vis 8a: R1 Gradient Penalty'); axes[0].set_xlabel('Kimg')
    axes[1].plot(df_reg.Kimg, df_reg.PL, color='teal', lw=2)
    axes[1].set_title('Vis 8b: Path Length Regularization'); axes[1].set_xlabel('Kimg')
    fig.suptitle('Vis 8: Dinamika Penalti Regularisasi (Anti Mode Collapse)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, 'vis8_penalty.png')
else:
    print('  [SKIP] Metrik R1/PL tidak tersedia (format log lama)')

# ─── VIS 9: Memory Usage ───────────────────────────────────────────────────────
print('[VIS 9] Profil Memori Hardware')
if len(df) and df.GPU.abs().sum() > 0:
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(df.Kimg, df.GPU, label='VRAM GPU (GB)', color='crimson', lw=2)
    ax.plot(df.Kimg, df.CPU, label='RAM CPU (GB)', color='steelblue', lw=2)
    ax.set_title('Vis 9: Utilisasi Memori Perangkat Keras Kaggle T4 x2')
    ax.set_xlabel('Kimg'); ax.set_ylabel('Gigabytes (GB)')
    ax.legend(); fig.tight_layout()
    save(fig, 'vis9_memory.png')
else:
    print('  [SKIP] Data memori tidak tersedia')

# ─── VIS 10: Pearson Correlation Heatmap ───────────────────────────────────────
print('[VIS 10] Pearson Correlation Heatmap')
if len(df) and len(df_fid):
    df_a = df.copy(); df_a['Kimg'] = df_a['Kimg'].astype(float)
    df_b = df_fid.copy(); df_b['Kimg'] = df_b['Kimg'].astype(float)
    df_m = pd.merge_asof(df_a.sort_values('Kimg'), df_b.sort_values('Kimg'), on='Kimg', direction='nearest')
    available_cols = [c for c in ['Loss_Fake','Loss_Real','Aug','Sign_Real','R1','FID'] if c in df_m.columns and df_m[c].notna().any()]
    if available_cols:
        labels = {'Loss_Fake':'Loss Generator','Loss_Real':'Loss Discriminator','Aug':'ADA p-value',
                  'Sign_Real':'Sign Real','R1':'R1 Penalty','FID':'FID Score'}
        df_corr = df_m[available_cols].rename(columns=labels).corr()
        fig, ax = plt.subplots(figsize=(max(8, len(available_cols)*1.5), max(6, len(available_cols)*1.2)))
        sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, linewidths=0.5, ax=ax)
        ax.set_title('Vis 10: Matriks Korelasi Kinerja Model (Pearson)')
        fig.tight_layout()
        save(fig, 'vis10_correlation.png')
else:
    print('  [SKIP] Butuh data log + FID')

# ─── SUMMARY ───────────────────────────────────────────────────────────────────
grafik = sorted(glob.glob(os.path.join(OUT_DIR, '*.png')))
print(f'\n✅ SELESAI! {len(grafik)} grafik tersimpan di:')
print(f'   {OUT_DIR}')
for g in grafik:
    print(f'   - {os.path.basename(g)}')
