import os
import glob
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG & STYLING ─────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'

BASE = r'c:\new era code\GAN_Kompre\Workspace_StyleGAN2\output stylegan2'
OUT_DIR_DASHBOARD = r'c:\new era code\GAN_Kompre\Workspace_StyleGAN2\output visualisasi sg2'
OUT_DIR_GRAFIK = r'c:\new era code\GAN_Kompre\Workspace_StyleGAN2\output grafik'

os.makedirs(OUT_DIR_DASHBOARD, exist_ok=True)
os.makedirs(OUT_DIR_GRAFIK, exist_ok=True)

# Professional SOTA Dark Palette
PALETTE = {
    'fid': '#2ecc71',      # Emerald Green
    'aug': '#9b59b6',      # Amethyst Purple
    'speed': '#e67e22',    # Carrot Orange
    'vram': '#e74c3c',     # Alizarin Red
    'ram': '#3498db',      # Peter River Blue
    'accent': '#f39c12',   # Orange Accent
    'loss_g': '#ff7f0e',   # Generator Loss
    'loss_d': '#1f77b4'    # Discriminator Loss
}
BG   = '#0f1117'
CARD = '#1a1d27'
TXT  = '#e8eaf0'
GRID = '#2a2d3e'

def style_ax(ax, title='', xlabel='Kimg', ylabel=''):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TXT, labelsize=11)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['top'].set_color(GRID)
    ax.spines['left'].set_color(GRID)
    ax.spines['right'].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    if title: ax.set_title(title, color=TXT, fontsize=14, fontweight='bold', pad=12)
    if xlabel: ax.set_xlabel(xlabel, color=TXT, fontsize=12)
    if ylabel: ax.set_ylabel(ylabel, color=TXT, fontsize=12)

# ─── HARVEST DATA ─────────────────────────────────────────────────────────────
stage_folders = [
    ('tahap 1', 1),
    ('tahap 2', 2),
    ('tahap 3', 3),
    ('tahap 4', 4),
    ('tahap 5', 5),
    ('tahap 6', 6),
    ('tahap 7', 7),
    ('tahap 8', 8),
    ('tahap 9', 9),
    ('tahap 10', 10),
    ('tahap 11', 11),
    ('tahap 12', 12),
    ('tahap 13', 13),
    ('tahap 14', 14),
    ('tahap 15', 15),
    ('tahap 16', 16),
    ('tahap 17', 17),
    ('tahap 18', 18),
    ('tahap 19', 19)
]

def parse_plain(line):
    d = {}
    for k, p in [('kimg', r'kimg\s+([\d\.]+)'), ('sec', r'sec/kimg\s+([\d\.]+)'),
                  ('aug', r'augment\s+([\d\.]+)'), ('cpu', r'cpumem\s+([\d\.]+)'),
                  ('gpu', r'gpumem\s+([\d\.]+)')]:
        m = re.search(p, line)
        if m: d[k] = float(m.group(1))
    return d if 'kimg' in d else None

# Cumulative offsets
stage_offsets = {}
current_offset = 0.0

for folder_name, t_num in stage_folders:
    stage_dir = os.path.join(BASE, folder_name)
    stage_offsets[t_num] = current_offset
    
    max_local_kimg = 0.0
    lf = os.path.join(stage_dir, 'log.txt')
    if os.path.exists(lf):
        with open(lf) as f:
            for raw in f:
                line = raw.strip()
                if line.startswith('{"tick"'):
                    try:
                        d = json.loads(line)
                        k = float(d.get('kimg', 0.0))
                        if k > max_local_kimg:
                            max_local_kimg = k
                    except: pass
                else:
                    d = parse_plain(line)
                    if d and d['kimg'] > max_local_kimg:
                        max_local_kimg = d['kimg']
    current_offset += max_local_kimg

print(f"Cumulative stage offsets: {stage_offsets}")

log_rows = []
fid_rows = []

for folder_name, t_num in stage_folders:
    stage_dir = os.path.join(BASE, folder_name)
    offset = stage_offsets[t_num]
    
    lf = os.path.join(stage_dir, 'log.txt')
    ff = os.path.join(stage_dir, 'metric-fid50k_full.jsonl')
    
    # Parse log.txt
    if os.path.exists(lf):
        with open(lf) as f:
            for raw in f:
                line = raw.strip()
                if line.startswith('{"tick"'):
                    try:
                        d = json.loads(line)
                        k = float(d.get('kimg', 0.0)) + offset
                        log_rows.append({
                            'Tahap': t_num,
                            'Kimg': k,
                            'Loss_Fake': d.get('Loss/scores/fake'),
                            'Loss_Real': d.get('Loss/scores/real'),
                            'Sec': d.get('Timing/sec_per_kimg', d.get('sec/kimg', 0.0)),
                            'Aug': d.get('Progress/augment', d.get('Augment/p', 0.0)),
                            'Sign_Real': d.get('Loss/signs/real'),
                            'Sign_Fake': d.get('Loss/signs/fake'),
                            'R1': d.get('Loss/r1_penalty'),
                            'PL': d.get('Loss/pl_penalty'),
                            'CPU': d.get('Resources/cpu_mem_gb', 0.0),
                            'GPU': d.get('Resources/peak_gpu_mem_gb', 0.0)
                        })
                    except: pass
                else:
                    d = parse_plain(line)
                    if d:
                        k = d['kimg'] + offset
                        log_rows.append({
                            'Tahap': t_num,
                            'Kimg': k,
                            'Loss_Fake': None,
                            'Loss_Real': None,
                            'Sec': d.get('sec', 0.0),
                            'Aug': d.get('aug', 0.0),
                            'Sign_Real': None,
                            'Sign_Fake': None,
                            'R1': None,
                            'PL': None,
                            'CPU': d.get('cpu', 0.0),
                            'GPU': d.get('gpu', 0.0)
                        })
                        
    # Parse metric-fid50k_full.jsonl
    if os.path.exists(ff):
        with open(ff) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    d = json.loads(line)
                    pkl = os.path.basename(d.get('snapshot_pkl', ''))
                    m_k = re.search(r'(\d+)', pkl)
                    if m_k:
                        local_k = float(m_k.group(1))
                    else:
                        continue
                    fid_rows.append({
                        'Kimg': offset + local_k,
                        'FID': d['results']['fid50k_full'],
                        'Tahap': t_num
                    })
                except Exception as e:
                    print(f"Error parsing FID line: {e}")

df = pd.DataFrame(log_rows).sort_values('Kimg').reset_index(drop=True)
dff = pd.DataFrame(fid_rows).sort_values('Kimg').drop_duplicates('Kimg').reset_index(drop=True)

print(f"Total Log entries: {len(df)} | Total FID entries: {len(dff)}")
if len(df) > 0 and len(dff) > 0:
    print(f"Overall Kimg range: {df.Kimg.min():.0f} - {df.Kimg.max():.0f}")
    best_fid_idx = dff.FID.idxmin()
    best_fid = dff.loc[best_fid_idx, 'FID']
    best_fid_kimg = dff.loc[best_fid_idx, 'Kimg']
    print(f"Best Record FID: {best_fid:.2f} @ Kimg {best_fid_kimg:.0f}")

# ─── VISUALISATIONS FOR OUTPUT VISUALISASI SG2 ───────────────────────────────

# 1. sg2_vis1_fid_progression.png
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)

if len(dff) > 0:
    ax.fill_between(dff.Kimg, dff.FID, alpha=0.15, color=PALETTE['fid'])
    ax.plot(dff.Kimg, dff.FID, color=PALETTE['fid'], lw=2.5, zorder=3)
    ax.scatter(dff.Kimg, dff.FID, color=PALETTE['fid'], s=60, zorder=4, edgecolors='white', lw=0.8)
    
    ax.axhline(50, color='#e74c3c', lw=1.5, linestyle='--', alpha=0.8, label='Target Skripsi (FID < 50)')
    ax.axhline(60, color=PALETTE['accent'], lw=1, linestyle=':', alpha=0.6, label='Batas SOTA (FID = 60)')
    
    best_idx = dff.FID.idxmin()
    bk, bf = dff.loc[best_idx, 'Kimg'], dff.loc[best_idx, 'FID']
    ax.annotate(f'Rekor Terbaik: {bf:.2f} FID\npada Kimg {bk:.0f}',
                xy=(bk, bf), xytext=(bk + 200, bf + 12),
                color='white', fontsize=11, fontweight='bold',
                arrowprops=dict(facecolor='white', shrink=0.08, width=1.5, headwidth=6),
                bbox=dict(boxstyle='round,pad=0.4', facecolor=CARD, edgecolor=PALETTE['fid'], alpha=0.9))

colors_tahap = ['#ffffff', '#cccccc']
max_k = dff.Kimg.max() if len(dff) > 0 else 7500
for i, (folder_name, t) in enumerate(stage_folders):
    start = stage_offsets[t]
    end = stage_offsets.get(t+1, df.Kimg.max() if len(df) > 0 else max_k)
    ax.axvspan(start, end, alpha=0.02, color=colors_tahap[i % 2])
    ax.axvline(start, color=GRID, lw=0.8, alpha=0.4)
    if start >= 0 and len(dff) > 0:
        ax.text(start + 20, dff.FID.max() * 0.95, f'T{t}', color='#8a90a6', fontsize=10, fontweight='bold')

style_ax(ax, f'Evolusi Skor FID (Frechet Inception Distance) StyleGAN2 — Kimg 0 s.d {max_k:.0f}', 'Kimg (Kumulatif)', 'Skor FID (Semakin Rendah = Semakin Bagus)')
ax.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID, fontsize=11, loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_DASHBOARD, 'sg2_vis1_fid_progression.png'), dpi=150, facecolor=BG)
fig.savefig(os.path.join(OUT_DIR_GRAFIK, 'vis2_fid.png'), dpi=150, facecolor=BG)
plt.close(fig)
print("Saved FID Progression plots.")

# 2. sg2_vis2_fid_per_tahap.png
# Calculate the final FID for each Stage
tahap_best_fid = []
for folder_name, t in stage_folders:
    stage_fids = dff[dff.Tahap == t]
    if not stage_fids.empty:
        # Get the minimum FID achieved during this Stage
        best_val = stage_fids.FID.min()
        tahap_best_fid.append({'Tahap': f'Tahap {t}', 'FID': best_val})
    else:
        # Fallback if no exact match (estimate or carry over)
        tahap_best_fid.append({'Tahap': f'Tahap {t}', 'FID': None})

df_tbest = pd.DataFrame(tahap_best_fid)
df_tbest['FID'] = df_tbest['FID'].ffill() # Carry over just in case

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)

bars = ax.bar(df_tbest.Tahap, df_tbest.FID, color=PALETTE['fid'], width=0.6, edgecolor='white', linewidth=0.5, alpha=0.9)
# Highlight Tahap 9 (where best FID was achieved)
best_stage_idx = df_tbest.FID.idxmin()
bars[best_stage_idx].set_color(PALETTE['accent'])
bars[best_stage_idx].set_edgecolor('white')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.2f}', ha='center', va='bottom', color=TXT, fontsize=10, fontweight='bold')

style_ax(ax, 'Perbandingan Capaian FID Terbaik per Tahap Pelatihan', '', 'Skor FID Terbaik')
ax.axhline(50, color='#e74c3c', lw=1.5, linestyle='--', alpha=0.8, label='Target Skripsi (FID < 50)')
ax.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_DASHBOARD, 'sg2_vis2_fid_per_tahap.png'), dpi=150, facecolor=BG)
plt.close(fig)
print("Saved Stage-wise FID Comparison plots.")

# 3. sg2_vis3_image_comparison.png (Grid of early vs late cervix fakes)
# Let's find fakes in Stage 1 and the latest available stage
early_fakes = sorted(glob.glob(os.path.join(BASE, 'tahap 1', 'fakes*.png')))
late_fakes = []
late_stage_name = 'tahap10'
for folder_name, _ in reversed(stage_folders):
    test_fakes = sorted(glob.glob(os.path.join(BASE, folder_name, 'fakes*.png')))
    if test_fakes:
        late_fakes = test_fakes
        late_stage_name = folder_name
        break

if early_fakes and late_fakes:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor(BG)
    
    img_early = mpimg.imread(early_fakes[0])
    img_late = mpimg.imread(late_fakes[-1])
    
    axes[0].imshow(img_early)
    axes[0].axis('off')
    axes[0].set_title('Awal Pelatihan (Kimg 0 - Noise Awal)', color=TXT, fontsize=14, fontweight='bold', pad=10)
    
    axes[1].imshow(img_late)
    axes[1].axis('off')
    axes[1].set_title(f'Akhir Pelatihan (Kimg {max_k:.0f} - Struktur Jelas)', color=TXT, fontsize=14, fontweight='bold', pad=10)
    
    fig.suptitle('Evolusi Visualisasi Hasil Generator StyleGAN2 (Cervix Dataset)', color=TXT, fontsize=18, fontweight='bold', y=0.96)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR_DASHBOARD, 'sg2_vis3_image_comparison.png'), dpi=150, facecolor=BG)
    fig.savefig(os.path.join(OUT_DIR_GRAFIK, 'vis5_komparasi.png'), dpi=150, facecolor=BG)
    plt.close(fig)
    print("Saved Image Comparison plots.")
else:
    print("Fakes not found, skipping image comparison plot.")

# 4. sg2_vis4_dashboard.png (Unified SOTA Training Dashboard)
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

# Subplot 1: FID Score Progression
ax1 = fig.add_subplot(gs[0, 0])
if len(dff) > 0:
    ax1.plot(dff.Kimg, dff.FID, color=PALETTE['fid'], lw=2)
    ax1.fill_between(dff.Kimg, dff.FID, alpha=0.1, color=PALETTE['fid'])
    ax1.axhline(50, color='#e74c3c', lw=1.2, linestyle='--', alpha=0.7)
    best_idx = dff.FID.idxmin()
    ax1.scatter(dff.loc[best_idx, 'Kimg'], dff.loc[best_idx, 'FID'], color=PALETTE['accent'], s=100, zorder=5, edgecolors='white')
    style_ax(ax1, f'Progress FID (Min: {dff.FID.min():.2f})', 'Kimg', 'FID')

# Subplot 2: Generator vs Discriminator Loss
ax2 = fig.add_subplot(gs[0, 1])
df_clean_loss = df.dropna(subset=['Loss_Fake', 'Loss_Real'])
if not df_clean_loss.empty:
    ax2.plot(df_clean_loss.Kimg, df_clean_loss.Loss_Real, label='Loss Discriminator (Real)', color=PALETTE['loss_d'], lw=1.5, alpha=0.8)
    ax2.plot(df_clean_loss.Kimg, df_clean_loss.Loss_Fake, label='Loss Generator (Fake)', color=PALETTE['loss_g'], lw=1.5, alpha=0.8)
    style_ax(ax2, 'Dinamika Loss GAN (Discriminator vs Generator)', 'Kimg', 'Loss Value')
    ax2.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID, fontsize=9)
else:
    # Fallback to general text or empty chart with grids
    style_ax(ax2, 'Loss Generator vs Discriminator (Belum Terpapar)', 'Kimg', 'Loss')

# Subplot 3: ADA Augment Intensity (p-value)
ax3 = fig.add_subplot(gs[1, 0])
if len(df) > 0:
    ax3.plot(df.Kimg, df.Aug, color=PALETTE['aug'], lw=2)
    ax3.fill_between(df.Kimg, df.Aug, alpha=0.15, color=PALETTE['aug'])
    style_ax(ax3, 'Intensitas Adaptive Data Augmentation (ADA p-value)', 'Kimg', 'Augmentation Probability (p)')

# Subplot 4: Computational Speed (sec/kimg)
ax4 = fig.add_subplot(gs[1, 1])
if len(df) > 0:
    ax4.plot(df.Kimg, df.Sec, color=PALETTE['speed'], lw=1.5)
    mean_speed = df.Sec.mean()
    ax4.axhline(mean_speed, color='white', linestyle=':', label=f'Mean Speed: {mean_speed:.1f} s/kimg')
    style_ax(ax4, 'Kecepatan Pelatihan (Detik per Kimg)', 'Kimg', 'Detik / Kimg')
    ax4.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID, fontsize=9)

fig.suptitle('Dashboard Komprehensif Evaluasi StyleGAN2-ADA (Dataset Serviks)', color=TXT, fontsize=20, fontweight='bold', y=0.96)
fig.savefig(os.path.join(OUT_DIR_DASHBOARD, 'sg2_vis4_dashboard.png'), dpi=150, facecolor=BG)
plt.close(fig)
print("Saved Unified Dashboard.")

# ─── EXTRA PLOTS FOR OUTPUT GRAFIK ──────────────────────────────────────────

# 5. vis1_loss.png (Loss plot)
if not df_clean_loss.empty:
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.plot(df_clean_loss.Kimg, df_clean_loss.Loss_Real, label='Loss Discriminator (Real)', color=PALETTE['loss_d'], lw=2, alpha=0.85)
    ax.plot(df_clean_loss.Kimg, df_clean_loss.Loss_Fake, label='Loss Generator (Fake)', color=PALETTE['loss_g'], lw=2, alpha=0.85)
    style_ax(ax, 'Dinamika Kerugian Model (Nash Equilibrium) — Generator vs Discriminator', 'Kimg', 'Loss Score')
    ax.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR_GRAFIK, 'vis1_loss.png'), dpi=150, facecolor=BG)
    plt.close(fig)

# 6. vis34_ada_efficiency.png
if len(df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor(BG)
    
    # ADA Augment
    axes[0].fill_between(df.Kimg, df.Aug, alpha=0.15, color=PALETTE['aug'])
    axes[0].plot(df.Kimg, df.Aug, color=PALETTE['aug'], lw=2.5)
    style_ax(axes[0], 'Intensitas Adaptive Data Augmentation (ADA p-value)', 'Kimg', 'Probabilitas p')
    
    # Training speed
    axes[1].plot(df.Kimg, df.Sec, color=PALETTE['speed'], lw=2)
    axes[1].axhline(df.Sec.mean(), color='white', linestyle=':', label=f'Rata-rata: {df.Sec.mean():.1f} s/kimg')
    style_ax(axes[1], 'Efisiensi Komputasi Pelatihan (Detik per Kimg)', 'Kimg', 'Detik per Kimg')
    axes[1].legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR_GRAFIK, 'vis34_ada_efficiency.png'), dpi=150, facecolor=BG)
    plt.close(fig)

# 7. vis6_rgb.png (RGB Color Intensity distribution)
if late_fakes:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    
    img_arr = np.array(Image.open(late_fakes[-1]).convert('RGB'))
    pixels = img_arr.reshape(-1, 3)
    # Random sampling to speed up plotting
    sample = pixels[np.random.choice(len(pixels), size=min(20000, len(pixels)), replace=False)]
    
    for i, (col, label) in enumerate(zip(['red', 'green', 'blue'], ['Red Channel', 'Green Channel', 'Blue Channel'])):
        sns.kdeplot(sample[:, i], label=label, color=col, fill=True, alpha=0.25, ax=ax)
        
    style_ax(ax, 'Distribusi Intensitas RGB Piksel Gambar Buatan (Generated Images)', 'Intensitas Piksel (0-255)', 'Kepadatan (Density)')
    ax.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR_GRAFIK, 'vis6_rgb.png'), dpi=150, facecolor=BG)
    plt.close(fig)

# 8. vis7_sign.png (Discriminator Confidence Sign)
df_clean_sign = df.dropna(subset=['Sign_Real', 'Sign_Fake'])
if not df_clean_sign.empty:
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    
    ax.plot(df_clean_sign.Kimg, df_clean_sign.Sign_Real, label='Real Confidence Sign', color='#2ecc71', lw=1.8, alpha=0.8)
    ax.plot(df_clean_sign.Kimg, df_clean_sign.Sign_Fake, label='Fake Confidence Sign', color='#e74c3c', lw=1.8, alpha=0.8)
    ax.axhline(0, color='white', linestyle='--', alpha=0.4)
    
    style_ax(ax, 'Tingkat Kepercayaan (Sign Accuracy) Diskriminator', 'Kimg', 'Confidence Score')
    ax.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR_GRAFIK, 'vis7_sign.png'), dpi=150, facecolor=BG)
    plt.close(fig)

# 9. vis8_penalty.png (R1 penalty and Path Length regularization)
df_clean_reg = df.dropna(subset=['R1', 'PL'])
if not df_clean_reg.empty:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor(BG)
    
    axes[0].plot(df_clean_reg.Kimg, df_clean_reg.R1, color='#9b59b6', lw=2)
    style_ax(axes[0], 'R1 Gradient Penalty (Pencegah Overfitting D)', 'Kimg', 'Penalti R1')
    
    axes[1].plot(df_clean_reg.Kimg, df_clean_reg.PL, color='#1abc9c', lw=2)
    style_ax(axes[1], 'Path Length Regularization (Penyederhana Latent)', 'Kimg', 'Regularisasi PL')
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR_GRAFIK, 'vis8_penalty.png'), dpi=150, facecolor=BG)
    plt.close(fig)

# 10. vis9_memory.png (Hardware VRAM usage)
# Check if CPU/GPU memory contains values
if len(df) > 0 and (df.GPU.abs().sum() > 0 or df.CPU.abs().sum() > 0):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    
    if df.GPU.abs().sum() > 0:
        ax.plot(df.Kimg, df.GPU, label='VRAM Peak GPU (GB)', color=PALETTE['vram'], lw=2)
    if df.CPU.abs().sum() > 0:
        ax.plot(df.Kimg, df.CPU, label='RAM CPU (GB)', color=PALETTE['ram'], lw=2)
        
    style_ax(ax, 'Penggunaan Memori Perangkat Keras Kaggle T4 GPU x2', 'Kimg', 'Ukuran Memori (Gigabytes)')
    ax.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR_GRAFIK, 'vis9_memory.png'), dpi=150, facecolor=BG)
    plt.close(fig)

# 11. vis10_correlation.png (Pearson Correlation Matrix)
if len(df) > 0 and len(dff) > 0:
    # Merge log dataframe and FID dataframe on nearest Kimg
    df_a = df.copy(); df_a['Kimg'] = df_a['Kimg'].astype(float)
    df_b = dff.copy(); df_b['Kimg'] = df_b['Kimg'].astype(float)
    df_m = pd.merge_asof(df_a.sort_values('Kimg'), df_b.sort_values('Kimg'), on='Kimg', direction='nearest')
    
    cols = ['Loss_Fake', 'Loss_Real', 'Aug', 'R1', 'FID']
    available_cols = [c for c in cols if c in df_m.columns and df_m[c].notna().any()]
    if available_cols:
        labels = {
            'Loss_Fake': 'Loss Gen',
            'Loss_Real': 'Loss Disc',
            'Aug': 'ADA p-val',
            'R1': 'R1 Penalty',
            'FID': 'Skor FID'
        }
        df_corr = df_m[available_cols].rename(columns=labels).corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(BG)
        
        sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                    cbar_kws={'label': 'Koefisien Korelasi Pearson'})
        ax.set_title('Matriks Korelasi Pearson Kinerja Model', color=TXT, fontsize=14, fontweight='bold', pad=12)
        ax.tick_params(colors=TXT)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR_GRAFIK, 'vis10_correlation.png'), dpi=150, facecolor=BG)
        plt.close(fig)

print("\n[SUCCESS] ALL VISUALISATIONS SUCCESSFULLY GENERATED!")
print(f"  Dashboard folder: {OUT_DIR_DASHBOARD}")
print(f"  Graphics folder: {OUT_DIR_GRAFIK}")
