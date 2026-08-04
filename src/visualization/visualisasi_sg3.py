import os, glob, json, re, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import seaborn as sns
from PIL import Image

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE    = r'c:\new era code\GAN_Kompre\Workspace_StyleGAN3\output stylegan3'
OUT_DIR = r'c:\new era code\GAN_Kompre\Workspace_StyleGAN3\output visualisasi sg3'
os.makedirs(OUT_DIR, exist_ok=True)

# Palette profesional
PALETTE = {'fid':'#2ecc71','aug':'#9b59b6','speed':'#e67e22',
           'vram':'#e74c3c','ram':'#3498db','accent':'#f39c12'}
BG   = '#0f1117'
CARD = '#1a1d27'
TXT  = '#e8eaf0'
GRID = '#2a2d3e'

def style_ax(ax, title='', xlabel='Kimg', ylabel=''):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TXT, labelsize=11)
    ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.spines[:].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    if title: ax.set_title(title, color=TXT, fontsize=14, fontweight='bold', pad=12)
    if xlabel: ax.set_xlabel(xlabel, color=TXT, fontsize=12)
    if ylabel: ax.set_ylabel(ylabel, color=TXT, fontsize=12)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {name}')

# ─── KIMG OFFSET PER TAHAP ─────────────────────────────────────────────────────
kimg_start = {1:0,2:300,3:600,4:900,5:1200,6:1300,7:1600,8:1900,9:2200,10:2500,11:2800}

# ─── HARVEST DATA ──────────────────────────────────────────────────────────────
log_rows = []
fid_rows = []

def parse_plain(line):
    d = {}
    for k,p in [('kimg',r'kimg\s+([\d\.]+)'),('sec',r'sec/kimg\s+([\d\.]+)'),
                ('aug',r'augment\s+([\d\.]+)'),('cpu',r'cpumem\s+([\d\.]+)'),
                ('gpu',r'gpumem\s+([\d\.]+)')]:
        m = re.search(p, line)
        if m: d[k] = float(m.group(1))
    return d if 'kimg' in d else None

for tahap_dir in sorted(glob.glob(os.path.join(BASE, 'tahap *'))):
    t = int(os.path.basename(tahap_dir).split()[-1])
    offset = kimg_start.get(t, (t-1)*300)
    lf = os.path.join(tahap_dir, 'log.txt')
    ff = os.path.join(tahap_dir, 'metric-fid50k_full.jsonl')

    if os.path.exists(lf):
        with open(lf) as f:
            for raw in f:
                d = parse_plain(raw.strip())
                if d:
                    log_rows.append(dict(
                        Tahap=t, Kimg=d['kimg']+offset,
                        Sec=d.get('sec',0), Aug=d.get('aug',0),
                        CPU=d.get('cpu',0), GPU=d.get('gpu',0)))

    if os.path.exists(ff):
        with open(ff) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                pkl = os.path.basename(d.get('snapshot_pkl',''))
                try: local_k = int(pkl.replace('network-snapshot-','').replace('.pkl',''))
                except: continue
                fid_rows.append(dict(Kimg=offset+local_k,
                    FID=d['results']['fid50k_full'], Tahap=t))

df  = pd.DataFrame(log_rows).sort_values('Kimg').reset_index(drop=True)
dff = pd.DataFrame(fid_rows).sort_values('Kimg').drop_duplicates('Kimg').reset_index(drop=True)
print(f'Log entries: {len(df)} | FID entries: {len(dff)}')
print(f'Kimg range : {df.Kimg.min():.0f} - {df.Kimg.max():.0f}')
print(f'Best FID   : {dff.FID.min():.2f} @ Kimg {dff.loc[dff.FID.idxmin(),"Kimg"]:.0f}')

# ─── VIS 1: FID FULL CURVE ─────────────────────────────────────────────────────
print('\n[1] FID Progression')
fig, ax = plt.subplots(figsize=(16,6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)

# Area
ax.fill_between(dff.Kimg, dff.FID, alpha=0.15, color=PALETTE['fid'])
ax.plot(dff.Kimg, dff.FID, color=PALETTE['fid'], lw=2.5, zorder=3)
ax.scatter(dff.Kimg, dff.FID, color=PALETTE['fid'], s=60, zorder=4, edgecolors='white', lw=0.8)

# Target lines
ax.axhline(50, color='#e74c3c', lw=1.5, linestyle='--', alpha=0.8, label='Target FID < 50')
ax.axhline(100, color=PALETTE['accent'], lw=1, linestyle=':', alpha=0.6, label='Batas Minimum (FID = 100)')

# Best FID annotation
best_idx = dff.FID.idxmin()
bk, bf = dff.loc[best_idx,'Kimg'], dff.loc[best_idx,'FID']
ax.annotate(f'Best: {bf:.2f}',
    xy=(bk,bf), xytext=(bk+150, bf+15),
    color='white', fontsize=12, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE['fid'], alpha=0.8))

# Shade tahap batas
colors_tahap = ['#ffffff','#cccccc']
prev = 0
for i, (t, start) in enumerate(sorted(kimg_start.items())):
    end = start + (100 if t==5 else 300)
    ax.axvspan(start, end, alpha=0.03, color=colors_tahap[i%2])
    ax.axvline(start, color=GRID, lw=0.5, alpha=0.5)
    if start > 0:
        ax.text(start+5, dff.FID.max()*0.97, f'T{t}', color='#555577', fontsize=8)

style_ax(ax, 'StyleGAN3 — Progres FID Score (0 – 2800 Kimg)', 'Kimg', 'FID Score')
ax.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID, fontsize=11)
ax.set_xlim(0, dff.Kimg.max()+50)
fig.tight_layout()
save(fig, 'sg3_vis1_fid_progression.png')

# ─── VIS 2: FID PER TAHAP (Bar Chart) ─────────────────────────────────────────
print('[2] FID per Tahap')
# Ambil FID akhir per tahap
tahap_fid = []
for t in sorted(kimg_start.keys()):
    sub = dff[dff.Tahap==t]
    if len(sub): tahap_fid.append((f'T{t}\n({kimg_start[t]+(100 if t==5 else 300)})', sub.FID.iloc[-1]))

labels = [x[0] for x in tahap_fid]
values = [x[1] for x in tahap_fid]
colors_bar = [PALETTE['fid'] if v == min(values) else '#4a7c59' for v in values]

fig, ax = plt.subplots(figsize=(14,6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)
bars = ax.bar(labels, values, color='#4a7c59', edgecolor=GRID, linewidth=0.8, width=0.6)
bars[values.index(min(values))].set_color(PALETTE['fid'])
for bar, val in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            f'{val:.1f}', ha='center', color=TXT, fontsize=11, fontweight='bold')
ax.axhline(50, color='#e74c3c', lw=1.5, linestyle='--', alpha=0.8, label='Target FID < 50')
style_ax(ax, 'StyleGAN3 — FID Akhir per Tahap Pelatihan', 'Tahap (Kimg Total)', 'FID Score')
ax.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID)
ax.set_ylim(0, max(values)*1.15)
fig.tight_layout()
save(fig, 'sg3_vis2_fid_per_tahap.png')

# ─── VIS 3: AUGMENTATION PROBABILITY ─────────────────────────────────────────
print('[3] ADA Augmentation')
fig, ax = plt.subplots(figsize=(16,5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)
ax.fill_between(df.Kimg, df.Aug, alpha=0.2, color=PALETTE['aug'])
ax.plot(df.Kimg, df.Aug, color=PALETTE['aug'], lw=2)
# Moving average
ma = df.Aug.rolling(10, min_periods=1).mean()
ax.plot(df.Kimg, ma, color='white', lw=1.5, linestyle='--', alpha=0.7, label='Moving Avg (10 tick)')
style_ax(ax, 'StyleGAN3 — Intensitas Augmentasi ADA (p-value)', 'Kimg', 'Probabilitas Augmentasi')
ax.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID)
fig.tight_layout()
save(fig, 'sg3_vis3_augmentation.png')

# ─── VIS 4: KECEPATAN TRAINING ────────────────────────────────────────────────
print('[4] Training Speed')
df_speed = df[df.Sec < 500]  # filter outlier maintenance
fig, ax = plt.subplots(figsize=(16,5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)
ax.fill_between(df_speed.Kimg, df_speed.Sec, alpha=0.2, color=PALETTE['speed'])
ax.plot(df_speed.Kimg, df_speed.Sec, color=PALETTE['speed'], lw=1.5, alpha=0.7)
ma_s = df_speed.Sec.rolling(15, min_periods=1).mean()
ax.plot(df_speed.Kimg, ma_s, color='white', lw=2, label=f'Moving Avg (~{ma_s.mean():.0f} s/kimg)')
ax.axhline(df_speed.Sec.mean(), color='#f39c12', lw=1.5, linestyle=':',
           label=f'Rata-rata: {df_speed.Sec.mean():.1f} s/kimg')
style_ax(ax, 'StyleGAN3 — Efisiensi Komputasi (Detik per Kimg)', 'Kimg', 'Detik / Kimg')
ax.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID)
fig.tight_layout()
save(fig, 'sg3_vis4_speed.png')

# ─── VIS 5: MEMORY USAGE ─────────────────────────────────────────────────────
print('[5] Memory Usage')
fig, ax = plt.subplots(figsize=(16,5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)
ax.fill_between(df.Kimg, df.GPU, alpha=0.2, color=PALETTE['vram'])
ax.plot(df.Kimg, df.GPU, color=PALETTE['vram'], lw=2, label=f'VRAM GPU (peak: {df.GPU.max():.1f} GB)')
ax.fill_between(df.Kimg, df.CPU, alpha=0.2, color=PALETTE['ram'])
ax.plot(df.Kimg, df.CPU, color=PALETTE['ram'], lw=2, label=f'RAM CPU (peak: {df.CPU.max():.1f} GB)')
style_ax(ax, 'StyleGAN3 — Utilisasi Memori Kaggle T4 x2 GPU (0–2800 Kimg)', 'Kimg', 'Gigabytes (GB)')
ax.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID, fontsize=12)
fig.tight_layout()
save(fig, 'sg3_vis5_memory.png')

# ─── VIS 6: KOMPARASI GAMBAR AWAL vs AKHIR ───────────────────────────────────
print('[6] Image Comparison')
all_fakes = []
for tahap_dir in sorted(glob.glob(os.path.join(BASE, 'tahap *'))):
    fakes = sorted(glob.glob(os.path.join(tahap_dir, 'fakes*.png')))
    all_fakes.extend(fakes)

if len(all_fakes) >= 2:
    fig = plt.figure(figsize=(20,9), facecolor=BG)
    gs = gridspec.GridSpec(1,2, figure=fig, wspace=0.04)
    for idx, (img_path, title, kimg_label) in enumerate([
        (all_fakes[0], 'Awal Pelatihan — Kimg 0\n(Noise Acak)', '0 Kimg'),
        (all_fakes[-1], f'Akhir Pelatihan — Kimg ~2800\n(Struktural Medis)', '~2800 Kimg')
    ]):
        ax = fig.add_subplot(gs[idx])
        ax.imshow(mpimg.imread(img_path))
        ax.axis('off')
        ax.set_title(title, color=TXT, fontsize=14,
                     fontweight='bold' if idx==1 else 'normal', pad=10)
        # Border highlight pada gambar terakhir
        if idx==1:
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE['fid']); spine.set_linewidth(3); spine.set_visible(True)
    fig.suptitle('StyleGAN3 — Evolusi Kualitas Generator (Awal vs Akhir)',
                 color=TXT, fontsize=16, fontweight='bold', y=1.01)
    save(fig, 'sg3_vis6_image_comparison.png')

# ─── VIS 7: FID HEATMAP per TAHAP x CHECKPOINT ───────────────────────────────
print('[7] FID Heatmap')
heat_data = {}
for t in sorted(kimg_start.keys()):
    sub = dff[dff.Tahap==t]
    offset = kimg_start[t]
    for _, row in sub.iterrows():
        local = int(row.Kimg - offset)
        heat_data[(t, local)] = row.FID

checkpoints = sorted(set(c for (t,c) in heat_data))
tahaps = sorted(kimg_start.keys())
matrix = np.full((len(tahaps), len(checkpoints)), np.nan)
for i,t in enumerate(tahaps):
    for j,c in enumerate(checkpoints):
        if (t,c) in heat_data:
            matrix[i,j] = heat_data[(t,c)]

fig, ax = plt.subplots(figsize=(14,8))
fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
mask = np.isnan(matrix)
sns.heatmap(matrix, annot=True, fmt='.1f', mask=mask,
            xticklabels=[f'+{c}' for c in checkpoints],
            yticklabels=[f'Tahap {t}' for t in tahaps],
            cmap='RdYlGn_r', linewidths=0.5, linecolor=BG,
            annot_kws={'size':11,'fontweight':'bold','color':'white'},
            cbar_kws={'label':'FID Score'}, ax=ax)
ax.set_title('StyleGAN3 — Heatmap FID per Tahap × Checkpoint (Kimg Relatif)',
             color=TXT, fontsize=14, fontweight='bold', pad=12)
ax.tick_params(colors=TXT, labelsize=11)
ax.set_xlabel('Checkpoint (Kimg dalam Tahap)', color=TXT, fontsize=12)
ax.set_ylabel('Tahap', color=TXT, fontsize=12)
fig.tight_layout()
save(fig, 'sg3_vis7_fid_heatmap.png')

# ─── VIS 8: DASHBOARD RINGKASAN ───────────────────────────────────────────────
print('[8] Summary Dashboard')
fig = plt.figure(figsize=(20,12), facecolor=BG)
gs = gridspec.GridSpec(2,3, figure=fig, hspace=0.4, wspace=0.35)

# FID mini
ax1 = fig.add_subplot(gs[0,:2])
ax1.set_facecolor(CARD)
ax1.fill_between(dff.Kimg, dff.FID, alpha=0.2, color=PALETTE['fid'])
ax1.plot(dff.Kimg, dff.FID, color=PALETTE['fid'], lw=2.5)
ax1.scatter([bk],[bf], s=150, color='white', zorder=5)
ax1.axhline(50, color='#e74c3c', lw=1.5, linestyle='--', alpha=0.8)
style_ax(ax1,'FID Progression (Penuh 2800 Kimg)','Kimg','FID')

# Stats card
ax2 = fig.add_subplot(gs[0,2])
ax2.set_facecolor(CARD); ax2.axis('off')
stats = [
    ('Best FID', f'{dff.FID.min():.2f}'),
    ('@ Kimg', f'{bk:.0f}'),
    ('Final FID', f'{dff.FID.iloc[-1]:.2f}'),
    ('Total Kimg', f'{dff.Kimg.max():.0f}'),
    ('Total Tahap', f'{len(kimg_start)}'),
    ('Avg Speed', f'{df[df.Sec<500].Sec.mean():.0f} s/kimg'),
    ('Peak VRAM', f'{df.GPU.max():.1f} GB'),
    ('Peak RAM', f'{df.CPU.max():.1f} GB'),
]
for i,(k,v) in enumerate(stats):
    ax2.text(0.05, 0.92-i*0.115, k+':', color='#aaaacc', fontsize=12, transform=ax2.transAxes)
    ax2.text(0.55, 0.92-i*0.115, v, color=PALETTE['fid'] if i<2 else TXT,
             fontsize=13, fontweight='bold', transform=ax2.transAxes)
ax2.set_title('Ringkasan', color=TXT, fontsize=13, fontweight='bold')

# ADA mini
ax3 = fig.add_subplot(gs[1,0])
ax3.set_facecolor(CARD)
ax3.fill_between(df.Kimg, df.Aug, alpha=0.25, color=PALETTE['aug'])
ax3.plot(df.Kimg, df.Aug, color=PALETTE['aug'], lw=1.8)
style_ax(ax3,'ADA Augmentation','Kimg','p-value')

# Speed mini
ax4 = fig.add_subplot(gs[1,1])
ax4.set_facecolor(CARD)
df_s = df[df.Sec<500]
ax4.plot(df_s.Kimg, df_s.Sec.rolling(15,min_periods=1).mean(), color=PALETTE['speed'], lw=2)
style_ax(ax4,'Efisiensi Komputasi','Kimg','s/kimg')

# Memory mini
ax5 = fig.add_subplot(gs[1,2])
ax5.set_facecolor(CARD)
ax5.plot(df.Kimg, df.GPU, color=PALETTE['vram'], lw=2, label='VRAM')
ax5.plot(df.Kimg, df.CPU, color=PALETTE['ram'], lw=2, label='RAM')
ax5.legend(facecolor=CARD, labelcolor=TXT, edgecolor=GRID, fontsize=10)
style_ax(ax5,'Memori Hardware','Kimg','GB')

fig.suptitle('StyleGAN3-T — Dashboard Visualisasi Skripsi (3000 Kimg Target)',
             color=TXT, fontsize=18, fontweight='bold', y=1.01)
save(fig, 'sg3_vis8_dashboard.png')

# ─── SELESAI ───────────────────────────────────────────────────────────────────
results = sorted(glob.glob(os.path.join(OUT_DIR, '*.png')))
print(f'\nSelesai! {len(results)} visualisasi tersimpan di:')
print(f'  {OUT_DIR}')
for r in results:
    kb = os.path.getsize(r)//1024
    print(f'  - {os.path.basename(r)} ({kb} KB)')
