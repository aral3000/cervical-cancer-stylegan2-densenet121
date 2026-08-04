import json

notebook_path = '8-sempro-tahap4-6k-8k_optimized.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Locate the cell that contains RESTORE_DIR and replace its content to include the missing checkpoint logic
    for c in nb['cells']:
        source_text = ''.join(c.get('source', []))
        if c['cell_type'] == 'code' and 'RESTORE_DIR =' in source_text:
            new_source = '''#Setup Checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# GUNAKAN MANAGER (Hanya simpan 3 file terakhir agar disk tidak penuh)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)

print("Checkpoint Manager telah didefinisikan. Variabel 'manager' siap digunakan.")

# ----------------- TAMBAHAN UNTUK TAHAP LANJUTAN -----------------
# Ganti dengan path folder checkpoint dari Tahap 3 sebelumnya!
RESTORE_DIR = '/kaggle/input/hasil-tahap-3-anda/training_checkpoints'

# Coba Load Checkpoint dari folder Input (Tahap sebelumnya)
latest_ckpt = tf.train.latest_checkpoint(RESTORE_DIR)
if latest_ckpt:
    checkpoint.restore(latest_ckpt)
    print(f"Sukses Me-load checkpoint dari Tahap Sebelumnya: {latest_ckpt}")
else:
    print("Peringatan: Checkpoint dari Tahap sebelumnya tidak ditemukan! Jangan lupa tambahkan input dataset.")
# ----------------------------------------------------------
'''
            c['source'] = [new_source]
            break

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print('✅ File 8-sempro-tahap4-6k-8k_optimized.ipynb telah berhasil diperbaiki secara struktural!')
    
except Exception as e:
    print("Error:", e)
