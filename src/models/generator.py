import json
import traceback

try:
    with open('8-sempro-tahap3-6k-8k.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Learning Rates and Label Smoothing
    for c in nb['cells']:
        source_text = "".join(c.get('source', []))
        if c['cell_type'] == 'code':
            # Label Smoothing & Cross Entropy
            if 'def discriminator_loss' in source_text and 'real_loss =' in source_text:
                new_source = '''# Helper function untuk Binary Cross Entropy
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# --- REVISI TAHAP 4: IMPLEMENTASI LABEL SMOOTHING ---
def discriminator_loss(real_output, fake_output):
    # LABEL SMOOTHING: Ubah target 'Asli' dari 1.0 menjadi 0.9
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    
    # Target 'Palsu' tetap 0.0
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
'''
                c['source'] = [new_source]

            # Optimizer updates
            if 'generator_optimizer =' in source_text and 'discriminator_optimizer =' in source_text and '1e-4' in source_text:
                new_source = '''# --- REVISI PARAMETER OPTIMIZER TAHAP 4 ---

# 1. Turunkan Learning Rate menjadi setengah atau seperempatnya
gen_lr = 2e-5  # Sedikit lebih lambat untuk generator
disc_lr = 5e-5 # Sedikit lebih lambat dari sebelumnya

# 2. Turunkan momentum Beta-1 (Ini membuat optimisasi tidak terlalu "agresif")
beta_1_val = 0.5 

print(f"Menggunakan Optimizer Adam Baru:")
print(f"  Generator LR: {gen_lr}")
print(f"  Discriminator LR: {disc_lr}")

# Terapkan optimizer baru
generator_optimizer = tf.keras.optimizers.Adam(gen_lr, beta_1=beta_1_val)
discriminator_optimizer = tf.keras.optimizers.Adam(disc_lr, beta_1=beta_1_val)
'''
                c['source'] = [new_source]

            # Best FID tracking and saving (In train block)
            if 'def train(' in source_text:
                new_train_block = '''# List untuk menampung riwayat skor FID
fid_history = []
epoch_history = []

# --- TAMBAHAN TAHAP 4: Variabel global untuk melacak skor FID terbaik ---
best_fid_score = float('inf')

def train(dataset, epochs):
    global best_fid_score # Panggil variabel global
    
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # 1. Kalkulasi FID setiap 10 epoch
        if (epoch + 1) % 20 == 0:
            # Generate gambar palsu untuk tes (sejumlah gambar asli)
            num_test = real_images_gt.shape[0]
            noise_test = tf.random.normal([num_test, noise_dim])
            generated_test = generator(noise_test, training=False).numpy()
            
            # Dapatkan aktivasi dan hitung skor
            act_fake = get_activations(generated_test, inception_model, batch_size=32)
            current_fid = calculate_fid(act_real, act_fake)
            
            # Simpan ke history
            fid_history.append(current_fid)
            epoch_history.append(epoch + 1)
            print(f"Epoch {epoch+1} - Skor FID: {current_fid:.2f}")

            # =======================================================
            # TAMBAHAN TAHAP 4: Logika Menyimpan Model TERBAIK
            if current_fid < best_fid_score:
                print(f"🌟 PERKEMBANGAN! FID membaik dari {best_fid_score:.2f} ke {current_fid:.2f}.")
                print("Menyimpan model Generator TERBAIK saat ini...")
                best_fid_score = current_fid
                
                # Buat folder jika belum ada (jaga-jaga)
                import os
                os.makedirs('saved_models', exist_ok=True)
                # Simpan terpisah agar tidak tertimpa checkpoint biasa
                generator.save('saved_models/generator_terbaik_sepanjang_masa.h5')
            # =======================================================

        # Visualisasi gambar berkala
        display.clear_output(wait=True)
        if (epoch + 1) % 50 == 0:
            generate_and_save_images(generator, epoch + 1, seed)
        # Simpan checkpoint rutin
        if (epoch + 1) % 15 == 0:
            manager.save()

        print(f'Waktu untuk epoch {epoch + 1} adalah {time.time()-start} detik')
'''
                c['source'] = [new_train_block]

            # Update RESTORE DIR
            if 'RESTORE_DIR =' in source_text:
                new_source = '''# ----------------- TAMBAHAN UNTUK TAHAP LANJUTAN -----------------
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

    # Save to a new notebook
    with open('8-sempro-tahap4-6k-8k_optimized.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Tahap 4 Notebook successfully created: 8-sempro-tahap4-6k-8k_optimized.ipynb")

except Exception as e:
    print("ERROR:")
    traceback.print_exc()
