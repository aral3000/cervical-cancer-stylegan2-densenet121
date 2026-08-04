# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang

Kanker serviks merupakan salah satu masalah kesehatan global yang sangat serius, ditandai dengan adanya pertumbuhan sel abnormal pada leher rahim. Berdasarkan data *Global Cancer Observatory* (GLOBOCAN) tahun 2022, penyakit ini menempati urutan keempat sebagai jenis kanker yang paling sering didiagnosis dan menjadi penyebab kematian akibat kanker pada wanita di seluruh dunia [1]. Di Indonesia, kanker serviks menduduki peringkat kedua dalam hal mortalitas kanker pada wanita dengan tingkat kematian mencapai sekitar 60% karena mayoritas pasien baru terdiagnosis ketika penyakit telah memasuki stadium lanjut [2]. Sebagai langkah penanganan di wilayah dengan keterbatasan sumber daya medis dan fasilitas laboratorium, Inspeksi Visual Asam Asetat (IVA) menjadi alternatif *screening* utama selain *Pap Smear* melalui pengamatan visual langsung terhadap perubahan warna epitel serviks menjadi putih (*acetowhite*) setelah diaplikasikan larutan asam asetat [3]. Penelitian ini menggunakan *dataset* digital hasil *screening* IVA yang mencakup informasi tekstur makroskopis dari lesi serviks. 

Meskipun metode IVA sangat terjangkau dan mudah diimplementasikan, efektivitas diagnosisnya sangat subjektif dan rentan terhadap inkonsistensi interpretasi antar pemeriksa medis (*inter-observer variability*). Untuk menstandarisasi serta meningkatkan akurasi interpretasi tersebut, pemanfaatan teknologi kecerdasan buatan berbasis *Deep Learning* dalam bidang *Computer Vision* medis menjadi sangat krusial. Namun, pengembangan model klasifikasi *Deep Learning* yang andal dan tangguh memerlukan dukungan *dataset* latih yang berskala besar. Di sisi lain, pengumpulan *dataset* citra medis serviks asli dalam jumlah masif sering kali terbentur oleh *data scarcity* serta regulasi etika privasi pasien yang sangat ketat [4]. Selain itu, karakteristik visual lesi serviks abnormal memiliki tingkat keberagaman tekstur dan batas epitel yang tinggi, sehingga kebutuhan akan data positif (abnormal) yang representatif menjadi sangat dominan. Meskipun *dataset* yang digunakan memiliki keterbatasan kuantitas—di mana kelas negatif (normal) sebanyak 331 citra bersumber dari data rekam medis RSMH Palembang dan kelas positif (abnormal) sebanyak 666 citra bersumber dari gabungan *dataset* publik (*iclx AnnoCerv main dataset* dan *Malhari Dataset*)—keterbatasan kuantitas data secara keseluruhan tetap menjadi tantangan utama dalam melatih model klasifikasi yang kokoh.

Pembangkitan citra sintetis menggunakan arsitektur generatif merupakan salah satu solusi inovatif untuk mengatasi keterbatasan data tersebut [5]. Namun, melatih model generatif pada citra medis dengan *dataset* yang sangat terbatas sering kali mengalami masalah ketidakstabilan pelatihan (*instability*) dan *overfitting*. Oleh karena itu, penelitian ini menerapkan arsitektur generatif modern berbasis *StyleGAN*. Dua varian utama yang dibandingkan dalam penelitian ini adalah *StyleGAN2* dengan *Adaptive Data Augmentation* (*StyleGAN2-ADA*) dan *StyleGAN3*. Arsitektur *StyleGAN3* dirancang untuk mengatasi masalah *texture sticking* pada *StyleGAN2* dengan mendesain ulang operasi konvolusi agar bersifat ekuivarian terhadap translasi dan rotasi. Namun, karakteristik tersebut justru membatasi kemampuan model dalam menangkap detail tekstur lesi patologis halus (*acetowhite*) pada kondisi *data scarcity* ekstrim (666 citra) tanpa didukung oleh waktu latih dan volume data yang masif. Sebaliknya, *StyleGAN2-ADA* yang mengintegrasikan sistem augmentasi adaptif non-bocor (*non-leaking*) secara dinamis terbukti lebih stabil dan unggul pada *dataset* kecil dengan mengontrol nilai $p$ pada rentang optimal 0,25 s.d. 0,32 [6]. Evaluasi kuantitatif menunjukkan *StyleGAN2-ADA* memecahkan rekor terbaik dengan skor *Fréchet Inception Distance* (FID) sebesar 49,62 pada *Kimg* kumulatif 15.240, mengungguli *StyleGAN3* yang hanya mencapai FID terbaik sebesar 62,66 pada *Kimg* kumulatif 2.000 [6].

Citra sintetis positif hasil pembangkitan dari arsitektur terbaik (yaitu *StyleGAN2-ADA*) tersebut selanjutnya diintegrasikan ke dalam model klasifikasi *downstream task* berbasis *Convolutional Neural Network* (CNN) untuk menguji *synthetic data efficacy* [7]. Penelitian ini membandingkan kinerja model CNN secara *head-to-head* pada dua skenario utama, yaitu Skenario A (*baseline*) yang dilatih menggunakan *train set* asli (terdiri atas citra positif asli dan negatif asli), dan Skenario B yang melatih model dengan mengganti 100% citra positif asli pada *train set* menggunakan citra positif sintetis hasil pembangkitan *StyleGAN2-ADA*, sementara kelas negatif tetap menggunakan citra asli. Evaluasi akhir kinerja model klasifikasi pada *test set* asli diukur secara komprehensif melalui metrik *Accuracy*, *Precision*, dan *Recall*. Melalui pendekatan ini, diharapkan dapat dibuktikan secara ilmiah bahwa citra sintetis hasil arsitektur generatif terbaik mampu merepresentasikan fitur klinis asli secara memadai untuk melatih model klasifikasi abnormalitas citra serviks.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang masalah yang telah dipaparkan, rumusan masalah dalam penelitian ini adalah sebagai berikut:
1. Bagaimana perbandingan kinerja generatif antara arsitektur *StyleGAN2-ADA* dan *StyleGAN3* dalam mensintesis citra kanker serviks IVA yang realistis pada kondisi *data scarcity*, diukur melalui metrik *Fréchet Inception Distance* (FID)?
2. Bagaimana pengaruh substitusi citra positif asli dengan citra positif sintetis hasil arsitektur generatif terbaik (*StyleGAN2-ADA*) pada *train set* terhadap kinerja klasifikasi model CNN (DenseNet121) jika diukur menggunakan metrik *Accuracy*, *Precision*, dan *Recall* pada *test set* asli?

## 1.3 Batasan Masalah

Penelitian tugas akhir ini dibatasi oleh beberapa ruang lingkup berikut guna menjaga fokus dan relevansi pembahasan:
1. *Dataset* citra serviks yang digunakan adalah citra hasil skrining Inspeksi Visual Asam Asetat (IVA) yang dikategorikan secara biner menjadi kelas positif (abnormal) dan kelas negatif (normal).
2. Pembangkitan citra sintetis dibatasi pada studi komparatif antara metode *StyleGAN2-ADA* dan *StyleGAN3* sebagai arsitektur generatif berbasis *Deep Learning*.
3. Dimensi piksel citra input dan output untuk seluruh eksperimen dibatasi pada ukuran resolusi target 128 × 128 piksel guna mengoptimalkan beban komputasi hardware tanpa menghilangkan detail visual lesi penting.
4. Eksperimen klasifikasi abnormalitas serviks dijalankan menggunakan model *Convolutional Neural Network* (CNN) dengan arsitektur DenseNet121, membandingkan Skenario A (*baseline* menggunakan *train set* asli) dan Skenario B (substitusi seluruh citra kelas positif pada *train set* menggunakan citra sintetis hasil arsitektur generatif terbaik).
5. Pelatihan model dilakukan di lingkungan komputasi awan (*cloud computing*) platform Kaggle dengan memanfaatkan akselerator *graphics processing unit* (GPU) yang tersedia.
6. Evaluasi performa model generatif diukur berdasarkan nilai *Fréchet Inception Distance* (FID), sedangkan evaluasi kinerja model klasifikasi dianalisis pada *test set* asli menggunakan parameter *Accuracy*, *Precision*, dan *Recall*.

## 1.4 Tujuan Penelitian

Adapun tujuan yang ingin dicapai melalui pelaksanaan penelitian ini adalah:
1. Membangun dan membandingkan model generatif berbasis *StyleGAN2-ADA* dan *StyleGAN3* untuk menghasilkan citra sintetis kanker serviks IVA yang representatif dengan pencapaian kualitas visual yang dibuktikan melalui skor FID yang rendah.
2. Mengevaluasi efikasi citra sintetis (*synthetic data efficacy*) dari arsitektur terbaik dengan mengukur kemampuan model CNN dalam mendeteksi citra kanker serviks asli setelah dilatih menggunakan data positif buatan.
3. Mengevaluasi secara komparatif performa klasifikasi abnormalitas citra serviks menggunakan CNN antara Skenario A (*baseline* dengan data asli) dan Skenario B (klasifikasi dengan substitusi citra sintetis hasil arsitektur terbaik).

## 1.5 Manfaat Penelitian

Penelitian ini diharapkan dapat memberikan manfaat sebagai berikut:
1. **Manfaat Akademis (Teoretis):** Memberikan referensi ilmiah mengenai efektivitas implementasi komparatif arsitektur *StyleGAN2-ADA* dan *StyleGAN3* dalam mengatasi masalah *data scarcity* dan *class imbalance* pada citra medis, serta kontribusinya terhadap optimasi kinerja model klasifikasi CNN.
2. **Manfaat Praktis (Klinis):** Membantu menyediakan metode alternatif untuk memproduksi data latihan sintetis berkualitas tinggi yang aman dari isu privasi pasien, yang nantinya dapat dimanfaatkan untuk memperkaya variasi bahan ajar digital bagi tenaga medis atau melatih sistem diagnosis berbantuan komputer (*computer-aided diagnosis*) lainnya.

## 1.6 Sistematika Penulisan

Keseluruhan laporan skripsi ini disusun secara sistematis ke dalam lima bab utama sebagai berikut:

* **BAB I PENDAHULUAN**  
Menguraikan dasar pemikiran penelitian yang meliputi latar belakang masalah mengenai kanker serviks, kendala keterbatasan data medis IVA, studi komparatif *StyleGAN2-ADA* vs *StyleGAN3*, rumusan masalah, batasan masalah penelitian, tujuan, manfaat penelitian, dan sistematika penulisan laporan.

* **BAB II TINJAUAN PUSTAKA**  
Memaparkan landasan teori yang relevan dan hasil kajian pustaka terdahulu. Bab ini mencakup pembahasan medis mengenai kanker serviks dan metode IVA, konsep dasar *Deep Learning*, teori umum *Generative Adversarial Networks* (GAN), arsitektur *StyleGAN2-ADA* dan *StyleGAN3*, konsep dasar klasifikasi menggunakan *Convolutional Neural Network* (CNN) DenseNet121, serta penjelasan metrik evaluasi FID beserta parameter klasifikasi (*Accuracy*, *Precision*, dan *Recall*).

* **BAB III METODOLOGI PENELITIAN**  
Menjelaskan secara terperinci tahapan perancangan dan alur eksperimen sistem. Pembahasan mencakup pengumpulan *dataset* citra serviks asli (positif dan negatif), tahapan pra-pemrosesan data (*resizing* dan normalisasi), perancangan arsitektur jaringan *StyleGAN2-ADA* dan *StyleGAN3*, skenario pelatihan generatif, pembagian *dataset* (*train*, *val*, dan *test set*), substitusi kelas positif pada *train set* menggunakan citra sintetis terbaik, perancangan model klasifikasi CNN DenseNet121, serta prosedur evaluasi pengujian Skenario A dan B.

* **BAB IV HASIL DAN ANALISIS**  
Menyajikan seluruh data hasil eksperimen yang diperoleh. Bab ini memaparkan hasil visualisasi citra sintetis kanker serviks, analisis perkembangan grafik nilai FID dari *StyleGAN2-ADA* dan *StyleGAN3*, perbandingan kinerja klasifikasi CNN pada Skenario A (*baseline*) dan Skenario B (substitusi citra sintetis), serta diskusi ilmiah mengenai temuan menarik dan fenomena yang terjadi selama pelatihan model.

* **BAB V KESIMPULAN DAN SARAN**  
Menyajikan kesimpulan akhir dari seluruh rangkaian hasil analisis penelitian yang menjawab rumusan masalah yang diajukan. Bagian ini juga menyertakan saran-saran konstruktif yang ditujukan bagi pengembangan penelitian lanjutan di masa mendatang agar dapat mencapai hasil yang lebih sempurna.
