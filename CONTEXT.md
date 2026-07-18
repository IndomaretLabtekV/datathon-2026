# CONTEXT.md - Datathon Task 1: Peramalan Kecepatan Lalu Lintas (v2)

Berkas ini adalah konteks tetap untuk sesi lanjutan Task 1, dipakai di *session* Claude Code yang baru. Baca berkas ini secara penuh, lalu baca `Datathon_Task_1_Kaggle_Notebook_Plan.md` sebelum menulis kode apa pun.

---

## 1. Status Progres Sampai Sekarang

- **v1 sudah dibuat, dieksekusi, dan disubmit.** Pendekatan: LightGBM (fitur riwayat kecepatan + statistik graf + teks) mengikuti `PROMPT_V1.md`.
- **Skor v1: 41.795.** Sebagai pembanding, sebuah rencana v2 terpisah (lihat bagian 3) melaporkan *shadow leaderboard*: baseline *persistence* sekitar `53,98`, baseline *mean-reversion* sekitar `41,65`, dan *mean-reversion* dengan *stride* 30 sekitar `40,16`. **v1 hanya sedikit lebih baik dari baseline naif *mean-reversion*.** Ini artinya v1 belum benar-benar mengeksploitasi sinyal graf jaringan jalan maupun teks peristiwa secara efektif, masih ada ruang perbaikan nyata untuk v2.
- Notebook v1 (`datathon-task1-v1.ipynb` atau nama serupa di root/scratchpad) tetap tersedia sebagai referensi dan sebagai *fallback* yang terbukti valid, meskipun skornya biasa saja.
- **Satu slot submisi sudah terpakai untuk v1.**

## 2. Anggaran Kritis yang Tersisa

- **Sisa waktu: sekitar 4 jam** dari total 5 jam ronde penyisihan.
- **Sisa submisi: 4 kali.** Ini sangat terbatas. Jangan pernah submit tanpa bukti perbaikan yang meyakinkan dari validasi lokal/*shadow* CV (bukan asumsi, bukan skor *training* saja). Perlakukan setiap slot submisi sebagai sumber daya berharga, bukan sesuatu yang dipakai untuk coba-coba.
- **Eksekutor v2: Claude Code dengan model Opus 4.8, effort xhigh** (bukan Fable 5 lagi). Baseline dan konteks dasar sudah diketahui dari v1, jadi tidak perlu eksplorasi seliar sesi pertama, fokus pada efisiensi token dan keputusan yang terarah.

## 3. Rencana Kandidat: `Datathon_Task_1_Kaggle_Notebook_Plan.md`

Ada satu berkas rencana tambahan, `Datathon_Task_1_Kaggle_Notebook_Plan.md`, dihasilkan oleh model lain (GPT 5.6 Sol, *max effort*), **tidak berbasis pada v1**. Rencana ini jauh lebih ambisius: baseline *mean-reversion* + model residual Graph-STID (utama) + Compact Graph WaveNet (penantang) + fitur teks opsional + *ensemble* per horizon, dengan alur kerja *test-driven development* (pytest) dan tiga mode eksekusi (`smoke`/`cv`/`final`).

**Status rencana ini: kandidat kuat, bukan perintah mutlak.** Boleh diimplementasikan sebagian atau seluruhnya, dengan penyesuaian berikut:

- Rencana itu mengasumsikan *skill* eksternal (`superpowers:subagent-driven-development`, `superpowers:executing-plans`) yang ketersediaannya di *environment* ini tidak dipastikan. Kalau tidak tersedia, kerjakan langsung tanpa orkestrasi *subagent* formal, jangan berhenti hanya karena *skill* itu tidak ada.
- Perangkat `pytest`/TDD boleh dipakai selama membantu korektnes dengan cepat dan murah (tensor sintetis kecil, seperti contoh pada rencana: 8 *node*, `hidden_dim=16`, 2 epoch). Jangan habiskan waktu besar untuk infrastruktur tes yang tidak esensial dalam sisa waktu 4 jam.
- Seluruh klaim numerik pada rencana tersebut (jumlah *timestep* M1/M2, persentase nilai nol, komposisi regime test 372/168, angka *shadow leaderboard*) **berasal dari analisis agen lain dan harus diverifikasi ulang terhadap dataset lokal sebelum dipercaya**, jangan diterima mentah-mentah.
- Bagian 13 rencana itu sendiri sudah punya jalur mundur: jika waktu tersisa kurang dari 90 menit, lewati Graph WaveNet dan fitur teks, pakai Graph-STID terbaik plus *mean-reversion* dan *ensemble* sederhana. Ikuti semangat ini, dan jangan ragu menyederhanakan lebih jauh lagi kalau situasi menuntut. Prioritas utama tetap satu: **submisi valid yang terbukti mengalahkan anchor 41.795 lewat CV asli**, bukan menyelesaikan seluruh rencana secara literal.

## 4. ATURAN KERAS: Tidak Ada Training Lokal

Ini aturan paling penting untuk sesi v2, tidak boleh dilanggar dalam kondisi apa pun:

- **Laptop lokal tidak punya GPU.** Menjalankan *training* nyata di sana lambat dan menyebabkan *overheat*.
- Eksekusi lokal **hanya** boleh berupa: penulisan/pemeriksaan kode, dan tes struktural/kontrak memakai data sintetis yang sangat kecil (mengikuti contoh pola rencana: beberapa *node*, dimensi kecil, satu-dua epoch), yang selesai dalam hitungan detik.
- **Dilarang** menjalankan *notebook* dengan `RUN_MODE` apa pun (`smoke`, `cv`, `final`) memakai data kompetisi asli secara lokal, termasuk mode `smoke` sekalipun, walau itu didesain ringan di dalam rencana. Tidak ada pengecualian.
- Seluruh *training* nyata (`smoke` untuk verifikasi awal, `cv` untuk validasi, `final` untuk model produksi) **dieksekusi manual oleh pengguna langsung di Kaggle**, bukan oleh Claude Code.
- Karena itu, notebook harus dirancang supaya pengguna bisa menjalankan tahapan itu sendiri di Kaggle secara mandiri (lewat *switch* `CFG` seperti pada rencana), lalu melaporkan hasil/log/error kembali untuk iterasi berikutnya. Jangan mengasumsikan hasil training yang belum benar-benar dijalankan dan dilaporkan penggunanya.

## 5. Struktur Data (tetap sama dari v1)

### 5.1 Lokal (untuk pengembangan/eksplorasi non-training)

```
C:\Users\riant\Downloads\datathon\
├── dataset\
│   │   sample_submission.csv
│   │
│   ├── static\
│   │       active_mask.npy
│   │       matrix.npy
│   │       Roads1260.json
│   │
│   ├── test\
│   │       test_texts.json
│   │       test_X_hist.npy
│   │
│   └── train\
│           train_speed_m1_1_11160.npy
│           train_speed_m2_1_5039.npy
│           train_text_m1_1_11160.json
│           train_text_m2_1_5039.json
│
└── ref\
        findit-v17.ipynb
```

### 5.2 Kaggle (target eksekusi akhir, wajib dipakai di notebook final)

```
/kaggle/input/competitions/datathon-task-1
/kaggle/input/competitions/datathon-task-1/dataset-task1
/kaggle/input/competitions/datathon-task-1/dataset-task1/static
/kaggle/input/competitions/datathon-task-1/dataset-task1/test
/kaggle/input/competitions/datathon-task-1/dataset-task1/train
/kaggle/input/competitions/datathon-task-1/dataset-task1/sample_submission.csv
```

Notebook harus mendeteksi environment secara otomatis dan tetap mengarah benar ke path Kaggle di atas saat dijalankan di sana.

### 5.3 Target Prediksi dan Metrik

- Horizon `h5` = +20 menit, `h10` = +40 menit, `h15` = +60 menit.
- Metrik: MSE, dirata-ratakan lintas semua sampel, horizon, dan segmen jalan.
- Format id submisi: `test_{sample}_h{horizon}_r{road}`, urutan id pada `sample_submission.csv` wajib dipertahankan persis.

## 6. Referensi Struktur Notebook: `findit-v17.ipynb`

Aturan gaya penulisan notebook dari v1 **tetap berlaku** untuk v2:

- Header `<hr>` + judul + subjudul, Daftar Isi dengan anchor, setiap bagian besar diawali/diakhiri `<hr>` + judul beranchor.
- Narasi Indonesia yang mengalir dan menjelaskan alasan, istilah asing/teknis di-*italic*.
- Blockquote `> ### Insights` setelah visualisasi penting di bagian EDA.
- Kelas konfigurasi terpusat (`Settings`/`CFG`) untuk seluruh *hyperparameter*, path, dan mode eksekusi.
- Bagian Daftar Pustaka dan penutup di akhir.

## 7. Aturan Wajib untuk Notebook Final

1. Seluruh narasi markdown berbahasa Indonesia.
2. **Tidak boleh ada karakter em dash ("—") di mana pun**, baik markdown maupun komentar/*docstring* kode.
3. Istilah asing/teknis ditulis *italic*.
4. Notebook harus bisa dieksekusi penuh dengan Run All di Kaggle GPU T4x2, memakai path Kaggle pada bagian 5.2.
5. Output akhir: satu `submission.csv` sesuai format `sample_submission.csv`, urutan id dipertahankan persis.
6. **Satu berkas notebook (`.ipynb`) saja** sebagai keluaran utama yang siap dijalankan di Kaggle. Berkas pendukung (tes, hasil CV, dsb.) boleh ada sebagai artefak kerja, tapi deliverable utamanya tetap satu notebook.

## 8. Prinsip Kompetisi

- Jangan memilih submisi final murni berdasarkan skor *public leaderboard*. Pakai validasi berbasis waktu (bukan *split* acak) karena ini data deret waktu.
- Dalam anggaran waktu dan submisi yang sangat ketat sekarang, **keandalan lebih bernilai daripada kompleksitas**. Satu perbaikan yang terbukti dan teruji lebih baik daripada arsitektur ambisius yang belum tervalidasi penuh.
- **Setiap kandidat submisi baru wajib melewati gerbang validasi eksplisit** (mengalahkan anchor 41.795 secara meyakinkan lewat CV asli dari hasil run Kaggle, bukan cuma skor training) sebelum dipakai memakai salah satu dari 4 slot submisi yang tersisa.
