# CONTEXT.md - Datathon Task 1: Peramalan Kecepatan Lalu Lintas

Berkas ini adalah konteks tetap untuk seluruh sesi pengerjaan Task 1. Baca berkas ini secara penuh sebelum membuka `PROMPT_V1.md`.

---

## 1. Ringkasan Kompetisi

Lalu lintas tidak bergerak acak. Ia merespons jaringan jalan, waktu, dan peristiwa yang terjadi di kota (kecelakaan, penutupan jalan, cuaca, kerumunan) yang dideskripsikan lewat aliran teks.

**Tugas**: berdasarkan riwayat kecepatan satu jam terakhir (15 *timestep*) pada 1.260 segmen jalan yang saling terhubung, ditambah teks peristiwa langsung, ramalkan kecepatan lalu lintas 20, 40, dan 60 menit ke depan.

**Tiga sumber sinyal per sampel**:
1. **Riwayat kecepatan** - jendela 15 *timestep* (satu jam) untuk seluruh 1.260 segmen.
2. **Teks peristiwa** - deskripsi bahasa natural tentang kejadian di jaringan jalan, sejenis konteks yang dibaca dispatcher manusia sebelum memprediksi perubahan lalu lintas.
3. **Jaringan jalan** - matriks adjacency 1.260 x 1.260 plus geometri dan metadata tiap segmen.

**Target prediksi**: kecepatan (km/jam) tiap segmen jalan pada tiga horizon:
- `h5` = +20 menit
- `h10` = +40 menit
- `h15` = +60 menit

**Metrik evaluasi**: MSE, dirata-ratakan lintas semua sampel, horizon, dan segmen jalan.

**Format submisi**: satu baris per (sampel, horizon, jalan), id berpola `test_{sample}_h{horizon}_r{road}`, contoh `test_00000_h5_r0`. Urutan id pada `sample_submission.csv` wajib dipertahankan persis.

---

## 2. Kendala Waktu dan Konteks Eksekusi

- Ronde 1 (penyisihan) berdurasi **5 jam**, jam 12.00 sampai 17.00.
- `PROMPT_V1.md` adalah prompt baseline pertama, dieksekusi oleh **Claude Code menggunakan model Fable 5, effort xhigh**.
- Ini bukan sesi eksplorasi bebas. Prioritas mutlak: satu pendekatan yang solid, terjustifikasi lewat EDA singkat, dan **jalan sampai selesai menghasilkan submisi valid**. Lebih baik satu pendekatan sederhana yang tuntas daripada pendekatan ambisius yang tidak sempat selesai.
- *Compute* target akhir: **Kaggle, GPU T4x2** (dua unit NVIDIA T4, 16 GB VRAM masing-masing). Tidak ada asumsi ketersediaan GPU kelas atas (A100/H100). Model atau *batch size* yang menuntut VRAM besar berisiko OOM atau lambat, pertimbangkan ini saat memilih pendekatan.

---

## 3. Struktur Data

### 3.1 Lokal (untuk pengembangan/eksplorasi)

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

### 3.2 Kaggle (target eksekusi akhir, wajib dipakai di notebook final)

```
/kaggle/input/competitions/datathon-task-1
/kaggle/input/competitions/datathon-task-1/dataset-task1
/kaggle/input/competitions/datathon-task-1/dataset-task1/static
/kaggle/input/competitions/datathon-task-1/dataset-task1/test
/kaggle/input/competitions/datathon-task-1/dataset-task1/train
/kaggle/input/competitions/datathon-task-1/dataset-task1/sample_submission.csv
```

Catatan penting: root lokal (`dataset/`) dan root Kaggle (`dataset-task1/`) punya nama folder berbeda meski isinya setara. Notebook harus mendeteksi environment secara otomatis (pola serupa `Path('/kaggle/working').exists()` pada `findit-v17.ipynb`) supaya path menyesuaikan sendiri. **Namun target akhir yang harus benar-benar berjalan adalah path Kaggle di atas.**

### 3.3 Deskripsi berkas

| Berkas | Isi |
|---|---|
| `train_speed_m1_*.npy` | Deret kecepatan kontinu, blok 1. Baris = *timestep*, kolom = 1.260 jalan. |
| `train_speed_m2_*.npy` | Deret kecepatan kontinu, blok 2 (jalan yang sama, kemungkinan rentang waktu berbeda). |
| `train_text_m1_*.json` | Teks peristiwa selaras dengan *timestep* blok 1. |
| `train_text_m2_*.json` | Teks peristiwa selaras dengan *timestep* blok 2. |
| `test_X_hist.npy` | Jendela riwayat test: satu jendela riwayat 15 *timestep* per sampel, seluruh jalan. |
| `test_texts.json` | Teks peristiwa, satu entri per sampel test. |
| `sample_submission.csv` | Daftar id lengkap yang harus diprediksi, terisi `0.0`. |
| `matrix.npy` | Adjacency antar 1.260 segmen jalan. |
| `Roads1260.json` | Geometri dan metadata per segmen (koordinat, panjang, id, dst). |
| `active_mask.npy` | Mask boolean yang mengaitkan grid spasial penuh dengan 1.260 segmen aktif. |

Angka pada nama berkas (`_1_11160`, `_1_5039`) kemungkinan menunjukkan jumlah baris/*timestep*. **Jangan berasumsi buta soal shape, dtype, atau isi persis tiap berkas.** Semua ini harus diverifikasi lewat inspeksi langsung di awal notebook (EDA), bukan ditebak dari deskripsi ini.

---

## 4. Referensi Struktur Notebook: `findit-v17.ipynb`

Berkas `ref/findit-v17.ipynb` (kompetisi klasifikasi *face anti-spoofing*, tidak ada hubungannya secara metode dengan Task 1 ini) dipakai **murni sebagai referensi struktur, format penulisan, gaya bahasa, dan cara penyampaian**. Jangan tiru pendekatan/metodenya (DINOv3, *subject-disjoint split*, dst), itu tidak relevan untuk peramalan lalu lintas.

Elemen yang wajib ditiru dari referensi:

- Header notebook: `<hr>` dekoratif + judul `<h1>` + subjudul `<h3>` abu-abu.
- Sel **Daftar Isi** di awal, dengan tautan anchor (`[Judul](#n)`) ke tiap bagian besar.
- Setiap bagian besar diawali `<hr>` + judul `<h1>` dengan anchor `<a name="n"></a>` + `<hr>` penutup.
- Narasi Indonesia yang mengalir dan menjelaskan alasan (bukan sekadar label kode), istilah asing/teknis di-*italic* (*training*, *forecasting*, *feature engineering*, dst).
- Blockquote `> ### Insights` setelah visualisasi penting di bagian EDA, isinya interpretasi singkat yang menuntun keputusan berikutnya.
- Kelas `Settings`/`CFG` terpusat di awal yang menampung seluruh *hyperparameter*, path, dan konfigurasi *run*.
- Sel *environment check* di awal (`!nvidia-smi`, versi Python/OS).
- Bagian **Daftar Pustaka** dan **Thanks** di akhir.

Alur besar bagian (sesuaikan judul dengan konteks peramalan lalu lintas, bukan disalin literal):

Pendahuluan -> Inisialisasi (environment, instalasi, *imports*, *seed*, `Settings`, deteksi path) -> Analisis Data Eksploratif -> Prapemrosesan/*Feature Engineering* -> Pemodelan -> Inferensi dan Submisi -> Pasca-Analisis -> Daftar Pustaka.

---

## 5. Aturan Wajib untuk Notebook Final

1. Seluruh narasi markdown berbahasa Indonesia.
2. **Tidak boleh ada karakter em dash ("—") di mana pun**, baik di sel markdown maupun komentar/*docstring* kode. Gunakan tanda hubung biasa ("-"), koma, atau kalimat baru.
3. Istilah asing/teknis ditulis *italic* mengikuti konvensi referensi.
4. Notebook harus bisa dieksekusi penuh dengan **Run All** langsung di Kaggle, GPU T4x2, memakai path Kaggle pada bagian 3.2.
5. Output akhir: satu `submission.csv` sesuai format `sample_submission.csv`, urutan id dipertahankan persis.
6. **Satu berkas notebook (`.ipynb`) saja** sebagai keluaran, siap pakai.

---

## 6. Prinsip Kompetisi dari Pengalaman Sebelumnya

- Jangan memilih submisi final murni berdasarkan skor *public leaderboard*. Pertimbangkan validasi yang stabil dan masuk akal secara metodologis, ini adalah data deret waktu, jadi validasi berbasis waktu (bukan *split* acak) jauh lebih jujur untuk mengukur performa nyata.
- Dalam anggaran waktu ketat, keandalan (pendekatan yang pasti selesai dan menghasilkan submisi valid) lebih bernilai daripada kompleksitas yang belum tentu sempat dieksekusi dan divalidasi dengan baik.
