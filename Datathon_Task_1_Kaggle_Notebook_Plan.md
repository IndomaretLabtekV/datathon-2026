# Rencana Kaggle Notebook Task 1 - Revisi v2

> Revisi ini disusun setelah mengevaluasi hasil konkret dari dua artefak:
> - `datathon-task1-v1.ipynb` - model LightGBM yang sudah disubmit.
> - `Datathon_Task_1_Graph_STID_Kaggle.ipynb` - model Residual Graph-STID yang sudah dijalankan dalam mode *smoke* di Kaggle.
>
> Tujuan revisi: memastikan sisa waktu dan sisa slot submisi digunakan untuk satu kandidat yang terbukti mengalahkan *anchor* 41.795, bukan mengejar arsitektur ambisius yang belum tervalidasi.

---

## 1. Ringkasan Evaluasi

### 1.1 Apa yang berhasil dari v1

- **Skor publik v1: 41.795.** Ini hanya sedikit lebih baik dari *baseline mean-reversion* sekitar 41.65, artinya fitur graf dan teks pada v1 belum benar-benar dieksploitasi.
- **Validasi lokal v1 terlalu optimis:** tabel validasi menunjukkan MSE rata-rata LightGBM 32.020, jauh di bawah skor publik. Jangan memakai angka validasi v1 sebagai estimasi publik.
- **Aturan nol dan validasi berbasis waktu sudah benar.** Keduanya harus dipertahankan di v2.

### 1.2 Apa yang berhasil dari Graph-STID (smoke run)

Notebook `Datathon_Task_1_Graph_STID_Kaggle.ipynb` sudah berjalan penuh di Kaggle dalam mode *smoke*:

| Metrik | Nilai smoke fold 2 | Baseline fold 2 | Delta |
| --- | --- | --- | --- |
| MSE gabungan | **35.5396** | 40.6703 | -5.1307 |
| MSE h5 | 30.8971 | 36.3948 | -5.4976 |
| MSE h10 | 35.9752 | 41.2705 | -5.2953 |
| MSE h15 | 39.7466 | 44.3457 | -4.5992 |

- Model berhasil mengalahkan *baseline* pada fold 2 dengan margin besar.
- *Acceptance gate* hanya gagal karena baru 1 fold yang dijalankan, bukan karena kualitas model.
- Waktu pelatihan terukur: sekitar **2.5 detik per step** pada GPU T4, sehingga rencana asli yang mengabaikan biaya waktu perlu direvisi.
- Semua *assertion* data, baseline, inisialisasi nol, dan submission lulus.

### 1.3 Apa yang perlu dikoreksi dari rencana asli

- **Graph WaveNet dan fitur teks terlalu mahal untuk sisa waktu.** Keduanya dihapus dari jalur utama dan dipindahkan ke *icebox*.
- **Pipeline pytest/TDD belum dibangun.** Karena waktu terbatas, pengujian mengandalkan *assertion* internal notebook dan *smoke run*, bukan suite pytest terpisah.
- **Konfigurasi asli tidak mengatur *training stride* / *max samples*.** Tanpa itu, full 3-fold CV akan memakan waktu 6-12 jam.

---

## 2. Kesimpulan Strategis

1. **Fokus utama adalah Residual Graph-STID.** Arsitektur ini sudah terbukti mengalahkan *baseline* dan cukup ringan.
2. **Jangan submit tanpa bukti dari CV 3 fold.** Satu slot submisi sangat berharga.
3. **Gunakan *training stride* atau *max samples* untuk masuk ke anggaran waktu.** Kehilangan sebagian data latih lebih aman daripada kehabisan waktu.
4. **Ensemble wajib:** gabungkan prediksi Graph-STID dengan *mean-reversion* per horizon menggunakan bobot dari OOF fold 0 dan 1.
5. **Training nyata tetap di Kaggle.** Lokal hanya untuk menulis/memeriksa kode dan tes sintetis kecil.

---

## 3. Perubahan Konfigurasi (Cell 1)

Tambahkan dan ubah beberapa kunci di `CFG`:

```python
CFG = {
    # ---- Path ----
    "DATA_DIR": None,
    "WORK_DIR": "/kaggle/working",

    # ---- Mode eksekusi ----
    "RUN_MODE": "cv",              # smoke | cv | final
    "MODEL": "graph_stid",         # graph_stid saja untuk jalur utama
    "FOLDS": [0, 1, 2],            # wajib 3 fold saat CV

    # ---- Sampling data latih (baru) ----
    "TRAIN_STRIDE": 3,             # stride antar origin training; turunkan jika masih ada waktu
    "TRAIN_MAX_SAMPLES": None,     # alternatif: batasi windows per fold, contoh 4096

    # ---- Arsitektur ----
    "USE_GRAPH": True,
    "GRAPH_HOPS": 2,
    "USE_ROAD_METADATA": True,
    "USE_TEXT": False,             # dihapus dari scope v2
    "HIDDEN_DIM": 128,
    "ROAD_EMBED_DIM": 32,
    "TEXT_EMBED_DIM": 0,

    # ---- Pelatihan ----
    "BATCH_SIZE": 32,              # naikkan dari 16 untuk mempercepat; turunkan ke 16 jika OOM
    "EPOCHS": 20,                  # cukup dengan early stopping
    "PATIENCE": 5,                 # lebih pendek agar tidak membuang waktu
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "DROPOUT": 0.10,
    "GRAD_CLIP": 1.0,
    "AMP": True,

    # ---- Lainnya ----
    "SEED": 42,
    "NUM_WORKERS": 2,
    "PREDICTION_CLIP": [0.0, 160.0],
    "MAKE_SUBMISSION": False,      # aktifkan hanya pada mode final / saat yakin
}
```

**Catatan implementasi:** `LazyTrafficDataset` dan `build_fold_origins` perlu diubah agar menghormati `TRAIN_STRIDE`. Contoh implementasi minimal:

```python
train_origins = np.arange(min_origin, min(train_end_idx, max_origin) + 1, CFG["TRAIN_STRIDE"])
```

Jika `TRAIN_MAX_SAMPLES` tidak `None`, ambil subset acak dari `train_origins` setelah stride.

---

## 4. Prioritas yang Diperbarui

| Prioritas | Metode | Peran | Keputusan |
| --- | --- | --- | --- |
| P0 | Mean-reversion | Baseline, komponen ensemble, dan pemeriksaan keselarasan target | Wajib, sudah ada |
| P1 | Residual Graph-STID | Model utama; satu-satunya model yang sudah tervalidasi | Wajib, fokus utama |
| P2 | Ensemble per horizon (Graph-STID + mean-reversion) | Menggabungkan sinyal komplementer | Wajib sebelum submit |
| P3 | Final training all-data | Maksimalkan data sebelum submisi final | Jalankan jika gate lulus |
| P4 | Compact Graph WaveNet | Penantang dengan graph-temporal eksplisit | *Icebox*, hanya jika tersisa >90 menit |
| P5 | Fitur teks | Memanfaatkan deskripsi peristiwa | Dihapus dari scope v2 |

---

## 5. Timeline dan Anggaran Waktu Realistis

Asumsi: **~2.5 detik per step** pada T4, batch size 32, dan `TRAIN_STRIDE=3`.

| Tahap | Langkah | Perkiraan waktu | Kondisi lanjut |
| --- | --- | --- | --- |
| 1 | Modifikasi konfigurasi + stride + *smoke* | 15-25 menit | Semua assertion lulus |
| 2 | CV 3 fold Graph-STID (stride=3, batch=32) | 1.5-2.5 jam | Mean MSE < 39 dan mengalahkan baseline pada >=2 fold |
| 3 | Ensemble OOF per horizon | 10-15 menit | Bobot terverifikasi pada fold 2 |
| 4 | Final training all-data + submission | 45-75 menit | Jalankan hanya jika gate lulus |
| 5 | Submission audit | 5-10 menit | ID, shape, dan rentang valid |

**Total rencana:** sekitar 3-4 jam. Jika waktu tersisa <90 menit saat tahap 2 selesai, lewati tahap 4 dan submit langsung dari hasil CV + ensemble.

---

## 6. Acceptance Gate untuk Submission

Sebelum memakai salah satu dari 4 slot submisi yang tersisa, kandidat harus memenuhi semua syarat berikut:

1. **Mean MSE 3-fold < 39.00.**
2. **Model mengalahkan baseline mean-reversion pada minimal 2 dari 3 fold.**
3. **MSE h15 tidak boleh lebih buruk dari baseline h15.**
4. **MSE salah satu regime tidak memburuk lebih dari 1% dari baseline regime yang sama.**
5. **Seluruh prediksi finite dan berada dalam rentang `[0.0, 160.0]`.**
6. **Variasi fold dilaporkan:** std MSE fold harus > 0 dan tidak boleh lebih besar dari mean/10.
7. **Estimasi publik harus di bawah anchor 41.795.** Gunakan:

```text
estimasi_public = public_anchor_v1 * local_model_mse / local_baseline_mse
```

Dengan angka v1: `public_anchor = 41.795` dan `local_baseline` dari CV 3 fold Graph-STID. Target estimasi publik < 39.

---

## 7. Langkah Eksekusi yang Direkomendasikan

### Langkah 1: Modifikasi notebook

- Tambahkan `TRAIN_STRIDE` dan `TRAIN_MAX_SAMPLES` ke `CFG`.
- Terapkan stride pada `build_fold_origins` untuk origin training.
- Naikkan `BATCH_SIZE` menjadi 32 (turunkan ke 16 jika OOM).
- Pastikan `RUN_MODE="smoke"` dan `FOLDS=[2]` saat verifikasi cepat.

### Langkah 2: Smoke run

- Jalankan *Save and Run All* di Kaggle dengan `RUN_MODE="smoke"`.
- Periksa: baseline mean-reversion wajar, model latih 2 epoch tanpa error, submission_smoke.csv valid.

### Langkah 3: CV 3 fold

- Ubah `CFG`:
  - `RUN_MODE = "cv"`
  - `FOLDS = [0, 1, 2]`
  - `TRAIN_STRIDE = 3`
  - `BATCH_SIZE = 32`
  - `EPOCHS = 20`, `PATIENCE = 5`
- Jalankan *Save and Run All* di Kaggle.
- Catat `notebook_cv_results.json` dan hasil *acceptance gate*.

### Langkah 4: Keputusan submit

- Jika gate lulus, submit hasil CV + ensemble sebagai kandidat pertama v2.
- Jika tidak lulus, laporkan hasil per fold dan kita iterasi hyperparameter (misalnya turunkan `TRAIN_STRIDE`, ubah `LR`, atau tambahkan *dropout*).

### Langkah 5: Final training (opsional, jika waktu cukup)

- Ubah `CFG`:
  - `RUN_MODE = "final"`
  - `TRAIN_STRIDE = 2` (atau 1 jika masih ada waktu)
  - `MAKE_SUBMISSION = True`
- Jalankan lagi untuk melatih pada seluruh data train dan menghasilkan `submission.csv` final.

---

## 8. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
| --- | --- | --- |
| Waktu pelatihan melebihi anggaran | Kehabisan waktu sebelum submit | Gunakan `TRAIN_STRIDE=3`, `BATCH_SIZE=32`, `PATIENCE=5`; jika masih lambat, naikkan stride atau batasi `EPOCHS` |
| OOM dengan batch 32 | Run gagal | Turunkan `BATCH_SIZE` ke 16 atau kurangi `HIDDEN_DIM` ke 96 |
| Overfit karena data latih sedikit | Skor publik lebih buruk dari CV | Pertahankan *early stopping*, *dropout*, dan validasi berbasis waktu |
| Ensemble tidak membantu | Hasil lebih buruk dari model tunggal | Hanya pakai ensemble jika bobot OOF memperbaiki fold 2 |
| Graph WaveNet memakan waktu tanpa hasil | Kehilangan slot submisi | Jangan jalankan WaveNet kecuali tersisa >90 menit setelah Graph-STID lolos gate |

---

## 9. Catatan untuk Iterasi Lanjut (v3 / kompetisi mendatang)

Jika masih ada waktu setelah Graph-STID stabil:

1. **Graph WaveNet:** jalankan *smoke* fold 2 untuk memeriksa apakah galatnya komplementer dengan Graph-STID.
2. **Fitur teks:** implementasi minimal dengan `HashingVectorizer` + proyeksi 32 dimensi, hanya jika peningkatan validasi >= 0.75 MSE.
3. **Kalibrasi residual per jalan:** koreksi bias per segmen dari residual validasi.
4. **Dua model terpisah per regime:** latih bobot terpisah untuk M1 dan M2 jika regime shift masih besar.

---

## 10. Referensi Cepat

- v1: `datathon-task1-v1.ipynb` - skor publik 41.795, validasi LightGBM 32.020.
- Graph-STID: `Datathon_Task_1_Graph_STID_Kaggle.ipynb` - smoke fold 2 MSE 35.5396.
- Context: `CONTEXT.md` - aturan keras "tidak ada training lokal" dan struktur path Kaggle.
