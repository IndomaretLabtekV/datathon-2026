# Kaggle Traffic Forecasting Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membuat satu notebook Kaggle yang dapat melatih, memvalidasi, membandingkan, dan menghasilkan submission model prakiraan kecepatan lalu lintas tanpa mengubah cell selain cell konfigurasi.

**Architecture:** Notebook memakai baseline mean-reversion sebagai prediksi dasar, kemudian model GPU mempelajari residualnya. Model utama adalah Residual Graph-STID yang menggabungkan riwayat kecepatan, agregasi graph, identitas ruas, regime, dan metadata jalan. Compact Graph WaveNet menjadi model penantang, sedangkan fitur teks hanya diaktifkan setelah model berbasis kecepatan lolos shadow leaderboard.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy sparse, scikit-learn, pandas, nbformat, pytest, Kaggle GPU

---

## 1. Konteks dan Batasan

- Durasi kompetisi: 5 jam.
- Batas submission: 5 kali.
- Riwayat masukan: 15 timestep untuk 1.260 ruas.
- Horizon target: `h5`, `h10`, dan `h15`.
- Metrik resmi: mean squared error (MSE) pada skala kecepatan asli.
- Data train terdiri atas dua regime yang berbeda:
  - M1: 11.160 timestep dan sekitar 1% nilai nol.
  - M2: 5.039 timestep dan sekitar 16,7% nilai nol.
- Test terdeteksi berisi 372 sampel M1 dan 168 sampel M2.
- Shadow leaderboard saat ini:
  - persistence: `53,98 ± 3,37`;
  - mean-reversion: `41,65 ± 2,22`;
  - mean-reversion dengan stride 30: `40,16 ± 2,11`.
- Workspace ini bukan repositori Git. Setiap checkpoint implementasi dilakukan melalui tes dan artefak keluaran, bukan commit.

## 2. Keputusan Metode

| Prioritas | Metode | Peran | Keputusan |
| --- | --- | --- | --- |
| P0 | Mean-reversion | Baseline, residual anchor, dan pemeriksaan keselarasan target | Wajib |
| P1 | Residual Graph-STID | Model utama yang cepat dan efisien | Wajib |
| P2 | Compact Graph WaveNet | Penantang dengan pemodelan graph-temporal eksplisit | Dikerjakan setelah Graph-STID stabil |
| P3 | Text-conditioned residual | Memanfaatkan teks event melalui hashing | Hanya dijalankan jika waktu mencukupi |
| P4 | Ensemble per horizon | Menggabungkan model yang memiliki galat saling melengkapi | Wajib sebelum kandidat submission final |

STID dipilih sebagai dasar karena identitas spasial atau embedding ruas merupakan komponen terpenting pada ablasinya dan biaya latihnya jauh lebih rendah daripada banyak STGNN. Temporal identity asli STID tidak dipakai karena timestamp test tidak tersedia. Sumber: [Spatial-Temporal Identity](https://finleywang.github.io/assets/pdf/2022_CIKM_STID.pdf).

Graph WaveNet dipilih sebagai penantang karena menggabungkan dilated temporal convolution, graph diffusion, dan adaptive dependency. Konfigurasi awal hanya memakai adjacency forward dan backward agar biaya pada 1.260 node tetap terkendali. Sumber: [Graph WaveNet](https://arxiv.org/abs/1906.00121).

AGCRN tidak menjadi model pertama karena adaptive graph generation memiliki biaya tinggi pada graph besar. Sumber: [AGCRN](https://proceedings.neurips.cc/paper/2020/file/ce1aad92b939420fc17005e5461e6f48-Paper.pdf).

## 3. Model Utama: Residual Graph-STID

Prediksi akhir menggunakan bentuk berikut.

```text
prediksi_akhir = prediksi_mean_reversion + residual_model
```

### 3.1 Masukan dinamis per ruas

- Riwayat speed yang dinormalisasi: 15 nilai.
- Agregasi tetangga forward 1-hop dan 2-hop: 30 nilai.
- Agregasi tetangga backward 1-hop dan 2-hop: 30 nilai.
- Statistik temporal:
  - speed terakhir;
  - mean 3, 5, dan 15 timestep;
  - perubahan 1 dan 5 timestep;
  - slope 5 dan 15 timestep;
  - simpangan baku 5 dan 15 timestep.
- Konteks jaringan global pada 15 timestep.

### 3.2 Masukan statis dan identitas

- Learned road embedding untuk 1.260 ruas.
- Total panjang jalan.
- Jumlah segmen pembentuk ruas.
- Proporsi `roadclass`.
- Proporsi `formway`.

### 3.3 Kepala prediksi

- Tiga residual head langsung untuk `h5`, `h10`, dan `h15`.
- Head terakhir diinisialisasi dengan bobot dan bias nol.
- Pada awal pelatihan, keluaran model sama dengan mean-reversion baseline.
- Ruas yang selalu nol pada regime tertentu dipaksa menghasilkan nol.
- Loss dihitung dengan MSE pada skala km/jam, bukan pada target yang telah dinormalisasi.

### 3.4 Regime

M1 dan M2 memakai arsitektur yang sama, tetapi bobot model dan statistik normalisasinya terpisah. Regime test ditentukan dari fraksi nilai nol pada history dengan prototipe yang dihitung dari train. Notebook harus menegaskan bahwa hasil klasifikasi test tetap `372/168`.

## 4. Model Penantang: Compact Graph WaveNet

- Input history: 15 timestep.
- Dilasi temporal: `[1, 2, 4, 8]` agar receptive field mencakup history penuh.
- Residual channels: 32.
- Skip channels: 64.
- Graph diffusion order: 2.
- Adjacency awal: forward dan backward yang dinormalisasi.
- Adaptive adjacency: nonaktif pada eksperimen pertama.
- Output langsung: tiga residual untuk `h5`, `h10`, dan `h15`.
- Prediksi akhir tetap menambahkan mean-reversion baseline.

Adaptive adjacency hanya diaktifkan jika model fixed-graph selesai dalam anggaran waktu dan shadow MSE masih menjanjikan. Jika diaktifkan, embedding graph berukuran 16 dan adjacency dibatasi dengan top-k 16 per node.

## 5. Fitur Teks Opsional

Teks hanya ditambahkan setelah Graph-STID berbasis speed lolos validation gate.

- Ambil teks pada timestep terakhir history train.
- Gunakan `HashingVectorizer` tanpa proses fit:
  - `n_features=256`;
  - `ngram_range=(1, 2)`;
  - `alternate_sign=False`;
  - `norm="l2"`.
- Proyeksikan 256 fitur menjadi embedding 32 dimensi.
- Gabungkan embedding teks dengan road embedding agar model dapat mempelajari dampak frasa event yang berbeda untuk setiap ruas.
- Bandingkan Graph-STID dengan dan tanpa teks pada fold yang sama.

Fitur teks dipertahankan hanya jika mean MSE membaik sekurang-kurangnya 0,75 dan tidak memperburuk lebih dari satu fold.

## 6. Satu Cell Konfigurasi

Cell pertama yang boleh diedit pengguna berisi konfigurasi berikut.

```python
CFG = {
    "DATA_DIR": None,
    "WORK_DIR": "/kaggle/working",
    "RUN_MODE": "smoke",          # smoke | cv | final
    "MODEL": "graph_stid",        # graph_stid | graph_wavenet
    "FOLDS": [2],                  # [2] untuk cepat, [0, 1, 2] untuk CV
    "USE_GRAPH": True,
    "GRAPH_HOPS": 2,
    "USE_ROAD_METADATA": True,
    "USE_TEXT": False,
    "HIDDEN_DIM": 128,
    "ROAD_EMBED_DIM": 32,
    "TEXT_EMBED_DIM": 32,
    "BATCH_SIZE": 16,
    "EPOCHS": 40,
    "PATIENCE": 7,
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "DROPOUT": 0.10,
    "GRAD_CLIP": 1.0,
    "AMP": True,
    "SEED": 42,
    "NUM_WORKERS": 2,
    "PREDICTION_CLIP": [0.0, 160.0],
    "MAKE_SUBMISSION": False,
}
```

Jika `DATA_DIR=None`, notebook mencari satu file `test/test_X_hist.npy` di bawah `/kaggle/input`, kemudian memakai direktori induk yang memuat folder `train`, `test`, dan `static`.

### 6.1 Mode eksekusi

| Mode | Perilaku |
| --- | --- |
| `smoke` | Memakai maksimum 512 window train, 2 epoch, dan fold terakhir untuk memeriksa pipeline |
| `cv` | Memakai fold dalam `CFG["FOLDS"]`, early stopping, dan menyimpan OOF prediction |
| `final` | Melatih dengan seluruh train memakai jumlah epoch hasil CV, melakukan inferensi test, dan membuat submission jika diaktifkan |

## 7. Desain Validasi

Shadow leaderboard harus identik dengan evaluator lokal.

| Fold | Train | Validation |
| --- | --- | --- |
| 0 | 0% sampai 55% | 55% sampai 70% |
| 1 | 0% sampai 70% | 70% sampai 85% |
| 2 | 0% sampai 85% | 85% sampai 100% |

Aturan wajib:

- Seluruh history validation berada setelah batas train.
- Target adalah indeks `origin + 5`, `origin + 10`, dan `origin + 15`.
- Normalizer, mean-reversion coefficient, text transform, dan model hanya memakai train prefix.
- MSE dihitung setelah mengembalikan prediksi ke skala kecepatan asli.
- Skor M1 dan M2 digabung dengan bobot `372/540` dan `168/540`.
- Laporan memuat MSE keseluruhan, per horizon, per regime, dan per fold.

## 8. Kriteria Layak Submission

Model belum boleh menggunakan slot submission jika salah satu syarat berikut tidak terpenuhi.

1. Mean shadow MSE harus lebih kecil dari `39,00`.
2. Model harus mengalahkan mean-reversion pada sekurang-kurangnya 2 dari 3 fold.
3. MSE `h15` tidak boleh lebih buruk daripada mean-reversion.
4. MSE salah satu regime tidak boleh memburuk lebih dari 1%.
5. Seluruh prediksi harus finite dan berada dalam rentang yang ditetapkan konfigurasi.
6. Variasi fold harus dilaporkan. Peningkatan yang lebih kecil daripada noise fold tidak dianggap cukup.

Submission pertama menjadi public anchor. Estimasi public score model berikutnya dihitung dengan:

```text
estimasi_public_model = public_anchor × local_model / local_anchor
```

## 9. Strategi Ensemble

- Kandidat awal: mean-reversion, Graph-STID, dan Graph WaveNet.
- Bobot dipilih terpisah untuk `h5`, `h10`, dan `h15`.
- Bobot harus nonnegatif dan berjumlah 1.
- Fold 0 dan 1 dipakai untuk memilih bobot.
- Fold 2 dipakai untuk memeriksa generalisasi bobot.
- Ensemble dipakai hanya jika memperbaiki fold 2 dan mean tiga fold.

Contoh ruang pencarian tiga model:

```python
WEIGHT_GRID = np.arange(0.0, 1.01, 0.05)
candidate_weights = [
    (w0, w1, 1.0 - w0 - w1)
    for w0 in WEIGHT_GRID
    for w1 in WEIGHT_GRID
    if 0.0 <= 1.0 - w0 - w1 <= 1.0
]
```

## 10. Struktur Notebook

| Cell | Jenis | Isi |
| ---: | --- | --- |
| 1 | Code | Satu-satunya cell `CFG` yang boleh diedit |
| 2 | Markdown | Tujuan, data, metric, dan aturan validasi |
| 3 | Code | Imports, seed, informasi GPU, dan pemeriksaan versi |
| 4 | Code | Auto-discovery path dan validasi file |
| 5 | Code | Load array, graph, metadata, dan teks |
| 6 | Code | Profil ringkas dan deteksi regime test |
| 7 | Code | Fold builder dan leakage assertions |
| 8 | Code | Baseline mean-reversion dan sanity score |
| 9 | Code | Normalizer dan graph feature precomputation |
| 10 | Code | Lazy window dataset dan data loader |
| 11 | Code | Residual Graph-STID definition |
| 12 | Code | Compact Graph WaveNet definition |
| 13 | Code | Training engine, AMP, checkpoint, dan resume |
| 14 | Code | Evaluasi raw-scale MSE |
| 15 | Code | Mode `smoke` atau `cv` |
| 16 | Code | Ringkasan ranking dan acceptance gate |
| 17 | Code | Final training |
| 18 | Code | Test inference dan regime routing |
| 19 | Code | OOF ensemble per horizon |
| 20 | Code | Submission generation dan assertions |
| 21 | Markdown | Ringkasan hasil yang diisi dari keluaran aktual |

## 11. Struktur File

- Create: `Datathon_Task_1_Graph_STID_Kaggle.ipynb`
  - Notebook mandiri yang dapat dijalankan melalui Kaggle Save and Run All.
- Create: `tests/test_kaggle_notebook_contract.py`
  - Memeriksa susunan cell, konfigurasi tunggal, simbol wajib, dan larangan nilai path yang tersebar.
- Create: `tests/conftest.py`
  - Mengeksekusi cell definisi yang bertanda `unit-test-api` dalam namespace terisolasi untuk tes sintetis.
- Create: `tests/test_notebook_model_contract.py`
  - Menjalankan fungsi data dan model pada tensor sintetis.
- Create at runtime: `outputs/notebook_cv_results.json`
  - Ringkasan fold, horizon, regime, epoch terbaik, dan waktu latih.
- Create at runtime: `outputs/oof_predictions.npz`
  - Prediksi out-of-fold kandidat ensemble.
- Create at runtime: `submission.csv`
  - Submission final dengan urutan ID dari `sample_submission.csv`.

## 12. Rencana Implementasi

### Task 1: Notebook skeleton dan konfigurasi tunggal

**Files:**
- Create: `Datathon_Task_1_Graph_STID_Kaggle.ipynb`
- Create: `tests/test_kaggle_notebook_contract.py`

- [ ] **Step 1: Tulis tes kontrak notebook yang gagal**

```python
import nbformat


def test_first_code_cell_is_the_only_config_cell():
    notebook = nbformat.read("Datathon_Task_1_Graph_STID_Kaggle.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells[0].source.startswith("CFG = {")
    assert sum("CFG = {" in cell.source for cell in code_cells) == 1


def test_notebook_exposes_three_run_modes():
    notebook = nbformat.read("Datathon_Task_1_Graph_STID_Kaggle.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert all(mode in source for mode in ['"smoke"', '"cv"', '"final"'])
```

- [ ] **Step 2: Jalankan tes dan pastikan gagal karena notebook belum ada**

Run:

```bash
conda run -n bdc2026-dinov3 python -m pytest tests/test_kaggle_notebook_contract.py -q
```

Expected: `FileNotFoundError` untuk `Datathon_Task_1_Graph_STID_Kaggle.ipynb`.

- [ ] **Step 3: Buat notebook dengan 21 cell sesuai Bagian 10**

Cell pertama harus memuat dictionary `CFG` persis seperti Bagian 6. Cell code lain hanya membaca `CFG` dan tidak mendefinisikan path atau hiperparameter baru yang dapat diubah pengguna.

Cell yang hanya berisi imports, fungsi, dataclass, atau definisi model diberi metadata tag `unit-test-api`. Cell bertanda tersebut tidak boleh memuat data kompetisi, memulai pelatihan, atau menulis artefak.

- [ ] **Step 4: Jalankan kembali tes kontrak**

Expected: seluruh tes pada `test_kaggle_notebook_contract.py` lulus.

### Task 2: Data discovery, target alignment, dan regime routing

**Files:**
- Modify: `Datathon_Task_1_Graph_STID_Kaggle.ipynb`
- Create: `tests/conftest.py`
- Create: `tests/test_notebook_model_contract.py`

- [ ] **Step 1: Buat fixture untuk mengeksekusi API notebook**

```python
from types import SimpleNamespace

import nbformat
import pytest


@pytest.fixture(scope="session")
def notebook_api():
    notebook = nbformat.read(
        "Datathon_Task_1_Graph_STID_Kaggle.ipynb",
        as_version=4,
    )
    namespace = {"__name__": "notebook_under_test"}
    for cell in notebook.cells:
        tags = cell.metadata.get("tags", [])
        if cell.cell_type == "code" and "unit-test-api" in tags:
            exec(compile(cell.source, "<notebook-cell>", "exec"), namespace)
    return SimpleNamespace(**namespace)
```

- [ ] **Step 2: Tulis tes target alignment dan regime inference**

```python
import numpy as np


def test_targets_use_exact_requested_horizons(notebook_api):
    series = np.arange(40, dtype=np.float32)[:, None]
    histories, targets = notebook_api.make_windows(
        series,
        origins=np.array([14]),
        history=15,
        horizons=(5, 10, 15),
    )
    np.testing.assert_array_equal(histories[0, :, 0], np.arange(15))
    np.testing.assert_array_equal(targets[0, :, 0], [19, 24, 29])


def test_regime_inference_uses_zero_fraction(notebook_api):
    prototypes = {"m1": 0.01, "m2": 0.17}
    histories = np.stack([
        np.ones((15, 100), dtype=np.float32),
        np.zeros((15, 100), dtype=np.float32),
    ])
    assert notebook_api.infer_regimes(histories, prototypes).tolist() == ["m1", "m2"]
```

- [ ] **Step 3: Jalankan tes dan pastikan simbol data belum tersedia**

Expected: gagal karena `make_windows` dan `infer_regimes` belum diekspor oleh cell data.

- [ ] **Step 4: Implementasikan discovery dan assertion data**

Notebook harus menolak eksekusi jika shape bukan `(540, 15, 1260)`, matrix bukan `(1260, 1260)`, terdapat nilai nonfinite, atau urutan road berubah.

- [ ] **Step 5: Jalankan tes sampai lulus**

Expected: target alignment dan regime inference lulus pada data sintetis.

### Task 3: Graph dan metadata preprocessing

**Files:**
- Modify: `Datathon_Task_1_Graph_STID_Kaggle.ipynb`
- Modify: `tests/test_notebook_model_contract.py`

- [ ] **Step 1: Tulis tes normalisasi graph**

```python
def test_normalized_adjacency_has_unit_nonempty_rows(notebook_api):
    adjacency = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    normalized = notebook_api.row_normalize(adjacency)
    np.testing.assert_allclose(normalized[:2].sum(axis=1), 1.0)
    np.testing.assert_array_equal(normalized[2], np.zeros(3, dtype=np.float32))
```

- [ ] **Step 2: Jalankan tes dan pastikan gagal karena fungsi belum ada**

- [ ] **Step 3: Implementasikan sparse forward/backward graph features**

Fitur graph dipraproses satu kali dengan `scipy.sparse.csr_matrix`. Hasil hop pertama menjadi masukan hop kedua. Statistik graph train, validation, dan test harus memakai transformasi yang sama.

- [ ] **Step 4: Agregasikan metadata 1.260 ruas**

Setiap elemen `Roads1260.json` diubah menjadi satu baris fitur yang memuat total panjang, jumlah segmen, distribusi `roadclass`, dan distribusi `formway`.

- [ ] **Step 5: Jalankan tes dan pemeriksaan finite**

Expected: seluruh matriks fitur memiliki 1.260 baris dan tidak mengandung `NaN` atau `Inf`.

### Task 4: Baseline parity

**Files:**
- Modify: `Datathon_Task_1_Graph_STID_Kaggle.ipynb`
- Modify: `tests/test_notebook_model_contract.py`

- [ ] **Step 1: Tulis tes bahwa head residual nol mempertahankan baseline**

```python
def test_zero_residual_returns_baseline(notebook_api):
    baseline = np.array([[[40.0], [42.0], [44.0]]], dtype=np.float32)
    residual = np.zeros_like(baseline)
    np.testing.assert_array_equal(notebook_api.compose_prediction(baseline, residual), baseline)
```

- [ ] **Step 2: Implementasikan mean-reversion per regime dan horizon**

Koefisien dihitung hanya dari train prefix. Ruas dengan denominator nol memakai koefisien nol dan mempertahankan nilai terakhir.

- [ ] **Step 3: Jalankan baseline pada seluruh shadow fold**

Expected:

```text
mean MSE ≈ 41.6541
test regimes = {m1: 372, m2: 168}
```

Selisih lebih dari `0,01` harus menghentikan pengerjaan model karena menandakan target atau weighting yang tidak selaras.

### Task 5: Residual Graph-STID

**Files:**
- Modify: `Datathon_Task_1_Graph_STID_Kaggle.ipynb`
- Modify: `tests/test_notebook_model_contract.py`

- [ ] **Step 1: Tulis tes shape dan zero initialization**

```python
import torch


def test_graph_stid_returns_three_horizons_and_starts_at_baseline(notebook_api):
    model = notebook_api.ResidualGraphSTID(
        roads=8,
        dynamic_dim=64,
        static_dim=6,
        hidden_dim=32,
        road_embed_dim=8,
        horizons=3,
        dropout=0.0,
    )
    dynamic = torch.randn(2, 8, 64)
    static = torch.randn(8, 6)
    baseline = torch.randn(2, 8, 3)
    prediction = model(dynamic, static, baseline)
    assert prediction.shape == (2, 8, 3)
    torch.testing.assert_close(prediction, baseline)
```

- [ ] **Step 2: Jalankan tes dan pastikan kelas belum tersedia**

- [ ] **Step 3: Implementasikan embedding, residual MLP, dan output head**

Gunakan tiga residual MLP block dengan `LayerNorm`, `GELU`, dropout, dan skip connection. Inisialisasikan layer terakhir dengan nol.

- [ ] **Step 4: Terapkan inactive-road mask setelah residual composition**

- [ ] **Step 5: Jalankan tes model**

Expected: shape `(2, 8, 3)`, prediksi awal sama dengan baseline, dan gradient finite.

### Task 6: Training engine dan checkpoint

**Files:**
- Modify: `Datathon_Task_1_Graph_STID_Kaggle.ipynb`
- Modify: `tests/test_notebook_model_contract.py`

- [ ] **Step 1: Tetapkan kontrak batch dan hasil pelatihan**

Setiap data loader menghasilkan dictionary dengan tensor `dynamic [N, D]`, `baseline [N, 3]`, `target [N, 3]`, dan `active_mask [N]`. Setelah proses batching, dimensi pertama menjadi batch. Metadata jalan `[N, S]` diberikan satu kali melalui argumen `static_features`.

```python
@dataclass
class TrainingResult:
    history: list[dict[str, float]]
    best_val_mse: float
    best_epoch: int
    checkpoint_path: str
```

- [ ] **Step 2: Tulis smoke test pelatihan**

```python
import torch
from torch.utils.data import DataLoader, Dataset


class TinyTrafficDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(42)
        self.dynamic = torch.randn(4, 8, 16, generator=generator)
        self.baseline = torch.randn(4, 8, 3, generator=generator)
        self.target = self.baseline + 0.1

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "dynamic": self.dynamic[index],
            "baseline": self.baseline[index],
            "target": self.target[index],
            "active_mask": torch.ones(8, dtype=torch.bool),
        }


def test_two_epoch_training_produces_finite_loss(notebook_api, tmp_path):
    loader = DataLoader(TinyTrafficDataset(), batch_size=2, shuffle=False)
    model = notebook_api.ResidualGraphSTID(
        roads=8,
        dynamic_dim=16,
        static_dim=4,
        hidden_dim=16,
        road_embed_dim=4,
        horizons=3,
        dropout=0.0,
    )
    result = notebook_api.train_model(
        model=model,
        train_loader=loader,
        val_loader=loader,
        static_features=torch.zeros(8, 4),
        epochs=2,
        patience=2,
        amp=False,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        checkpoint_path=tmp_path / "tiny.pt",
    )
    assert len(result.history) == 2
    assert np.isfinite(result.best_val_mse)
    assert (tmp_path / "tiny.pt").exists()
```

- [ ] **Step 3: Implementasikan AdamW, AMP, gradient clipping, dan cosine decay**

Gunakan `torch.amp.autocast`, `GradScaler`, `clip_grad_norm_`, checkpoint epoch terbaik, dan early stopping berdasarkan raw validation MSE.

- [ ] **Step 4: Implementasikan resume**

Checkpoint menyimpan model, optimizer, scheduler, scaler, epoch, best MSE, konfigurasi, dan seed.

- [ ] **Step 5: Jalankan smoke test lokal dan Kaggle**

Expected: dua epoch selesai, loss finite, checkpoint dapat dimuat, dan hasil inferensi sebelum serta sesudah reload identik dalam toleransi `1e-5`.

### Task 7: Shadow CV dan acceptance gate

**Files:**
- Modify: `Datathon_Task_1_Graph_STID_Kaggle.ipynb`
- Create at runtime: `outputs/notebook_cv_results.json`
- Create at runtime: `outputs/oof_predictions.npz`

- [ ] **Step 1: Jalankan Graph-STID pada fold 2**

Gunakan `RUN_MODE="cv"` dan `FOLDS=[2]`. Catat waktu per epoch sebelum menentukan jumlah epoch maksimum.

- [ ] **Step 2: Terapkan rapid gate**

Lanjutkan ke tiga fold hanya jika fold 2 mengalahkan mean-reversion fold 2 sekurang-kurangnya 1,0 MSE.

- [ ] **Step 3: Jalankan fold 0, 1, dan 2**

Simpan OOF prediction dalam raw scale dan laporan JSON yang memuat per-fold, per-horizon, serta per-regime MSE.

- [ ] **Step 4: Evaluasi acceptance gate Bagian 8**

Notebook mencetak `SUBMIT_CANDIDATE=True` hanya jika seluruh syarat terpenuhi.

### Task 8: Compact Graph WaveNet

**Files:**
- Modify: `Datathon_Task_1_Graph_STID_Kaggle.ipynb`
- Modify: `tests/test_notebook_model_contract.py`

- [ ] **Step 1: Tulis tes receptive field dan output**

```python
def test_graph_wavenet_covers_history_and_outputs_selected_horizons(notebook_api):
    model = notebook_api.CompactGraphWaveNet(
        roads=8,
        history=15,
        horizons=3,
        dilations=(1, 2, 4, 8),
        residual_channels=8,
        skip_channels=16,
    )
    assert model.receptive_field >= 15
    output = model(torch.randn(2, 1, 8, 15), torch.randn(2, 8, 3))
    assert output.shape == (2, 8, 3)
```

- [ ] **Step 2: Implementasikan gated temporal convolution dan diffusion graph convolution**

- [ ] **Step 3: Jalankan fold 2 fixed-graph**

- [ ] **Step 4: Lanjutkan tiga fold hanya jika rapid gate terpenuhi**

Adaptive adjacency tidak diaktifkan sebelum fixed-graph selesai dan tercatat durasi serta peak VRAM-nya.

### Task 9: Text-conditioned residual

**Files:**
- Modify: `Datathon_Task_1_Graph_STID_Kaggle.ipynb`
- Modify: `tests/test_notebook_model_contract.py`

- [ ] **Step 1: Tulis tes determinisme hashing**

```python
def test_text_hashing_is_deterministic(notebook_api):
    texts = ["road closure on bridge", "construction on highway"]
    first = notebook_api.hash_texts(texts)
    second = notebook_api.hash_texts(texts)
    np.testing.assert_array_equal(first.toarray(), second.toarray())
```

- [ ] **Step 2: Implementasikan hashing dan proyeksi teks**

- [ ] **Step 3: Bandingkan ablation pada fold 2 yang sama**

Cabang teks dipertahankan hanya jika memenuhi syarat Bagian 5.

### Task 10: Ensemble, final training, dan submission

**Files:**
- Modify: `Datathon_Task_1_Graph_STID_Kaggle.ipynb`
- Modify: `tests/test_kaggle_notebook_contract.py`
- Create at runtime: `submission.csv`

- [ ] **Step 1: Pilih bobot ensemble dengan fold 0 dan 1**

- [ ] **Step 2: Verifikasi bobot pada fold 2**

- [ ] **Step 3: Latih final model pada seluruh train**

Jumlah epoch final adalah median best epoch dari shadow CV. Seluruh normalizer dan baseline state di-fit ulang pada semua data train.

- [ ] **Step 4: Lakukan inference test per regime**

Prediksi dikembalikan ke urutan asli 540 sampel, kemudian disusun sebagai `[540, 3, 1260]` dengan urutan horizon `[h5, h10, h15]`.

- [ ] **Step 5: Tulis tes submission contract**

```python
def test_submission_preserves_official_ids():
    sample = pd.read_csv("Dataset/sample_submission.csv")
    submission = pd.read_csv("submission.csv")
    assert submission.shape == (2_041_200, 2)
    assert submission.columns.tolist() == ["id", "speed"]
    assert submission["id"].equals(sample["id"])
    assert np.isfinite(submission["speed"]).all()
    assert submission["speed"].between(0.0, 160.0).all()
```

- [ ] **Step 6: Jalankan Save and Run All di Kaggle**

Notebook harus dapat berjalan dari kernel bersih, menemukan dataset, memuat checkpoint bila tersedia, dan menghasilkan `submission.csv` tanpa perubahan di luar cell konfigurasi.

## 13. Urutan Waktu Eksekusi

| Tahap | Anggaran | Kondisi lanjut |
| --- | ---: | --- |
| Skeleton, data, baseline parity | 20-30 menit | Baseline `41,6541 ± 0,01` |
| Graph-STID smoke dan fold 2 | 20-40 menit | Fold 2 membaik sekurang-kurangnya 1,0 |
| Graph-STID tiga-fold CV | 45-75 menit | Mean MSE `<39,00` dan stabil |
| Compact Graph WaveNet fold 2 | 30-50 menit | Lebih baik atau galatnya komplementer |
| Text ablation | 20-30 menit | Peningkatan sekurang-kurangnya 0,75 |
| Ensemble dan final training | 35-60 menit | Seluruh acceptance gate lulus |
| Submission audit | 10 menit | Semua assertion lulus |

Jika waktu tersisa kurang dari 90 menit, lewati Graph WaveNet dan fitur teks. Gunakan Graph-STID terbaik, mean-reversion, dan beberapa seed untuk ensemble final.

## 14. Perintah Verifikasi

```bash
conda run -n bdc2026-dinov3 python -m pytest \
  tests/test_shadow_leaderboard.py \
  tests/test_kaggle_notebook_contract.py \
  tests/test_notebook_model_contract.py -q
```

```bash
conda run -n bdc2026-dinov3 python -m jupyter nbconvert \
  --to notebook \
  --execute Datathon_Task_1_Graph_STID_Kaggle.ipynb \
  --output /tmp/Datathon_Task_1_Graph_STID_Kaggle.executed.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Kriteria selesai:

- seluruh tes lulus;
- notebook dapat dieksekusi dari awal pada mode `smoke`;
- baseline parity tercapai;
- output training dan validation finite;
- checkpoint dapat dilanjutkan;
- `submission.csv` mempertahankan seluruh ID resmi;
- tidak ada hasil yang diklaim sebelum benar-benar dijalankan.

## 15. Self-Review Plan

- Cakupan spec: konfigurasi tunggal, data, split, baseline, Graph-STID, Graph WaveNet, teks, ensemble, final training, dan submission memiliki task masing-masing.
- Konsistensi tipe: seluruh model menerima baseline `[B, N, 3]` dan menghasilkan prediksi `[B, N, 3]`.
- Konsistensi evaluasi: semua skor memakai raw-scale MSE dan bobot regime test yang sama.
- Cakupan risiko: leakage, regime shift, graph besar, waktu Kaggle, checkpoint, dan submission order memiliki assertion eksplisit.
- Workspace tidak memiliki Git sehingga tidak ada langkah commit atau integrasi branch.
