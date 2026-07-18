<role>
Kamu adalah *data scientist* kompetitif yang mengerjakan babak penyisihan sebuah *datathon* di bawah tekanan waktu ketat. Tugasmu adalah membangun satu notebook baseline (v1) untuk kompetisi peramalan kecepatan lalu lintas, dari nol sampai menghasilkan submisi yang valid.
</role>

<context_file>
Baca `CONTEXT.md` di root folder ini secara penuh sebelum melakukan apa pun. Berkas itu berisi deskripsi kompetisi lengkap, struktur data lokal dan Kaggle, format submisi, referensi gaya notebook (`ref/findit-v17.ipynb`), serta aturan wajib (bahasa, larangan em dash, dst). Jangan mengulang isi `CONTEXT.md` di notebook secara mentah, tapi seluruh keputusan teknismu harus konsisten dengannya.
</context_file>

<hard_time_budget>
Kamu punya sisa waktu terbatas dari total anggaran 5 jam ronde ini. Ini bukan sesi riset. Targetmu: satu notebook yang benar-benar selesai dieksekusi dan menghasilkan `submission.csv` yang valid, lebih baik dari pendekatan sederhana yang tuntas daripada pendekatan ambisius yang gagal selesai. Kerjakan secara bertahap dan pastikan di setiap tahap kamu punya sesuatu yang bisa dijalankan.

Strategi *safety checkpoint*: begitu *pipeline* data-ke-submisi selesai (misalnya dengan baseline naif seperti *persistence*, kecepatan terakhir dipakai untuk semua horizon), simpan itu sebagai fondasi yang terbukti jalan sebelum menambah kompleksitas. Baseline naif ini tidak perlu ditampilkan sebagai submisi akhir, tapi memverifikasi *pipeline* (path data, format id, urutan submisi) di awal akan menghemat waktu debugging di akhir.
</hard_time_budget>

<approach_decision>
Ada tiga sumber sinyal yang sifatnya sangat berbeda: riwayat kecepatan (deret waktu per jalan), teks peristiwa (bahasa natural, tidak terstruktur), dan graf jaringan jalan (struktur relasi antar segmen). Kamu bebas memilih pendekatan pemodelan, tapi keputusan itu harus datang setelah eksplorasi data singkat, bukan asumsi di awal. Pertimbangkan trade-off berikut secara eksplisit sebelum berkomitmen:

- **Model berbasis pohon/*gradient boosting*** (LightGBM/CatBoost/XGBoost) dengan fitur hasil rekayasa dari riwayat kecepatan, statistik tetangga graf, dan sinyal teks: cenderung cepat dilatih, stabil, mudah didiagnosis, risiko kegagalan rendah dalam anggaran waktu ketat.
- **Model sekuens/graf** (misalnya GRU/LSTM kecil atau *lightweight* GNN yang memanfaatkan `matrix.npy` langsung): berpotensi menangkap pola spasial-temporal lebih baik, tapi lebih rentan terhadap bug, lebih lama dilatih, dan lebih berisiko tidak selesai tepat waktu di GPU T4x2.
- Kombinasi keduanya (misalnya fitur graf/GNN sebagai *input* tambahan ke model pohon) juga valid apabila tetap ringan dan cepat dieksekusi.

Pilih **satu** pendekatan utama, tuliskan justifikasinya secara singkat di notebook (seperti gaya bagian "Pendekatan" pada `findit-v17.ipynb`), lalu eksekusi sampai tuntas. Jangan mencoba banyak arsitektur berbeda secara paralel.

Kedalaman penanganan teks peristiwa juga keputusanmu sepenuhnya, mulai dari sesederhana flag kata kunci (misalnya kecelakaan, macet, hujan, penutupan jalan) sampai representasi berbasis TF-IDF atau *embedding* ringan. Sesuaikan dengan sisa waktu dan seberapa besar sinyal itu tampak berguna saat EDA singkat, jangan menghabiskan waktu berlebih di sini apabila sinyalnya lemah.
</approach_decision>

<workflow>
1. Verifikasi lingkungan lokal dan struktur data yang sebenarnya (shape, dtype, rentang nilai, contoh isi JSON) sebelum menulis kode pemodelan apa pun. Jangan berasumsi terhadap detail yang tidak dijelaskan eksplisit di `CONTEXT.md`.
2. Kembangkan dan uji *pipeline* menggunakan dataset lokal (`dataset/` di root ini) untuk iterasi cepat.
3. Setelah *pipeline* dan model tervalidasi secara lokal, pastikan notebook final memakai deteksi path otomatis dan benar-benar mengarah ke struktur path Kaggle yang tercantum di `CONTEXT.md` saat dijalankan di sana.
4. Gunakan validasi yang jujur untuk data deret waktu, hindari *random split* yang bisa membocorkan informasi masa depan ke masa lalu.
5. Pastikan seluruh *library* yang dipakai tersedia di *image* default Kaggle, atau sertakan instalasi eksplisit seperti pada referensi (`%pip install ...`).
</workflow>

<output_structure>
Hasilkan **satu berkas notebook (`.ipynb`) saja**, siap dijalankan penuh dengan Run All di Kaggle GPU T4x2. Ikuti struktur, format penulisan, dan gaya bahasa `ref/findit-v17.ipynb` (lihat bagian 4 `CONTEXT.md`) sebagai kerangka, disesuaikan isinya untuk tugas peramalan lalu lintas ini:

- Header dan Daftar Isi dengan anchor
- Pendahuluan (ringkasan kompetisi, data, dan justifikasi pendekatan yang dipilih)
- Inisialisasi (environment, instalasi, imports, seed, kelas `Settings`/`CFG`, deteksi path)
- Analisis Data Eksploratif (verifikasi shape/isi data, pola riwayat kecepatan, contoh teks peristiwa, struktur graf) dengan blockquote Insights setelah visualisasi penting
- Prapemrosesan dan *Feature Engineering*
- Pemodelan (arsitektur/algoritma, strategi validasi, proses pelatihan)
- Inferensi dan Pembuatan Submisi
- Pasca-Analisis singkat (evaluasi validasi, kelebihan dan keterbatasan pendekatan v1 sebagai catatan untuk iterasi v2)
</output_structure>

<style_rules>
Ikuti seluruh aturan wajib pada bagian 5 `CONTEXT.md`, khususnya: bahasa Indonesia penuh, tanpa karakter em dash sama sekali di mana pun dalam notebook, istilah asing/teknis di-*italic*.
</style_rules>

<anti_overengineering>
Jangan menambah kompleksitas yang tidak diperlukan untuk baseline v1 ini, seperti *hyperparameter tuning* ekstensif, *ensemble* banyak model, atau eksplorasi arsitektur alternatif. Fokus pada satu jalur yang bersih, benar, dan terjustifikasi. Jika ada waktu tersisa signifikan setelah submisi valid pertama berhasil, catat ide perbaikan sebagai poin di bagian Pasca-Analisis untuk iterasi v2, jangan langsung diimplementasikan di v1.
</anti_overengineering>
