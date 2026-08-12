# Dataset

The Datathon 2026 Task 1 dataset is **not redistributed in this repository**.

Download it from the official Kaggle competition page:

<https://www.kaggle.com/competitions/datathon-task-1/>

For local execution, place the files in this structure:

```text
dataset/
├── sample_submission.csv
├── static/
│   ├── active_mask.npy
│   ├── matrix.npy
│   └── Roads1260.json
├── test/
│   ├── test_X_hist.npy
│   └── test_texts.json
└── train/
    ├── train_speed_m1_1_11160.npy
    ├── train_speed_m2_1_5039.npy
    ├── train_text_m1_1_11160.json
    └── train_text_m2_1_5039.json
```

The notebook also detects the corresponding Kaggle competition directory automatically when running on Kaggle.
