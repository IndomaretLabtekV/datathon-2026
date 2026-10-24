# Wikipedia Navigation Next-Click Prediction

Predicting the next Wikipedia link a navigator will click while moving from a current article toward a target article.

This repository contains the cleaned implementation developed for **Datathon 2026 Task 2: Wiki Article Next Click Prediction**, where the system reached **0.448 private accuracy** and advanced to the **semifinal round**.

> Competition: [Datathon Task 2 on Kaggle](https://www.kaggle.com/competitions/datathon-task-2/)

## Problem

Each sample describes a navigation state `(current_article, target_article)`. The goal is to predict the exact article link that the navigator clicks next from the current page.

The main challenge is that the dataset does not provide an explicit outgoing-link graph. Candidate links therefore have to be reconstructed from the rendered Wikipedia screenshots before they can be ranked.

**Metric:** exact-match accuracy.

## Approach

The final pipeline treats the task as **candidate generation + learning-to-rank**:

1. **Screenshot link reconstruction**
   - Detect Wikipedia hyperlink-blue regions from page screenshots.
   - Run EasyOCR on link regions.
   - Normalize and fuzzy-match extracted anchor text against known article titles with RapidFuzz.
   - Build an internal directed article graph from the recovered links.

2. **Candidate generation**
   - OCR-reconstructed outgoing links.
   - Fold-safe empirical transitions from labeled training states.
   - Target-conditioned current-article candidates.
   - Sequence-derived candidates when supported by the state ordering audit.

3. **Feature engineering**
   - Directed and undirected graph distance to the target.
   - Graph progress and article degree statistics.
   - TF-IDF title/category similarity.
   - Sentence-Transformer semantic similarity.
   - Cross-encoder navigation relevance.
   - OCR confidence, title-match confidence, link position, order, and occurrence count.
   - Fold-safe behavioral frequencies such as `P(candidate | current)` and `P(candidate | target)`.

4. **Reranking**
   - LightGBM LambdaRank for within-state ranking.
   - A LightGBM binary ranker as a complementary model.
   - OOF-selected blending with semantic and graph-based scores.
   - High-confidence routing rules are enabled only when they improve out-of-fold accuracy.

## Repository Structure

```text
.
├── notebooks/
│   └── wikipedia_navigation_next_click.ipynb
├── dataset/
│   └── README.md
├── .gitignore
├── README.md
└── requirements.txt
```

The competition dataset is intentionally **not included** in this repository.

## Dataset Setup

Download the data from the [competition page](https://www.kaggle.com/competitions/datathon-task-2/) and place it under `dataset/`:

```text
dataset/
├── articles.csv
├── categories.csv
├── states_train.csv
├── states_test.csv
├── sample_submission.csv
└── screenshots/
    ├── 0.png
    ├── 1.png
    └── ...
```

The original competition data contains **4,604 articles**, **9,000 labeled training states**, and **6,000 test states**.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Then open:

```text
notebooks/wikipedia_navigation_next_click.ipynb
```

For the simplest path resolution, start Jupyter from the repository root.

## Outputs

When run locally, generated artifacts are written to:

```text
outputs/wikinav_v2/
```

The main submission file is:

```text
outputs/wikinav_v2/submission_wikinav_v2.csv
```

The notebook also exports diagnostics such as candidate recall, fold metrics, feature importance, OOF routing results, OCR statistics, and submission-source audits.

## Notes

- OCR and pretrained transformer inference are substantially faster with a CUDA-capable GPU, but the OCR initialization falls back to CPU when CUDA is unavailable.
- Sentence-Transformer, cross-encoder, and EasyOCR model weights may need to be downloaded on first use.
- Training-label-derived candidate edges are rebuilt fold-by-fold during validation to avoid leaking validation labels into candidate generation.

## Result

| Metric | Score |
|---|---:|
| Private leaderboard accuracy | **0.448** |

## Tech Stack

`Python` · `LightGBM` · `EasyOCR` · `OpenCV` · `Sentence-Transformers` · `scikit-learn` · `SciPy` · `pandas`
