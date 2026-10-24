# Datathon 2026

Solutions for **Datathon 2026**: two machine learning competition tasks and one innovation track. Every entry reached the **semifinal round**.

| Track | Problem | Approach | Result |
|---|---|---|---|
| [Task 1](task1-traffic-forecasting/) | City traffic speed forecasting (1,260 roads, +20/+40/+60 min) | Regime-routed residual LightGBM on a per-road mean-reversion anchor | **38.831** private MSE |
| [Task 2](task2-wiki-next-click/) | Wikipedia next-click prediction | OCR link reconstruction + LightGBM LambdaRank reranking | **0.448** private accuracy |
| [Innovation](innovation/) | Child growth screening for Posyandu | Home photo-based HAZ estimation + longitudinal LightGBM risk ranking with SHAP | [Paper](innovation/Tunas-Paper.pdf) · [Code](https://github.com/IndomaretLabtekV/tunas-posyandu) |

## Task 1: City Traffic Speed Forecasting

Predict road speeds at three horizons from one hour of history, the road network, and live event text.

- Routes each test window to one of two traffic regimes by zero-speed fraction.
- Builds a per-road anchor: `road_mean + c × (last_speed - road_mean)`, with shrunk coefficients.
- Six LightGBM models (2 regimes × 3 horizons) predict the residual on top of the anchor, using lag, road, network, neighbor-graph, geometry, and event-text features.
- Chronological CV (weighted MSE 33.74) gates the final run; structural zeros are restored after prediction.

Details: [`task1-traffic-forecasting/README.md`](task1-traffic-forecasting/README.md) · [Kaggle](https://www.kaggle.com/competitions/datathon-task-1/)

## Task 2: Wikipedia Next-Click Prediction

Predict the exact link a navigator clicks next from `(current_article, target_article)`, with no link graph provided.

- Reconstructs outgoing links from page screenshots (color masking, EasyOCR, RapidFuzz title matching).
- Generates candidates from OCR links, fold-safe empirical transitions, and target-conditioned priors.
- Ranks with graph distance, TF-IDF, Sentence-Transformer, cross-encoder, OCR, and behavioral features.
- LightGBM LambdaRank + binary ranker, blended with OOF-selected weights.

Details: [`task2-wiki-next-click/README.md`](task2-wiki-next-click/README.md) · [Kaggle](https://www.kaggle.com/competitions/datathon-task-2/)

## Innovation: Tunas

**Tunas** is a home-based monthly growth screening and follow-up prioritization system for Posyandu, connecting parents, Posyandu cadres, and Puskesmas nutritionists.

- Parents submit a monthly photo and weight; a CV pipeline estimates body length, converts it to WHO HAZ, and stores it as an unverified measurement.
- Low-quality images or concerning growth signals are queued for cadre verification, then reviewed by a nutritionist.
- A LightGBM model (M2) ranks the risk of deterioration about three months ahead, explained with SHAP.
- Evaluated on synthetic cohorts with grouped and temporal holdouts: AUPRC 0.274 / 0.298, Recall@20% 0.482 / 0.436. WHO HAZ conversion matches 18 reference cases within 0.026.

Paper (Indonesian): [`innovation/Tunas-Paper.pdf`](innovation/Tunas-Paper.pdf) · Source code: [IndomaretLabtekV/tunas-posyandu](https://github.com/IndomaretLabtekV/tunas-posyandu)

## Repository Structure

```text
.
├── task1-traffic-forecasting/
│   ├── notebooks/trafficcast_v3.ipynb
│   ├── dataset/README.md
│   └── requirements.txt
├── task2-wiki-next-click/
│   ├── notebooks/wikipedia_navigation_next_click.ipynb
│   ├── dataset/README.md
│   └── requirements.txt
└── innovation/
    └── Tunas-Paper.pdf
```

## Quick Start

Each task is self-contained. Competition data is not included; follow each task's `dataset/README.md` to download it.

```bash
cd task1-traffic-forecasting      # or task2-wiki-next-click
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Task 1 is designed to run on Kaggle; Task 2 benefits from a CUDA GPU for OCR and transformer inference.
