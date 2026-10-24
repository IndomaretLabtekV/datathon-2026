# City Traffic Speed Forecasting

**Datathon 2026 Task 1 — Semifinalist**

A traffic-speed forecasting pipeline for a city network of **1,260 road segments**, predicting speeds at **+20, +40, and +60 minutes** from one hour of recent history, road-network information, and live event text.

**Private leaderboard MSE: 38.831**

## Overview

The competition provides three complementary signals for every forecasting state:

- **Speed history:** 15 recent timesteps across 1,260 road segments.
- **Event text:** natural-language descriptions of incidents and conditions affecting the network.
- **Road network:** adjacency, geometry, and metadata for all road segments.

The final solution, **TrafficCast v3**, uses a routed collection of **six residual LightGBM models**, one for each `(traffic regime, forecast horizon)` pair. Instead of predicting speed directly, each model predicts a correction on top of a per-road mean-reversion forecast.

Competition page: [Datathon Task 1: City Network Traffic Forecasting](https://www.kaggle.com/competitions/datathon-task-1/)

## Result

| Metric | Value |
|---|---:|
| Private leaderboard MSE | **38.831** |
| Internal weighted time-based CV MSE | **33.7376** |
| Forecast horizons | +20, +40, +60 minutes |
| Road segments | 1,260 |
| Final residual models | 6 LightGBM models |

The internal CV score is included as model-development evidence and should not be compared directly with the private leaderboard because the validation distribution is only a proxy for the hidden test set.

## Approach

### 1. Traffic-regime routing

The two training blocks have noticeably different structural-zero patterns. Test windows are routed to the closest training regime using their zero-speed fraction, allowing the model to handle the two operating conditions separately.

### 2. Per-road mean-reversion anchor

For every road and horizon, the pipeline first builds a strong deterministic baseline:

```text
anchor = road_mean + c_road,horizon × (last_speed - road_mean)
```

The per-road coefficient is estimated from historical autocorrelation and shrunk toward a robust central value to reduce instability on sparse roads.

### 3. Feature engineering

The residual models combine compact signals from several sources:

- recent speed lags and temporal summaries;
- per-road historical statistics and identity;
- global network state;
- lightweight neighbor-graph anomaly features;
- road geometry and metadata;
- event-text keyword counts;
- the mean-reversion anchor and horizon-specific coefficient.

### 4. Residual LightGBM models

Rather than learning raw future speed, LightGBM predicts:

```text
residual = future_speed - anchor
```

A separate model is trained for each of the three horizons inside each traffic regime. This keeps each learner specialized while retaining the stability of the deterministic anchor.

### 5. Time-based validation

Validation uses chronological folds instead of random splits to avoid future leakage. Model acceptance is gated by weighted CV performance against the anchor before the final training and submission path is enabled.

For the second regime, training examples also receive mild recency weighting because its later observations more closely resemble the test distribution.

### 6. Post-processing

Final predictions:

1. add the predicted residual to the anchor;
2. restore structural zero values when the latest road reading is zero;
3. clip speeds to the valid range;
4. preserve the exact ordering of `sample_submission.csv`.

## Repository Structure

```text
task1-traffic-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
│   └── README.md
└── notebooks/
    └── trafficcast_v3.ipynb
```

The competition dataset is intentionally **not included** in this repository.

## Setup

### Option A: Kaggle

The notebook was designed to run directly on Kaggle and automatically searches under `/kaggle/input` for the competition data.

1. Open `notebooks/trafficcast_v3.ipynb` in Kaggle.
2. Attach the **Datathon Task 1** competition dataset.
3. Choose the desired `CFG.RUN_MODE`:
   - `smoke` for a lightweight pipeline check;
   - `cv` for validation only;
   - `final` for validation, full training, and `submission.csv` generation.
4. Run all cells.

### Option B: Local environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/trafficcast_v3.ipynb
```

Download the competition data separately and place it under `dataset/` using the layout documented in [`dataset/README.md`](dataset/README.md).

Full training is computationally heavy, so Kaggle is the recommended environment for reproducing the final run.

## Submission Format

The competition expects one prediction per road, horizon, and test sample:

```csv
id,speed
test_00000_h5_r0,44.0
test_00000_h5_r1,73.0
...
```

with horizons `h5`, `h10`, and `h15`, corresponding to +20, +40, and +60 minutes.

## Notes

- Dataset files and generated submissions are excluded through `.gitignore`.
- The notebook keeps its original executed outputs so the development-time validation results remain inspectable.
- The final portfolio artifact focuses on the submitted residual LightGBM solution rather than retaining unrelated experimental notebooks and research scratch files.
