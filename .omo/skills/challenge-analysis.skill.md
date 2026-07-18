---
name: challenge-analysis
description: "Analysis of the Datathon Task 1 traffic forecasting challenge. Profiles the 3 data modalities (time series, graph, text), identifies the problem type as multi-horizon spatiotemporal regression, and maps to the correct domain skills. Trigger: 'analyze challenge', 'what type of problem', 'data analysis', 'challenge profile'."
---

# Challenge Analysis: Datathon Task 1

## Problem Classification

| Property | Value |
|---|---|
| **Problem type** | Multivariate multi-horizon spatiotemporal regression |
| **Primary modality** | Time series (15-step window, 1260 series) |
| **Secondary modality** | Graph (1260-node road network adjacency) |
| **Tertiary modality** | Text (natural language event descriptions) |
| **Target** | Speed (km/h) at 3 horizons: +20min (h5), +40min (h10), +60min (h15) |
| **Metric** | Mean Squared Error (MSE), averaged across samples, horizons, roads |
| **Output format** | `test_{sample}_h{horizon}_r{road}` per row, 2,041,200 rows total |

## Data Profile (from actual inspection)

| File | Shape / Size | Type | Notes |
|---|---|---|---|
| `train_speed_m1_1_11160.npy` | (11160, 1260) | float32 | ~1% zero values |
| `train_speed_m2_1_5039.npy` | (5039, 1260) | float32 | ~16.7% zero values |
| `test_X_hist.npy` | (540, 15, 1260) | float32 | 540 samples, 15-timestep windows |
| `matrix.npy` | (1260, 1260) | int8 | ~99.7% sparse (0.3% density) |
| `active_mask.npy` | boolean mask | | Maps spatial grid to 1260 active segments |
| `test_texts.json` | dict, 540 keys | str | `test_XXXXX` -> concatenated event sentences |
| `train_text_m1_1_11160.json` | dict, 11160 keys | str | `m1_N` -> event sentences per timestep |
| `sample_submission.csv` | (2041200, 2) | str, float | id + speed columns |

## Modality Analysis

### 1. Time Series (primary signal)
- 1260 parallel road speed series, each with 11160 (M1) + 5039 (M2) timesteps
- Two regimes with different zero-value fractions (~1% vs ~17%)
- Test: 540 sliding windows of 15 timesteps each

### 2. Graph Structure (spatial signal)
- 1260x1260 adjacency matrix, extremely sparse
- Each road segment connects to ~3-4 neighbors on average
- Enables forward/backward 1-hop and 2-hop neighbor feature extraction

### 3. Text Events (context signal)
- Natural language descriptions: road closures, construction, accidents, announcements
- Written in English with Chinese road names
- Good for HashingVectorizer or TF-IDF feature extraction

## Recommended Approach

Residual learning framework:
```
final_prediction = mean_reversion_baseline + residual_model_prediction
```

- **Mean-reversion baseline**: Statistical forecast using recent speed history
- **Residual Graph-STID**: Lightweight neural net (road embeddings + dynamic features + graph aggregation)
- **Compact Graph WaveNet**: Graph-temporal model with dilated convolutions (alternative)
- **Text features**: Optional, gated behind validation gates

## Skill Map

| Component | Domain skill | Key guidance |
|---|---|---|
| Temporal features, lags, multi-horizon | `domain-time-series` | Temporal split, no future leakage, rolling backtest |
| Graph adjacency, neighbor aggregation, GNN | `domain-graph` | Sparse adjacency, GCN/GAT for node-level regression |
| Text vectorization, hashing features | `domain-nlp-llm` | HashingVectorizer for speed |
| Multi-modality fusion | `domain-multimodal` | Early fusion vs late fusion |
| Train/val split for time series | `data-rigor-and-leakage` | Group split by time; train strictly before val |
| Evaluation per horizon and regime | `rigorous-evaluation` | Per-horizon MSE, per-regime MSE |
| Kaggle notebook production | `notebook-delivery` | Auto-detect Kaggle paths, Run All compatibility |
| GPU training with limited VRAM | `training-optimization` | AMP, gradient clipping, checkpoint/resume |

## Key Constraints

- **Time budget**: 5 hours total. One solid approach > multiple incomplete experiments.
- **Compute budget**: T4x2 GPUs (16GB VRAM each).
- **Submission limit**: 5. Use temporal CV for model selection.
- **Regime shift**: M1 and M2 have different zero-value distributions. Test has 372 M1-like + 168 M2-like samples.
