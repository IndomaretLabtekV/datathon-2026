---
name: spatiotemporal-forecast
description: "Custom workflow for spatiotemporal traffic speed forecasting on 1260 road segments with 3 horizons, graph adjacency, and text events. Load domain-time-series, domain-graph, domain-nlp-llm, data-rigor-and-leakage, and training-optimization in parallel with this skill. Trigger: 'traffic forecast', 'spatiotemporal', 'speed prediction', 'residual graph-stid', 'graph wavenet'."
---

# Spatiotemporal Traffic Speed Forecasting Pipeline

## Overview

Residual learning: a strong statistical baseline handles the bulk, a lightweight neural network learns residuals.

```
prediction = mean_reversion_baseline + residual_model(input_features)
```

## Required Skills (load in order)

1. `dataset-profiling` -- verify data shapes first
2. `data-rigor-and-leakage` -- temporal fold construction
3. `domain-time-series` -- feature engineering and baseline
4. `domain-graph` -- adjacency preprocessing
5. `domain-nlp-llm` -- text feature extraction (optional)
6. `training-optimization` -- training loop with T4 constraints
7. `rigorous-evaluation` -- MSE evaluation protocol

## Workflow Steps

### Phase 1: Data Discovery & Validation

Assert shapes:
```
train_speed_m1: (11160, 1260)    test_X_hist: (540, 15, 1260)
train_speed_m2: (5039, 1260)     matrix: (1260, 1260)
```

Checks: non-finite detection, regime inference via zero fraction, verify test counts = 372 M1 + 168 M2.

### Phase 2: Temporal Validation Folds

| Fold | Train | Validation |
|---|---|---|
| 0 | 0% -- 55% | 55% -- 70% |
| 1 | 0% -- 70% | 70% -- 85% |
| 2 | 0% -- 85% | 85% -- 100% |

Windows of 15 timesteps, targets at origin + 5/10/15. Fit all stats on train prefix only.

### Phase 3: Feature Engineering

**Dynamic features**: raw history (normalized), mean(3/5/15), diff(1/5), slope(5/15), std(5/15), global context mean.

**Graph features**: sparse forward/backward normalized adjacency, 1-hop and 2-hop neighbor aggregation, road metadata (length, segments, roadclass, formway).

**Static features**: learned road embedding (1260 x embed_dim) + metadata.

**Text features** (optional): HashingVectorizer(256, (1,2), norm=l2) -> 32-dim projection.

### Phase 4: Models

**Residual Graph-STID**: road embedding -> MLP on dynamic + static -> 3 residual blocks (Linear + LayerNorm + GELU + Dropout) -> 3 zero-initialized heads. Separate weights per regime.

**Compact Graph WaveNet** (alternative): 4 dilated temporal conv layers (1,2,4,8), gated TCN, graph diffusion convolution, skip connections.

### Phase 5: Training Loop

AdamW (1e-3, 1e-4), cosine decay + warmup, raw-scale MSE loss, batch 16, AMP, grad clip 1.0, dropout 0.10, patience 7.

### Phase 6: Evaluation

Per fold, per horizon, per regime. Weighted: (372 * M1_mse + 168 * M2_mse) / 540.

### Phase 7: Ensemble

Per-horizon weighted average, grid search on folds 0+1, validate on fold 2.

### Phase 8: Submission

Train on all data, infer test per regime, write in sample order, assert shape and finite values.

## ponytail: shortcuts

- Mean-reversion first validates the whole pipeline before neural code runs.
- Text is gated: only after speed-only models prove themselves.
- Graph WaveNet is a fallback, implement only if Graph-STID passes with margin.
- Single CFG cell, no scattered hyperparameters.
