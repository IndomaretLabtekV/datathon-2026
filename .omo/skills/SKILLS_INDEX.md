# Project Skills Index

> **Project**: Datathon Task 1 -- Traffic Speed Forecasting
>
> **Challenge type**: Multimodal Spatiotemporal Regression (Time Series + Graph + Text)
>
> **Target environment**: Kaggle, GPU T4x2

## Global Skills Mapped to This Challenge

| Skill file | Why it applies | When to load |
|---|---|---|
| `dataset-profiling.skill.md` | Inspect data shapes, types, sparsity before any modeling | First step |
| `data-rigor-and-leakage.skill.md` | Temporal folds, future-leakage assertions, group split by time | Before building validation |
| `domain-time-series.skill.md` | Multi-horizon forecasting, temporal features, backtesting | Model design phase |
| `domain-graph.skill.md` | Road network adjacency (1260x1260 sparse), GNN architectures | Model architecture phase |
| `domain-nlp-llm.skill.md` | Text event processing (HashingVectorizer, embeddings) | Feature engineering phase |
| `domain-multimodal.skill.md` | Combining 3 heterogeneous signals (time series + graph + text) | Fusion design phase |
| `training-optimization.skill.md` | T4x2 constraints, AMP, grad clipping, checkpoint/resume | Training loop design |
| `rigorous-evaluation.skill.md` | MSE reporting per horizon, regime, fold | Evaluation phase |
| `notebook-delivery.skill.md` | Kaggle notebook output, path auto-detection, Run All compat | Final notebook generation |
| `ml-research-methodology.skill.md` | Full workflow orchestrator | Top-level process guide |

## Project-Specific Skills

| Skill file | Purpose |
|---|---|
| `challenge-analysis.skill.md` | Full challenge type analysis, data modalities, approach selection |
| `spatiotemporal-forecast.skill.md` | Custom pipeline for this traffic forecasting competition |

## Signal Modalities

```
Time Series (15 timestep, 1260 roads)
       |
       |----> Temporal features (lags, stats, trends) ------>|
       |                                                     |
Graph (1260x1260 adj, 99.7% sparse)  ---> Graph features --->|---> RESIDUAL PREDICTION
       |                                (neighbor agg,        |          +
       |                                 network stats)       |    Mean-reversion
       |                                                     |      baseline
Text (event descriptions)  ---> Text features (hash/embed)-->|
```

## Recommended Workflow Order

1. `dataset-profiling.skill.md` -- profile actual data shapes
2. `data-rigor-and-leakage.skill.md` -- build temporal folds with assertions
3. `domain-time-series.skill.md` -- feature engineering, baseline
4. `domain-graph.skill.md` -- adjacency preprocessing, GNN architecture
5. `domain-nlp-llm.skill.md` -- text vectorization
6. `training-optimization.skill.md` -- train loop with T4 constraints
7. `rigorous-evaluation.skill.md` -- MSE per horizon/regime/fold
8. `notebook-delivery.skill.md` -- produce final Kaggle .ipynb
