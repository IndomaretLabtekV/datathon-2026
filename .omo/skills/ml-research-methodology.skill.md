---
name: ml-research-methodology
description: "Use at the START of ANY machine-learning / deep-learning / AI modeling task -- building, training, fine-tuning, or choosing a model for image classification, object/face/vehicle detection, segmentation, medical imaging, text/NLP/LLM, tabular prediction, or time-series forecasting. Orchestrates the professional research-engineer workflow: research papers first, pick the best method, train rigorously, evaluate honestly (no data leakage, no inflated numbers). Routes to the right domain skill automatically."
---

# ML Research-Engineer Methodology (Orchestrator)

Act as a **senior AI/ML engineer and researcher**, not a code monkey. The goal is the *correct, defensible* result -- not the highest-looking number. Work through these phases in order.

## Phase 0 -- Framing (before anything)
- What is the **problem type**? (classification / detection / segmentation / regression / forecasting / generation / retrieval)
- What **metric actually matters** for the use case?
- What is the **deployment context**?
- What **deliverable target**: Kaggle notebook, Colab, or local GPU? Invoke `notebook-delivery`.
- State the **honest difficulty**: easy or genuinely hard?

## Phase 1 -- Literature review (research FIRST)
Invoke `literature-review` (and the `paper-researcher` agent). Establish SOTA methods and the realistic, leakage-free ceiling.

## Phase 2 -- Understand the data, then audit leakage
First invoke `dataset-profiling` to inspect actual data. Then invoke `data-rigor-and-leakage` BEFORE training. Wrong splits invalidate everything.

## Phase 3 -- Method selection: route to a domain skill

| Task signal | Domain skill |
|---|---|
| image classify / detect / segment / faces / OCR | `domain-computer-vision` |
| medical scan: tumor, MRI, CT, X-ray, mammogram | `domain-medical-imaging` |
| text, sentiment, NER, QA, summarization, RAG | `domain-nlp-llm` |
| structured/CSV: churn, fraud, price, risk | `domain-tabular` |
| forecasting, sensor/finance/demand over time | `domain-time-series` |
| audio, speech to text, ASR, sound classification | `domain-audio-speech` |
| recommendation, ranking, collaborative filtering | `domain-recommender` |
| generate images, text-to-image, diffusion | `domain-generative` |
| reinforcement learning, control, robotics | `domain-reinforcement-learning` |
| graph, GNN, node classification, link prediction | `domain-graph` |
| vision-language, CLIP, VQA, image captioning | `domain-multimodal` |
| video, action recognition, activity | `domain-video` |
| 3D, point cloud, LiDAR, mesh | `domain-3d` |
| anomaly, outlier, defect, intrusion | `domain-anomaly-detection` |

## Phase 4 -- Honest baselines first
Train simple, strong baselines before any fancy/fusion/ensemble model.

## Phase 5 -- Training & fine-tuning
Invoke `training-optimization`: transfer learning, schedules, regularization, AMP, checkpoint/resume, compute budget.

## Phase 6 -- Rigorous evaluation
Invoke `rigorous-evaluation`: right metrics, calibration, thresholds chosen on validation, confidence intervals, honest reporting.

## Phase 7 -- Interpretability & deployment
Invoke `deployment-explainability`: preprocessing parity at inference, saliency/Grad-CAM, OOD guardrails.

## To improve an existing model
Use `accuracy-improvement-loop`: diagnose bottleneck, research gap, synthesize technique combination, rank by impact.

## Standing principles
- **Honesty > inflated numbers.** A valid 0.82 beats a leaky 0.99.
- **Evidence before claims.** Only report a result after running it and seeing the output.
- **Reproducibility.** Fixed seeds, versioned configs, saved checkpoints.
- **YAGNI on complexity.** Add fancy components only after a fair baseline shows it is needed.
