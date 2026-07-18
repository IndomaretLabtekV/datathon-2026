---
name: training-optimization
description: "Use when training or fine-tuning a deep model and tuning it for best accuracy. Covers transfer learning, two-phase fine-tuning, discriminative/layer-wise LR, LR schedules + warmup, regularization tuned to over/under-fit (dropout, weight decay, label smoothing, MixUp), mixed precision, gradient checkpointing, EMA, checkpoint/resume, and fitting a GPU/time budget. Trigger: 'fine-tune', 'train the model', 'best hyperparameters', 'overfitting', 'learning rate', 'out of memory', 'training is too slow'."
---

# Training & Fine-Tuning

Best-practice recipe for training a strong model, and how to fix over/under-fitting.

## Training recipe (default that works)
1. **Transfer learning**: start from pretrained weights; train from scratch only with lots of data.
2. **Two phases**: (1) freeze backbone, train the new head; (2) unfreeze and fine-tune end-to-end at a lower LR.
3. **Discriminative / layer-wise LR**: small LR for early layers, higher for the head. Biggest single tuning win.
4. **Schedule**: cosine decay with a short warmup. Log the LR curve.
5. **Optimizer**: AdamW (decoupled weight decay). Gradient clipping ~1.0.
6. **Selection**: checkpoint the best epoch by the **validation** metric; early stopping.

## Diagnose then regularize (don't guess)
- **Overfitting** (train acc rising while val loss rising/plateauing): raise weight decay, dropout, label smoothing; add MixUp; stronger-but-domain-appropriate augmentation; reduce epochs/patience; smaller head.
- **Underfitting** (train acc low too): reduce regularization, raise LR / capacity / resolution, train longer, check data quality.
- Match augmentation to the domain -- do **not** use aggressive ops that destroy the signal.

## Compute & memory (fit the budget)
- **Mixed precision (AMP)** always on GPU.
- **Gradient checkpointing** to fit high resolution / big models (~3x memory saving, ~30% slower).
- **Small batch + gradient accumulation** to keep effective batch size when VRAM is tight; **freeze BatchNorm** at very small batch (LayerNorm backbones are safer there).
- **Preprocess once to disk** if per-epoch CPU work starves the GPU.
- Budget = seconds/epoch x epochs x models. Measure on a quick run first. Prefer one honest fit over a run that dies mid-way.

## Robustness
- Fixed seeds; **checkpoint per epoch + resume** so interrupted long run continues, not restarts.
- EMA of weights often helps generalization -- but verify it beats the best checkpoint on val before keeping it.
