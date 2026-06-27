# minDrive-JEPA

**Zero-Label Driving Scenario Complexity Detection via Joint Embedding Predictive Architecture**

> Paper submitted to arXiv — link coming soon.

---

## Overview

minDrive-JEPA applies Joint-Embedding Predictive Architecture (JEPA) to structured autonomous driving scenarios from the [nuPlan](https://www.nuscenes.org/nuplan) dataset.
The model learns compact latent representations through masked prediction — without any pixel-level reconstruction or human-provided labels.
The learned representations capture scenario complexity and can be used zero-shot as a difficulty signal for scenario curation, data weighting, or downstream planning tasks.

Key results:
- Strong rank correlation between learned latent surprise and objective scenario tags (e.g., `cut_in`, `stationary_in_traffic`, `on_intersection`)
- Representations transfer to downstream retrieval tasks (top-k precision)
- EMA target encoder is the critical component: ablations show collapse without it

---

## Project Structure

```
src/mindrive_jepa/
  models/        — JEPA encoder, predictor, target encoder
  training/      — training loop, EMA update
  data/          — nuPlan data loading and tokenization
  evaluation/    — tag correlation, retrieval metrics
  visualization/ — plotting utilities

scripts/
  train.py           — main training entry point
  evaluate.py        — evaluation against scenario tags
  preprocess_data.py — nuPlan → processed tensor format

configs/
  default.yaml       — full training configuration
  no_ema.yaml        — ablation: no EMA target encoder
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.x, nuPlan SDK.
See `requirements.txt` for the full dependency list.

---

## Training

```bash
python scripts/train.py --config configs/default.yaml
```

Checkpoints are saved to `checkpoints/` by default (configurable in YAML).

---

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt
```

Outputs tag correlation scores and top-k precision metrics.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{jaiswal2026mindrive,
  title   = {Zero-Label Driving Scenario Complexity Detection
             via Joint Embedding Predictive Architecture},
  author  = {Jaiswal, Santosh},
  journal = {arXiv preprint arXiv:2506.XXXXX},
  year    = {2026},
  url     = {https://arxiv.org/abs/2506.XXXXX}
}
```

> **Note:** Replace `2506.XXXXX` with the actual arXiv ID once the paper goes live (expected within a few days of submission).

---

## License

MIT
