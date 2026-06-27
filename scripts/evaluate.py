"""
scripts/evaluate.py
===================
Score all processed scenarios with SurpriseScore, print top-N most
surprising, and save a ranked CSV to outputs/surprise_scores.csv.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --top 20
"""

import argparse
import csv
import pathlib
import sys

import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mindrive_jepa.evaluation.surprise_score import SurpriseScore


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate minDrive-JEPA surprise scores")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data_dir",   default=None,
                   help="Override processed data dir from config")
    p.add_argument("--top",        type=int, default=10,
                   help="Number of top surprising scenarios to print")
    p.add_argument("--output",     default="outputs/surprise_scores.csv",
                   help="Path to save ranked CSV")
    return p.parse_args()


def main():
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]

    cfg = yaml.safe_load((root / args.config).read_text())

    data_dir = pathlib.Path(args.data_dir) if args.data_dir else \
               root / cfg["data"]["processed_dir"]

    # Load all scenario tensors
    files = sorted(data_dir.glob("scenario_*.pt"))
    if not files:
        print(f"No scenario_*.pt files found in {data_dir}")
        sys.exit(1)

    print(f"Loading {len(files)} scenarios from {data_dir} ...")
    tensors = [torch.load(f, weights_only=True) for f in files]

    # Build scorer
    ckpt_path = root / args.checkpoint
    device    = cfg["training"]["device"]
    scorer    = SurpriseScore(ckpt_path, cfg, device=device)
    print(f"Loaded checkpoint: {ckpt_path}")

    # Calibrate on the full dataset
    percentile = cfg.get("evaluation", {}).get("surprise_threshold_percentile", 90)
    print(f"Calibrating threshold at {percentile}th percentile ...")
    threshold = scorer.calibrate(tensors, percentile=percentile)
    print(f"Surprise threshold: {threshold:.4f}\n")

    # Score all scenarios
    scores = scorer.score_batch(tensors)

    # Rank by surprise (highest first)
    ranked = sorted(
        zip(files, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # Print top-N
    n_surprising = sum(1 for _, s in ranked if s > threshold)
    print(f"Top-{args.top} most surprising scenarios  "
          f"({n_surprising}/{len(scores)} above threshold):")
    print(f"  {'Rank':<5} {'Scenario':<25} {'Score':>8}  {'Flag'}")
    print(f"  {'-'*5} {'-'*25} {'-'*8}  {'-'*4}")
    for rank, (f, s) in enumerate(ranked[:args.top], 1):
        flag = "⚠️  SURPRISING" if s > threshold else ""
        print(f"  {rank:<5} {f.stem:<25} {s:8.4f}  {flag}")

    # Save CSV
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "scenario", "surprise_score", "is_surprising"])
        for rank, (f, s) in enumerate(ranked, 1):
            writer.writerow([rank, f.stem, f"{s:.6f}", s > threshold])

    print(f"\nSaved ranked scores to {out_path}")
    print(f"Stats — min: {min(scores):.4f}  "
          f"max: {max(scores):.4f}  "
          f"mean: {sum(scores)/len(scores):.4f}  "
          f"threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()

