"""
scripts/train.py
================
Main training entry point for minDrive-JEPA.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --resume checkpoints/last.pt
    python scripts/train.py --data_dir data/toy_slice   # quick smoke-test on 5 samples
"""

import argparse
import pathlib
import sys

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

# Make sure the src package is importable when running as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mindrive_jepa.data.tokenizer import SceneTokenizer
from mindrive_jepa.models import MinDriveJEPA
from mindrive_jepa.training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train minDrive-JEPA")
    p.add_argument("--config",   default="configs/default.yaml",
                   help="Path to YAML config file")
    p.add_argument("--data_dir", default=None,
                   help="Override processed data directory from config")
    p.add_argument("--resume",   default=None,
                   help="Path to checkpoint to resume training from")
    return p.parse_args()


def load_dataset(data_dir: pathlib.Path) -> torch.Tensor:
    """
    Load all scenario_*.pt files and stack into a single tensor [N, 50, 21, 6].
    """
    files = sorted(data_dir.glob("scenario_*.pt"))
    if not files:
        raise FileNotFoundError(f"No scenario_*.pt files found in {data_dir}")

    tensors = [torch.load(f, weights_only=True) for f in files]
    data    = torch.stack(tensors, dim=0)   # [N, T, N+1, D]
    print(f"Loaded {len(tensors)} scenarios from {data_dir}  →  shape {tuple(data.shape)}")
    return data


def main():
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]

    # Load config
    cfg_path = root / args.config
    cfg      = yaml.safe_load(cfg_path.read_text())

    # Resolve data directory
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else \
               root / cfg["data"]["processed_dir"]

    # Load dataset
    data    = load_dataset(data_dir)
    dataset = TensorDataset(data)

    # Train / val split
    train_split = cfg["data"]["train_split"]
    n_train     = int(len(dataset) * train_split)
    n_val       = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train} scenarios  |  Val: {n_val} scenarios")

    # DataLoaders
    batch_size   = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model
    model = MinDriveJEPA(cfg["data"], cfg["model"])
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {trainable_params:,} trainable params  ({total_params:,} total)")

    # Build trainer
    trainer = Trainer(model, cfg, train_loader, val_loader)

    # Optionally resume from checkpoint
    if args.resume:
        resume_path = pathlib.Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = root / resume_path
        trainer.load_checkpoint(resume_path)

    # Train
    trainer.run()


if __name__ == "__main__":
    main()

