"""
trainer.py
==========
Manages the full training loop for MinDriveJEPA.

  Trainer.run()  →  trains for max_epochs, saves checkpoints, logs to TensorBoard
"""

import math
import pathlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from mindrive_jepa.models import MinDriveJEPA
from mindrive_jepa.training.collapse_monitor import CollapseMonitor


class Trainer:
    def __init__(
        self,
        model:        MinDriveJEPA,
        config:       dict,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ):
        self.model        = model
        self.train_cfg    = config["training"]
        self.model_cfg    = config["model"]

        self.device       = torch.device(self.train_cfg["device"])
        self.max_epochs   = self.train_cfg["max_epochs"]
        self.warmup_epochs= self.train_cfg["warmup_epochs"]
        self.log_every    = self.train_cfg["log_every_n_steps"]
        self.ckpt_dir     = pathlib.Path(self.train_cfg["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr           = float(self.train_cfg["learning_rate"]),
            weight_decay = float(self.train_cfg["weight_decay"]),
        )

        # Warmup + cosine annealing
        self.scheduler = self._build_scheduler()

        self.collapse_monitor = CollapseMonitor()
        self.writer           = SummaryWriter(log_dir="runs/mindrive_jepa")

        self.best_val_loss = float("inf")
        self.global_step   = 0

    # ------------------------------------------------------------------
    def _build_scheduler(self):
        warmup_steps = self.warmup_epochs * len(self.train_loader)
        total_steps  = self.max_epochs   * len(self.train_loader)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)      # linear warmup
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (scenarios,) in enumerate(self.train_loader):
            scenarios = scenarios.to(self.device)          # [B, 50, 21, 6]

            # Split into context (first half) and target (second half)
            T          = scenarios.size(1)
            mid        = T // 2
            ctx_window = scenarios[:, :mid]                # [B, 25, 21, 6]
            tgt_window = scenarios[:, mid:]                # [B, 25, 21, 6]

            self.optimizer.zero_grad()

            out  = self.model(ctx_window, tgt_window)
            loss = out["loss"]

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            self.optimizer.step()
            self.model.update_target_encoder(self.model_cfg["ema_decay"])
            self.scheduler.step()

            # Collapse monitoring
            collapse_info = self.collapse_monitor.update(
                out["context_latent"].detach()
            )

            total_loss     += loss.item()
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar("train/loss",         loss.item(),                      self.global_step)
                self.writer.add_scalar("train/jepa_loss",    out["jepa_loss"].item(),           self.global_step)
                self.writer.add_scalar("train/position_loss",out["position_loss"].item(),       self.global_step)
                self.writer.add_scalar("train/variance",     collapse_info["variance"],         self.global_step)
                self.writer.add_scalar("train/cosine_sim",   collapse_info["cosine_sim"],       self.global_step)
                self.writer.add_scalar("train/lr",           lr,                                self.global_step)

                if collapse_info["collapsed"]:
                    print(f"  ⚠️  {self.collapse_monitor.summary()}")

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        for (scenarios,) in self.val_loader:
            scenarios  = scenarios.to(self.device)
            T          = scenarios.size(1)
            mid        = T // 2
            ctx_window = scenarios[:, :mid]
            tgt_window = scenarios[:, mid:]

            out        = self.model(ctx_window, tgt_window)
            total_loss += out["loss"].item()

        avg = total_loss / len(self.val_loader)
        self.writer.add_scalar("val/loss", avg, epoch)
        return avg

    # ------------------------------------------------------------------
    def _save_checkpoint(self, path: pathlib.Path, epoch: int, val_loss: float):
        torch.save(
            {
                "epoch":              epoch,
                "val_loss":           val_loss,
                "model_state_dict":   self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: pathlib.Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"Resumed from {path}  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
        return ckpt["epoch"]

    # ------------------------------------------------------------------
    def run(self):
        print(f"Training on {self.device}  |  {self.max_epochs} epochs  |  "
              f"{len(self.train_loader)} train batches / {len(self.val_loader)} val batches")
        t0 = time.time()

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss   = self._val_epoch(epoch)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{self.max_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}  "
                f"elapsed={elapsed:.1f}s  "
                f"{self.collapse_monitor.summary()}"
            )

            # Save last checkpoint every epoch
            self._save_checkpoint(self.ckpt_dir / "last.pt", epoch, val_loss)

            # Save best checkpoint if val loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(self.ckpt_dir / "best.pt", epoch, val_loss)
                print(f"  → New best val loss: {val_loss:.6f}  (saved best.pt)")

        self.writer.close()
        print(f"\nTraining complete in {time.time()-t0:.1f}s  |  best val loss: {self.best_val_loss:.6f}")


# ----------------------------------------------------------------------
# Smoke test — does NOT use real data, just verifies the Trainer wiring
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import yaml, pathlib

    root = pathlib.Path(__file__).resolve().parents[3]
    cfg  = yaml.safe_load((root / "configs/default.yaml").read_text())

    # Tiny fake dataset: 64 scenarios of shape [50, 21, 6]
    N   = 64
    T   = cfg["data"]["sequence_len"]         # 50
    N1  = cfg["data"]["max_agents"] + 1       # 21
    D   = cfg["data"]["agent_feat_dim"]        # 6

    fake_data   = torch.randn(N, T, N1, D)
    dataset     = TensorDataset(fake_data)
    n_train     = int(N * 0.8)
    n_val       = N - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    # Override to 3 epochs and cpu so the smoke test is fast
    cfg["training"]["max_epochs"]  = 3
    cfg["training"]["warmup_epochs"] = 1
    cfg["training"]["device"]      = "cpu"
    cfg["training"]["checkpoint_dir"] = str(root / "checkpoints")

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"])

    model   = MinDriveJEPA(cfg["data"], cfg["model"])
    trainer = Trainer(model, cfg, train_loader, val_loader)
    trainer.run()

    print("\nSmoke test passed.")

