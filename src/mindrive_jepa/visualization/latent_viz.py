"""
latent_viz.py
=============
UMAP projection of context latent vectors coloured by surprise score.

For every scenario in the dataset we encode the context window
(first 25 frames) through the trained context_encoder to get a
128-dimensional latent vector.  UMAP compresses those 1322 vectors
down to 2D so we can scatter-plot them.

Colour mapping:
  - Hue = surprise score  (blue=routine → red=surprising)
  - Size = surprise score  (larger dot = more surprising)
  - Stars  = top-10 most surprising (labelled)

Output: outputs/latent_umap.png
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import yaml
from umap import UMAP


def build_latents(
    data_dir:    pathlib.Path,
    ckpt_path:   pathlib.Path,
    config:      dict,
    device:      str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """
    Encode every scenario's context window and return
    (latent_matrix [N, 128], scenario_names [N]).
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
    from mindrive_jepa.models import MinDriveJEPA

    dev = torch.device(device)

    model = MinDriveJEPA(config["data"], config["model"])
    ckpt  = torch.load(ckpt_path, map_location=dev, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    files = sorted(data_dir.glob("scenario_*.pt"))
    latents, names = [], []

    with torch.no_grad():
        for f in files:
            tensor = torch.load(f, weights_only=True).to(dev)   # [T, N+1, D]
            mid    = tensor.size(0) // 2
            ctx    = tensor[:mid].unsqueeze(0)                    # [1, T/2, N+1, D]
            z      = model.context_encoder(ctx)                   # [1, 128]
            latents.append(z.squeeze(0).cpu().numpy())
            names.append(f.stem)

    return np.stack(latents, axis=0), names   # [N, 128], [N]


def plot_umap(
    latents:    np.ndarray,
    scores:     np.ndarray,
    names:      list[str],
    output_path: pathlib.Path,
    threshold:  float | None = None,
    top_k:      int = 10,
) -> pathlib.Path:
    """
    Project latents with UMAP, colour by surprise score, save PNG.

    Args:
        latents     : [N, 128]
        scores      : [N]  surprise score per scenario
        names       : [N]  scenario names
        output_path : where to save the PNG
        threshold   : surprise threshold (drawn as colourbar line if provided)
        top_k       : label the top-k most surprising points

    Returns:
        output_path
    """
    print("Running UMAP (this takes ~10–20 seconds) ...")
    reducer  = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    random_state=42, verbose=False)
    embedding = reducer.fit_transform(latents)   # [N, 2]

    # Normalise scores for colour mapping
    vmin, vmax = scores.min(), scores.max()
    norm   = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap   = cm.coolwarm
    colours = cmap(norm(scores))
    sizes   = 10 + 60 * (scores - vmin) / (vmax - vmin + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=scores, cmap="coolwarm", vmin=vmin, vmax=vmax,
        s=sizes, alpha=0.7, linewidths=0.3, edgecolors="white",
        zorder=2,
    )

    # Label + star the top-k most surprising
    top_idx = np.argsort(scores)[::-1][:top_k]
    for i in top_idx:
        ax.scatter(embedding[i, 0], embedding[i, 1],
                   marker="*", s=200, color="crimson",
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(
            names[i].replace("scenario_", "#"),
            (embedding[i, 0], embedding[i, 1]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=6, color="crimson",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
            zorder=6,
        )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Surprise Score", fontsize=9)
    if threshold is not None:
        cbar.ax.axhline(threshold, color="black", linewidth=1.5, linestyle="--")
        cbar.ax.text(1.6, threshold, f"  threshold\n  {threshold:.2f}",
                     va="center", fontsize=7, transform=cbar.ax.transData)

    ax.set_title(
        f"UMAP of Context Latents — {len(latents)} scenarios\n"
        f"Colour = surprise score  |  ★ = top-{top_k} most surprising",
        fontsize=10,
    )
    ax.set_xlabel("UMAP dim 1", fontsize=9)
    ax.set_ylabel("UMAP dim 2", fontsize=9)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import csv

    root       = pathlib.Path(__file__).resolve().parents[3]
    cfg        = yaml.safe_load((root / "configs/default.yaml").read_text())
    ckpt_path  = root / "checkpoints" / "best.pt"
    data_dir   = root / cfg["data"]["processed_dir"]
    scores_csv = root / "outputs" / "surprise_scores.csv"
    output_path = root / "outputs" / "latent_umap.png"

    if not ckpt_path.exists():
        print("Run  python scripts/train.py  first.")
        raise SystemExit(1)
    if not scores_csv.exists():
        print("Run  python scripts/evaluate.py  first.")
        raise SystemExit(1)

    # Load pre-computed surprise scores (already ran evaluate.py)
    score_map: dict[str, float] = {}
    threshold = None
    with open(scores_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            score_map[row["scenario"]] = float(row["surprise_score"])

    # Build latents
    device  = cfg["training"]["device"]
    latents, names = build_latents(data_dir, ckpt_path, cfg, device=device)

    # Align scores to latent order
    scores = np.array([score_map[n] for n in names], dtype=np.float32)

    pct = cfg.get("evaluation", {}).get("surprise_threshold_percentile", 90)
    threshold = float(np.percentile(scores, pct))

    plot_umap(latents, scores, names, output_path,
              threshold=threshold, top_k=10)
    print("Done.")

