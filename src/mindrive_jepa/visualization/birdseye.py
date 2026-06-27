"""
birdseye.py
===========
Bird's-eye view plots of driving scenarios coloured by agent type,
with surprise scores in the title.

Tensor layout: [T, N+1, D]
  dim 0 (T)   : timestep  (50 frames = 5 seconds)
  dim 1 (N+1) : agents — index 0 is ego, 1..20 are other agents
  dim 2 (D=6) : [x, y, vx, vy, heading, type]
                  x, y       normalised (÷50m, ego-centred)
                  type float : 0=ego, 1=vehicle, 2=pedestrian, 3=cyclist/other

Agent colours
  ego        → black star marker
  vehicle    → steelblue
  pedestrian → tomato
  cyclist    → mediumseagreen
  inactive   → grey (all-zero rows)
"""

import csv
import pathlib

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


# ── colour map ────────────────────────────────────────────────────────────────
_TYPE_COLOUR = {
    0: "black",          # ego
    1: "steelblue",      # vehicle
    2: "tomato",         # pedestrian
    3: "mediumseagreen", # cyclist / other
}


def _agent_colour(type_float: float) -> str:
    key = int(round(type_float))
    return _TYPE_COLOUR.get(key, "grey")


# ── single panel ──────────────────────────────────────────────────────────────
def plot_scenario_bev(
    tensor: torch.Tensor,
    title:  str,
    ax:     plt.Axes,
    alpha_trail: float = 0.5,
) -> None:
    """
    Draw one bird's-eye view panel onto *ax*.

    Args:
        tensor : [T, N+1, D]  scenario tensor (normalised coords)
        title  : panel title (e.g. "scenario_00019  score=4.73 ⚠️")
        ax     : matplotlib Axes to draw onto
        alpha_trail: opacity of trajectory lines
    """
    data = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
    T, N1, D = data.shape

    # ── dashed ±1 normalised box (= ±50 m in real world) ─────────────────────
    box = plt.Rectangle((-1, -1), 2, 2,
                        linewidth=1, edgecolor="grey",
                        linestyle="--", facecolor="none", zorder=1)
    ax.add_patch(box)

    # ── draw each agent ───────────────────────────────────────────────────────
    for agent_idx in range(N1):
        traj = data[:, agent_idx, :]   # [T, D]

        # Skip inactive agents (all positions zero after t=0 for non-ego)
        if agent_idx > 0:
            active = np.any(np.abs(traj[:, :2]) > 1e-6, axis=1)
            if active.sum() < 2:
                continue

        xs = traj[:, 0]
        ys = traj[:, 1]
        type_val = traj[0, 5]          # type stored in feature dim 5

        if agent_idx == 0:
            # Ego: star at origin (t=0) with trajectory
            ax.plot(xs, ys, color="black", linewidth=1.5,
                    alpha=alpha_trail, zorder=3)
            ax.plot(xs[0], ys[0], marker="*", color="black",
                    markersize=12, zorder=5)
        else:
            colour = _agent_colour(type_val)
            ax.plot(xs, ys, color=colour, linewidth=0.8,
                    alpha=alpha_trail, zorder=2)
            # Dot at final position
            ax.plot(xs[-1], ys[-1], marker="o", color=colour,
                    markersize=4, zorder=4)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, pad=3)
    ax.tick_params(labelsize=10)
    ax.set_xlabel("x (norm.)", fontsize=10)
    ax.set_ylabel("y (norm.)", fontsize=10)


# ── grid of K scenarios ───────────────────────────────────────────────────────
def plot_top_k(
    scores_csv:  str | pathlib.Path,
    data_dir:    str | pathlib.Path,
    output_dir:  str | pathlib.Path,
    k:           int = 9,
    cols:        int = 3,
) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Read outputs/surprise_scores.csv, load top-k and bottom-k scenarios,
    and save two PNG grids.

    Returns:
        (top_png_path, bottom_png_path)
    """
    scores_csv = pathlib.Path(scores_csv)
    data_dir   = pathlib.Path(data_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV (already sorted highest-first)
    rows = []
    with open(scores_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((row["scenario"], float(row["surprise_score"]),
                         row["is_surprising"] == "True"))

    top_rows    = rows[:k]
    bottom_rows = rows[-k:][::-1]   # least surprising, ascending

    def _make_grid(scenario_rows, out_path, suptitle):
        rows_count = (k + cols - 1) // cols
        fig, axes  = plt.subplots(rows_count, cols,
                                  figsize=(cols * 4, rows_count * 4))
        axes = np.array(axes).flatten()

        for i, (name, score, is_surp) in enumerate(scenario_rows):
            pt_path = data_dir / f"{name}.pt"
            if not pt_path.exists():
                axes[i].set_visible(False)
                continue

            tensor = torch.load(pt_path, weights_only=True)
            flag   = " ⚠️" if is_surp else ""
            title  = f"{name}\nscore={score:.3f}{flag}"
            plot_scenario_bev(tensor, title, axes[i])

        # Hide any unused panels
        for j in range(len(scenario_rows), len(axes)):
            axes[j].set_visible(False)

        # Legend
        legend_handles = [
            mpatches.Patch(color="black",          label="ego"),
            mpatches.Patch(color="steelblue",      label="vehicle"),
            mpatches.Patch(color="tomato",         label="pedestrian"),
            mpatches.Patch(color="mediumseagreen", label="cyclist/other"),
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.0))

        fig.suptitle(suptitle, fontsize=16, y=1.01)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    top_png    = output_dir / "top_surprising.png"
    bottom_png = output_dir / "bottom_routine.png"

    _make_grid(top_rows,    top_png,    f"Top-{k} Most Surprising Scenarios")
    _make_grid(bottom_rows, bottom_png, f"Top-{k} Most Routine Scenarios")

    return top_png, bottom_png


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib, yaml

    root       = pathlib.Path(__file__).resolve().parents[3]
    cfg        = yaml.safe_load((root / "configs/default.yaml").read_text())
    scores_csv = root / "outputs" / "surprise_scores.csv"
    data_dir   = root / cfg["data"]["processed_dir"]
    output_dir = root / "outputs"

    if not scores_csv.exists():
        print("Run  python scripts/evaluate.py  first to generate surprise_scores.csv")
        raise SystemExit(1)

    top_png, bottom_png = plot_top_k(scores_csv, data_dir, output_dir, k=9)
    print(f"\nOpen these files to compare:")
    print(f"  Most surprising : {top_png}")
    print(f"  Most routine    : {bottom_png}")

