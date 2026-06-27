"""
pygame_demo.py
==============
Live animated replay of driving scenarios with real-time surprise score bar.

Plays through the top-N most surprising scenarios (from outputs/surprise_scores.csv),
then loops. Draws agents frame-by-frame at 20 fps.

Controls:
  SPACE   pause / resume
  →       skip to next scenario
  ←       go back to previous scenario
  Q / ESC quit

Agent colours:
  ★  black  — ego vehicle (always at centre at t=0)
  ●  steel blue   — vehicle
  ●  tomato       — pedestrian
  ●  sea green    — cyclist / other

Usage:
    python src/mindrive_jepa/visualization/pygame_demo.py
    python src/mindrive_jepa/visualization/pygame_demo.py --top 20
    python src/mindrive_jepa/visualization/pygame_demo.py --fps 10
"""

import argparse
import csv
import math
import pathlib
import sys

import torch


# ── constants ─────────────────────────────────────────────────────────────────
WIDTH,  HEIGHT   = 900, 700
PANEL_H          = 80          # height of the info bar at the bottom
SCENE_H          = HEIGHT - PANEL_H
CX, CY           = WIDTH // 2, SCENE_H // 2
SCALE            = 250          # pixels per normalised unit (1 unit = 50 m)
FPS_DEFAULT      = 20

# Colours (RGB)
BG_COLOUR        = (20,  20,  30)
GRID_COLOUR      = (45,  45,  60)
EGO_COLOUR       = (240, 240, 240)
AGENT_COLOURS    = {
    1: (100, 160, 220),   # vehicle   — steel blue
    2: (220,  80,  70),   # pedestrian— tomato
    3: ( 80, 180, 120),   # cyclist   — sea green
}
TRAIL_ALPHA      = 160          # 0-255
BAR_BG           = ( 50,  50,  50)
BAR_LOW          = ( 60, 160,  80)   # green
BAR_HIGH         = (220,  50,  50)   # red
TEXT_COLOUR      = (220, 220, 220)
WARN_COLOUR      = (255, 100,  50)


# ── helpers ───────────────────────────────────────────────────────────────────
def norm_to_px(x: float, y: float) -> tuple[int, int]:
    """Convert normalised scene coords to pixel coords."""
    px = int(CX + x * SCALE)
    py = int(CY - y * SCALE)   # y-axis flipped (screen y increases downward)
    return px, py


def agent_colour(type_float: float) -> tuple[int, int, int]:
    key = int(round(float(type_float)))
    return AGENT_COLOURS.get(key, (150, 150, 150))


def draw_grid(surface, pygame):
    """Faint grid lines at ±0.5 and ±1.0 normalised units."""
    for v in [-1.0, -0.5, 0.5, 1.0]:
        px, _  = norm_to_px(v, 0)
        _,  py = norm_to_px(0, v)
        pygame.draw.line(surface, GRID_COLOUR, (px, 0),     (px, SCENE_H))
        pygame.draw.line(surface, GRID_COLOUR, (0,  py),    (WIDTH, py))
    # ±1 box
    tl = norm_to_px(-1,  1)
    br = norm_to_px( 1, -1)
    w  = br[0] - tl[0]
    h  = br[1] - tl[1]
    pygame.draw.rect(surface, (90, 90, 110), (*tl, w, h), 1)


def draw_info_bar(surface, pygame, font_big, font_small,
                  name: str, score: float, threshold: float,
                  frame: int, total_frames: int,
                  paused: bool, scenario_idx: int, total_scenarios: int):
    """Draw the bottom panel with scenario name, surprise bar, and controls."""
    bar_top = SCENE_H
    pygame.draw.rect(surface, (10, 10, 18), (0, bar_top, WIDTH, PANEL_H))

    # Scenario name + score
    is_surp = score > threshold
    colour  = WARN_COLOUR if is_surp else TEXT_COLOUR
    flag    = "  ⚠  SURPRISING" if is_surp else ""
    label   = font_big.render(
        f"{name}   score={score:.3f}{flag}",
        True, colour)
    surface.blit(label, (12, bar_top + 6))

    # Surprise bar
    bar_x, bar_y = 12, bar_top + 36
    bar_w        = WIDTH - 24
    bar_h        = 14
    pygame.draw.rect(surface, BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
    fill_frac  = min(score / (threshold * 1.5 + 1e-8), 1.0)
    fill_w     = int(bar_w * fill_frac)
    bar_colour = BAR_HIGH if is_surp else BAR_LOW
    if fill_w > 0:
        pygame.draw.rect(surface, bar_colour,
                         (bar_x, bar_y, fill_w, bar_h), border_radius=4)
    # Threshold marker
    thr_x = bar_x + int(bar_w * (threshold / (threshold * 1.5 + 1e-8)))
    pygame.draw.line(surface, (200, 200, 80),
                     (thr_x, bar_y - 2), (thr_x, bar_y + bar_h + 2), 2)

    # Controls + frame counter
    pause_str = "[PAUSED]" if paused else ""
    info = font_small.render(
        f"frame {frame+1}/{total_frames}  |  "
        f"scenario {scenario_idx+1}/{total_scenarios}  |  "
        f"SPACE=pause  →=next  ←=prev  Q=quit  {pause_str}",
        True, (140, 140, 140))
    surface.blit(info, (12, bar_top + 56))


def draw_scenario_frame(surface, pygame, tensor_np, frame_idx: int):
    """Draw all agents at a single timestep."""
    frame = tensor_np[frame_idx]    # [N+1, 6]

    for agent_idx in range(frame.shape[0]):
        x, y   = float(frame[agent_idx, 0]), float(frame[agent_idx, 1])
        type_v = float(frame[agent_idx, 5])

        # Skip inactive agents (all zeros)
        if agent_idx > 0 and abs(x) < 1e-6 and abs(y) < 1e-6:
            continue

        px, py = norm_to_px(x, y)

        if agent_idx == 0:
            # Ego — white circle with cross
            pygame.draw.circle(surface, EGO_COLOUR, (px, py), 8)
            pygame.draw.line(surface, BG_COLOUR, (px-6, py), (px+6, py), 2)
            pygame.draw.line(surface, BG_COLOUR, (px, py-6), (px, py+6), 2)
        else:
            col = agent_colour(type_v)
            r   = 5 if int(round(type_v)) == 1 else 4   # vehicles slightly bigger
            pygame.draw.circle(surface, col, (px, py), r)


def draw_trails(surface, pygame, tensor_np, frame_idx: int):
    """Draw faded trajectory trails up to current frame."""
    for agent_idx in range(tensor_np.shape[1]):
        pts = []
        for t in range(frame_idx + 1):
            x, y   = float(tensor_np[t, agent_idx, 0]), float(tensor_np[t, agent_idx, 1])
            type_v = float(tensor_np[t, agent_idx, 5])
            if agent_idx > 0 and abs(x) < 1e-6 and abs(y) < 1e-6:
                continue
            pts.append(norm_to_px(x, y))

        if len(pts) < 2:
            continue

        col = EGO_COLOUR if agent_idx == 0 else agent_colour(type_v)
        # Draw each segment with decreasing alpha (older = more faded)
        for i in range(1, len(pts)):
            alpha = int(TRAIL_ALPHA * i / len(pts))
            r, g, b = col
            faded = (
                int(r * alpha / 255 + BG_COLOUR[0] * (255 - alpha) / 255),
                int(g * alpha / 255 + BG_COLOUR[1] * (255 - alpha) / 255),
                int(b * alpha / 255 + BG_COLOUR[2] * (255 - alpha) / 255),
            )
            pygame.draw.line(surface, faded, pts[i-1], pts[i], 1)


# ── main ──────────────────────────────────────────────────────────────────────
def run_demo_from_playlist(
    playlist:    list[tuple[str, float]],
    data_dir:    pathlib.Path,
    threshold:   float,
    fps:         int = FPS_DEFAULT,
    show_trails: bool = True,
):
    import pygame  # imported here so the file is importable without pygame

    pygame.init()
    pygame.display.set_caption("minDrive-JEPA — Surprise Score Demo")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock  = pygame.time.Clock()

    font_big   = pygame.font.SysFont("monospace", 14, bold=True)
    font_small = pygame.font.SysFont("monospace", 11)

    rows = playlist   # list of (scenario_name, score)

    scenario_idx = 0
    frame_idx    = 0
    paused       = False
    tensor_np    = None

    def load_scenario(idx):
        name, score = rows[idx]
        pt = data_dir / f"{name}.pt"
        t  = torch.load(pt, weights_only=True).numpy()
        return t, name, score

    tensor_np, cur_name, cur_score = load_scenario(scenario_idx)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    scenario_idx = (scenario_idx + 1) % len(rows)
                    frame_idx    = 0
                    tensor_np, cur_name, cur_score = load_scenario(scenario_idx)
                elif event.key == pygame.K_LEFT:
                    scenario_idx = (scenario_idx - 1) % len(rows)
                    frame_idx    = 0
                    tensor_np, cur_name, cur_score = load_scenario(scenario_idx)

        # Draw frame
        screen.fill(BG_COLOUR)
        draw_grid(screen, pygame)

        if show_trails:
            draw_trails(screen, pygame, tensor_np, frame_idx)
        draw_scenario_frame(screen, pygame, tensor_np, frame_idx)

        draw_info_bar(screen, pygame, font_big, font_small,
                      cur_name, cur_score, threshold,
                      frame_idx, tensor_np.shape[0],
                      paused, scenario_idx, len(rows))

        pygame.display.flip()
        clock.tick(fps)

        # Advance frame
        if not paused:
            frame_idx += 1
            if frame_idx >= tensor_np.shape[0]:
                # Auto-advance to next scenario
                scenario_idx = (scenario_idx + 1) % len(rows)
                frame_idx    = 0
                tensor_np, cur_name, cur_score = load_scenario(scenario_idx)

    pygame.quit()


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_playlist(
    scores_csv: pathlib.Path,
    top_n: int,
    all_scenarios: bool,
) -> list[tuple[str, float]]:
    """
    Build a playlist of (scenario_name, score) pairs.

    Default (top_n=10):  bottom-5 routine (green) then top-5 surprising (red)
                         so the viewer sees the contrast.
    --top N:             bottom-N/2 routine then top-N/2 surprising.
    --all:               all scenarios ordered low→high (green→red progression).
    """
    rows = []
    with open(scores_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append((row["scenario"], float(row["surprise_score"])))
    # rows is already sorted highest-first (from evaluate.py)

    if all_scenarios:
        return list(reversed(rows))   # low → high

    half = max(1, top_n // 2)
    routine    = list(reversed(rows[-half:]))   # least surprising first
    surprising = rows[:half]                    # most surprising last
    return routine + surprising


def main():
    p = argparse.ArgumentParser(description="minDrive-JEPA Pygame Demo")
    p.add_argument("--top",   type=int, default=10,
                   help="Show N scenarios: bottom-N/2 routine then top-N/2 surprising")
    p.add_argument("--fps",   type=int, default=FPS_DEFAULT, help="Playback FPS")
    p.add_argument("--all",   action="store_true",
                   help="Show all scenarios ordered low→high surprise (green→red)")
    p.add_argument("--no-trails", action="store_true", help="Disable trajectory trails")
    args = p.parse_args()

    root       = pathlib.Path(__file__).resolve().parents[3]
    import yaml
    cfg        = yaml.safe_load((root / "configs/default.yaml").read_text())
    scores_csv = root / "outputs" / "surprise_scores.csv"
    data_dir   = root / cfg["data"]["processed_dir"]

    if not scores_csv.exists():
        print("Run  python scripts/evaluate.py  first.")
        sys.exit(1)

    import numpy as np
    all_scores = []
    with open(scores_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            all_scores.append(float(row["surprise_score"]))
    pct       = cfg.get("evaluation", {}).get("surprise_threshold_percentile", 90)
    threshold = float(np.percentile(all_scores, pct))

    playlist = build_playlist(scores_csv, top_n=args.top, all_scenarios=args.all)
    half     = max(1, args.top // 2)
    print(f"Playlist: {half} routine (green) → {half} surprising (red)  |  "
          f"threshold={threshold:.3f}  |  {args.fps} fps")
    print("Controls: SPACE=pause  →=next  ←=prev  Q=quit")

    run_demo_from_playlist(playlist, data_dir, threshold,
                           fps=args.fps,
                           show_trails=not args.no_trails)


if __name__ == "__main__":
    main()

