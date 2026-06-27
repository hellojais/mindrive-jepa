import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


# Maps NuPlanReader integer labels → agent type float for the tensor feature
# ego=0.0 (set separately), vehicle=1.0, pedestrian=2.0, cyclist=3.0, other=3.0
_LABEL_TO_TYPE_FLOAT = {0: 1.0, 1: 2.0, 2: 3.0, 3: 3.0}


class SceneTokenizer:
    """Converts raw scenario dicts from NuPlanReader into normalized [T, N+1, D] tensors."""

    def __init__(self, config: dict):
        """config: the 'data' section of default.yaml."""
        self.sequence_len = config['sequence_len']            # 50
        self.max_agents = config['max_agents']                # 20
        self.agent_feat_dim = config['agent_feat_dim']        # 6
        self.position_scale_m = config['position_scale_m']   # 50.0
        self.velocity_scale_ms = config['velocity_scale_ms'] # 10.0

    def tokenize_scenario(self, scenario: dict) -> torch.Tensor:
        """
        Converts one scenario dict into a float32 tensor of shape [T, N+1, D].
          T   = sequence_len (50 timesteps at 10 Hz)
          N+1 = max_agents + 1 (row 0 = ego, rows 1..N = other agents)
          D   = agent_feat_dim (6: x, y, vx, vy, heading, agent_type)

        Normalization:
          positions  = (world_pos - ego_pos_at_t0) / position_scale_m
          velocities = raw_velocity / velocity_scale_ms
          heading    = raw yaw in radians, unchanged
          agent_type = ego=0.0, vehicle=1.0, pedestrian=2.0, cyclist/other=3.0
        """
        ego_frames = scenario['ego_trajectory']   # ~100 dicts at 20 Hz
        agent_tracks = scenario['agent_tracks']   # all agent detections in window

        # Step 1: Resample ego trajectory to exactly sequence_len timesteps
        # linspace picks evenly-spaced indices → robust to any number of raw frames
        n_frames = len(ego_frames)
        indices = np.linspace(0, n_frames - 1, self.sequence_len, dtype=int)
        sampled_ego = [ego_frames[i] for i in indices]

        # Step 2: Ego-centric origin — ego's world position at t=0
        ego_x0 = sampled_ego[0]['x']
        ego_y0 = sampled_ego[0]['y']

        # Step 3: Group agent detections by timestamp for O(1) lookup per frame
        agents_by_ts: Dict[int, List[dict]] = defaultdict(list)
        for track in agent_tracks:
            agents_by_ts[track['timestamp']].append(track)

        # At t=0, rank all agents by distance from ego origin; keep closest max_agents
        ts0 = sampled_ego[0]['timestamp']
        agents_at_t0 = agents_by_ts.get(ts0, [])
        agents_at_t0_sorted = sorted(
            agents_at_t0,
            key=lambda a: (a['x'] - ego_x0) ** 2 + (a['y'] - ego_y0) ** 2,
        )
        selected_tokens = [a['track_token'] for a in agents_at_t0_sorted[:self.max_agents]]

        # Cache type float per track token (read from first appearance at t=0)
        token_to_type: Dict[str, float] = {
            a['track_token']: _LABEL_TO_TYPE_FLOAT.get(a['label'], 3.0)
            for a in agents_at_t0_sorted
        }

        # Step 4: Build tensor — allocate zeros, then fill frame by frame
        T  = self.sequence_len
        N1 = self.max_agents + 1  # +1 for ego
        D  = self.agent_feat_dim
        tensor = np.zeros((T, N1, D), dtype=np.float32)

        for t_idx, ego_frame in enumerate(sampled_ego):
            ts = ego_frame['timestamp']

            # Row 0: ego vehicle
            tensor[t_idx, 0, 0] = (ego_frame['x'] - ego_x0) / self.position_scale_m
            tensor[t_idx, 0, 1] = (ego_frame['y'] - ego_y0) / self.position_scale_m
            tensor[t_idx, 0, 2] = ego_frame['vx'] / self.velocity_scale_ms
            tensor[t_idx, 0, 3] = ego_frame['vy'] / self.velocity_scale_ms
            tensor[t_idx, 0, 4] = ego_frame['heading']
            tensor[t_idx, 0, 5] = 0.0  # ego type

            # Rows 1..max_agents: selected agents (zeros if not detected this frame)
            agents_at_ts = {a['track_token']: a for a in agents_by_ts.get(ts, [])}
            for a_idx, token in enumerate(selected_tokens):
                if token not in agents_at_ts:
                    continue  # agent not detected this frame → stays zeros
                a = agents_at_ts[token]
                tensor[t_idx, a_idx + 1, 0] = (a['x'] - ego_x0) / self.position_scale_m
                tensor[t_idx, a_idx + 1, 1] = (a['y'] - ego_y0) / self.position_scale_m
                tensor[t_idx, a_idx + 1, 2] = a['vx'] / self.velocity_scale_ms
                tensor[t_idx, a_idx + 1, 3] = a['vy'] / self.velocity_scale_ms
                tensor[t_idx, a_idx + 1, 4] = a['heading']
                tensor[t_idx, a_idx + 1, 5] = token_to_type.get(token, 3.0)

        return torch.from_numpy(tensor)  # [T, N+1, D], float32

    def tokenize_dataset(self, scenarios: List[dict]) -> tuple:
        """
        Tokenizes a list of scenario dicts with a tqdm progress bar.
        Skips and warns on individual failures.

        Returns:
            tensors  : List[Tensor]  — [T, N+1, D] per scenario
            metadata : List[dict]   — {scenario_id, db_path} per tensor (same order)
        """
        tensors  = []
        metadata = []
        for scenario in tqdm(scenarios, desc="Tokenizing scenarios"):
            try:
                tensors.append(self.tokenize_scenario(scenario))
                metadata.append({
                    'scenario_id': scenario.get('scenario_id', ''),
                    'db_path':     scenario.get('db_path', ''),
                })
            except Exception as e:
                sid = scenario.get('scenario_id', '?')[:12]
                print(f"  WARNING: skipping scenario {sid}: {e}")
        return tensors, metadata

    def save_processed(
        self,
        tensors:    List[torch.Tensor],
        output_dir: str,
        metadata:   List[dict] | None = None,
    ):
        """
        Saves each tensor as output_dir/scenario_NNNNN.pt.
        Writes manifest.json with count, shape, and per-scenario source metadata.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for i, tensor in enumerate(tensors):
            torch.save(tensor, out_path / f"scenario_{i:05d}.pt")

        manifest = {
            'count':     len(tensors),
            'shape':     list(tensors[0].shape) if tensors else [],
            'dtype':     str(tensors[0].dtype) if tensors else 'float32',
            'scenarios': [
                {
                    'file':        f"scenario_{i:05d}.pt",
                    'scenario_id': (metadata[i]['scenario_id'] if metadata else ''),
                    'db_path':     (metadata[i]['db_path']     if metadata else ''),
                }
                for i in range(len(tensors))
            ],
        }
        with open(out_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Saved {len(tensors)} tensors to {output_dir}  shape={manifest['shape']}")

    def load_processed(self, input_dir: str) -> List[torch.Tensor]:
        """Loads all scenario_*.pt files from input_dir in sorted order."""
        in_path = Path(input_dir)
        pt_files = sorted(in_path.glob('scenario_*.pt'))
        if not pt_files:
            raise FileNotFoundError(f"No scenario_*.pt files found in {input_dir}")
        tensors = [torch.load(f, weights_only=True) for f in pt_files]
        print(f"Loaded {len(tensors)} tensors from {input_dir}")
        return tensors
