import math
import sqlite3
from pathlib import Path
from typing import Dict, List


# Maps nuPlan category names → integer labels
# 0=vehicle, 1=pedestrian, 2=cyclist, 3=other
CATEGORY_TO_LABEL = {
    'vehicle': 0,
    'pedestrian': 1,
    'bicycle': 2,
    'traffic_cone': 3,
    'barrier': 3,
    'czone_sign': 3,
    'generic_object': 3,
}


def _quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Convert quaternion to yaw angle (rotation around Z axis) in radians."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


class NuPlanReader:
    """Reads a single nuPlan mini SQLite log file and returns raw scenario data."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # allows column access by name
        self.cursor = self.conn.cursor()

        # Print actual table names so schema mismatches surface immediately (Patch 6)
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in self.cursor.fetchall()]
        print(f"[NuPlanReader] {Path(self.db_path).name} — tables: {tables}")

        expected = ['lidar_pc', 'ego_pose', 'lidar_box', 'track', 'category', 'scene']
        for t in expected:
            if t not in tables:
                print(f"  WARNING: expected table '{t}' not found — "
                      f"nuplan_reader.py may need adjustments.")

    def get_scenario_tokens(self) -> List[str]:
        """Returns list of hex-encoded scene token strings from this log file."""
        self.cursor.execute("SELECT HEX(token) FROM scene")
        return [row[0] for row in self.cursor.fetchall()]

    def get_ego_trajectory(self, start_ts: int, end_ts: int) -> List[Dict]:
        """
        Returns ego vehicle poses between start_ts and end_ts (microseconds, inclusive).
        Joins lidar_pc → ego_pose to get position, velocity, and orientation.
        Heading is computed from quaternion (qw, qx, qy, qz) → yaw in radians.

        Returns list of dicts: {timestamp, x, y, vx, vy, heading}
        """
        query = """
            SELECT lp.timestamp,
                   ep.x, ep.y, ep.vx, ep.vy,
                   ep.qw, ep.qx, ep.qy, ep.qz
            FROM lidar_pc lp
            JOIN ego_pose ep ON lp.ego_pose_token = ep.token
            WHERE lp.timestamp BETWEEN ? AND ?
            ORDER BY lp.timestamp
        """
        self.cursor.execute(query, (start_ts, end_ts))
        return [
            {
                'timestamp': row['timestamp'],
                'x': row['x'],
                'y': row['y'],
                'vx': row['vx'],
                'vy': row['vy'],
                'heading': _quat_to_yaw(row['qw'], row['qx'], row['qy'], row['qz']),
            }
            for row in self.cursor.fetchall()
        ]

    def get_agent_tracks(self, start_ts: int, end_ts: int) -> List[Dict]:
        """
        Returns all detected agent bounding boxes between start_ts and end_ts (microseconds).
        Joins lidar_box → lidar_pc (for timestamp) → track → category (for label).

        Returns list of dicts: {timestamp, track_token, x, y, vx, vy, heading, label}
        label: 0=vehicle, 1=pedestrian, 2=cyclist, 3=other
        """
        query = """
            SELECT lp.timestamp,
                   HEX(lb.track_token) AS track_token,
                   lb.x, lb.y, lb.vx, lb.vy, lb.yaw,
                   c.name AS category_name
            FROM lidar_box lb
            JOIN lidar_pc lp ON lb.lidar_pc_token = lp.token
            JOIN track t     ON lb.track_token = t.token
            JOIN category c  ON t.category_token = c.token
            WHERE lp.timestamp BETWEEN ? AND ?
            ORDER BY lp.timestamp, lb.track_token
        """
        self.cursor.execute(query, (start_ts, end_ts))
        return [
            {
                'timestamp': row['timestamp'],
                'track_token': row['track_token'],
                'x': row['x'],
                'y': row['y'],
                'vx': row['vx'],
                'vy': row['vy'],
                'heading': row['yaw'],
                'label': CATEGORY_TO_LABEL.get(row['category_name'], 3),
            }
            for row in self.cursor.fetchall()
        ]

    def get_all_scenarios(self, duration_sec: float = 5.0) -> List[Dict]:
        """
        Returns a list of scenario dicts, one per scene in this log file.
        Each scenario covers a duration_sec window starting from the scene's first frame.

        Filters out:
        - Scenes shorter than duration_sec
        - Scenes with fewer than 2 unique agents

        Each dict: {scenario_id, db_path, ego_trajectory, agent_tracks, start_ts, end_ts}
        """
        duration_us = int(duration_sec * 1_000_000)

        self.cursor.execute("SELECT token, HEX(token) AS hex_token FROM scene")
        scenes = self.cursor.fetchall()

        scenarios = []
        for scene in scenes:
            scene_token_bytes = scene['token']
            scenario_id = scene['hex_token']

            # Get timestamp range for frames belonging to this scene
            self.cursor.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM lidar_pc WHERE scene_token = ?",
                (scene_token_bytes,)
            )
            ts_row = self.cursor.fetchone()
            if ts_row[0] is None:
                continue  # no frames for this scene

            min_ts, max_ts = ts_row[0], ts_row[1]

            # Skip scenes shorter than requested duration
            if (max_ts - min_ts) < duration_us:
                continue

            start_ts = min_ts
            end_ts = min_ts + duration_us

            ego_trajectory = self.get_ego_trajectory(start_ts, end_ts)
            if not ego_trajectory:
                continue

            agent_tracks = self.get_agent_tracks(start_ts, end_ts)

            # Skip scenes with fewer than 2 unique agents
            unique_agents = len({t['track_token'] for t in agent_tracks})
            if unique_agents < 2:
                continue

            scenarios.append({
                'scenario_id': scenario_id,
                'db_path': self.db_path,
                'ego_trajectory': ego_trajectory,
                'agent_tracks': agent_tracks,
                'start_ts': start_ts,
                'end_ts': end_ts,
            })

        return scenarios

    def close(self):
        """Close the SQLite connection."""
        self.conn.close()
